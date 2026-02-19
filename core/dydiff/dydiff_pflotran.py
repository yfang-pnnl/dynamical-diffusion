import os
import torch
import torch.nn.functional as F
import torchmetrics
import numpy as np

from dydiff.dydiff import DynamicalLDMCleanedWithEncoderCondition, DynamicalLatentDiffusion

class DynamicalLDMForPFLOTRANWithStaticCondition(DynamicalLDMCleanedWithEncoderCondition):
    def __init__(self, *args,
                 total_length=40,
                 input_length=20,
                 num_vis=0,
                 validation_save_dir="",
                 validate_kwargs={},
                 unconditional_guidance_scale=1.,
                 static_latent_channels=8,   # channels passed into UNet as static maps
                 static_downsample=2,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.input_length = input_length
        self.total_length = total_length
        self.num_vis = num_vis
        self.validation_save_dir = validation_save_dir
        self.validate_kwargs = validate_kwargs
        self.unconditional_guidance_scale = unconditional_guidance_scale

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()

        # Map (perm, poro) -> static maps that match the latent resolution.
        # Use static_downsample=2 for VAE (ch_mult=[1,2]) or 1 for Identity.
        self.static_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, kernel_size=3, stride=static_downsample, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, static_latent_channels, kernel_size=3, stride=1, padding=1),
        )

    @torch.no_grad()
    def get_input(self, batch, k, k_prev, log_mode=False):
        # cond sequence (known frames)
        x_c = super(DynamicalLatentDiffusion, self).get_input(batch, self.cond_stage_key).to(self.device)

        # future frames (target sequence)
        x = super(DynamicalLatentDiffusion, self).get_input(batch, k).to(self.device)

        # prev (DyDiff uses it)
        x_prev = super(DynamicalLatentDiffusion, self).get_input(batch, k_prev).to(self.device)

        # static maps
        static = batch["static"].to(self.device)  # (B,2,H,W)

        # inpainting-style: concatenate cond+future into one total sequence before encoding
        if self.model.conditioning_key.startswith("concat-video-mask"):
            x_total = torch.cat([x_c, x], dim=1)  # (B, (P+S)*1, H, W)
        else:
            x_total = x

        # encode sequences to latent
        z = self.batched_encode(x_total)
        z_prev = self.batched_encode(x_prev)
        z_c = self.batched_encode(x_c)

        # encode static to latent resolution and pass as extra c_concat element
        z_static = self.static_encoder(static)
        if z_static.shape[-2:] != z_c.shape[-2:]:
            z_static = F.interpolate(z_static, size=z_c.shape[-2:], mode="bilinear", align_corners=False)

        out = [z, z_prev, [z_c, z_static]]

        if log_mode:
            x_rec = self.batched_decode(z)
            x_prev_rec = self.batched_decode(z_prev)
            x_c_rec = self.batched_decode(z_c)
            out.extend([x_total, x_prev, x_rec, x_prev_rec, x_c, x_c_rec, static])

        return out

    @torch.no_grad()
    def log_images(self, batch, ddim_steps=50, ddim_eta=0., use_ema_scope=True, **kwargs):
        """Override parent to handle extra 'static' return value."""
        from contextlib import nullcontext
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, z_prev, cond_list, x, x_prev, x_rec, x_prev_rec, x_c, x_c_rec, static = \
            self.get_input(batch, self.first_stage_key, self.first_stage_key_prev, log_mode=True)
        
        log["inputs"] = x
        log["prev"] = x_prev
        log["cond"] = x_c
        log["inputs_rec"] = x_rec
        log["prev_rec"] = x_prev_rec
        log["cond_rec"] = x_c_rec

        # get denoise row
        with ema_scope("Sampling"):
            samples, (intermediates, pred_x0) = self.sample_log(
                x_prev=z_prev, cond=cond_list, batch_size=z.shape[0], 
                ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta
            )
        x_samples = self.batched_decode(samples)
        log["samples"] = x_samples
        return log

    def validation_step(self, batch, batch_idx):
        z, z_prev, cond_list, x_total, x_prev, x_rec, x_prev_rec, x_c, x_c_rec, static = \
            self.get_input(batch, self.first_stage_key, self.first_stage_key_prev, log_mode=True)

        with self.ema_scope():
            z_sample, _ = self.sample_log(
                x_prev=z_prev, cond=cond_list, batch_size=z.shape[0],
                unconditional_guidance_scale=self.unconditional_guidance_scale,
                unconditional_conditioning=None,
                **self.validate_kwargs
            )
        x_sample = self.batched_decode(z_sample)

        # compare in log space by default (consistent with training); evaluation script will expm1
        self.valid_mse(x_sample, x_total)
        self.valid_mae(x_sample, x_total)

    def test_step(self, batch, batch_idx):
        z, z_prev, cond_list, x_total, *_ = self.get_input(batch, self.first_stage_key, self.first_stage_key_prev, log_mode=True)

        with self.ema_scope():
            z_sample, _ = self.sample_log(x_prev=z_prev, cond=cond_list, batch_size=z.shape[0], **self.validate_kwargs)
        x_sample = self.batched_decode(z_sample)

        # save gt/preds like turbulence
        save_root = os.path.join(self.validation_save_dir, f'output_for_evaluation_{self.global_step}')
        os.makedirs(save_root, exist_ok=True)

        x_gather = self.concat_all_gather(x_total)
        if self.trainer.is_global_zero:
            # reshape: (-1, total_length, x_channels=1, H=40, W=28)
            x_gather = x_gather.view(-1, self.total_length, self.x_channels, 40, 28).cpu().numpy()
            np.save(os.path.join(save_root, f'gt_{batch_idx}.npy'), x_gather)

        # number of stochastic samples
        num_samples = 2  # Ultra-fast for testing (increase to 10-50 for final evaluation)
        for sidx in range(num_samples):
            with self.ema_scope():
                z_i, _ = self.sample_log(x_prev=z_prev, cond=cond_list, batch_size=z.shape[0], **self.validate_kwargs)
            x_i = self.batched_decode(z_i)
            x_i_gather = self.concat_all_gather(x_i)
            if self.trainer.is_global_zero:
                x_i_gather = x_i_gather.view(-1, self.total_length, self.x_channels, 40, 28).cpu().numpy()
                np.save(os.path.join(save_root, f'pred_{batch_idx}_sample_{sidx}.npy'), x_i_gather)

    def concat_all_gather(self, x):
        """Gather tensors from all GPUs (for DDP). For single GPU, returns x as-is."""
        x = self.all_gather(x)
        if isinstance(x, torch.Tensor):
            return x
        # For multi-GPU, concatenate tensors from different GPUs
        x = torch.concat([_.to(self.device) for _ in x])
        return x
