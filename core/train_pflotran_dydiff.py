import os
import pytorch_lightning as pl
from datasets.pflotran.pflotran_datamodule import PFLOTRANDataModuleForDyDiff
from logger.logger_turbulence import ImageLoggerWithKeyToConcat

import argparse
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from pytorch_lightning.callbacks import ModelCheckpoint


def create_data_module(cfg):
    data_module = PFLOTRANDataModuleForDyDiff(cfg.data, cfg.training.batch_size, cfg.training.num_workers)
    return data_module

def create_model(cfg):
    latent_channels = int(getattr(cfg.model.params, "z_channels", 3))
    cfg.model.params.total_length = cfg.data.total_length
    cfg.model.params.input_length = cfg.data.input_length
    cfg.model.params.num_vis = cfg.eval.num_vis
    cfg.model.params.validation_save_dir = cfg.training.logger.save_dir

    cfg.model.params.channels = cfg.data.total_length - cfg.data.input_length
    cfg.model.params.unet_config.params.in_channels += cfg.data.total_length
    cfg.model.params.unet_config.params.out_channels = cfg.data.total_length - cfg.data.input_length
    
    if "ensemble" in cfg.model.params.keys() and cfg.model.params.ensemble == True:
        # cfg.model.params.channels *= 2
        cfg.model.params.unet_config.params.in_channels += cfg.model.params.unet_config.params.out_channels

    cfg.model.params.channels *= latent_channels
    cfg.model.params.unet_config.params.in_channels *= latent_channels
    cfg.model.params.unet_config.params.out_channels *= latent_channels
    if cfg.model.params.conditioning_key == "mcvd":
        cfg.model.params.unet_config.params.cond_channels *= latent_channels
        
    if cfg.model.params.conditioning_key == "concat-video":
        cfg.model.params.unet_config.params.in_channels = (cfg.data.input_length + 1) * latent_channels
        cfg.model.params.unet_config.params.out_channels = latent_channels
        cfg.model.params.unet_config.params.num_video_frames = cfg.data.total_length - cfg.data.input_length
    elif cfg.model.params.conditioning_key.startswith("concat-video-mask"):
        '''
        cfg.model.params.unet_config.params.in_channels = 3 * 2 + 1 # 3 * 2 for video&concat, 1 for mask
        if "1st" in cfg.model.params.conditioning_key:
            cfg.model.params.unet_config.params.in_channels += 1
        '''
        #Fang
        base = latent_channels * 2 + 1
        if "1st" in cfg.model.params.conditioning_key:
            base += 1
        base += cfg.model.params.static_latent_channels
        cfg.model.params.unet_config.params.in_channels = base

        cfg.model.params.unet_config.params.out_channels = latent_channels
        cfg.model.params.unet_config.params.num_video_frames = cfg.data.total_length
    
    cfg.model.params.ckpt_path = cfg.ckpt_path

    model = instantiate_from_config(cfg.model)
    for k, v in cfg.training.model_attrs.items():
        setattr(model, k, v)
    return model

def create_loggers(cfg):
    image_logger = ImageLoggerWithKeyToConcat(
        batch_frequency=cfg.training.logger.logger_freq,
        save_dir=cfg.training.logger.save_dir,
        keys_to_concat=["inputs", "samples"],
        log_images_kwargs=dict(cfg.model.params.validate_kwargs)
    )
    checkpoint_logger = ModelCheckpoint(
        dirpath=cfg.training.logger.save_dir,
        every_n_epochs=cfg.training.logger.checkpoint_freq_epochs if 'checkpoint_freq_epochs' in cfg.training.logger else 1,
        every_n_train_steps=cfg.training.logger.checkpoint_freq if 'checkpoint_freq' in cfg.training.logger else 0,
        save_top_k=-1
    )
    return [image_logger, checkpoint_logger]


if __name__=='__main__':
    # Parse arguments
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs (1=single GPU, >1=DDP multi-GPU)')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--model_root', type=str)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument("--test", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument('--find_unused_parameters', type=str2bool, nargs="?", const=True, default=False,
                        help='Enable find_unused_parameters for DDP (use if you get DDP errors)')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, vars(args))
    config_file_name = os.path.basename(args.config_file)
    config_file_name = config_file_name.split('.')[0]

    cfg.training.logger.save_dir = os.path.join(cfg.training.logger.save_dir, config_file_name)
    
    # Optimize num_workers for multi-GPU (reduce overhead)
    if args.n_gpu > 1 and cfg.training.num_workers > 2:
        original_workers = cfg.training.num_workers
        cfg.training.num_workers = min(2, cfg.training.num_workers)
        print(f"Multi-GPU mode: Reduced num_workers from {original_workers} to {cfg.training.num_workers}")
    
    print("=" * 80)
    print(f"Training Configuration:")
    print(f"  Mode: {'DDP Multi-GPU' if args.n_gpu > 1 else 'Single GPU'}")
    print(f"  GPUs: {args.n_gpu}")
    print(f"  Batch size per GPU: {cfg.training.batch_size}")
    print(f"  Effective batch size: {cfg.training.batch_size * args.n_gpu * cfg.training.accumulate_grad_batches}")
    print(f"  Num workers: {cfg.training.num_workers}")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    
    # model
    model = create_model(cfg)

    # data module
    data_module = create_data_module(cfg)

    # loggers
    loggers = create_loggers(cfg)

    # Configure trainer kwargs
    trainer_kwargs = dict(
        accelerator="gpu",
        devices=args.n_gpu,
        precision=16,
        callbacks=loggers,
        default_root_dir=cfg.training.logger.save_dir,
        max_steps=cfg.training.max_iterations,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=int(cfg.training.validation_freq) if cfg.training.validation_freq is not None else float(1.),
        num_sanity_val_steps=0,
        sync_batchnorm=True if args.n_gpu > 1 else False,  # Sync batch norm across GPUs
        gradient_clip_val=cfg.training.gradient_clip_val if 'gradient_clip_val' in cfg.training else None,  # Optional gradient clipping
    )
    
    # Configure DDP strategy if using multiple GPUs
    if args.n_gpu > 1:
        from pytorch_lightning.strategies import DDPStrategy
        trainer_kwargs['strategy'] = DDPStrategy(
            find_unused_parameters=args.find_unused_parameters,
            gradient_as_bucket_view=True  # More efficient gradient syncing
        )
        print(f"Using DDP strategy with find_unused_parameters={args.find_unused_parameters}")
    else:
        print("Using single GPU strategy (auto)")

    # trainer
    trainer = pl.Trainer(**trainer_kwargs)
    # Train!
    if not args.test:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=args.resume)


#python train_pflotran_dydiff.py \
#  --config_file models/pflotran/diff_pflotran_static.yaml

