"""
Evaluate PFLOTRAN diffusion model checkpoints and generate predictions.
This script generates ensemble predictions on the test set for visualization and metric computation.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from pathlib import Path

# Add core to path
import sys
sys.path.insert(0, 'core')

from datasets.pflotran.pflotran import PFLOTRANDatasetConcStatic
from ldm.util import instantiate_from_config


def load_model_from_checkpoint(config_path, checkpoint_path):
    """Load trained model from checkpoint."""
    cfg = OmegaConf.load(config_path)
    
    # Build model config (same as train script)
    latent_channels = int(getattr(cfg.model.params, "z_channels", 3))
    cfg.model.params.total_length = cfg.data.total_length
    cfg.model.params.input_length = cfg.data.input_length
    cfg.model.params.num_vis = cfg.eval.num_vis
    cfg.model.params.validation_save_dir = cfg.training.logger.save_dir
    cfg.model.params.channels = cfg.data.total_length - cfg.data.input_length
    cfg.model.params.unet_config.params.in_channels += cfg.data.total_length
    cfg.model.params.unet_config.params.out_channels = cfg.data.total_length - cfg.data.input_length
    
    cfg.model.params.channels *= latent_channels
    cfg.model.params.unet_config.params.in_channels *= latent_channels
    cfg.model.params.unet_config.params.out_channels *= latent_channels
    
    if cfg.model.params.conditioning_key.startswith("concat-video-mask"):
        base = latent_channels * 2 + 1
        if "1st" in cfg.model.params.conditioning_key:
            base += 1
        base += cfg.model.params.static_latent_channels
        cfg.model.params.unet_config.params.in_channels = base
        cfg.model.params.unet_config.params.out_channels = latent_channels
        cfg.model.params.unet_config.params.num_video_frames = cfg.data.total_length
    
    # Create model
    model = instantiate_from_config(cfg.model)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.cuda()
    
    return model, cfg


def generate_predictions(model, cfg, num_samples=50, num_test_cases=10, output_dir=None):
    """Generate predictions on test set."""
    
    if output_dir is None:
        output_dir = f"logs/custom_pflotran/pflotran_identity/predictions_step{model.global_step}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating predictions...")
    print(f"  Number of ensemble samples: {num_samples}")
    print(f"  Number of test cases: {num_test_cases}")
    print(f"  Output directory: {output_dir}")
    
    # Load test data
    test_idx = np.load(cfg.data.test_idx_path)
    test_dataset = PFLOTRANDatasetConcStatic(
        perm_path=cfg.data.perm_path,
        poro_path=cfg.data.poro_path,
        conc_path=cfg.data.conc_path,
        indices=test_idx,
        input_length=cfg.data.input_length,
        pred_length=cfg.data.pred_length,
        t_keep=cfg.data.t_keep,
        dt=getattr(cfg.data, "dt", 1),
        mmap=True,
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Generate predictions
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_test_cases:
                break
                
            print(f"\n  Processing test case {batch_idx + 1}/{num_test_cases}...")
            
            # Move batch to GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()
            
            # Get ground truth
            z, z_prev, cond_list, x_total, *_ = model.get_input(
                batch, model.first_stage_key, model.first_stage_key_prev, log_mode=True
            )
            
            # Save ground truth (in normalized log space, as model outputs)
            x_total_np = x_total.view(-1, cfg.data.total_length, 1, 40, 28).cpu().numpy()
            np.save(os.path.join(output_dir, f'gt_{batch_idx}.npy'), x_total_np)
            
            # Generate ensemble predictions
            predictions = []
            for sidx in range(num_samples):
                if (sidx + 1) % 10 == 0:
                    print(f"    Sample {sidx + 1}/{num_samples}")
                
                with model.ema_scope():
                    z_sample, _ = model.sample_log(
                        x_prev=z_prev, 
                        cond=cond_list, 
                        batch_size=z.shape[0],
                        **cfg.model.params.validate_kwargs
                    )
                
                x_sample = model.batched_decode(z_sample)
                x_sample_np = x_sample.view(-1, cfg.data.total_length, 1, 40, 28).cpu().numpy()
                
                # Save individual sample
                np.save(os.path.join(output_dir, f'pred_{batch_idx}_sample_{sidx}.npy'), x_sample_np)
                predictions.append(x_sample_np)
            
            print(f"    ✓ Saved {num_samples} predictions for case {batch_idx}")
    
    print(f"\n✓ All predictions saved to: {output_dir}")
    return output_dir


def visualize_predictions(output_dir, cfg, num_vis_cases=5):
    """Create quick visualizations of predictions."""
    
    print(f"\nCreating visualizations...")
    
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    input_length = cfg.data.input_length
    
    for case_idx in range(num_vis_cases):
        gt_path = os.path.join(output_dir, f'gt_{case_idx}.npy')
        if not os.path.exists(gt_path):
            continue
        
        gt = np.load(gt_path)[0]  # Shape: (total_length, 1, H, W)
        
        # Load all prediction samples for this case
        pred_files = sorted([f for f in os.listdir(output_dir) if f.startswith(f'pred_{case_idx}_sample_')])
        if len(pred_files) == 0:
            continue
        
        preds = np.stack([np.load(os.path.join(output_dir, f))[0] for f in pred_files], axis=0)
        # preds: (num_samples, total_length, 1, H, W)
        
        # Inverse transform: normalized log1p -> concentration
        # Note: This assumes data is in log space. If you have stats file, use it.
        gt_conc = np.expm1(gt[:, 0])  # (total_length, H, W)
        preds_conc = np.expm1(preds[:, :, 0])  # (num_samples, total_length, H, W)
        
        # Compute mean and std
        pred_mean = preds_conc.mean(axis=0)  # (total_length, H, W)
        pred_std = preds_conc.std(axis=0)
        
        # Select timesteps to visualize
        vis_times = [input_length, input_length + 5, input_length + 10, input_length + 15, input_length + 19]
        vis_times = [t for t in vis_times if t < gt_conc.shape[0]]
        
        fig, axes = plt.subplots(3, len(vis_times), figsize=(4*len(vis_times), 10))
        
        for idx, t in enumerate(vis_times):
            vmax = max(gt_conc[t].max(), pred_mean[t].max())
            
            # Ground truth
            im0 = axes[0, idx].imshow(gt_conc[t], cmap='viridis', vmin=0, vmax=vmax)
            axes[0, idx].set_title(f't={t} (Day {t})\nGround Truth')
            axes[0, idx].axis('off')
            plt.colorbar(im0, ax=axes[0, idx], fraction=0.046)
            
            # Prediction mean
            im1 = axes[1, idx].imshow(pred_mean[t], cmap='viridis', vmin=0, vmax=vmax)
            axes[1, idx].set_title(f'Prediction Mean')
            axes[1, idx].axis('off')
            plt.colorbar(im1, ax=axes[1, idx], fraction=0.046)
            
            # Prediction std
            im2 = axes[2, idx].imshow(pred_std[t], cmap='hot', vmin=0)
            axes[2, idx].set_title(f'Prediction Std')
            axes[2, idx].axis('off')
            plt.colorbar(im2, ax=axes[2, idx], fraction=0.046)
        
        plt.suptitle(f'Test Case {case_idx} - Concentration (mol/L)', fontsize=14, y=0.995)
        plt.tight_layout()
        
        save_path = os.path.join(vis_dir, f'case_{case_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved visualization: {save_path}")
    
    print(f"\n✓ Visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PFLOTRAN diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='core/models/pflotran_identity.yaml', 
                        help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=50, 
                        help='Number of ensemble samples per test case')
    parser.add_argument('--num_cases', type=int, default=10, 
                        help='Number of test cases to evaluate')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for predictions')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualizations after predictions')
    parser.add_argument('--num_vis', type=int, default=5, 
                        help='Number of cases to visualize')
    
    args = parser.parse_args()
    
    # Load model
    model, cfg = load_model_from_checkpoint(args.config, args.checkpoint)
    
    # Generate predictions
    output_dir = generate_predictions(
        model, cfg, 
        num_samples=args.num_samples, 
        num_test_cases=args.num_cases,
        output_dir=args.output_dir
    )
    
    # Visualize
    if args.visualize:
        visualize_predictions(output_dir, cfg, num_vis_cases=args.num_vis)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    print(f"\nPredictions: {output_dir}")
    if args.visualize:
        print(f"Visualizations: {output_dir}/visualizations/")
    print("\nNext steps:")
    print("  - View visualizations")
    print("  - Run metric evaluation: python core/evaluation/plot_pflotran_metrics.py \\")
    print(f"      --model_output {output_dir} \\")
    print(f"      --stats_path <path_to_stats.npy>")
    print("")


if __name__ == "__main__":
    main()
