#!/usr/bin/env python3
"""
Evaluate PFLOTRAN diffusion model from a checkpoint.
Runs inference on validation/test data and saves predictions.

Usage:
    python core/eval_pflotran_checkpoint.py \
        --config core/models/pflotran_identity.yaml \
        --checkpoint logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/epoch=X-step=XXXX.ckpt \
        --num_samples 5 \
        --output_dir ./evaluation_results \
        --split val
"""

import argparse
import os
import yaml
import numpy as np
import torch
from tqdm import tqdm

from dydiff.dydiff_pflotran import DynamicalLDMForPFLOTRANWithStaticCondition
from datasets.pflotran.pflotran_datamodule import PFLOTRANDataModule


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_checkpoint(checkpoint_path, config):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model with config params
    model_config = config['model']['params']
    model = DynamicalLDMForPFLOTRANWithStaticCondition.load_from_checkpoint(
        checkpoint_path,
        strict=False,  # In case some params changed
        **model_config
    )
    
    model.eval()
    model.cuda()
    
    return model


def run_inference(model, dataloader, num_samples=5, output_dir='./outputs'):
    """
    Run inference on dataset and save predictions.
    
    Args:
        model: Trained diffusion model
        dataloader: DataLoader for val/test set
        num_samples: Number of ensemble samples per input
        output_dir: Directory to save predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRunning inference with {num_samples} samples per case...")
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to GPU
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
            
            # Ground truth
            gt = batch['image']  # (B, T, C, H, W)
            
            # Generate multiple samples
            samples = []
            for sample_idx in range(num_samples):
                # Run DDIM sampling
                pred = model.sample(
                    batch,
                    ddim_steps=50,
                    ddim_eta=0.0,
                    return_intermediates=False
                )
                samples.append(pred.cpu().numpy())
            
            # Stack samples: (num_samples, B, T, C, H, W)
            samples = np.stack(samples, axis=0)
            
            # Save individual predictions
            batch_size = gt.shape[0]
            for i in range(batch_size):
                case_id = batch_idx * batch_size + i
                
                # Save ground truth (only once)
                gt_path = os.path.join(output_dir, f"gt_{case_id}.npy")
                if not os.path.exists(gt_path):
                    np.save(gt_path, gt[i:i+1].cpu().numpy())
                
                # Save each sample
                for sample_idx in range(num_samples):
                    pred_path = os.path.join(output_dir, f"pred_{case_id}_sample_{sample_idx}.npy")
                    np.save(pred_path, samples[sample_idx, i:i+1])
                
                all_ground_truth.append(gt[i:i+1].cpu().numpy())
                all_predictions.append(samples[:, i:i+1])  # (num_samples, 1, T, C, H, W)
    
    print(f"\nSaved predictions to: {output_dir}")
    print(f"Total cases evaluated: {len(all_ground_truth)}")
    
    return all_predictions, all_ground_truth


def compute_metrics(predictions, ground_truth):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of arrays, each (num_samples, 1, T, C, H, W)
        ground_truth: List of arrays, each (1, T, C, H, W)
    """
    print("\nComputing metrics...")
    
    mse_list = []
    mae_list = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Ensemble mean
        pred_mean = pred.mean(axis=0)  # (1, T, C, H, W)
        
        # MSE and MAE
        mse = np.mean((pred_mean - gt) ** 2)
        mae = np.mean(np.abs(pred_mean - gt))
        
        mse_list.append(mse)
        mae_list.append(mae)
    
    metrics = {
        'MSE': np.mean(mse_list),
        'MAE': np.mean(mae_list),
        'RMSE': np.sqrt(np.mean(mse_list))
    }
    
    print("\nMetrics (ensemble mean):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PFLOTRAN diffusion model from checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of ensemble samples")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Update output directory with checkpoint info
    ckpt_name = os.path.basename(args.checkpoint).replace('.ckpt', '')
    args.output_dir = os.path.join(args.output_dir, ckpt_name, args.split)
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config)
    
    # Setup datamodule
    print("\nSetting up data...")
    datamodule = PFLOTRANDataModule(
        perm_path=config['data']['perm_path'],
        poro_path=config['data']['poro_path'],
        conc_path=config['data']['conc_path'],
        train_idx_path=config['data']['train_idx_path'],
        val_idx_path=config['data']['val_idx_path'],
        test_idx_path=config['data']['test_idx_path'],
        input_length=config['data']['input_length'],
        pred_length=config['data']['pred_length'],
        total_length=config['data']['total_length'],
        t_keep=config['data']['t_keep'],
        dt=config['data']['dt'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False  # No shuffling for evaluation
    )
    
    datamodule.setup()
    
    # Get appropriate dataloader
    if args.split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()
    
    print(f"Evaluating on {args.split} set ({len(dataloader.dataset)} samples)")
    
    # Run inference
    predictions, ground_truth = run_inference(
        model, 
        dataloader, 
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.npy")
    np.save(metrics_path, metrics)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print("\n✓ Evaluation complete!")
    print(f"\nTo visualize results, run:")
    print(f"  python core/evaluation/plot_pflotran_metrics.py --input_dir {args.output_dir}")


if __name__ == "__main__":
    main()
