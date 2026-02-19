"""
Simple visualization script for PFLOTRAN predictions.
Loads saved predictions and creates publication-quality figures.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_case(output_dir, case_id):
    """Load ground truth and predictions for a specific case."""
    gt_path = os.path.join(output_dir, f'gt_{case_id}.npy')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    
    gt = np.load(gt_path)[0]  # Shape: (total_length, C, H, W)
    
    # Load all prediction samples
    pred_files = sorted([f for f in os.listdir(output_dir) if f.startswith(f'pred_{case_id}_sample_')])
    if len(pred_files) == 0:
        raise FileNotFoundError(f"No predictions found for case {case_id}")
    
    preds = np.stack([np.load(os.path.join(output_dir, f))[0] for f in pred_files], axis=0)
    
    return gt, preds


def visualize_case(gt, preds, input_length=20, case_id=0, save_path=None):
    """
    Create visualization for one test case.
    
    Args:
        gt: (total_length, C, H, W) ground truth in log space  
        preds: (num_samples, total_length, C, H, W) predictions in log space
        input_length: number of conditioning frames
        case_id: case identifier
        save_path: where to save figure (if None, displays instead)
    """
    # Inverse transform to physical concentration
    gt_conc = np.expm1(gt[:, 0])  # (T, H, W)
    preds_conc = np.expm1(preds[:, :, 0])  # (K, T, H, W)
    
    # Compute statistics
    pred_mean = preds_conc.mean(axis=0)  # (T, H, W)
    pred_std = preds_conc.std(axis=0)
    pred_min = preds_conc.min(axis=0)
    pred_max = preds_conc.max(axis=0)
    
    # Select timesteps to visualize (input + future)
    total_length = gt_conc.shape[0]
    vis_times = [
        input_length - 1,  # Last input frame
        input_length,      # First prediction
        input_length + 5,
        input_length + 10,
        input_length + 15,
        total_length - 1   # Last prediction
    ]
    vis_times = [t for t in vis_times if 0 <= t < total_length]
    
    # Create figure
    fig, axes = plt.subplots(4, len(vis_times), figsize=(4*len(vis_times), 14))
    
    for idx, t in enumerate(vis_times):
        vmax = max(gt_conc[t].max(), pred_mean[t].max()) * 1.1
        
        # Row 1: Ground Truth
        im0 = axes[0, idx].imshow(gt_conc[t], cmap='viridis', vmin=0, vmax=vmax)
        if t < input_length:
            title = f'Day {t}\n(Input)'
        else:
            title = f'Day {t}\n(Pred +{t-input_length})'
        axes[0, idx].set_title(title, fontsize=12)
        axes[0, idx].axis('off')
        plt.colorbar(im0, ax=axes[0, idx], fraction=0.046, pad=0.04)
        
        # Row 2: Prediction Mean
        im1 = axes[1, idx].imshow(pred_mean[t], cmap='viridis', vmin=0, vmax=vmax)
        axes[1, idx].axis('off')
        plt.colorbar(im1, ax=axes[1, idx], fraction=0.046, pad=0.04)
        
        # Row 3: Prediction Std (Uncertainty)
        im2 = axes[2, idx].imshow(pred_std[t], cmap='hot', vmin=0)
        axes[2, idx].axis('off')
        plt.colorbar(im2, ax=axes[2, idx], fraction=0.046, pad=0.04)
        
        # Row 4: Absolute Error
        abs_error = np.abs(pred_mean[t] - gt_conc[t])
        im3 = axes[3, idx].imshow(abs_error, cmap='Reds', vmin=0, vmax=vmax*0.5)
        axes[3, idx].axis('off')
        plt.colorbar(im3, ax=axes[3, idx], fraction=0.046, pad=0.04)
    
    # Row labels
    row_labels = ['Ground Truth', 'Prediction Mean', 'Uncertainty (Std)', 'Absolute Error']
    for ax, label in zip(axes[:, 0], row_labels):
        ax.text(-0.15, 0.5, label, transform=ax.transAxes, 
                fontsize=13, fontweight='bold', va='center', ha='right', rotation=90)
    
    plt.suptitle(f'Test Case {case_id} - Concentration Prediction (mol/L)\n'
                 f'{preds.shape[0]} Ensemble Members', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def create_ensemble_comparison(gt, preds, t_idx, input_length=20, save_path=None):
    """
    Show multiple ensemble members side-by-side for one timestep.
    """
    gt_conc = np.expm1(gt[t_idx, 0])
    preds_conc = np.expm1(preds[:, t_idx, 0])  # (K, H, W)
    
    # Select 8 random samples + mean
    num_show = min(8, preds_conc.shape[0])
    sample_indices = np.random.choice(preds_conc.shape[0], num_show, replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    vmax = max(gt_conc.max(), preds_conc.mean(axis=0).max()) * 1.1
    
    # Plot GT
    im = axes[0].imshow(gt_conc, cmap='viridis', vmin=0, vmax=vmax)
    axes[0].set_title('Ground Truth', fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    
    # Plot samples
    for i, idx in enumerate(sample_indices):
        ax_idx = i + 1
        im = axes[ax_idx].imshow(preds_conc[idx], cmap='viridis', vmin=0, vmax=vmax)
        axes[ax_idx].set_title(f'Sample {idx}')
        axes[ax_idx].axis('off')
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046)
    
    # Plot mean
    pred_mean = preds_conc.mean(axis=0)
    im = axes[8].imshow(pred_mean, cmap='viridis', vmin=0, vmax=vmax)
    axes[8].set_title(f'Ensemble Mean', fontweight='bold')
    axes[8].axis('off')
    plt.colorbar(im, ax=axes[8], fraction=0.046)
    
    day = t_idx
    pred_day = t_idx - input_length if t_idx >= input_length else None
    title = f'Ensemble Comparison at Day {day}'
    if pred_day is not None and pred_day >= 0:
        title += f' (Prediction +{pred_day})'
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def create_temporal_evolution(gt, preds, input_length=20, save_path=None):
    """
    Create animation-style figure showing temporal evolution.
    """
    gt_conc = np.expm1(gt[:, 0])  # (T, H, W)
    pred_mean = np.expm1(preds[:, :, 0]).mean(axis=0)  # (T, H, W)
    
    total_length = gt_conc.shape[0]
    num_frames = 10
    timesteps = np.linspace(0, total_length-1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, num_frames, figsize=(2.5*num_frames, 5.5))
    
    for idx, t in enumerate(timesteps):
        vmax = max(gt_conc[t].max(), pred_mean[t].max()) * 1.1
        
        # Ground truth
        im0 = axes[0, idx].imshow(gt_conc[t], cmap='viridis', vmin=0, vmax=vmax)
        if t < input_length:
            axes[0, idx].set_title(f'Day {t}\n(Input)', fontsize=10)
        else:
            axes[0, idx].set_title(f'Day {t}\n(Pred +{t-input_length})', fontsize=10)
        axes[0, idx].axis('off')
        
        # Prediction
        im1 = axes[1, idx].imshow(pred_mean[t], cmap='viridis', vmin=0, vmax=vmax)
        axes[1, idx].axis('off')
    
    axes[0, 0].text(-0.1, 0.5, 'Ground Truth', transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    axes[1, 0].text(-0.1, 0.5, 'Prediction', transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    
    plt.suptitle('Temporal Evolution of Concentration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize PFLOTRAN predictions')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing gt_*.npy and pred_*.npy files')
    parser.add_argument('--case_id', type=int, default=0,
                        help='Test case ID to visualize')
    parser.add_argument('--input_length', type=int, default=20,
                        help='Number of input frames')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save figures (default: output_dir/visualizations)')
    parser.add_argument('--all_plots', action='store_true',
                        help='Generate all plot types')
    
    args = parser.parse_args()
    
    # Set up save directory
    if args.save_dir is None:
        args.save_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nLoading predictions from: {args.output_dir}")
    print(f"Visualizing case: {args.case_id}")
    print(f"Saving to: {args.save_dir}\n")
    
    # Load data
    try:
        gt, preds = load_case(args.output_dir, args.case_id)
        print(f"✓ Loaded ground truth: {gt.shape}")
        print(f"✓ Loaded predictions: {preds.shape} ({preds.shape[0]} ensemble members)\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Main visualization
    save_path = os.path.join(args.save_dir, f'case_{args.case_id}_full.png')
    visualize_case(gt, preds, args.input_length, args.case_id, save_path)
    
    if args.all_plots:
        # Ensemble comparison at last prediction timestep
        t_last = gt.shape[0] - 1
        save_path = os.path.join(args.save_dir, f'case_{args.case_id}_ensemble.png')
        create_ensemble_comparison(gt, preds, t_last, args.input_length, save_path)
        
        # Temporal evolution
        save_path = os.path.join(args.save_dir, f'case_{args.case_id}_temporal.png')
        create_temporal_evolution(gt, preds, args.input_length, save_path)
    
    print(f"\n✓ All visualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
