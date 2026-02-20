#!/usr/bin/env python3
"""
Plot time series at a specific spatial location.
Shows concentration evolution over time at one point.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_timeseries_at_location(output_dir, case_id=0, z=20, x=10, input_length=20, save_path=None):
    """
    Plot time series at a specific spatial location (z, x).
    
    Args:
        output_dir: Directory with saved predictions
        case_id: Which test case
        z: Z coordinate (height/row index, 0-39 for 40x28 grid)
        x: X coordinate (width/column index, 0-27 for 40x28 grid)
        input_length: Number of input frames
        save_path: Where to save figure
    """
    # Load ground truth
    gt = np.load(os.path.join(output_dir, f'gt_{case_id}.npy'))[0]  # (T, C, H, W)
    
    # Load all prediction samples
    pred_files = sorted(glob.glob(os.path.join(output_dir, f'pred_{case_id}_sample_*.npy')))
    preds = np.stack([np.load(f)[0] for f in pred_files], axis=0)  # (K, T, C, H, W)
    
    # Convert from log space to physical concentration
    gt_conc = np.expm1(gt[:, 0])  # (T, H, W)
    preds_conc = np.expm1(preds[:, :, 0])  # (K, T, H, W)
    
    # Extract time series at location (z, x)
    gt_ts = gt_conc[:, z, x]  # (T,)
    preds_ts = preds_conc[:, :, z, x]  # (K, T)
    
    # Replace exact zeros with small value for log plotting
    min_nonzero = 1e-10
    gt_ts = np.where(gt_ts == 0, min_nonzero, gt_ts)
    preds_ts = np.where(preds_ts == 0, min_nonzero, preds_ts)
    
    # Compute statistics
    pred_mean = preds_ts.mean(axis=0)  # (T,)
    pred_std = preds_ts.std(axis=0)
    pred_min = preds_ts.min(axis=0)
    pred_max = preds_ts.max(axis=0)
    
    # Time axis
    total_length = gt_ts.shape[0]
    time = np.arange(total_length)
    
    # Create figure (wider to accommodate legend outside)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot ground truth
    ax.plot(time, gt_ts, 'k-', linewidth=2, label='Ground Truth', marker='o', markersize=4)
    
    # Plot prediction mean
    ax.plot(time, pred_mean, 'b-', linewidth=2, label='Prediction Mean', marker='s', markersize=4)
    
    # Plot uncertainty bands (clipped to valid range for log scale: [1e-10, 1])
    min_val = 1e-10
    ax.fill_between(time, np.clip(pred_mean - pred_std, min_val, 1), np.clip(pred_mean + pred_std, min_val, 1), 
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax.fill_between(time, np.clip(pred_min, min_val, 1), np.clip(pred_max, min_val, 1), 
                     alpha=0.15, color='blue', label='Min-Max Range')
    
    # Plot individual ensemble members (first 5 for clarity)
    for i in range(min(5, preds_ts.shape[0])):
        ax.plot(time, preds_ts[i], 'c-', alpha=0.3, linewidth=0.5)
    
    # Mark the boundary between input and prediction
    ax.axvline(input_length - 0.5, color='red', linestyle='--', linewidth=2, 
               label=f'Input/Prediction Boundary (t={input_length})')
    
    # Shading for input vs prediction regions
    ax.axvspan(-0.5, input_length - 0.5, alpha=0.1, color='green', label='Input Region')
    ax.axvspan(input_length - 0.5, total_length - 0.5, alpha=0.1, color='orange', label='Prediction Region')
    
    # Labels and formatting
    ax.set_xlabel('Time Step (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Concentration (log scale)', fontsize=12, fontweight='bold')
    ax.set_title(f'Time Series at Location (z={z}, x={x}) - Case {case_id}\n'
                 f'{preds_ts.shape[0]} Ensemble Members', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1.5)  # From minimum non-zero value to slightly above 1
    
    # Add horizontal line at maximum concentration
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Statistics text box
    input_mae = np.abs(gt_ts[:input_length] - pred_mean[:input_length]).mean()
    pred_mae = np.abs(gt_ts[input_length:] - pred_mean[input_length:]).mean()
    corr = np.corrcoef(gt_ts[input_length:], pred_mean[input_length:])[0, 1]
    
    stats_text = f'Prediction Region Statistics:\n'
    stats_text += f'MAE: {pred_mae:.6f}\n'
    stats_text += f'Correlation: {corr:.4f}\n'
    stats_text += f'GT Peak: {gt_ts[input_length:].max():.4f}\n'
    stats_text += f'Pred Peak: {pred_mean[input_length:].max():.4f}\n'
    stats_text += f'\nLog scale plot\n'
    stats_text += f'Range: [1e-10, 1]'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show() 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved time series plot: {save_path}")
        print(f"  (Log scale plot - range: [1e-10, 1])")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot time series at specific location')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing predictions')
    parser.add_argument('--case_id', type=int, default=0,
                        help='Test case ID')
    parser.add_argument('--z', type=int, default=20,
                        help='Z coordinate (row, 0-39)')
    parser.add_argument('--x', type=int, default=10,
                        help='X coordinate (column, 0-27)')
    parser.add_argument('--input_length', type=int, default=20,
                        help='Number of input frames')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save figure')
    
    args = parser.parse_args()
    
    # Set up save directory
    if args.save_dir is None:
        args.save_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nPlotting time series at location (z={args.z}, x={args.x})")
    print(f"Case: {args.case_id}")
    print(f"Output dir: {args.output_dir}\n")
    
    save_path = os.path.join(args.save_dir, f'case_{args.case_id}_timeseries_z{args.z}_x{args.x}.png')
    plot_timeseries_at_location(args.output_dir, args.case_id, args.z, args.x, 
                                  args.input_length, save_path)
    
    print(f"\n✓ Time series plot saved to: {save_path}")


if __name__ == "__main__":
    main()
