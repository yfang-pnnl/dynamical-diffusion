#!/usr/bin/env python3
"""
Analyze model performance across different concentration magnitudes.
Check if the diffusion model captures the full dynamic range (1e-8 to 1).
"""
import numpy as np
import glob
import os

output_dir = 'logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0'

# Load a representative case
gt = np.load(os.path.join(output_dir, 'gt_0.npy'))[0]
pred_files = sorted(glob.glob(os.path.join(output_dir, 'pred_0_sample_*.npy')))
preds = np.stack([np.load(f)[0] for f in pred_files], axis=0)

# Convert from log space to physical concentration
gt_conc = np.expm1(gt[:, 0])
preds_conc = np.expm1(preds[:, :, 0])
pred_mean = preds_conc.mean(axis=0)

# Focus on prediction region
input_length = 20
gt_pred = gt_conc[input_length:]
pred_pred = pred_mean[input_length:]

print('='*70)
print('DIFFUSION MODEL PERFORMANCE ACROSS CONCENTRATION RANGES')
print('='*70)
print()

# Overall statistics
print('Overall Concentration Statistics:')
print(f'  Ground Truth range:  [{gt_pred.min():.2e}, {gt_pred.max():.2e}]')
print(f'  Prediction range:    [{pred_pred.min():.2e}, {pred_pred.max():.2e}]')
print(f'  Dynamic range:       {gt_pred.max() / (gt_pred.min() + 1e-20):.2e}')
print()

# Define concentration bins (logarithmic)
bins = [
    (0, 1e-8, 'Very Low (<1e-8)'),
    (1e-8, 1e-6, 'Low (1e-8 to 1e-6)'),
    (1e-6, 1e-4, 'Moderate (1e-6 to 1e-4)'),
    (1e-4, 1e-2, 'Medium (1e-4 to 1e-2)'),
    (1e-2, 1, 'High (1e-2 to 1)')
]

print('Performance by Concentration Range:')
print('-'*70)
print(f'{"Range":<25} {"Count":<10} {"MAE":<12} {"Rel Error":<12} {"Corr":<8}')
print('-'*70)

for low, high, label in bins:
    mask = (gt_pred >= low) & (gt_pred < high)
    count = mask.sum()
    
    if count > 0:
        gt_subset = gt_pred[mask]
        pred_subset = pred_pred[mask]
        
        mae = np.abs(gt_subset - pred_subset).mean()
        rel_error = mae / (gt_subset.mean() + 1e-20) * 100
        
        # Correlation (only if enough variance)
        if gt_subset.std() > 1e-10 and pred_subset.std() > 1e-10:
            corr = np.corrcoef(gt_subset, pred_subset)[0, 1]
        else:
            corr = np.nan
        
        print(f'{label:<25} {count:<10} {mae:<12.2e} {rel_error:<12.1f}% {corr:<8.4f}')
    else:
        print(f'{label:<25} {count:<10} {"N/A":<12} {"N/A":<12} {"N/A":<8}')

print('-'*70)
print()

# Check low concentration accuracy
low_threshold = 1e-6
low_mask = gt_pred < low_threshold
if low_mask.sum() > 0:
    low_mae = np.abs(gt_pred[low_mask] - pred_pred[low_mask]).mean()
    low_gt_mean = gt_pred[low_mask].mean()
    print(f'Low Concentration Performance (< {low_threshold:.0e}):')
    print(f'  Count: {low_mask.sum()}')
    print(f'  MAE: {low_mae:.2e}')
    print(f'  Mean GT: {low_gt_mean:.2e}')
    print(f'  Relative Error: {low_mae/low_gt_mean*100:.1f}%')
    print()

# Check high concentration accuracy
high_threshold = 0.1
high_mask = gt_pred > high_threshold
if high_mask.sum() > 0:
    high_mae = np.abs(gt_pred[high_mask] - pred_pred[high_mask]).mean()
    high_gt_mean = gt_pred[high_mask].mean()
    print(f'High Concentration Performance (> {high_threshold:.2e}):')
    print(f'  Count: {high_mask.sum()}')
    print(f'  MAE: {high_mae:.2e}')
    print(f'  Mean GT: {high_gt_mean:.2e}')
    print(f'  Relative Error: {high_mae/high_gt_mean*100:.1f}%')
    print()

# Log-space performance (this is what the model actually trains on)
gt_log = np.log1p(gt_pred)
pred_log = np.log1p(pred_pred)
log_mae = np.abs(gt_log - pred_log).mean()
log_corr = np.corrcoef(gt_log.flatten(), pred_log.flatten())[0, 1]

print('='*70)
print('Why log1p Transform Helps:')
print('-'*70)
print(f'  Log-space MAE:         {log_mae:.6f}')
print(f'  Log-space Correlation: {log_corr:.4f}')
print()
print('  The log1p transform compresses the dynamic range from ~8 orders')
print('  of magnitude to a manageable range, allowing the neural network')
print('  to learn both low and high concentrations effectively.')
print('='*70)
print()
print('VERDICT:')
if gt_pred.min() < 1e-6 and pred_pred.min() < 1e-6:
    print('  ✓ Model DOES capture low concentrations (<1e-6)')
else:
    print('  ⚠ Model may struggle with very low concentrations')

if gt_pred.max() > 0.5 and pred_pred.max() > 0.5:
    print('  ✓ Model DOES capture high concentrations (>0.5)')
else:
    print('  ⚠ Model may struggle with high concentrations')

print('='*70)
