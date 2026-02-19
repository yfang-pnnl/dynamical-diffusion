#!/usr/bin/env python3
import numpy as np
import glob
import os

# Load one example to check prediction quality
output_dir = 'logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0'

gt = np.load(os.path.join(output_dir, 'gt_0.npy'))[0]

# Load all prediction samples for case 0
pred_files = sorted(glob.glob(os.path.join(output_dir, 'pred_0_sample_*.npy')))
pred_samples = [np.load(f)[0] for f in pred_files]
preds = np.stack(pred_samples)

# Convert from log space
gt_conc = np.expm1(gt[:, 0])
preds_conc = np.expm1(preds[:, :, 0])
pred_mean = preds_conc.mean(axis=0)
pred_std = preds_conc.std(axis=0)

# Compute metrics for prediction frames only (after input_length=20)
input_length = 20
gt_pred = gt_conc[input_length:]
pred_pred = pred_mean[input_length:]

# MSE and relative error
mse = ((gt_pred - pred_pred)**2).mean()
mae = np.abs(gt_pred - pred_pred).mean()
rel_error = mae / gt_pred.mean() * 100

# Correlation
corr = np.corrcoef(gt_pred.flatten(), pred_pred.flatten())[0, 1]

# Peak values
gt_max = gt_pred.max()
pred_max = pred_pred.max()
peak_error = abs(gt_max - pred_max) / gt_max * 100

print('='*60)
print('PFLOTRAN Diffusion Model Performance - Case 0')
print('='*60)
print(f'Number of ensemble samples: {len(pred_samples)}')
print(f'Input frames: {input_length}, Prediction frames: {len(gt_pred)}')
print()
print('Prediction Quality Metrics:')
print(f'  MSE (Mean Squared Error):    {mse:.6f}')
print(f'  MAE (Mean Absolute Error):   {mae:.6f}')
print(f'  Relative Error:              {rel_error:.2f}%')
print(f'  Correlation coefficient:     {corr:.4f}')
print()
print('Concentration Statistics:')
print(f'  GT mean concentration:       {gt_pred.mean():.4f}')
print(f'  Pred mean concentration:     {pred_pred.mean():.4f}')
print(f'  GT peak concentration:       {gt_max:.4f}')
print(f'  Pred peak concentration:     {pred_max:.4f}')
print(f'  Peak prediction error:       {peak_error:.2f}%')
print()
print('Uncertainty Quantification:')
print(f'  Mean std dev across space:   {pred_std.mean():.4f}')
print(f'  Max std dev:                 {pred_std.max():.4f}')
print('='*60)
print()
print('Performance Assessment:')
if corr > 0.9:
    print('  ✓ EXCELLENT: Correlation > 0.9 - Strong predictive skill')
elif corr > 0.7:
    print('  ✓ GOOD: Correlation 0.7-0.9 - Good predictive skill')
elif corr > 0.5:
    print('  ⚠ FAIR: Correlation 0.5-0.7 - Moderate predictive skill')
else:
    print('  ✗ POOR: Correlation < 0.5 - Low predictive skill')

if rel_error < 10:
    print('  ✓ EXCELLENT: Relative error < 10%')
elif rel_error < 25:
    print('  ✓ GOOD: Relative error 10-25%')
else:
    print('  ⚠ FAIR: Relative error > 25%')
print('='*60)
