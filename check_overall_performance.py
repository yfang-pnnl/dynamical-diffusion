#!/usr/bin/env python3
import numpy as np
import glob
import os

output_dir = 'logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0'

# Find all ground truth files
gt_files = sorted(glob.glob(os.path.join(output_dir, 'gt_*.npy')))
num_cases = len(gt_files)

print('='*60)
print(f'Analyzing {num_cases} test cases...')
print('='*60)
print()

# Storage for metrics
correlations = []
rel_errors = []
peak_errors = []
maes = []
input_length = 20

# Analyze each case
for i, gt_file in enumerate(gt_files[:10]):  # First 10 cases for speed
    case_id = int(os.path.basename(gt_file).replace('gt_', '').replace('.npy', ''))
    
    # Load ground truth
    gt = np.load(gt_file)[0]
    
    # Load predictions
    pred_files = sorted(glob.glob(os.path.join(output_dir, f'pred_{case_id}_sample_*.npy')))
    if len(pred_files) == 0:
        continue
    
    pred_samples = [np.load(f)[0] for f in pred_files]
    preds = np.stack(pred_samples)
    
    # Convert from log space
    gt_conc = np.expm1(gt[:, 0])
    preds_conc = np.expm1(preds[:, :, 0])
    pred_mean = preds_conc.mean(axis=0)
    
    # Metrics for prediction frames only
    gt_pred = gt_conc[input_length:]
    pred_pred = pred_mean[input_length:]
    
    mae = np.abs(gt_pred - pred_pred).mean()
    rel_error = mae / (gt_pred.mean() + 1e-10) * 100
    corr = np.corrcoef(gt_pred.flatten(), pred_pred.flatten())[0, 1]
    peak_error = abs(gt_pred.max() - pred_pred.max()) / (gt_pred.max() + 1e-10) * 100
    
    correlations.append(corr)
    rel_errors.append(rel_error)
    peak_errors.append(peak_error)
    maes.append(mae)
    
    if i < 3:  # Show first 3 cases
        print(f'Case {case_id}: Corr={corr:.4f}, Peak Error={peak_error:.2f}%, MAE={mae:.6f}')

# Summary statistics
correlations = np.array(correlations)
rel_errors = np.array(rel_errors)
peak_errors = np.array(peak_errors)
maes = np.array(maes)

print()
print('='*60)
print('OVERALL MODEL PERFORMANCE (across all analyzed cases)')
print('='*60)
print(f'Cases analyzed: {len(correlations)}')
print()
print('Correlation Coefficient:')
print(f'  Mean:   {correlations.mean():.4f}')
print(f'  Median: {np.median(correlations):.4f}')
print(f'  Min:    {correlations.min():.4f}')
print(f'  Max:    {correlations.max():.4f}')
print(f'  Std:    {correlations.std():.4f}')
print()
print('Peak Prediction Error (%):')
print(f'  Mean:   {peak_errors.mean():.2f}%')
print(f'  Median: {np.median(peak_errors):.2f}%')
print(f'  Min:    {peak_errors.min():.2f}%')
print(f'  Max:    {peak_errors.max():.2f}%')
print()
print('Mean Absolute Error:')
print(f'  Mean:   {maes.mean():.6f}')
print(f'  Median: {np.median(maes):.6f}')
print()
print('='*60)
print('Overall Assessment:')
if correlations.mean() > 0.9:
    print('  ✓ EXCELLENT: Mean correlation > 0.9')
elif correlations.mean() > 0.7:
    print('  ✓ GOOD: Mean correlation 0.7-0.9')
else:
    print('  ⚠ FAIR: Mean correlation < 0.7')

if peak_errors.mean() < 10:
    print('  ✓ EXCELLENT: Mean peak error < 10%')
elif peak_errors.mean() < 25:
    print('  ✓ GOOD: Mean peak error 10-25%')
else:
    print('  ⚠ NEEDS IMPROVEMENT: Mean peak error > 25%')
print('='*60)
