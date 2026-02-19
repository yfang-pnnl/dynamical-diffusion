#!/usr/bin/env python3
"""
Verify what happens in the input region (days 0-19).
Check if GT and predictions match in the conditioning region.
"""
import numpy as np
import glob
import os

output_dir = 'logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0'

# Load case 0
gt = np.load(os.path.join(output_dir, 'gt_0.npy'))[0]
pred_files = sorted(glob.glob(os.path.join(output_dir, 'pred_0_sample_*.npy')))
preds = np.stack([np.load(f)[0] for f in pred_files[:5]], axis=0)  # First 5 samples

# Convert from log space
gt_conc = np.expm1(gt[:, 0])
preds_conc = np.expm1(preds[:, :, 0])

input_length = 20
z, x = 19, 10  # The location we're interested in

# Extract time series
gt_ts = gt_conc[:, z, x]
preds_ts = preds_conc[:, :, z, x]  # (K, T)

print('='*70)
print('INPUT REGION ANALYSIS (Days 0-19)')
print('='*70)
print(f'Location: (z={z}, x={x})')
print()

# Compare input region (days 0-19)
gt_input = gt_ts[:input_length]
preds_input = preds_ts[:, :input_length]  # (K, 20)
pred_mean_input = preds_input.mean(axis=0)

print('Comparison at Day 0 (first input frame):')
print(f'  Ground Truth:      {gt_input[0]:.6e}')
print(f'  Prediction Mean:   {pred_mean_input[0]:.6e}')
print(f'  Difference:        {abs(gt_input[0] - pred_mean_input[0]):.6e}')
print()

print('Comparison at Day 10 (middle of input):')
print(f'  Ground Truth:      {gt_input[10]:.6e}')
print(f'  Prediction Mean:   {pred_mean_input[10]:.6e}')
print(f'  Difference:        {abs(gt_input[10] - pred_mean_input[10]):.6e}')
print()

print('Comparison at Day 19 (last input frame):')
print(f'  Ground Truth:      {gt_input[19]:.6e}')
print(f'  Prediction Mean:   {pred_mean_input[19]:.6e}')
print(f'  Difference:        {abs(gt_input[19] - pred_mean_input[19]):.6e}')
print()

# Overall statistics for input region
input_diff = np.abs(gt_input - pred_mean_input)
print('Input Region (Days 0-19) Statistics:')
print(f'  Mean Absolute Difference: {input_diff.mean():.6e}')
print(f'  Max Absolute Difference:  {input_diff.max():.6e}')
print(f'  Correlation:              {np.corrcoef(gt_input, pred_mean_input)[0,1]:.6f}')
print()

print('='*70)
print('PREDICTION REGION ANALYSIS (Days 20-39)')
print('='*70)

# Compare prediction region (days 20-39)
gt_pred = gt_ts[input_length:]
preds_pred = preds_ts[:, input_length:]
pred_mean_pred = preds_pred.mean(axis=0)

print('Comparison at Day 20 (first prediction):')
print(f'  Ground Truth:      {gt_pred[0]:.6e}')
print(f'  Prediction Mean:   {pred_mean_pred[0]:.6e}')
print(f'  Difference:        {abs(gt_pred[0] - pred_mean_pred[0]):.6e}')
print()

print('Comparison at Day 30 (middle of prediction):')
print(f'  Ground Truth:      {gt_pred[10]:.6e}')
print(f'  Prediction Mean:   {pred_mean_pred[10]:.6e}')
print(f'  Difference:        {abs(gt_pred[10] - pred_mean_pred[10]):.6e}')
print()

print('Comparison at Day 39 (last prediction):')
print(f'  Ground Truth:      {gt_pred[19]:.6e}')
print(f'  Prediction Mean:   {pred_mean_pred[19]:.6e}')
print(f'  Difference:        {abs(gt_pred[19] - pred_mean_pred[19]):.6e}')
print()

pred_diff = np.abs(gt_pred - pred_mean_pred)
print('Prediction Region (Days 20-39) Statistics:')
print(f'  Mean Absolute Difference: {pred_diff.mean():.6e}')
print(f'  Max Absolute Difference:  {pred_diff.max():.6e}')
print(f'  Correlation:              {np.corrcoef(gt_pred, pred_mean_pred)[0,1]:.6f}')
print()

print('='*70)
print('INTERPRETATION:')
print('='*70)
if input_diff.mean() < 1e-5:
    print('✓ Input region (days 0-19): GT and Prediction are VERY SIMILAR')
    print('  → The model reconstructs the conditioning input accurately')
else:
    print('⚠ Input region (days 0-19): GT and Prediction are DIFFERENT')
    print('  → This is unusual - the model sees these frames as input')

print()
if pred_diff.mean() > input_diff.mean() * 10:
    print('✓ Prediction region (days 20-39): Larger differences as expected')
    print('  → This is where the model forecasts unseen future')
else:
    print('⚠ Prediction region: Similar accuracy to input region')
    print('  → Model is very accurate at forecasting!')
print('='*70)
