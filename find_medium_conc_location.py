#!/usr/bin/env python3
"""
Find a grid location with concentration in the range [1e-4, 1e-2]
and explain the temporal structure.
"""
import numpy as np
import glob
import os

output_dir = 'logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0'

# Load case 0
gt = np.load(os.path.join(output_dir, 'gt_0.npy'))[0]
pred_files = sorted(glob.glob(os.path.join(output_dir, 'pred_0_sample_*.npy')))
preds = np.stack([np.load(f)[0] for f in pred_files], axis=0)

# Convert from log space
gt_conc = np.expm1(gt[:, 0])  # (T=40, H=40, W=28)
preds_conc = np.expm1(preds[:, :, 0])  # (K, T=40, H=40, W=28)
pred_mean = preds_conc.mean(axis=0)

input_length = 20

print('='*70)
print('TEMPORAL STRUCTURE EXPLANATION')
print('='*70)
print()
print('Data structure:')
print(f'  Total timesteps in this window: 40 (days 0-39)')
print(f'  Input frames (conditioning): 0-19 (first 20 days)')
print(f'  Prediction frames (forecast): 20-39 (next 20 days)')
print()
print('What the model does:')
print('  1. Takes days 0-19 as INPUT (conditioned on these)')
print('  2. Predicts days 20-39 as OUTPUT (generates these)')
print()
print('Ground truth vs Prediction:')
print('  Days 0-19: GT and Pred SHOULD BE SIMILAR (model sees this)')
print('  Days 20-39: GT and Pred COMPARISON shows forecast accuracy')
print()
print('Starting day of case 0:')

# Load the test index to see which realization this is
test_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/test_idx_eval.npy')
print(f'  Realization: {test_idx[0]}')
print(f'  Window start: Day 0 of 240-day simulation')
print(f'  Window covers: Days 0-39 from the full PFLOTRAN simulation')
print()
print('='*70)
print()

# Find locations with concentrations in [1e-4, 1e-2] during prediction window
pred_region = gt_conc[input_length:]  # Days 20-39
target_min, target_max = 1e-4, 1e-2

# Find all locations that have mean concentration in this range
H, W = pred_region.shape[1], pred_region.shape[2]
candidates = []

for z in range(H):
    for x in range(W):
        ts = pred_region[:, z, x]
        mean_conc = ts.mean()
        max_conc = ts.max()
        
        if target_min <= mean_conc <= target_max:
            candidates.append({
                'z': z, 
                'x': x, 
                'mean': mean_conc,
                'max': max_conc,
                'variance': ts.var()
            })

# Sort by variance (most interesting dynamics)
candidates.sort(key=lambda c: c['variance'], reverse=True)

print('LOCATIONS WITH MEDIUM CONCENTRATION (1e-4 to 1e-2):')
print('-'*70)
print(f'Found {len(candidates)} locations')
print()

if len(candidates) > 0:
    print('Top 5 locations (sorted by variance - most dynamic):')
    print(f'{"#":<4} {"z":<6} {"x":<6} {"Mean Conc":<15} {"Max Conc":<15} {"Variance":<12}')
    print('-'*70)
    for i, c in enumerate(candidates[:5]):
        print(f'{i+1:<4} {c["z"]:<6} {c["x"]:<6} {c["mean"]:<15.2e} {c["max"]:<15.2e} {c["variance"]:<12.2e}')
    
    print()
    print('='*70)
    print('RECOMMENDATION:')
    best = candidates[0]
    print(f'  Use location (z={best["z"]}, x={best["x"]}) for time series plot')
    print(f'  This location has:')
    print(f'    - Mean concentration: {best["mean"]:.2e}')
    print(f'    - Max concentration: {best["max"]:.2e}')
    print(f'    - Interesting temporal dynamics (high variance)')
    print()
    print('Run this command:')
    print(f'  python plot_timeseries.py \\')
    print(f'    --output_dir {output_dir} \\')
    print(f'    --case_id 0 \\')
    print(f'    --z {best["z"]} \\')
    print(f'    --x {best["x"]}')
    print('='*70)
else:
    print('No locations found with mean concentration in [1e-4, 1e-2]')
    print('Try a different range or case.')
