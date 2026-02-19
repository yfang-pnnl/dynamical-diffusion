#!/usr/bin/env python3
"""
Verify that the concentration scale is correct by comparing
the saved predictions with the original data.
"""
import numpy as np
import glob

# Load original concentration data
conc_raw = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/conc.npy', mmap_mode='r')
test_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/test_idx_eval.npy')

# Get the first test realization
first_test_real = test_idx[0]

# Load saved ground truth from evaluation
output_dir = 'logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0'
gt_saved = np.load(f'{output_dir}/gt_0.npy')[0]  # (T, C, H, W) in log space

# Convert from log space to original
gt_conc = np.expm1(gt_saved[:, 0])  # (T, H, W)

# Get corresponding slices from raw data
# Case 0 should correspond to first window of first test realization
# With dt=1, start=0, total_length=40
raw_slice = conc_raw[first_test_real, 0:40]  # (T, H, W)

print('='*60)
print('Concentration Scale Verification')
print('='*60)
print(f'Test realization index: {first_test_real}')
print(f'Time range: 0-39 days')
print()
print('Sample values at location (z=20, x=10):')
print(f'  Raw data (original):     {raw_slice[:5, 20, 10]}')
print(f'  Saved GT (after expm1):  {gt_conc[:5, 20, 10]}')
print()
print('Difference (should be ~0):')
diff = np.abs(raw_slice - gt_conc).max()
print(f'  Max absolute difference: {diff:.10f}')
print()
if diff < 1e-5:
    print('✓ VERIFIED: Concentrations are in original scale!')
else:
    print('⚠ WARNING: Scale mismatch detected')
print()
print('Statistics:')
print(f'  Raw data range: [{raw_slice.min():.6f}, {raw_slice.max():.6f}]')
print(f'  Saved GT range: [{gt_conc.min():.6f}, {gt_conc.max():.6f}]')
print('='*60)
