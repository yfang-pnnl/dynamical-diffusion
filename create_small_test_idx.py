#!/usr/bin/env python3
import numpy as np

# Load original test indices
test_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/test_idx.npy')
print(f'Original test indices: {len(test_idx)} cases')

# Create a small subset for quick evaluation (20 cases)
test_idx_small = test_idx[:20]
np.save('/qfs/projects/dl_calibration/d3m045/test_dd/output/test_idx_small.npy', test_idx_small)
print(f'Created test_idx_small.npy with {len(test_idx_small)} cases')
