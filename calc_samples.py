import numpy as np

# Load data
idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/train_idx.npy')
conc = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/conc.npy', mmap_mode='r')

# Config params from pflotran_identity.yaml
input_length = 20
pred_length = 20
total_length = 40
t_keep = 240
dt = 1

# Calculate as per pflotran.py __init__ method
T = min(conc.shape[1], t_keep)
T_eff = (T - 1) // dt + 1
max_start = T_eff - total_length
num_windows = max_start + 1
num_realizations = len(idx)
total_samples = num_realizations * num_windows

print(f'Conc shape: {conc.shape}')
print(f'Training realizations: {num_realizations}')
print(f'T (timesteps kept): {T}')
print(f'T_eff (effective timesteps): {T_eff}')
print(f'max_start: {max_start}')
print(f'Windows per realization: {num_windows}')
print(f'Total samples: {total_samples}')
print(f'Expected batches (batch_size=1): {total_samples}')
print(f'\nActual batches from log: 57,888')
print(f'Discrepancy: {57888 - total_samples}')
