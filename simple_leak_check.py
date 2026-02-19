import numpy as np

# Load data
train_real = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/train_idx.npy')
test_real = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/test_idx_eval.npy')

print("REALIZATION-LEVEL SPLIT CHECK")
print("="*60)
print(f"Train realizations: {sorted(train_real)}")
print(f"Test realizations: {sorted(test_real)}")

overlap = set(train_real) & set(test_real)
if overlap:
    print(f"\n❌ DATA LEAKAGE: {len(overlap)} realizations in both train and test!")
    print(f"   Overlapping IDs: {sorted(overlap)}")
else:
    print(f"\n✓ NO LEAKAGE: Train and test use completely separate realizations")

print(f"\nSummary:")
print(f"  Train: {len(train_real)} realizations")
print(f"  Test:  {len(test_real)} realizations")
print(f"  Total: {len(train_real) + len(test_real)} realizations")
