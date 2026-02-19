import numpy as np

# Load indices
train_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/train_idx.npy')
val_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/val_idx.npy')
test_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/test_idx.npy')
test_eval_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/test_idx_eval.npy')

print("="*60)
print("DATA SPLIT ANALYSIS")
print("="*60)

print(f"\nNumber of windows:")
print(f"  Train:     {len(train_idx)}")
print(f"  Val:       {len(val_idx)}")
print(f"  Test (full):     {len(test_idx)}")
print(f"  Test (eval):     {len(test_eval_idx)}")

# Get unique realizations
train_real = np.unique(train_idx[:,0])
val_real = np.unique(val_idx[:,0])
test_real = np.unique(test_idx[:,0])
test_eval_real = np.unique(test_eval_idx[:,0])

print(f"\nRealizations:")
print(f"  Train:     {sorted(train_real)}")
print(f"  Val:       {sorted(val_real)}")
print(f"  Test:      {sorted(test_real)}")
print(f"  Test eval: {sorted(test_eval_real)}")

# Check for overlaps
print(f"\n" + "="*60)
print("CHECKING FOR DATA LEAKAGE")
print("="*60)

train_set = set(train_real)
val_set = set(val_real)
test_set = set(test_real)
test_eval_set = set(test_eval_real)

train_val_overlap = train_set & val_set
train_test_overlap = train_set & test_set
train_test_eval_overlap = train_set & test_eval_set
val_test_overlap = val_set & test_set

if train_val_overlap:
    print(f"\n❌ LEAKAGE: Train and Val share realizations: {sorted(train_val_overlap)}")
else:
    print(f"\n✓ No overlap between Train and Val")

if train_test_overlap:
    print(f"❌ LEAKAGE: Train and Test share realizations: {sorted(train_test_overlap)}")
else:
    print(f"✓ No overlap between Train and Test")

if train_test_eval_overlap:
    print(f"❌ LEAKAGE: Train and Test_Eval share realizations: {sorted(train_test_eval_overlap)}")
else:
    print(f"✓ No overlap between Train and Test_Eval")

if val_test_overlap:
    print(f"❌ LEAKAGE: Val and Test share realizations: {sorted(val_test_overlap)}")
else:
    print(f"✓ No overlap between Val and Test")

# Check data statistics
print(f"\n" + "="*60)
print("DATA DISTRIBUTION COMPARISON")
print("="*60)

conc = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/conc.npy')
print(f"\nFull dataset shape: {conc.shape}")  # (n_realizations, n_timesteps, H, W)

# Sample data from each split
train_windows = conc[train_idx[:,0], train_idx[:,1]]  # (n_train, H, W)
test_windows = conc[test_eval_idx[:,0], test_eval_idx[:,1]]  # (n_test, H, W)

print(f"\nTrain windows shape: {train_windows.shape}")
print(f"Test windows shape: {test_windows.shape}")

# Convert to log1p space (as used in training)
train_log = np.log1p(train_windows)
test_log = np.log1p(test_windows)

print(f"\nLog1p space statistics:")
print(f"  Train: mean={train_log.mean():.4f}, std={train_log.std():.4f}, min={train_log.min():.4f}, max={train_log.max():.4f}")
print(f"  Test:  mean={test_log.mean():.4f}, std={test_log.std():.4f}, min={test_log.min():.4f}, max={test_log.max():.4f}")

print(f"\nOriginal scale statistics:")
print(f"  Train: mean={train_windows.mean():.4e}, std={train_windows.std():.4e}, min={train_windows.min():.4e}, max={train_windows.max():.4e}")
print(f"  Test:  mean={test_windows.mean():.4e}, std={test_windows.std():.4e}, min={test_windows.min():.4e}, max={test_windows.max():.4e}")

# Check fraction of near-zero values
train_near_zero = (train_windows < 1e-8).mean()
test_near_zero = (test_windows < 1e-8).mean()

print(f"\nFraction of near-zero values (<1e-8):")
print(f"  Train: {train_near_zero:.2%}")
print(f"  Test:  {test_near_zero:.2%}")

# Distribution similarity
print(f"\n" + "="*60)
print("VERDICT")
print("="*60)

has_leakage = bool(train_val_overlap or train_test_overlap or train_test_eval_overlap)
stats_similar = abs(train_log.mean() - test_log.mean()) < 0.1

if has_leakage:
    print("\n❌ DATA LEAKAGE DETECTED!")
    print("   The model saw test data during training.")
    print("   This explains perfect training but poor test performance.")
elif not stats_similar:
    print("\n⚠ DISTRIBUTION MISMATCH")
    print("   Train and test have very different distributions.")
    print("   Model may not generalize well to test domain.")
else:
    print("\n✓ Split appears valid (no leakage, similar distributions)")
    print("   Poor test performance is likely due to:")
    print("   - Model overfitting (too complex)")
    print("   - Insufficient training")
    print("   - Model architecture issues")
