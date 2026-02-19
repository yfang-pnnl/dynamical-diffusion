import numpy as np

train_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/train_idx.npy')
test_eval_idx = np.load('/qfs/projects/dl_calibration/d3m045/test_dd/output/test_idx_eval.npy')

print("Train indices:", len(train_idx), "windows")
print("Test eval indices:", len(test_eval_idx), "windows")

print("\nTrain realizations:", np.unique(train_idx[:,0]))
print("Test eval realizations:", np.unique(test_eval_idx[:,0]))

# Check overlap
train_real = set(np.unique(train_idx[:,0]))
test_real = set(np.unique(test_eval_idx[:,0]))
overlap = train_real & test_real

if overlap:
    print(f"\nWARNING: Realizations {overlap} appear in both train and test!")
else:
    print("\n✓ No realization overlap between train and test")
