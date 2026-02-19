import numpy as np

pred_dir = "logs/custom_pflotran/pflotran_identity/output_for_evaluation_0"

gt = np.load(f"{pred_dir}/gt_0.npy")
pred_0 = np.load(f"{pred_dir}/pred_0_sample_0.npy")

print("Ground truth shape:", gt.shape)
print("Prediction shape:", pred_0.shape)

print("\nLog1p space ranges:")
print("  GT:", gt.min(), "to", gt.max())
print("  Pred:", pred_0.min(), "to", pred_0.max())

gt_conc = np.expm1(gt)
pred_conc = np.expm1(pred_0)

print("\nOriginal concentration scale:")
print("  GT:", gt_conc.min(), "to", gt_conc.max())
print("  Pred:", pred_conc.min(), "to", pred_conc.max())

neg = (pred_conc < 0).sum()
print("\nNegative values:", neg, "out of", pred_conc.size)

# Correlation on prediction region
gt_region = gt[0, 20:40]
pred_region = pred_0[0, 20:40]
corr = np.corrcoef(gt_region.flatten(), pred_region.flatten())[0,1]
print("\nCorrelation (prediction region):", corr)
