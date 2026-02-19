import numpy as np

pred_dir = "logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0"

# Load one case
gt = np.load(f"{pred_dir}/gt_0.npy")
pred_0 = np.load(f"{pred_dir}/pred_0_sample_0.npy")

print("Shapes:")
print(f"  GT: {gt.shape}")
print(f"  Pred: {pred_0.shape}")

print("\nLog1p space ranges:")
print(f"  GT: [{gt.min():.3f}, {gt.max():.3f}]")
print(f"  Pred: [{pred_0.min():.3f}, {pred_0.max():.3f}]")

# Denormalize
gt_conc = np.expm1(gt)
pred_conc = np.expm1(pred_0)

print("\nOriginal concentration scale:")
print(f"  GT: [{gt_conc.min():.3e}, {gt_conc.max():.3e}]")
print(f"  Pred: [{pred_conc.min():.3e}, {pred_conc.max():.3e}]")

# Check for negatives
neg_count = (pred_conc < 0).sum()
print(f"\nNegative predictions: {neg_count}/{pred_conc.size}")

# Correlation on prediction region (days 20-39)
gt_pred = gt[0, 20:40]
pred_pred = pred_0[0, 20:40]
corr = np.corrcoef(gt_pred.flatten(), pred_pred.flatten())[0,1]
print(f"\nCorrelation (prediction region): {corr:.4f}")

# Check specific location z=19, x=10
z, x = 19, 10
print(f"\nAt location (z={z}, x={x}):")
print(f"  GT pred region: [{gt_conc[0,20:40,0,z,x].min():.3e}, {gt_conc[0,20:40,0,z,x].max():.3e}]")
print(f"  Pred pred region: [{pred_conc[0,20:40,0,z,x].min():.3e}, {pred_conc[0,20:40,0,z,x].max():.3e}]")
