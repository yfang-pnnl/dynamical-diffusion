import numpy as np
import os

# Load the predictions and ground truth for case 0
pred_dir = "logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0"

# Load ground truth
gt = np.load(f"{pred_dir}/gt_0.npy")
print(f"Ground truth shape: {gt.shape}")
print(f"GT range: [{gt.min():.2e}, {gt.max():.2e}]")

# Load prediction samples (50 samples)
pred_samples = []
for i in range(50):
    pred_file = f"{pred_dir}/pred_0_sample_{i}.npy"
    if os.path.exists(pred_file):
        pred = np.load(pred_file)
        pred_samples.append(pred)
        
print(f"\nLoaded {len(pred_samples)} prediction samples")
if len(pred_samples) > 0:
    pred_arr = np.array(pred_samples)
    print(f"Prediction array shape: {pred_arr.shape}")
    print(f"Prediction range: [{pred_arr.min():.2e}, {pred_arr.max():.2e}]")
    
    # Check the location z=19, x=10
    z, x = 19, 10
    
    # Ground truth at this location (timesteps 20-39 prediction region)
    gt_timeseries = gt[20:40, z, x]  # prediction region
    print(f"\n=== GT at (z={z}, x={x}) prediction region ===")
    print(f"Range: [{gt_timeseries.min():.2e}, {gt_timeseries.max():.2e}]")
    print(f"Mean: {gt_timeseries.mean():.2e}")
    print(f"Values: {gt_timeseries}")
    
    # Predictions at this location
    pred_timeseries = pred_arr[:, 20:40, z, x]  # [50 samples, 20 timesteps]
    print(f"\n=== Predictions at (z={z}, x={x}) prediction region ===")
    print(f"Range: [{pred_timeseries.min():.2e}, {pred_timeseries.max():.2e}]")
    print(f"Mean: {pred_timeseries.mean():.2e}") 
    print(f"Std: {pred_timeseries.std():.2e}")
    
    # Check for zeros or near-zeros
    near_zero = (pred_timeseries < 1e-8).sum()
    print(f"Number of near-zero values (<1e-8): {near_zero} out of {pred_timeseries.size}")
    
    # Check mean across samples for each timestep
    print(f"\n=== Mean prediction per timestep (prediction region) ===")
    for t in range(20):
        mean_val = pred_timeseries[:, t].mean()
        min_val = pred_timeseries[:, t].min()
        max_val = pred_timeseries[:, t].max()
        zero_count = (pred_timeseries[:, t] < 1e-8).sum()
        print(f"t={t+20}: mean={mean_val:.2e}, min={min_val:.2e}, max={max_val:.2e}, zeros={zero_count}/50")
    
    # Check sample 0 specifically
    print(f"\n=== Sample 0 full timeseries (all 40 timesteps) ===")
    sample_0 = pred_arr[0, :, z, x]
    for t in range(40):
        print(f"t={t}: {sample_0[t]:.2e}")
    
    # Check if this is specific to this location or widespread
    print(f"\n=== Checking if near-zero issue is widespread ===")
    # Count near-zeros across entire spatial domain for each sample
    for i in range(min(5, len(pred_samples))):
        pred_full = pred_arr[i, 20:40, :, :]  # prediction region only
        near_zero_count = (pred_full < 1e-8).sum()
        total_count = pred_full.size
        print(f"Sample {i}: {near_zero_count}/{total_count} ({100*near_zero_count/total_count:.1f}%) near-zero values")
