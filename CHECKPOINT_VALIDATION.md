# Checkpoint Validation Guide

## Files Created
1. **`core/eval_pflotran_checkpoint.py`** - Full evaluation script
2. **`core/quick_check_checkpoint.py`** - Quick validation check
3. **`core/eval_job.sub`** - Slurm job for evaluation

## Usage

### 1. Wait for Checkpoints to Save
Training saves checkpoints every 2000 iterations:
```bash
# Check for checkpoints
ls -lh logs/custom_pflotran/pflotran_identity/lightning_logs/version_*/checkpoints/
```

First checkpoint appears at ~8 minutes (iteration 2000).

### 2. Quick Check (Fast Validation)
Verify a checkpoint works:
```bash
python core/quick_check_checkpoint.py \
    logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/epoch=0-step=2000.ckpt
```

This runs one batch through the model (~30 seconds).

### 3. Full Evaluation
Run complete evaluation with ensemble sampling:

**Option A: Interactive**
```bash
python core/eval_pflotran_checkpoint.py \
    --config core/models/pflotran_identity.yaml \
    --checkpoint logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/epoch=0-step=2000.ckpt \
    --num_samples 5 \
    --output_dir ./evaluation_results \
    --split val
```

**Option B: Submit as job**
```bash
sbatch core/eval_job.sub \
    logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/epoch=0-step=2000.ckpt
```

### 4. Visualize Results
After evaluation completes:
```bash
python core/evaluation/plot_pflotran_metrics.py \
    --input_dir evaluation_results/epoch=0-step=2000/val/
```

## What Gets Saved

**During Evaluation:**
- `evaluation_results/<checkpoint>/val/gt_<i>.npy` - Ground truth (1, T, C, H, W)
- `evaluation_results/<checkpoint>/val/pred_<i>_sample_<j>.npy` - Predictions (1, T, C, H, W)
- `evaluation_results/<checkpoint>/val/metrics.npy` - MSE, MAE, RMSE metrics

**Checkpoint Timing:**
- Iteration 2000: ~8 min (first checkpoint)
- Iteration 4000: ~17 min
- Iteration 6000: ~26 min
- Iteration 8000: ~35 min
- Iteration 10000: ~44 min (final)

## Resume Training from Checkpoint

To resume training from a checkpoint:
```yaml
# In core/models/pflotran_identity.yaml
ckpt_path: ./logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/epoch=0-step=2000.ckpt
```

Then run:
```bash
sbatch core/job.sub
```

## Example Workflow

```bash
# 1. Training is running...
squeue -u $USER

# 2. Check for first checkpoint (after ~8 min)
ls -lh logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/

# 3. Quick validation check
python core/quick_check_checkpoint.py \
    logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/epoch=0-step=2000.ckpt

# 4. Run full evaluation
sbatch core/eval_job.sub \
    logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/epoch=0-step=2000.ckpt

# 5. Check evaluation progress
tail -f logs_t/eval_<jobid>_0000.out

# 6. Visualize results
python core/evaluation/plot_pflotran_metrics.py \
    --input_dir evaluation_results/epoch=0-step=2000/val/
```
