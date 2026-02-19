# Evaluation & Visualization Summary

## Quick Reference

### Option 1: One-Command Evaluation (Fastest)
```bash
./evaluate_quick.sh
```
- Uses latest checkpoint
- Generates 50 ensemble predictions for 5 test cases  
- Creates visualizations automatically
- **Time: ~5-10 minutes**

### Option 2: Custom Evaluation
```bash
python core/evaluation/evaluate_pflotran.py \
    --checkpoint logs/custom_pflotran/pflotran_identity/epoch=15-step=6000.ckpt \
    --num_samples 50 \
    --num_cases 10 \
    --visualize
```

### Option 3: Batch Job (for large evaluations)
```bash
sbatch job_evaluate.sub
```

## Available Checkpoints

```bash
ls -lh logs/custom_pflotran/pflotran_identity/*.ckpt
```

Expected output:
- `epoch=5-step=2000.ckpt` (420M)
- `epoch=10-step=4000.ckpt` (420M)
- `epoch=15-step=6000.ckpt` (420M)

## What You Get

### Predictions
- Location: `logs/custom_pflotran/pflotran_identity/predictions_step6000/`
- Files: `gt_0.npy`, `pred_0_sample_0.npy` to `pred_0_sample_49.npy`
- Format: (1, 40, 1, 40, 28) - batch, time, channels, height, width

### Visualizations
- Location: `logs/custom_pflotran/pflotran_identity/predictions_step6000/visualizations/`
- Files:
  - `case_0_full.png` - Complete comparison (GT, prediction, uncertainty, error)
  - `case_0_ensemble.png` - Individual ensemble members
  - `case_0_temporal.png` - Time evolution

## View Results

```bash
# List generated files
ls -lh logs/custom_pflotran/pflotran_identity/predictions_step*/visualizations/

# View with image viewer (if X11 forwarding enabled)
firefox logs/custom_pflotran/pflotran_identity/predictions_step6000/visualizations/*.png &

# Or copy to local machine
scp -r deception02:/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/logs/custom_pflotran/pflotran_identity/predictions_step6000/visualizations/ ./
```

## Customization

### Evaluate specific checkpoint
```bash
./evaluate_quick.sh logs/custom_pflotran/pflotran_identity/epoch=10-step=4000.ckpt
```

### More ensemble samples (better uncertainty)
```bash
python core/evaluation/evaluate_pflotran.py \
    --checkpoint logs/custom_pflotran/pflotran_identity/epoch=15-step=6000.ckpt \
    --num_samples 100 \
    --num_cases 5
```

### Specific test case visualization
```bash
python core/evaluation/visualize_pflotran.py \
    --output_dir logs/custom_pflotran/pflotran_identity/predictions_step6000 \
    --case_id 3 \
    --all_plots
```

## Files Created

### Scripts
- ✓ `evaluate_quick.sh` - One-command evaluation
- ✓ `job_evaluate.sub` - Slurm batch job
- ✓ `core/evaluation/evaluate_pflotran.py` - Main evaluation script
- ✓ `core/evaluation/visualize_pflotran.py` - Visualization script

### Documentation
- ✓ `EVALUATION_GUIDE.md` - Complete guide with examples

## Troubleshooting

**"Checkpoint not found"**
- Check path: `ls logs/custom_pflotran/pflotran_identity/*.ckpt`

**"No module named 'datasets'"**
- Make sure you're in project directory
- Activate conda: `conda activate dydiff`

**"CUDA out of memory"**
- Reduce `--num_samples` to 20 or 10

**Visualizations not showing**
- Check if X11 forwarding is enabled: `echo $DISPLAY`
- Copy files to local machine instead

## Complete Example

```bash
# 1. Activate environment
conda activate dydiff

# 2. Run evaluation
./evaluate_quick.sh

# 3. View results (example output shown)
$ ls logs/custom_pflotran/pflotran_identity/predictions_step6000/visualizations/
case_0_ensemble.png
case_0_full.png
case_0_temporal.png
case_1_ensemble.png
case_1_full.png
case_1_temporal.png
...

# 4. Check prediction accuracy in visualizations
#    - Look at "Absolute Error" row in *_full.png
#    - Check "Uncertainty (Std)" for confidence
#    - Compare GT vs Prediction Mean

# 5. For publication, generate specific cases
python core/evaluation/visualize_pflotran.py \
    --output_dir logs/custom_pflotran/pflotran_identity/predictions_step6000 \
    --case_id 0 \
    --all_plots
```

## Next Steps

After evaluation, you can:
1. Compare different checkpoints to find best model
2. Analyze uncertainty quantification quality
3. Use predictions for downstream tasks (calibration, forecasting)
4. Compute quantitative metrics with `plot_pflotran_metrics.py`

For detailed instructions, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
