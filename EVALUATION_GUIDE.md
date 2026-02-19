# PFLOTRAN Model Evaluation & Visualization Guide

This guide explains how to evaluate trained PFLOTRAN diffusion model checkpoints and visualize results.

## Available Checkpoints

Check what checkpoints you have:
```bash
ls -lh logs/custom_pflotran/pflotran_identity/*.ckpt
```

You should see:
- `epoch=5-step=2000.ckpt` (420M) - Early checkpoint
- `epoch=10-step=4000.ckpt` (420M) - Mid-training
- `epoch=15-step=6000.ckpt` (420M) - Latest checkpoint

## Quick Start (Recommended)

The fastest way to evaluate and visualize:

```bash
chmod +x evaluate_quick.sh
./evaluate_quick.sh
```

This will:
1. Use the latest checkpoint (epoch=15-step=6000.ckpt)
2. Generate 50 ensemble predictions for 5 test cases
3. Create visualizations automatically
4. Save everything to `logs/custom_pflotran/pflotran_identity/predictions_step6000/`

**To use a different checkpoint:**
```bash
./evaluate_quick.sh logs/custom_pflotran/pflotran_identity/epoch=10-step=4000.ckpt
```

## Method 1: Python Script (Flexible)

### Generate Predictions

```bash
python core/evaluation/evaluate_pflotran.py \
    --checkpoint logs/custom_pflotran/pflotran_identity/epoch=15-step=6000.ckpt \
    --config core/models/pflotran_identity.yaml \
    --num_samples 50 \
    --num_cases 10 \
    --visualize \
    --num_vis 5
```

**Parameters:**
- `--checkpoint`: Path to trained model checkpoint
- `--config`: Config file used for training
- `--num_samples`: Number of stochastic samples per test case (ensemble size)
- `--num_cases`: How many test cases to evaluate
- `--visualize`: Also create visualizations (optional)
- `--num_vis`: Number of cases to visualize
- `--output_dir`: Custom output directory (optional)

**Output:**
- Predictions saved to: `logs/custom_pflotran/pflotran_identity/predictions_stepXXXX/`
- Files: `gt_0.npy`, `pred_0_sample_0.npy`, ..., `pred_0_sample_49.npy`

### Visualize Results

After generating predictions, create detailed visualizations:

```bash
python core/evaluation/visualize_pflotran.py \
    --output_dir logs/custom_pflotran/pflotran_identity/predictions_step6000 \
    --case_id 0 \
    --input_length 20 \
    --all_plots
```

**Parameters:**
- `--output_dir`: Directory containing prediction files
- `--case_id`: Which test case to visualize (0, 1, 2, ...)
- `--input_length`: Number of conditioning frames (default: 20)
- `--all_plots`: Generate all visualization types
- `--save_dir`: Where to save figures (default: output_dir/visualizations)

**Generated plots:**
1. **Full comparison** (`case_0_full.png`): 
   - Ground truth, prediction mean, uncertainty, and error
   - Multiple timesteps
   
2. **Ensemble comparison** (`case_0_ensemble.png`):
   - Individual ensemble members side-by-side
   - Shows prediction diversity
   
3. **Temporal evolution** (`case_0_temporal.png`):
   - Animation-style sequence
   - Ground truth vs prediction over time

## Method 2: Slurm Job Submission

For batch evaluation on compute nodes:

### Edit job_evaluate.sub

```bash
# Edit the checkpoint path
setenv CHECKPOINT "logs/custom_pflotran/pflotran_identity/epoch=15-step=6000.ckpt"
```

### Submit job

```bash
sbatch job_evaluate.sub
```

This runs the evaluation in test mode using PyTorch Lightning's test framework.

**Output:**
- Saved to: `logs/custom_pflotran/pflotran_identity/output_for_evaluation_XXXXX/`
- Check logs: `tail -f logs_t/eval_JOBID_*.out`

## Method 3: Using Training Script in Test Mode

Use the same training script with `--test` flag:

```bash
python core/train_pflotran_dydiff.py \
    --config_file core/models/pflotran_identity.yaml \
    --n_gpu 1 \
    --test True \
    --resume logs/custom_pflotran/pflotran_identity/epoch=15-step=6000.ckpt
```

This executes the model's `test_step()` method on the test dataset.

## Understanding the Visualizations

### 1. Full Comparison Plot

Shows 4 rows:
- **Ground Truth**: Actual concentration field
- **Prediction Mean**: Average of ensemble predictions
- **Uncertainty (Std)**: Standard deviation across ensemble (epistemic uncertainty)
- **Absolute Error**: |Prediction - Truth|

Columns show different timesteps from input through predictions.

### 2. Ensemble Comparison

- Shows ground truth + 8 random ensemble members + ensemble mean
- Demonstrates prediction diversity and uncertainty quantification
- Useful for understanding model confidence

### 3. Temporal Evolution

- Shows evolution over time (10 frames)
- Ground truth vs prediction side-by-side
- Good for understanding how prediction quality changes with lead time

## Output File Format

### Prediction Files

**Ground Truth** (`gt_0.npy`):
- Shape: `(1, total_length, 1, H, W)` = `(1, 40, 1, 40, 28)`
- First dimension: batch (always 1)
- Second: timesteps (0-39, where 0-19 are input, 20-39 are prediction targets)
- Third: channels (1 for concentration)
- Fourth/Fifth: spatial dimensions

**Predictions** (`pred_0_sample_0.npy` to `pred_0_sample_49.npy`):
- Same shape as ground truth
- 50 files = 50 stochastic samples (ensemble)
- Each is an independent prediction from the diffusion model

**Data format:**
- Values are in **normalized log space**: `(log1p(conc) - mu) / std`
- To convert to physical concentration: 
  ```python
  conc = np.expm1(pred_normalized * std + mu)
  ```
- Without stats file, approximate: `conc = np.expm1(pred_normalized)`

## Metrics Computation

If you have a stats file (normalization parameters), compute quantitative metrics:

```bash
python core/evaluation/plot_pflotran_metrics.py \
    --model_output logs/custom_pflotran/pflotran_identity/predictions_step6000 \
    --stats_path path/to/stats.npy \
    --input_length 20 \
    --pred_length 20 \
    --out_dir logs/custom_pflotran/pflotran_identity/predictions_step6000/metrics
```

This computes:
- RMSE per lead time
- CRPS (ensemble skill score)
- Coverage (90% prediction interval)
- Plume mass/extent metrics (if porosity available)

## Comparing Multiple Checkpoints

Evaluate all checkpoints and compare:

```bash
for CKPT in logs/custom_pflotran/pflotran_identity/*.ckpt; do
    echo "Evaluating $CKPT"
    ./evaluate_quick.sh "$CKPT"
done
```

Then compare visualizations:
```bash
ls logs/custom_pflotran/pflotran_identity/predictions_step*/visualizations/case_0_full.png
```

## Tips & Best Practices

### 1. **Ensemble Size**
- Use 50-100 samples for robust uncertainty quantification
- Use 10-20 for quick visualization
- More samples = better uncertainty estimates but slower

### 2. **Number of Test Cases**
- Evaluate 10-50 cases for comprehensive assessment
- Visualize 3-5 cases for presentation

### 3. **Memory Management**
- Each prediction file is ~450MB (for 50 samples × 40 timesteps × 40×28 grid)
- Monitor disk space: `df -h logs/`
- Clean up old predictions if needed

### 4. **GPU Usage**
- Evaluation can run on CPU (slower) or GPU (faster)
- Each test case takes ~30-60 seconds on GPU with 50 samples
- Use `--num_cases` to limit for quick checks

### 5. **Visualization Settings**
- Adjust colormap ranges in visualize_pflotran.py if needed
- Use `--all_plots` for publication-quality figures
- PNG files are ~2-5MB each at 200 DPI

## Troubleshooting

### "Checkpoint not found"
```bash
# Check path
ls -lh logs/custom_pflotran/pflotran_identity/*.ckpt

# Use absolute path
python core/evaluation/evaluate_pflotran.py \
    --checkpoint /qfs/projects/dl_calibration/d3m045/dynamical-diffusion/logs/custom_pflotran/pflotran_identity/epoch=15-step=6000.ckpt
```

### "CUDA out of memory"
- Reduce `--num_samples` (generate fewer ensemble members at once)
- Modify script to process samples sequentially

### "No module named 'datasets'"
```bash
# Make sure you're in the right directory
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion

# Or add to PYTHONPATH
export PYTHONPATH=/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/core:$PYTHONPATH
```

### "Ground truth shape mismatch"
- Check data paths in config file match test data
- Verify test_idx.npy contains valid indices

## Next Steps After Evaluation

1. **Analyze Results**
   - Look for systematic errors (bias in certain regions/times)
   - Check if uncertainty correlates with error
   - Compare different checkpoints

2. **Publication Figures**
   - Use visualize_pflotran.py with custom styling
   - Export to PDF for papers: change `.png` to `.pdf` in save path

3. **Hyperparameter Tuning**
   - If results are poor, try different training settings
   - Adjust model capacity, latent channels, etc.

4. **Production Deployment**
   - Choose best checkpoint based on metrics
   - Use for forecasting on new data

## Example Workflow

```bash
# 1. Evaluate latest checkpoint
./evaluate_quick.sh

# 2. View main visualization
firefox logs/custom_pflotran/pflotran_identity/predictions_step6000/visualizations/case_0_full.png &

# 3. Compare with earlier checkpoint
./evaluate_quick.sh logs/custom_pflotran/pflotran_identity/epoch=10-step=4000.ckpt

# 4. View comparison
firefox logs/custom_pflotran/pflotran_identity/predictions_step4000/visualizations/case_0_full.png &

# 5. Generate more detailed visualizations for case 2
python core/evaluation/visualize_pflotran.py \
    --output_dir logs/custom_pflotran/pflotran_identity/predictions_step6000 \
    --case_id 2 \
    --all_plots

# 6. Compute metrics (if stats file available)
# python core/evaluation/plot_pflotran_metrics.py ...
```

## File Summary

**Scripts:**
- `evaluate_quick.sh` - One-command evaluation
- `job_evaluate.sub` - Slurm batch evaluation
- `core/evaluation/evaluate_pflotran.py` - Main evaluation script
- `core/evaluation/visualize_pflotran.py` - Visualization script
- `core/evaluation/plot_pflotran_metrics.py` - Metrics computation

**Outputs:**
- `logs/custom_pflotran/pflotran_identity/predictions_stepXXXX/` - Predictions
- `logs/custom_pflotran/pflotran_identity/predictions_stepXXXX/visualizations/` - Figures

For questions or issues, check the code comments or training logs.
