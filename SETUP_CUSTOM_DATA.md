# Guide: Using Dynamical Diffusion for Your Custom PFLOTRAN Simulations

This guide explains how to adapt the dynamical diffusion model for your own PFLOTRAN ensemble.

## Overview

Your setup:
- **Grid size**: 28 × 40 (nx=28, nz=40)
- **Realizations**: 300 ensemble members
- **Temporal data**: First 240 days of solute concentration
- **Data format**: *.npy files

The model will learn to predict future solute concentration fields conditioned on:
1. **Context**: Previous 20 days of concentration
2. **Static properties**: Permeability and porosity fields
3. **Diffusion**: A learned diffusion process for uncertainty quantification

## Step 1: Prepare Your Data

### 1.1 Organize Your Concentration Data

Organize your concentration files in a single directory. The script expects one of these naming conventions:
```
conc_0.npy, conc_1.npy, ..., conc_299.npy
# OR
realization_0.npy, realization_1.npy, ...
# OR
0.npy, 1.npy, ..., 299.npy
```

Each file should have shape: `(240, 40, 28)` representing (timesteps, nz, nx).

### 1.2 Run Data Preparation Script

```bash
# Activate your environment
conda activate dydiff

# Create a directory for processed data
mkdir -p /path/to/your/data

# Run preparation script
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion
python prepare_custom_data.py \
  --conc_dir /path/to/your/raw/concentration/files \
  --output_dir /path/to/your/data \
  --n_realizations 300 \
  --n_timesteps 240 \
  --nx 28 \
  --nz 40
```

**Optional**: If you have permeability and porosity fields saved separately:

```bash
python prepare_custom_data.py \
  --conc_dir /path/to/concentration/files \
  --output_dir /path/to/your/data \
  --perm_values /path/to/perm.npy \
  --poro_values /path/to/poro.npy \
  --n_realizations 300 \
  --n_timesteps 240
```

Expected outputs:
- `conc.npy`: (300, 240, 40, 28) - concentration time series
- `perm.npy`: (300, 40, 28) - permeability fields
- `poro.npy`: (300, 40, 28) - porosity fields

If you don't provide `--perm_values` or `--poro_values`, the script creates constant fields (perm=1.0, poro=0.3) that you can later modify.

### 1.3 Create Train/Val/Test Splits

```bash
# Create splits directory
mkdir -p /path/to/your/data/splits

# Generate splits (default: 80% train, 16% val, 4% test)
python create_data_splits.py \
  --output_dir /path/to/your/data/splits \
  --n_realizations 300 \
  --train_ratio 0.80 \
  --val_ratio 0.16 \
  --seed 42
```

This creates:
- `train_idx.npy`: Indices for 192 training realizations
- `val_idx.npy`: Indices for 48 validation realizations
- `test_idx.npy`: Indices for 60 test realizations

## Step 2: Configure Your Model

### 2.1 Copy and Edit Configuration

```bash
cp core/models/custom_pflotran.yaml core/models/your_custom_config.yaml
```

Edit `your_custom_config.yaml` and update:

```yaml
data:
  perm_path: /path/to/your/data/perm.npy
  poro_path: /path/to/your/data/poro.npy
  conc_path: /path/to/your/data/conc.npy
  train_idx_path: /path/to/your/data/splits/train_idx.npy
  val_idx_path: /path/to/your/data/splits/val_idx.npy
  test_idx_path: /path/to/your/data/splits/test_idx.npy
```

### 2.2 Adjust Hyperparameters (Optional)

```yaml
data:
  pred_length: 10        # Predict 10 days instead of 20
  input_length: 30       # Use 30 days of context
  t_keep: 200            # Use only first 200 days instead of 240

training:
  batch_size: 8          # Reduce if running out of GPU memory
  learning_rate: 5e-5    # Lower learning rate for finetuning
  max_iterations: 100000 # Fewer iterations if data is similar
```

## Step 3: Train the Model

### 3.1 Basic Training

```bash
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion

python core/train_pflotran_dydiff.py \
  --config_file core/models/your_custom_config.yaml \
  --n_gpu 1 \
  --data_root /path/to/your/data \
  --model_root ./logs
```

### 3.2 Multi-GPU Training

```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/your_custom_config.yaml \
  --n_gpu 4 \
  --data_root /path/to/your/data \
  --model_root ./logs
```

### 3.3 Resume Training from Checkpoint

```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/your_custom_config.yaml \
  --n_gpu 1 \
  --resume logs/checkpoint-epoch=10.ckpt
```

## Step 4: Evaluate and Use the Model

Once training is complete, you can:

1. **View training logs**: Check `logs/your_custom_config/` for:
   - Checkpoints every 10k steps
   - Validation visualizations every 5k steps
   - Loss curves

2. **Inference**: Use a trained checkpoint to generate predictions (see evaluation examples in `core/evaluation/`)

## Understanding the Model

### Architecture

The model consists of:

1. **Stage 1 (VAE)**: Compresses concentration from (20×40×28) → latent (3×5×4)
   - Encodes spatial-temporal structure efficiently
   - Allows training on limited memory

2. **Stage 2 (Diffusion)**: Learns to denoise latent representations
   - Input: Previous 20 days (latent + static maps)
   - Output: Predicts next 20 days
   - Conditioned on permeability and porosity

3. **Conditioning**: Uses spatial static fields (perm, poro) as guidance
   - Injected through concatenation with temporal data
   - Helps transfer between simulations with different properties

### Key Concepts

- **Input/Context (20 days)**: Historical concentration
- **Prediction (20 days)**: Future concentration to predict
- **Static Conditioning**: Spatial fields of perm/poro help guide predictions
- **Uncertainty**: Diffusion model naturally captures prediction uncertainty

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size in config
```yaml
training:
  batch_size: 4  # or even 2
```

### Issue: Data Loading Fails

**Check**:
1. Are all files present and readable?
   ```bash
   ls -lh /path/to/your/data/*.npy
   ```

2. Do shapes match?
   ```bash
   python -c "import numpy as np; print(np.load('/path/to/your/data/conc.npy').shape)"
   # Should print: (300, 240, 40, 28)
   ```

3. Are split indices valid?
   ```bash
   python -c "import numpy as np; idx=np.load('/path/to/splits/train_idx.npy'); print(f'Train indices: min={idx.min()}, max={idx.max()}, len={len(idx)}')"
   # Should show: min=0, max<300, len=192 (or your split size)
   ```

### Issue: Poor Predictions

**Check and adjust**:
1. **Input/prediction length**: Try `input_length: 30, pred_length: 10` for shorter-term predictions
2. **Conditioning strength**: Ensure permeability/porosity fields are meaningful, not all constant
3. **Training time**: This model typically needs 50k-200k steps
4. **Learning rate**: Try `1e-5` or `5e-5` if loss oscillates

## Advanced Customization

### Using Pre-trained Checkpoint

If you have a checkpoint from the original PFLOTRAN experiment, you can finetune:

```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/your_custom_config.yaml \
  --ckpt_path /path/to/pretrained/checkpoint.ckpt \
  --n_gpu 1
```

### Different Grid Resolution

If your grid is different (e.g., 64×64), update the config:

```yaml
first_stage_config:
  params:
    ddconfig:
      resolution: 64  # Change from 40

unet_config:
  params:
    attention_resolutions: [8, 4, 2, 1]  # Adjust based on new size
```

### Adjusting Time Windows

For different temporal resolution:

```yaml
data:
  dt: 2  # Use every other day (skip days)
  t_keep: 120  # Use first 120 days
  input_length: 10  # 10 days of context
  pred_length: 10  # Predict 10 days
```

## File Structure After Setup

```
/path/to/your/data/
├── conc.npy              # (300, 240, 40, 28)
├── perm.npy              # (300, 40, 28)
├── poro.npy              # (300, 40, 28)
└── splits/
    ├── train_idx.npy     # 192 indices
    ├── val_idx.npy       # 48 indices
    └── test_idx.npy      # 60 indices

logs/your_custom_config/
├── checkpoints/          # Saved models
│   ├── epoch=0-step=10000.ckpt
│   └── epoch=1-step=20000.ckpt
├── config.yaml           # Saved configuration
└── samples/              # Validation visualizations
    ├── epoch=0-step=5000.png
    └── epoch=0-step=10000.png
```

## Next Steps

After training:

1. **Generate predictions** using your trained model
2. **Evaluate metrics** (RMSE, MAE, SSIM) against test set
3. **Ensemble forecasting** - generate multiple samples for uncertainty
4. **Uncertainty quantification** - analyze ensemble statistics

See `core/evaluation/evaluate_turbulence.py` for example evaluation scripts.
