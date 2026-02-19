# Custom PFLOTRAN Data Setup - Quick Reference

This document summarizes the scripts and steps to set up your custom PFLOTRAN simulations for training with the dynamical diffusion model.

## Your Data Requirements

- **Grid**: 28 × 40 (nx=28, nz=40)
- **Realizations**: 300
- **Temporal data**: First 240 days of solute concentration
- **Format**: *.npy files

## Files We've Created For You

### 1. **SETUP_CUSTOM_DATA.md** (Detailed Guide)
   Comprehensive step-by-step guide with troubleshooting, advanced options, and architecture explanations.

### 2. **prepare_custom_data.py** (Main Data Preparation)
   Combines your individual concentration files into the unified format expected by the model.
   
   ```bash
   python prepare_custom_data.py \
     --conc_dir /path/to/concentration/files \
     --output_dir /path/to/output \
     --n_realizations 300 \
     --n_timesteps 240 \
     --nx 28 --nz 40
   ```

### 3. **create_data_splits.py** (Train/Val/Test Splits)
   Creates indices for splitting 300 realizations into training/validation/test sets.
   
   ```bash
   python create_data_splits.py \
     --output_dir /path/to/output/splits \
     --n_realizations 300 \
     --train_ratio 0.80 \
     --val_ratio 0.16
   ```

### 4. **setup_quick_start.sh** (Interactive Setup)
   Guided interactive script that walks you through the entire setup process.
   
   ```bash
   bash setup_quick_start.sh
   ```

### 5. **core/models/custom_pflotran.yaml** (Training Configuration)
   Template configuration file. Copy and edit this to point to your data.

### 6. **utils_extract_properties.py** (Property Extraction Utilities)
   Helper functions if you have permeability/porosity in HDF5, CSV, or other formats.
   
   ```bash
   python utils_extract_properties.py validate --perm perm.npy --poro poro.npy --conc conc.npy
   ```

## Quick Start (3 Steps)

### Step 1: Prepare Your Data
```bash
conda activate dydiff
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion

python prepare_custom_data.py \
  --conc_dir /your/concentration/data \
  --output_dir /your/data/directory \
  --n_realizations 300 \
  --n_timesteps 240
```

### Step 2: Create Splits
```bash
python create_data_splits.py \
  --output_dir /your/data/directory/splits \
  --n_realizations 300
```

### Step 3: Train the Model
```bash
# Copy and edit the config template
cp core/models/custom_pflotran.yaml core/models/my_config.yaml
# Edit my_config.yaml and update the data paths

# Start training
python core/train_pflotran_dydiff.py \
  --config_file core/models/my_config.yaml \
  --n_gpu 1
```

## What Each Step Does

### `prepare_custom_data.py`
**Inputs**:
- Individual *.npy files: `conc_0.npy`, `conc_1.npy`, ..., `conc_299.npy`
- Each file: shape `(240, 40, 28)` for (time, nz, nx)

**Outputs**:
- `conc.npy`: (300, 240, 40, 28)
- `perm.npy`: (300, 40, 28) - constant 1.0 if not provided
- `poro.npy`: (300, 40, 28) - constant 0.3 if not provided

### `create_data_splits.py`
**Outputs** (to `splits/` directory):
- `train_idx.npy`: 192 indices for training
- `val_idx.npy`: 48 indices for validation
- `test_idx.npy`: 60 indices for testing

### `train_pflotran_dydiff.py`
**Trains** the diffusion model with:
- Input: 20 days of concentration history + permeability/porosity fields
- Output: Predicts next 20 days of concentration
- Saves checkpoints every 10k steps
- Validates every 5k steps

## Expected Outputs

After running all steps, you'll have:
```
/your/data/directory/
├── conc.npy              # Combined concentration tensor
├── perm.npy              # Permeability fields
├── poro.npy              # Porosity fields
└── splits/
    ├── train_idx.npy
    ├── val_idx.npy
    └── test_idx.npy

./logs/custom_pflotran/    # Training logs (created during training)
├── checkpoints/
├── samples/
└── tensorboard/
```

## What If I Have Perm/Poro Data Already?

### Option A: Pre-computed Spatial Fields
If you already have `perm.npy` and `poro.npy` with shape (300, 40, 28):
```bash
python prepare_custom_data.py \
  --conc_dir /your/concentration/data \
  --output_dir /your/data/directory \
  --perm_values /path/to/perm.npy \
  --poro_values /path/to/poro.npy
```

### Option B: Scalar Values Per Realization
If you only have 1 perm and 1 poro value per realization (shape: 300,):
```bash
python prepare_custom_data.py \
  --conc_dir /your/concentration/data \
  --output_dir /your/data/directory \
  --perm_values /path/to/perm_scalars.npy \
  --poro_values /path/to/poro_scalars.npy
```
The script will automatically expand them to spatial fields.

### Option C: HDF5 or CSV Files
Use the utilities script:
```bash
# From HDF5
python utils_extract_properties.py from_hdf5 \
  --input data.h5 \
  --datasets permeability porosity \
  --output /extracted/

# Then pass to prepare_custom_data
python prepare_custom_data.py \
  --conc_dir ... \
  --perm_values /extracted/permeability.npy \
  --poro_values /extracted/porosity.npy
```

## Validating Your Setup

After data preparation, validate everything:
```bash
python utils_extract_properties.py validate \
  --perm /your/data/directory/perm.npy \
  --poro /your/data/directory/poro.npy \
  --conc /your/data/directory/conc.npy
```

This will check:
- Correct shapes
- No NaN or Inf values
- Value ranges
- Overall consistency

## Still at different grid size?

If your grid is different (e.g., 64×64 instead of 28×40), tell `prepare_custom_data.py`:
```bash
python prepare_custom_data.py \
  --conc_dir ... \
  --output_dir ... \
  --nx 64 --nz 64
```

Then update `custom_pflotran.yaml`:
```yaml
first_stage_config:
  params:
    ddconfig:
      resolution: 64  # Change from 40
```

## Model Training Parameters to Try

In `custom_pflotran.yaml`, adjust based on your data:

```yaml
data:
  input_length: 30   # 30 days of context (default: 20)
  pred_length: 10    # Predict 10 days (default: 20)
  
training:
  batch_size: 8      # Reduce if OOM (default: 16)
  learning_rate: 1e-4  # Start here, try 5e-5 if loss oscillates
  max_iterations: 100000  # Fewer iterations initially
```

## Troubleshooting

**"FileNotFoundError: Could not find concentration file"**
- Check your file naming: need `conc_0.npy`, `conc_1.npy`, etc.
- Or pass `--conc_dir` with the correct directory

**"Not enough timesteps"**
- Your `--n_timesteps` is too small for `input_length + pred_length`
- Use `--n_timesteps 240` as specified

**"Out of Memory" during training**
- Reduce `batch_size` in config: 16 → 8 → 4
- Or reduce `max_iterations` for shorter training runs

**Training loss not decreasing**
- Check if data is normalized (should be in reasonable range)
- Try lower learning rate: 1e-4 → 5e-5
- Increase training iterations

## Next Steps After Training

1. **Inference**: Generate predictions using your trained checkpoint
2. **Evaluation**: Compare predictions to held-out test data
3. **Uncertainty Quantification**: Sample from the diffusion model for ensemble forecasts
4. **Analysis**: See scripts in `core/evaluation/`

## Questions?

- Check `SETUP_CUSTOM_DATA.md` for detailed information
- Review inline comments in the Python scripts
- Look at existing PFLOTRAN setup in `core/models/pflotran/`

---

**Environment**: Use `conda activate dydiff` before running any commands.

**Location**: All scripts are in `/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/`
