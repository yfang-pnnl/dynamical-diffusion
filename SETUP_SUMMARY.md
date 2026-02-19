# Summary: Dynamical Diffusion Setup for Custom PFLOTRAN Data

## What We Created For You

We've prepared a complete setup for training the dynamical diffusion model on your PFLOTRAN ensemble. Here's what's included:

### 📚 Documentation
1. **README_CUSTOM_SETUP.md** - Quick reference and overview
2. **SETUP_CUSTOM_DATA.md** - Detailed guide with all options and troubleshooting

### 🐍 Python Scripts
1. **prepare_custom_data.py** - Combine your concentration files into unified format
2. **create_data_splits.py** - Create train/val/test split indices
3. **utils_extract_properties.py** - Utility functions for handling perm/poro data

### ⚙️ Configuration
1. **core/models/custom_pflotran.yaml** - Template config (copy and edit for your data)

### 🚀 Optional Interactive Setup
1. **setup_quick_start.sh** - Guided script that walks you through the entire process

### 🔧 Bug Fixes
- Fixed a bug in `core/train_pflotran_dydiff.py` (was using wrong class name)

---

## Your 3-Step Setup

### Step 1: Prepare Data
```bash
conda activate dydiff
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion

python prepare_custom_data.py \
  --conc_dir /path/to/your/conc_*.npy/files \
  --output_dir /path/to/processed/data \
  --n_realizations 300 \
  --n_timesteps 240 \
  --nx 28 \
  --nz 40
```

**Output**: `conc.npy`, `perm.npy`, `poro.npy` (each 300 realizations)

### Step 2: Create Splits
```bash
python create_data_splits.py \
  --output_dir /path/to/processed/data/splits \
  --n_realizations 300
```

**Output**: `train_idx.npy`, `val_idx.npy`, `test_idx.npy` (192, 48, 60 samples)

### Step 3: Train Model
```bash
# Copy and customize config
cp core/models/custom_pflotran.yaml core/models/my_config.yaml
# Edit my_config.yaml to point to your data paths

# Train
python core/train_pflotran_dydiff.py \
  --config_file core/models/my_config.yaml \
  --n_gpu 1
```

---

## What the Model Does

- **Input**: 20 days of solute concentration + permeability/porosity fields
- **Output**: Predicts next 20 days of concentration
- **Learns**: Spatial-temporal patterns and uncertainty via diffusion model
- **Conditions on**: Static geological properties (perm, poro)

### Architecture Overview

```
Concentration (20 days)    Perm/Porosity
       ↓                         ↓
    VAE Encoder              Static Encoder
       ↓                         ↓
   Latent (3×5×4) ←--- Conditioning Injection
       ↓
    Diffusion UNet
       ↓
   Latent (3×5×4)
       ↓
    VAE Decoder
       ↓
    Concentration (20 days)
```

---

## Data Format Detail

Your data should become:

```
conc.npy           (300, 240, 40, 28)    # All concentrations
perm.npy           (300, 40, 28)         # All permeability fields
poro.npy           (300, 40, 28)         # All porosity fields
splits/
  train_idx.npy    (192,)                # Indices 0-299 for training
  val_idx.npy      (48,)                 # Indices for validation
  test_idx.npy     (60,)                 # Indices for testing
```

- **N=300**: Number of realizations (ensemble members)
- **T=240**: Time steps (days)
- **H=40**: Spatial dimension z (nz)
- **W=28**: Spatial dimension x (nx)

---

## If You Have Permeability/Porosity Data

### Already in (300, 40, 28) format?
```bash
python prepare_custom_data.py \
  --conc_dir ... \
  --output_dir ... \
  --perm_values /path/to/your/perm.npy \
  --poro_values /path/to/your/poro.npy
```

### Just scalar values (300,)?
```bash
python prepare_custom_data.py \
  --conc_dir ... \
  --output_dir ... \
  --perm_values /path/to/perm_300.npy \
  --poro_values /path/to/poro_300.npy
```
Script automatically expands to spatial fields.

### In HDF5 or CSV?
```bash
# Extract first
python utils_extract_properties.py from_hdf5 \
  --input data.h5 \
  --datasets permeability porosity \
  --output ./extracted/

# Then use extracted as above
```

---

## Validating Your Setup

After data preparation run:
```bash
python utils_extract_properties.py validate \
  --perm /path/to/perm.npy \
  --poro /path/to/poro.npy \
  --conc /path/to/conc.npy
```

Checks:
- ✓ Correct shapes
- ✓ No NaN/Inf values
- ✓ Value ranges reasonable
- ✓ Consistency across files

---

## Configuration Tips

### For Fast Testing (fewer steps)
```yaml
training:
  max_iterations: 10000
  batch_size: 8
  logger:
    logger_freq: 1000
    checkpoint_freq: 5000
```

### For Best Results
```yaml
training:
  max_iterations: 200000
  batch_size: 16
  learning_rate: 1e-4
  num_workers: 8
```

### For Large GPU (>32GB VRAM)
```yaml
training:
  batch_size: 32
  max_iterations: 400000
```

### For Small GPU (<16GB VRAM)
```yaml
training:
  batch_size: 4
  max_iterations: 150000
```

---

## Training Output

During training you'll see:
```
Epoch 1, Step 5000:   [Training loss plots]
Epoch 1, Step 5000:   [Sample predictions]
Epoch 2, Step 10000:  Training checkpoint saved
                      Validation metrics computed
                      ...
```

All saved to: `./logs/custom_pflotran/`

### Monitoring Training
- **Checkpoints**: `./logs/custom_pflotran/checkpoints/`
- **Visualizations**: `./logs/custom_pflotran/samples/`
- **Metrics**: TensorBoard log (if enabled)

---

## Next Steps After Training

1. **Inference**: Generate predictions on test set
2. **Metrics**: Compute RMSE, MAE, SSIM vs ground truth
3. **Uncertainty**: Sample multiple times for ensemble forecasts
4. **Analysis**: Examine where model succeeds/fails

See `core/evaluation/` for example scripts.

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| FileNotFoundError | Check file naming (conc_0.npy, conc_1.npy, ...) |
| Shape mismatch | Use `--nx 28 --nz 40` as specified |
| Out of Memory | Reduce batch_size: 16 → 8 → 4 |
| Loss not decreasing | Lower learning rate (1e-4 → 5e-5) or use more iterations |
| NaN in data | Check for negative concentrations (use log1p) |

See SETUP_CUSTOM_DATA.md for detailed troubleshooting.

---

## Environment

Use conda environment:
```bash
conda activate dydiff
```

All scripts are in:
```
/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/
```

---

## Questions?

1. **Quick reference**: README_CUSTOM_SETUP.md
2. **Detailed guide**: SETUP_CUSTOM_DATA.md
3. **Code documentation**: Inline comments in Python scripts
4. **Example configs**: core/models/diff_pflotran_static.yaml

---

**Status**: ✓ Ready to use! Follow the 3-step setup above.
