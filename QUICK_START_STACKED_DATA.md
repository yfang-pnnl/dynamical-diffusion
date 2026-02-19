# Quick Start for Your Stacked Data

## Your Data Setup

You already have combined files:
- `perm.npy`: (1500, nz, nx) - **1500 realizations**
- `poro.npy`: (1500, nz, nx) - **1500 realizations**
- `conc.npy`: (300, 7300, nz, nx) - **300 realizations, 7300 timesteps**
- `material.npy`: (nz, nx) or (300, nz, nx) - **mask (0 = skip these locations)**
- First 300 members in perm/poro correspond to the 300 in conc

## Quick Setup (2 Steps)

### Step 1: Extract Relevant Subset (WITH Material Mask)

```bash
conda activate dydiff
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion

python prepare_existing_stacked_data.py \
  --perm_path /path/to/your/perm.npy \
  --poro_path /path/to/your/poro.npy \
  --conc_path /path/to/your/conc.npy \
  --material_path /path/to/your/material.npy \
  --output_dir /path/to/processed/data \
  --n_members 300 \
  --n_timesteps 240 \
  --timestep_start 0
```

**What this does**:
- Extracts first 300 realizations from perm/poro (matching your conc data)
- Extracts first 240 timesteps from concentration (or specify different range)
- **Applies material mask**: Sets masked regions (where material==0) to 0
- Validates data quality (checks for NaN, Inf, etc.)
- Saves processed subset ready for training

**About the Material Mask**:
- If `material.npy` is (nz, nx): Same mask for all realizations
- If `material.npy` is (300, nz, nx): Different mask per realization
- Locations where mask==0 are set to 0 in concentration data
- Perm/poro values are kept as-is (not masked) so the model knows the material properties

**Output**:
- `processed/data/perm.npy`: (300, nz, nx)
- `processed/data/poro.npy`: (300, nz, nx)
- `processed/data/conc.npy`: (300, 240, nz, nx)
- `processed/data/metadata.txt`: Info about processing

### Step 2: Create Data Splits

```bash
python create_data_splits.py \
  --output_dir /path/to/processed/data/splits \
  --n_realizations 300
```

**Output**:
- `splits/train_idx.npy`: 192 indices (80%)
- `splits/val_idx.npy`: 48 indices (16%)
- `splits/test_idx.npy`: 60 indices (4%)

### Step 3: Configure and Train

```bash
# Copy config template
cp core/models/custom_pflotran.yaml core/models/my_data.yaml

# Edit my_data.yaml - update paths:
#   perm_path: /path/to/processed/data/perm.npy
#   poro_path: /path/to/processed/data/poro.npy
#   conc_path: /path/to/processed/data/conc.npy
#   train_idx_path: /path/to/processed/data/splits/train_idx.npy
#   val_idx_path: /path/to/processed/data/splits/val_idx.npy
#   test_idx_path: /path/to/processed/data/splits/test_idx.npy
#   t_keep: 240  # Match n_timesteps

# Start training
python core/train_pflotran_dydiff.py \
  --config_file core/models/my_data.yaml \
  --n_gpu 1
```

---

## Options

### Without Material Mask

If you don't want to apply the mask (or don't have one), simply omit `--material_path`:
```bash
python prepare_existing_stacked_data.py \
  --perm_path /path/to/perm.npy \
  --poro_path /path/to/poro.npy \
  --conc_path /path/to/conc.npy \
  --output_dir /path/to/processed/data \
  --n_members 300 \
  --n_timesteps 240
```

### Custom Mask Value

By default, masked regions are set to 0. To use a different value (e.g., NaN):
```bash
python prepare_existing_stacked_data.py \
  --perm_path /path/to/perm.npy \
  --poro_path /path/to/poro.npy \
  --conc_path /path/to/conc.npy \
  --material_path /path/to/material.npy \
  --output_dir /path/to/processed/data \
  --mask_value nan \
  --n_members 300 \
  --n_timesteps 240
```

**Note**: Using NaN might cause issues during training. Stick with 0.0 unless you have a specific reason.

### Use Different Timestep Range

If you want timesteps 100-340 (240 days starting from day 100):
```bash
python prepare_existing_stacked_data.py \
  --perm_path /path/to/perm.npy \
  --poro_path /path/to/poro.npy \
  --conc_path /path/to/conc.npy \
  --output_dir /path/to/processed/data \
  --n_members 300 \
  --n_timesteps 240 \
  --timestep_start 100
```

### Use All 7300 Timesteps

```bash
python prepare_existing_stacked_data.py \
  --perm_path /path/to/perm.npy \
  --poro_path /path/to/poro.npy \
  --conc_path /path/to/conc.npy \
  --output_dir /path/to/processed/data \
  --n_members 300 \
  --n_timesteps 7300
```

Then update config:
```yaml
data:
  t_keep: 7300
  input_length: 20
  pred_length: 20
```

### Use Different Number of Realizations

If you want to use more realizations (e.g., 500):
```bash
python prepare_existing_stacked_data.py \
  --perm_path /path/to/perm.npy \
  --poro_path /path/to/poro.npy \
  --conc_path /path/to/conc.npy \
  --output_dir /path/to/processed/data \
  --n_members 500 \
  --n_timesteps 240
```

**Note**: This assumes you have 500 members with concentration data. If conc only has 300, this will fail. You'd need to handle the mismatch differently.

---

## Validation

After Step 1, verify your data:
```bash
python inspect_data.py prepared \
  --perm /path/to/processed/data/perm.npy \
  --poro /path/to/processed/data/poro.npy \
  --conc /path/to/processed/data/conc.npy
```

Should show:
```
Shapes:
  perm: (300, 40, 28)  (or your nz, nx)
  poro: (300, 40, 28)
  conc: (300, 240, 40, 28)

Value Ranges:
  perm: [min, max]
  poro: [min, max]
  conc: [min, max]

NaN/Inf Check:
  perm NaNs: 0, Infs: 0
  poro NaNs: 0, Infs: 0
  conc NaNs: 0, Infs: 0

✓ All checks passed!
```

---

## What You DON'T Need

Since your data is already stacked, you **don't need**:
- `prepare_custom_data.py` (this was for combining individual files)

You **DO need**:
- ✓ `prepare_existing_stacked_data.py` (new script for your case)
- ✓ `create_data_splits.py`
- ✓ `inspect_data.py` (for validation)
- ✓ `core/models/custom_pflotran.yaml` (config template)

---

## Example with Actual Paths

Assuming your data is at `/qfs/projects/dl_calibration/d3m045/dataset/`:

```bash
conda activate dydiff
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion

# Step 1: Extract subset
python prepare_existing_stacked_data.py \
  --perm_path /qfs/projects/dl_calibration/d3m045/dataset/perm.npy \
  --poro_path /qfs/projects/dl_calibration/d3m045/dataset/poro.npy \
  --conc_path /qfs/projects/dl_calibration/d3m045/dataset/conc.npy \
  --output_dir /qfs/projects/dl_calibration/d3m045/processed_data \
  --n_members 300 \
  --n_timesteps 240

# Step 2: Create splits
python create_data_splits.py \
  --output_dir /qfs/projects/dl_calibration/d3m045/processed_data/splits \
  --n_realizations 300

# Step 3: Validate
python inspect_data.py prepared \
  --perm /qfs/projects/dl_calibration/d3m045/processed_data/perm.npy \
  --poro /qfs/projects/dl_calibration/d3m045/processed_data/poro.npy \
  --conc /qfs/projects/dl_calibration/d3m045/processed_data/conc.npy

# Step 4: Train
cp core/models/custom_pflotran.yaml core/models/my_data.yaml
# Edit my_data.yaml with paths above
python core/train_pflotran_dydiff.py \
  --config_file core/models/my_data.yaml \
  --n_gpu 1
```

---

## Troubleshooting

### "Material mask shape doesn't match spatial dimensions"
**Issue**: Mask dimensions don't match your data grid.
**Fix**: Check shapes:
```bash
python -c "import numpy as np; \
  print('perm:', np.load('/path/to/perm.npy').shape); \
  print('conc:', np.load('/path/to/conc.npy').shape); \
  print('material:', np.load('/path/to/material.npy').shape)"
```
Material should be either (nz, nx) or (300, nz, nx).

### "Conc has only 300 members, need N"
**Issue**: Trying to extract more members than available in concentration data.
**Fix**: Use `--n_members 300` or less. Your concentration only has 300 realizations.

### "Perm and poro spatial dimensions don't match"
**Issue**: nz, nx dimensions differ between files.
**Fix**: Check your data shapes:
```bash
python -c "import numpy as np; \
  print('perm:', np.load('/path/to/perm.npy').shape); \
  print('poro:', np.load('/path/to/poro.npy').shape); \
  print('conc:', np.load('/path/to/conc.npy').shape)"
```

### "Conc has only T timesteps, need N"
**Issue**: Trying to extract more timesteps than available.
**Fix**: Check how many timesteps you have:
```bash
python -c "import numpy as np; \
  print('Total timesteps:', np.load('/path/to/conc.npy').shape[1])"
```
Then use `--n_timesteps` accordingly.

---

## Time Estimates

| Step | Time | Notes |
|------|------|-------|
| Extract subset | 1-5 min | Depends on file size and I/O |
| Create splits | <1 min | Very fast |
| Validation | 1 min | Quick check |
| Training | 6-24 hrs | Depends on GPU |

---

## Summary

Your workflow is simpler than the general case since data is already stacked:

1. ✓ Run `prepare_existing_stacked_data.py` to extract relevant subset
2. ✓ Run `create_data_splits.py` to create train/val/test indices
3. ✓ Edit config with paths
4. ✓ Train!

**Total setup time**: ~10 minutes
