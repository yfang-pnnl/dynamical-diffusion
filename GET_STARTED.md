# 🚀 GET STARTED - Custom PFLOTRAN Data Setup

**Welcome!** This guide gets you up and running in under 30 minutes.

## What You Have
- **Grid**: 28 × 40 (nx=28, nz=40)
- **Realizations**: 300 ensemble members
- **Timesteps**: 240 days of solute concentration
- **Format**: Individual *.npy files (one per realization)
- **Conda env**: `dydiff`

## What You'll Get
A trained diffusion model that:
- Takes 20 days of concentration history + permeability/porosity fields
- Predicts next 20 days of concentration
- Generates ensemble forecasts for uncertainty quantification

---

## ⚡ Quick Start (3 Commands!)

### 1️⃣ Prepare Your Data (5 min)
```bash
conda activate dydiff
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion

python prepare_custom_data.py \
  --conc_dir /path/to/your/concentration/files \
  --output_dir /path/to/save/processed/data \
  --n_realizations 300 \
  --n_timesteps 240 \
  --nx 28 \
  --nz 40
```

**Where**:
- `--conc_dir`: Directory containing `conc_0.npy`, `conc_1.npy`, ..., `conc_299.npy`
- `--output_dir`: Where to save combined files (creates if doesn't exist)

**Creates**:
- `conc.npy` (300, 240, 40, 28)
- `perm.npy` (300, 40, 28)
- `poro.npy` (300, 40, 28)

**Optional**: If you have perm/poro files, add:
```bash
  --perm_values /path/to/permeability.npy \
  --poro_values /path/to/porosity.npy
```

---

### 2️⃣ Create Data Splits (1 min)
```bash
python create_data_splits.py \
  --output_dir /path/to/save/processed/data/splits \
  --n_realizations 300
```

**Creates**:
- `splits/train_idx.npy` (192 training samples - 80%)
- `splits/val_idx.npy` (48 validation - 16%)
- `splits/test_idx.npy` (60 test - 4%)

---

### 3️⃣ Train the Model
```bash
# Copy and edit config
cp core/models/custom_pflotran.yaml core/models/my_data.yaml

# Edit my_data.yaml - update these paths:
#   perm_path: /path/to/save/processed/data/perm.npy
#   poro_path: /path/to/save/processed/data/poro.npy
#   conc_path: /path/to/save/processed/data/conc.npy
#   train_idx_path: /path/to/save/processed/data/splits/train_idx.npy
#   val_idx_path: /path/to/save/processed/data/splits/val_idx.npy
#   test_idx_path: /path/to/save/processed/data/splits/test_idx.npy

# Start training!
python core/train_pflotran_dydiff.py \
  --config_file core/models/my_data.yaml \
  --n_gpu 1
```

**Training runs for ~200k iterations** (adjust `max_iterations` in config if needed)
- Logs saved to `./logs/custom_pflotran/`
- Checkpoints every 10k steps
- Validation visualizations every 5k steps

---

## ✅ Verify Everything is Working

### After Step 1 (Prepare Data)
```bash
python inspect_data.py prepared \
  --perm /path/to/save/processed/data/perm.npy \
  --poro /path/to/save/processed/data/poro.npy \
  --conc /path/to/save/processed/data/conc.npy
```

Should show:
- ✓ Correct shapes: perm=(300,40,28), poro=(300,40,28), conc=(300,240,40,28)
- ✓ No NaNs or Infs
- ✓ Reasonable value ranges

### After Step 2 (Create Splits)
```bash
python -c "import numpy as np; \
  train=np.load('/path/to/splits/train_idx.npy'); \
  val=np.load('/path/to/splits/val_idx.npy'); \
  test=np.load('/path/to/splits/test_idx.npy'); \
  print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}'); \
  print(f'All unique: {len(set(train)|set(val)|set(test)) == len(train)+len(val)+len(test)}')"
```

Should show:
- Train: 192, Val: 48, Test: 60
- All unique: True

---

## 🎨 Optional: Visualize Your Data

```bash
# View distribution of values
python inspect_data.py dist \
  --perm perm.npy --poro poro.npy --conc conc.npy \
  --savefig distributions.png

# See temporal evolution
python inspect_data.py temporal \
  --conc conc.npy \
  --savefig temporal.png

# View a sample field
python inspect_data.py sample \
  --file perm.npy --sample 0 --title "Permeability Field #0" \
  --savefig sample_perm.png
```

---

## 🐛 Troubleshooting

### "FileNotFoundError: Could not find concentration file"
**Fix**: Check file naming. Files should be named exactly:
- `conc_0.npy`, `conc_1.npy`, ..., `conc_299.npy`

Or use alternative naming: `realization_0.npy`, `sim_0.npy`, or `0.npy`

### "Shape mismatch"
**Fix**: Verify your data shape:
```bash
python -c "import numpy as np; print(np.load('/path/to/conc_0.npy').shape)"
```
Should be: `(240, 40, 28)` for (time, nz, nx)

### "Out of Memory" during training
**Fix**: Edit `my_data.yaml`:
```yaml
training:
  batch_size: 8  # or even 4
```

### Training loss not decreasing
**Fix 1**: Try lower learning rate in config:
```yaml
training:
  model_attrs:
    learning_rate: 5e-5  # instead of 1e-4
```

**Fix 2**: Check if data needs normalization:
```bash
python inspect_data.py prepared --perm perm.npy --poro poro.npy --conc conc.npy
```
Look at value ranges - should be reasonable (not 1e10 or 1e-20)

---

## 📊 Monitoring Training

During training, watch the logs:
```bash
# View logs
tail -f ./logs/custom_pflotran/train.log

# Check latest checkpoint
ls -lht ./logs/custom_pflotran/checkpoints/ | head -5

# View validation samples
ls ./logs/custom_pflotran/samples/
```

Training metrics to watch:
- **Loss decreasing?** Good! Keep training.
- **Loss flat/increasing?** Adjust learning rate or batch size.
- **NaN loss?** Data issue - check for NaN/Inf in input data.

---

## 📚 More Documentation

Once you're comfortable with the basics:

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **SETUP_SUMMARY.md** | Visual overview | After this guide |
| **README_CUSTOM_SETUP.md** | Quick reference | When you need specific info |
| **SETUP_CUSTOM_DATA.md** | Full details | For advanced customization |
| **FILES_CREATED.md** | All scripts explained | To understand what each file does |

---

## 🎓 Understanding the Model

### Input
- **Concentration history**: Last 20 days of solute concentration (20, 40, 28)
- **Static fields**: Permeability and porosity (2, 40, 28)

### Output
- **Future concentration**: Next 20 days (20, 40, 28)

### Architecture
1. **VAE Encoder**: Compress concentration to latent space (3, 5, 4)
2. **Diffusion Model**: Learn to denoise latent + condition on perm/poro
3. **VAE Decoder**: Decode back to concentration space

### Why Diffusion?
- **Uncertainty**: Generate multiple samples for ensemble forecasts
- **Quality**: Better than deterministic prediction for complex dynamics
- **Conditioning**: Properly accounts for geological heterogeneity

---

## 🚀 After Training

Once you have a trained checkpoint:

1. **Generate predictions**:
   ```bash
   python core/train_pflotran_dydiff.py \
     --config_file core/models/my_data.yaml \
     --test --resume logs/custom_pflotran/checkpoint.ckpt
   ```

2. **Evaluate metrics**: See `core/evaluation/` for examples

3. **Ensemble forecasting**: Sample multiple times from the diffusion model

4. **Analyze uncertainty**: Compare prediction spread vs actual values

---

## ⏱️ Time Estimates

| Step | Time | Notes |
|------|------|-------|
| Data preparation | 5-10 min | Depends on file I/O speed |
| Create splits | <1 min | Very fast |
| Edit config | 2-5 min | Copy and update paths |
| Training | Hours-Days | Depends on GPU (V100: ~12hrs, A100: ~6hrs) |

**Total setup time**: ~30 minutes
**Total training time**: 6-24 hours (can start and leave running)

---

## 🎯 Success Checklist

Before starting training:
- [ ] Conda environment activated: `conda activate dydiff`
- [ ] Data prepared and validated (no NaNs)
- [ ] Splits created (192/48/60)
- [ ] Config file edited with correct paths
- [ ] GPU available (check with `nvidia-smi`) or willing to use CPU

During training:
- [ ] Loss is decreasing (check logs)
- [ ] Validation samples look reasonable
- [ ] No errors in terminal

After training:
- [ ] Checkpoint files saved
- [ ] Can generate predictions
- [ ] Results make physical sense

---

## 💬 Quick Command Reference

```bash
# Activate environment
conda activate dydiff

# Prepare data
python prepare_custom_data.py --conc_dir DIR --output_dir OUT --n_realizations 300 --n_timesteps 240 --nx 28 --nz 40

# Create splits
python create_data_splits.py --output_dir OUT/splits --n_realizations 300

# Validate
python inspect_data.py prepared --perm OUT/perm.npy --poro OUT/poro.npy --conc OUT/conc.npy

# Train
python core/train_pflotran_dydiff.py --config_file core/models/my_data.yaml --n_gpu 1

# Test
python core/train_pflotran_dydiff.py --config_file core/models/my_data.yaml --test --resume CKPT
```

---

## 🎉 You're Ready!

Follow the 3 steps above, and you'll have a working dynamical diffusion model for your PFLOTRAN data.

**Questions?** Check the other documentation files or inspect the Python scripts (they have detailed docstrings).

**Good luck!** 🚀
