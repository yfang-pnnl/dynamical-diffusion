# Files Created for Custom PFLOTRAN Setup

## 📋 Summary

We've created a complete toolkit for adapting the dynamical diffusion model to your PFLOTRAN simulations. Below is a comprehensive list of all files and what they do.

---

## 📖 Documentation Files (Read These First!)

### 1. **SETUP_SUMMARY.md** ← Start here!
   - **What**: Quick overview and visual summary
   - **Read time**: 5 minutes
   - **Contains**: 
     - Your 3-step setup process
     - Architecture overview
     - Common solutions
     - Tips for different GPU setups

### 2. **README_CUSTOM_SETUP.md** 
   - **What**: Quick reference guide
   - **Read time**: 10 minutes
   - **Contains**:
     - File descriptions
     - Step-by-step instructions
     - Troubleshooting guide
     - Next steps after training

### 3. **SETUP_CUSTOM_DATA.md** 
   - **What**: Comprehensive detailed guide
   - **Read time**: 20+ minutes
   - **Contains**:
     - Complete walkthrough with all options
     - Data format details
     - Advanced customization
     - Architecture explanation
     - Expected outputs and file structure

---

## 🐍 Python Data Preparation Scripts

### 1. **prepare_custom_data.py**
   - **Location**: `/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/`
   - **Purpose**: Combine your individual *.npy concentration files into unified tensors
   - **Input**: Your concentration files (conc_0.npy, conc_1.npy, ..., conc_299.npy)
   - **Output**: 
     - `conc.npy` (300, 240, 40, 28)
     - `perm.npy` (300, 40, 28) 
     - `poro.npy` (300, 40, 28)
   - **Usage**:
     ```bash
     python prepare_custom_data.py \
       --conc_dir /your/data \
       --output_dir /output \
       --n_realizations 300 \
       --n_timesteps 240 \
       --nx 28 --nz 40
     ```

### 2. **create_data_splits.py**
   - **Location**: `/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/`
   - **Purpose**: Create train/val/test split indices (80/16/4)
   - **Input**: Total count of realizations
   - **Output**:
     - `train_idx.npy` (192 indices)
     - `val_idx.npy` (48 indices)
     - `test_idx.npy` (60 indices)
   - **Usage**:
     ```bash
     python create_data_splits.py \
       --output_dir /output/splits \
       --n_realizations 300
     ```

### 3. **utils_extract_properties.py**
   - **Location**: `/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/`
   - **Purpose**: Helper utilities for permeability/porosity data
   - **Features**:
     - Extract from HDF5 files
     - Load from text/CSV files
     - Normalize values
     - Validate data formats
   - **Usage**:
     ```bash
     # Validate prepared data
     python utils_extract_properties.py validate \
       --perm perm.npy --poro poro.npy --conc conc.npy
     
     # Extract from HDF5
     python utils_extract_properties.py from_hdf5 \
       --input data.h5 --datasets permeability porosity --output ./
     ```

### 4. **inspect_data.py**
   - **Location**: `/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/`
   - **Purpose**: Inspect and visualize your data
   - **Features**:
     - Examine raw files
     - Check prepared data stats
     - Compare distributions
     - Analyze temporal evolution
     - Visualize sample fields
   - **Usage**:
     ```bash
     # Inspect raw files
     python inspect_data.py raw --conc_dir /your/data
     
     # Check prepared data
     python inspect_data.py prepared \
       --perm perm.npy --poro poro.npy --conc conc.npy
     
     # Compare distributions
     python inspect_data.py dist \
       --perm perm.npy --poro poro.npy --conc conc.npy --savefig dist.png
     
     # Temporal analysis
     python inspect_data.py temporal --conc conc.npy --savefig temporal.png
     
     # Visualize sample
     python inspect_data.py sample --file perm.npy --sample 0 --title "Permeability"
     ```

---

## ⚙️ Configuration Files

### 1. **core/models/custom_pflotran.yaml**
   - **Location**: `/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/core/models/`
   - **Purpose**: Template configuration for training
   - **What to edit**: 
     - Data paths (your processed data directory)
     - Hyperparameters (learning rate, batch size, etc.)
     - Network architecture settings
   - **How to use**:
     ```bash
     # Copy template
     cp core/models/custom_pflotran.yaml core/models/my_config.yaml
     
     # Edit with your paths
     # nano core/models/my_config.yaml
     
     # Use for training
     python core/train_pflotran_dydiff.py --config_file core/models/my_config.yaml
     ```

---

## 🚀 Automation Scripts

### 1. **setup_quick_start.sh**
   - **Location**: `/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/`
   - **Purpose**: Interactive guided setup (optional, for convenience)
   - **Features**:
     - Asks you questions about your data
     - Runs data preparation automatically
     - Creates splits
     - Generates configured YAML file
     - Shows next steps
   - **Usage**:
     ```bash
     bash setup_quick_start.sh
     ```
   - **Note**: You can skip this and do steps manually if you prefer

---

## 🔧 Bug Fixes Applied

### File: **core/train_pflotran_dydiff.py**
   - **Bug**: Line 15 was using wrong class name `PFLOTRANDatasetConcStatic`
   - **Fix**: Changed to correct class `PFLOTRANDataModuleForDyDiff`
   - **Status**: ✓ Fixed

---

## 📂 Expected Directory Structure After Setup

```
/your/data/directory/
├── conc.npy              # (300, 240, 40, 28) - concentration time series
├── perm.npy              # (300, 40, 28) - permeability
├── poro.npy              # (300, 40, 28) - porosity
└── splits/
    ├── train_idx.npy     # 192 training indices
    ├── val_idx.npy       # 48 validation indices
    └── test_idx.npy      # 60 test indices

/qfs/projects/dl_calibration/d3m045/dynamical-diffusion/
├── core/models/
│   ├── custom_pflotran.yaml     # Your edited config template
│   └── my_config.yaml           # Your actual config (if you prefer custom name)
│
├── prepare_custom_data.py        # Data preparation script
├── create_data_splits.py         # Split creation script
├── utils_extract_properties.py   # Property extraction utilities
├── inspect_data.py               # Data inspection tool
├── setup_quick_start.sh          # Interactive setup (optional)
│
├── SETUP_SUMMARY.md              # Quick overview (START HERE)
├── README_CUSTOM_SETUP.md        # Quick reference
├── SETUP_CUSTOM_DATA.md          # Detailed guide
├── FILES_CREATED.md              # This file
│
└── logs/custom_pflotran/         # Created during training
    ├── checkpoints/              # Model checkpoints
    ├── samples/                  # Validation visualizations
    └── config.yaml               # Saved config
```

---

## 🎯 What to Do Next

### Immediate (What You'll Do First)
1. **Read**: SETUP_SUMMARY.md (5 min)
2. **Prepare**: Run `prepare_custom_data.py` (5 min)
3. **Split**: Run `create_data_splits.py` (1 min)
4. **Validate**: Run inspection script (5 min)
5. **Configure**: Edit config file (5 min)
6. **Train**: Start training (varies by GPU)

### Short Term (During Training)
1. Monitor training in `logs/custom_pflotran/`
2. Check validation visualizations
3. Adjust hyperparameters if needed

### Long Term (After Training)
1. Evaluate on test set
2. Generate predictions
3. Uncertainty quantification
4. Analysis and interpretation

---

## 💡 Tips for Success

### Data Preparation
- ✓ Keep all concentration files in one directory
- ✓ Use consistent naming: `conc_0.npy`, `conc_1.npy`, etc.
- ✓ Run `inspect_data.py` to check before training
- ✓ Look for NaNs, Infs, or extreme values

### Training
- ✓ Start with smaller `max_iterations` (50k) to test
- ✓ Monitor loss curves in logs
- ✓ Save checkpoints often (`checkpoint_freq: 5000`)
- ✓ Use TensorBoard or similar for visualization

### Debugging
- If shapes don't match, use `--nx` and `--nz` flags
- If data validation fails, check raw files with `inspect_data.py raw`
- If training is slow, reduce `batch_size` or `num_workers`
- If loss is bad, check learning rate and data normalization

---

## 📚 External Resources

- **Original Paper**: Dynamical Diffusion for temporal prediction
- **Configuration Syntax**: OmegaConf (used for .yaml configs)
- **Training Framework**: PyTorch Lightning
- **Your Data**: PFLOTRAN simulations with permeability/porosity variations

---

## ✅ Checklist

Before training, make sure you have:
- [ ] Read SETUP_SUMMARY.md
- [ ] Run `prepare_custom_data.py` successfully
- [ ] Run `create_data_splits.py` successfully
- [ ] Validated data with `inspect_data.py validate`
- [ ] Edited config file with correct data paths
- [ ] Activated environment: `conda activate dydiff`
- [ ] GPU available: `nvidia-smi` (or CPU if preferred)

---

## 🆘 Getting Help

### Documentation
- See "README_CUSTOM_SETUP.md" for quick reference
- See "SETUP_CUSTOM_DATA.md" for detailed information

### Scripts Help
- Each Python script has `--help` flag:
  ```bash
  python prepare_custom_data.py --help
  python inspect_data.py --help
  ```

### Debugging
- Use `inspect_data.py` to understand your data
- Check logs in `./logs/custom_pflotran/`
- Look at inline script documentation

### Original Code
- See `core/models/pflotran/diff_pflotran_static.yaml` for reference config
- See `core/datasets/pflotran/` for data loading details

---

**Status**: ✅ All setup files created and tested

**Last Updated**: February 13, 2026

**Environment**: `dydiff` conda environment
