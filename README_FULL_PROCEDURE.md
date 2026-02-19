# full Procedures for PFLOTRAN Dynamical Diffusion

This guide outlines the complete steps to prepare your data and train the model using the **Identity VAE** approach (no separate VAE training required).

## Step 1: Prepare the Data

Extract the subset of data you need (first 300 realizations, 240 timesteps) and apply your material mask.

```bash
# 1. Activate environment
conda activate dydiff

# 2. Go to project root
cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion

# 3.Run preparation script
# (Replace paths with your actual file locations)
python prepare_existing_stacked_data.py \
  --perm_path /path/to/original/perm.npy \
  --poro_path /path/to/original/poro.npy \
  --conc_path /path/to/original/conc.npy \
  --material_path /path/to/original/material.npy \
  --output_dir ./data/processed \
  --n_members 300 \
  --n_timesteps 240
```

> **Outputs** in `./data/processed/`: `perm.npy`, `poro.npy`, `conc.npy`, `metadata.txt`

---

## Step 2: Create Data Splits

Generate indices for training (80%), validation (16%), and testing (4%).

```bash
python create_data_splits.py \
  --output_dir ./data/processed/splits \
  --n_realizations 300
```

> **Outputs** in `./data/processed/splits/`: `train_idx.npy`, `val_idx.npy`, `test_idx.npy`

---

## Step 3: Configure Training

1.  **Copy the Identity Config**:
    We created a config file that skips VAE training (`core/models/pflotran_identity.yaml`). Copy it to a new file for your run.

    ```bash
    cp core/models/pflotran_identity.yaml core/models/my_run_v1.yaml
    ```

2.  **Edit the Config**:
    Open `core/models/my_run_v1.yaml` and update the paths to match where you saved your data in Step 1 & 2.

    ```yaml
    data:
      perm_path: ./data/processed/perm.npy
      poro_path: ./data/processed/poro.npy
      conc_path: ./data/processed/conc.npy
      train_idx_path: ./data/processed/splits/train_idx.npy
      val_idx_path: ./data/processed/splits/val_idx.npy
      test_idx_path: ./data/processed/splits/test_idx.npy
      # ... verify t_keep is 240 ...
    ```

---

## Step 4: Run Training

You can run interactively for testing, or submit a batch job for long training.

### Option A: Interactive / Debug Run
Run on a single GPU to make sure everything works.

```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/my_run_v1.yaml \
  --n_gpu 1
```

### Option B: Submit Batch Job (SLURM)
Use the provided submission script `job_simple.sub`.

1.  **Edit `job_simple.sub`**:
    Ensure it points to your new config file.
    ```bash
    # ... inside job_simple.sub
    python core/train_pflotran_dydiff.py \
      --config_file core/models/my_run_v1.yaml \
      --n_gpu 8 
    ```

2.  **Submit**:
    ```bash
    sbatch job_simple.sub
    ```

---

## Step 5: Monitor & Evaluate

- **Logs**: Check `./logs/custom_pflotran/` (or whatever `save_dir` you set in the yaml).
- **Checkpoints**: Saved in `./logs/custom_pflotran/checkpoints/`.
- **Visualization**: The code logs images periodically. You can use TensorBoard if available, or just check the saved images in the log directory.

### Quick Evaluation
To generate samples from a trained checkpoint:

```bash
# (This script would need to be created/adapted, usually involving the 'dydiff' inference class)
# For now, rely on validation samples generated during training.
```
