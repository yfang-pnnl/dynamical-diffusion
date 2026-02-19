# PyTorch 2.0 Upgrade Guide

## What's Updated

### 1. Environment (environment.yaml)
- **PyTorch**: 1.12.1 → 2.0.1
- **Python**: 3.8.5 → 3.10
- **PyTorch Lightning**: 1.5.0 → 2.0.0
- **CUDA**: 11.3 → 11.8
- **xformers**: 0.0.16 → 0.0.20
- **torchmetrics**: 0.6.0 → 0.11.0

### 2. Code (train_pflotran_dydiff.py)
- Updated Trainer API: `gpus=` → `accelerator="gpu", devices=`
- Compatible with PyTorch Lightning 2.0

## Upgrade Instructions

### Option 1: Automated (Recommended)
```bash
chmod +x upgrade_pytorch.sh
./upgrade_pytorch.sh
```

### Option 2: Manual
```bash
# Backup current packages
conda activate dydiff
conda list --export > dydiff_old_packages.txt

# Remove old environment
conda deactivate
conda env remove -n dydiff -y

# Create new environment
conda env create -f environment.yaml

# Activate and verify
conda activate dydiff
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Expected Performance Improvements

### Single GPU
- **Before (PyTorch 1.13)**: ~12 it/s with vanilla attention
- **After (PyTorch 2.0)**: ~30-36 it/s with scaled_dot_product_attention
- **Speedup**: 2.5-3× faster

### Multi-GPU (4 GPUs with DDP)
- **Before**: Not practical (DDP overhead too high)
- **After**: ~120-140 it/s (4 × 30-35 it/s)
- **Training time**: 22 hours → ~3-4 hours

## Using Multi-GPU After Upgrade

Edit `job_simple.sub`:
```bash
#SBATCH --gres=gpu:4
```

Edit command:
```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 4
```

Edit `core/models/pflotran_identity.yaml`:
```yaml
training:
  batch_size: 1          # Per GPU
  num_workers: 1         # Reduce for multi-GPU
  accumulate_grad_batches: 1  # Can reduce since effective batch = 4
```

## Verify Installation

```bash
conda activate dydiff
python << EOF
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'SDP attention: {hasattr(torch.nn.functional, "scaled_dot_product_attention")}')

# Test SDP
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    q = torch.randn(1, 8, 10, 64).cuda()
    k = torch.randn(1, 8, 10, 64).cuda()
    v = torch.randn(1, 8, 10, 64).cuda()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f'SDP test: ✓ Passed (output shape: {out.shape})')
EOF
```

## Rollback (if needed)

```bash
conda env remove -n dydiff -y
conda env create -f dydiff_old_packages.txt
```

## Important Notes

1. **Cancel current training**: `scancel 509128` before upgrading
2. **Checkpoints compatible**: Can resume from old checkpoints
3. **Results comparable**: Same model architecture, just faster training
4. **Test first**: Run 100 iterations to verify before long training
