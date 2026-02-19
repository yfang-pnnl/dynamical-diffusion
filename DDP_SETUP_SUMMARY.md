# DDP and Single GPU Training - Quick Reference

## Files Updated

### Training Code
- **[core/train_pflotran_dydiff.py](core/train_pflotran_dydiff.py)**
  - ✓ Auto-detects single vs multi-GPU mode
  - ✓ Optimized DDP strategy with `gradient_as_bucket_view`
  - ✓ Auto-reduces num_workers for multi-GPU
  - ✓ Synchronized batch norm for multi-GPU
  - ✓ Prints training configuration on startup
  - ✓ Added `--find_unused_parameters` flag for debugging

### Configuration
- **[core/models/pflotran_identity.yaml](core/models/pflotran_identity.yaml)**
  - ✓ Optimized `num_workers: 2` (works for both modes)
  - ✓ Added comments explaining effective batch size

### Job Scripts
- **[job_simple.sub](job_simple.sub)** - Single GPU (1 GPU, 3 hours)
- **[job_multi_gpu.sub](job_multi_gpu.sub)** - Multi-GPU (4 GPUs, 6 hours)

### Documentation
- **[GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)** - Complete guide
- **[test_gpu_modes.sh](test_gpu_modes.sh)** - Test script

## Usage

### Single GPU
```bash
sbatch job_simple.sub
```

### Multi-GPU (4 GPUs)
```bash
sbatch job_multi_gpu.sub
```

### Custom
```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu <1 or 4>
```

## Key Features

### Automatic Mode Detection
The code automatically configures itself based on `--n_gpu`:

| Parameter | Single GPU | Multi-GPU (4) |
|-----------|------------|---------------|
| Strategy | None | DDPStrategy |
| num_workers | 2 | 2 (auto-checked) |
| Batch norm | Regular | Synchronized |
| Effective batch | 1×4=4 | 4×1×4=16 |
| Speed | ~30-36 it/s | ~120-140 it/s |

### Training Output
You'll see:
```
================================================================================
Training Configuration:
  Mode: DDP Multi-GPU        (or "Single GPU")
  GPUs: 4                    (or 1)
  Batch size per GPU: 1
  Effective batch size: 16   (or 4)
  Num workers: 2
================================================================================
```

## Testing

### Quick test (recommended before full run)
```bash
# Make executable
chmod +x test_gpu_modes.sh

# Run tests
./test_gpu_modes.sh
```

### Manual test
```bash
# Test single GPU (100 iterations)
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 1 \
  --max_iterations 100

# Test multi-GPU (100 iterations)
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 4 \
  --max_iterations 100
```

## Performance Expectations (PyTorch 2.0)

### Single GPU
- Speed: ~30-36 it/s
- Full training (200k iter): ~22 hours
- Needs 3× 3-hour jobs (use resume flag)

### Multi-GPU (4 GPUs)
- Speed: ~120-140 it/s  
- Full training (200k iter): ~5-6 hours
- Fits in one 6-hour job ✓

## Troubleshooting

### DDP Issues
If you get DDP errors:
```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 4 \
  --find_unused_parameters True
```

### Check CUDA/GPU
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

### Monitor Training
```bash
# Watch logs
tail -f logs_t/pflo_JOBID_*.out

# Check GPU usage (on compute node)
nvidia-smi
```

## What Changed from Old Code

**Before (PyTorch 1.13)**:
- Only vanilla attention (slow)
- DDP had issues with batch_size=1
- `gpus=` parameter (deprecated in Lightning 2.0)
- Manual strategy configuration

**After (PyTorch 2.0)**:
- Scaled dot product attention (2-3× faster)
- Optimized DDP with gradient bucketing
- Modern `accelerator=` + `devices=` API
- Auto-configured strategy based on GPU count
- Synchronized batch norm for multi-GPU

## Next Steps

1. **Test both modes** (optional but recommended):
   ```bash
   chmod +x test_gpu_modes.sh
   ./test_gpu_modes.sh
   ```

2. **Start production training**:
   ```bash
   # Recommended: Multi-GPU for speed
   sbatch job_multi_gpu.sub
   
   # Or single GPU for testing
   sbatch job_simple.sub
   ```

3. **Monitor progress**:
   ```bash
   squeue -u $USER
   tail -f logs_t/pflo_*_*.out
   ```

4. **Resume if needed** (job timeout):
   ```bash
   # Find checkpoint
   ls -lh logs/custom_pflotran/pflotran_identity/*.ckpt
   
   # Resume training (edit job script to add)
   --resume logs/custom_pflotran/pflotran_identity/checkpoint_step_XXXXX.ckpt
   ```
