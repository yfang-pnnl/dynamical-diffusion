# GPU Training Modes Guide

The training code now supports both **single GPU** and **multi-GPU (DDP)** training with PyTorch 2.0.

## Quick Start

### Single GPU (Slower, but simpler)
```bash
sbatch job_simple.sub
```
- **Speed**: ~30-36 it/s (with PyTorch 2.0 SDP attention)
- **Training time**: ~22 hours for 200k iterations
- **Best for**: Testing, debugging, small-scale runs

### Multi-GPU DDP (Faster, recommended)
```bash
sbatch job_multi_gpu.sub
```
- **Speed**: ~120-140 it/s (4 GPUs × ~30-35 it/s)
- **Training time**: ~5-6 hours for 200k iterations
- **Best for**: Full training runs

## Performance Comparison

| Mode | GPUs | Speed | 200k iter Time | Effective Batch |
|------|------|-------|----------------|-----------------|
| Single GPU | 1 | ~30-36 it/s | ~22 hours | 1×1×4 = 4 |
| DDP | 4 | ~120-140 it/s | ~5-6 hours | 4×1×4 = 16 |

## Job Scripts

### job_simple.sub
- Single GPU training
- 3 hour time limit
- Partition: `a100_80_shared`
- Use for quick tests

### job_multi_gpu.sub
- 4 GPU DDP training
- 6 hour time limit (can fit full training)
- Partition: `a100_80`
- Use for production runs

## Configuration Details

### What the code does automatically:

**Single GPU mode** (`--n_gpu 1`):
- Strategy: None (standard single GPU)
- num_workers: Uses config value (2-8)
- Batch norm: Regular
- Simpler, less overhead

**Multi-GPU mode** (`--n_gpu 4`):
- Strategy: DDPStrategy with optimizations
- num_workers: Auto-reduced to 2 (less multiprocessing overhead)
- Batch norm: Synchronized across GPUs
- Gradients synchronized after each batch

### Key Parameters

From `pflotran_identity.yaml`:
```yaml
training:
  batch_size: 1              # Per GPU
  accumulate_grad_batches: 4 # Accumulate before optimizer step
  num_workers: 2             # Auto-adjusted for multi-GPU
```

**Effective batch size**:
- Single GPU: `1 × 4 = 4`
- 4 GPUs: `4 × 1 × 4 = 16`

## Advanced Usage

### Resume from checkpoint
```bash
# Single GPU
sbatch job_simple.sub --resume logs/custom_pflotran/pflotran_identity/checkpoint_step_10000.ckpt

# Multi-GPU
sbatch job_multi_gpu.sub --resume logs/custom_pflotran/pflotran_identity/checkpoint_step_10000.ckpt
```

### Custom number of GPUs
```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 2
```

### Enable find_unused_parameters (if DDP errors occur)
```bash
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 4 \
  --find_unused_parameters True
```

## Troubleshooting

### DDP hangs at initialization
- Reduce `num_workers` to 0 or 1
- Check firewall settings (DDP uses network communication)
- Verify all GPUs are on same node

### Out of memory with multi-GPU
- Despite using more GPUs, each GPU still holds full model
- Reduce `batch_size` if needed
- Current settings (batch_size=1) should work fine

### Inconsistent results between single/multi-GPU
- DDP shuffles data differently on each GPU
- Set `pl.seed_everything(42)` for reproducibility
- Results should be statistically similar, not identical

### "unused parameter" errors
```bash
# Add this flag
--find_unused_parameters True
```

## Monitoring

### Check job status
```bash
squeue -u $USER
```

### Watch training progress
```bash
# Single GPU
tail -f logs_t/pflo_JOBID_*.out

# Multi-GPU (only rank 0 prints)
tail -f logs_t/pflo_JOBID_*.out
```

### Compare speeds
The output shows:
```
Training Configuration:
  Mode: DDP Multi-GPU (or Single GPU)
  GPUs: 4
  Batch size per GPU: 1
  Effective batch size: 16
  ...
```

And progress:
```
Epoch 0:   6%|▋ | 3750/48240 [05:10<1:14:40, 12.08it/s, loss=0.00542]
```

## Recommendations

1. **For development/testing**: Use single GPU (`job_simple.sub`)
   - Faster startup
   - Easier debugging
   - Good for short runs

2. **For production training**: Use multi-GPU (`job_multi_gpu.sub`)
   - 4× faster
   - Completes in one job (∼6 hours vs 3×3 hour jobs)
   - More efficient use of cluster resources

3. **PyTorch 2.0 benefits**:
   - Both modes benefit from SDP attention (2-3× faster than PyTorch 1.x)
   - Better DDP performance
   - More memory efficient

## Example Workflow

```bash
# 1. Test with single GPU (100 iterations)
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 1 \
  --max_iterations 100

# 2. Verify multi-GPU works (100 iterations)
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 4 \
  --max_iterations 100

# 3. Run full training with multi-GPU
sbatch job_multi_gpu.sub
```
