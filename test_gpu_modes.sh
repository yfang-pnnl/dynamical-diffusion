#!/bin/csh
# Test both single GPU and multi-GPU modes
# This runs a short test (100 iterations) to verify everything works

echo "==================================="
echo "Testing GPU Training Modes"
echo "==================================="
echo ""

# Activate environment
source /people/d3m045/.cshrc_vs
conda activate dydiff

# Test 1: Single GPU
echo "Test 1: Single GPU (100 iterations)"
echo "-----------------------------------"
python core/train_pflotran_dydiff.py \
  --config_file core/models/pflotran_identity.yaml \
  --n_gpu 1 \
  --max_iterations 100

if ($status == 0) then
    echo "✓ Single GPU test PASSED"
else
    echo "✗ Single GPU test FAILED"
    exit 1
endif

echo ""
echo ""

# Test 2: Multi-GPU (if available)
echo "Test 2: Multi-GPU DDP (100 iterations)"
echo "---------------------------------------"

# Check if GPUs are available
set gpu_count=`nvidia-smi -L | wc -l`
echo "Available GPUs: $gpu_count"

if ($gpu_count >= 4) then
    python core/train_pflotran_dydiff.py \
      --config_file core/models/pflotran_identity.yaml \
      --n_gpu 4 \
      --max_iterations 100
    
    if ($status == 0) then
        echo "✓ Multi-GPU test PASSED"
    else
        echo "✗ Multi-GPU test FAILED"
        exit 1
    endif
else
    echo "⚠ Skipping multi-GPU test (need 4 GPUs, only $gpu_count available)"
    echo "  To test multi-GPU, run on a compute node with 4 GPUs"
endif

echo ""
echo "==================================="
echo "All tests completed successfully!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  - For single GPU training: sbatch job_simple.sub"
echo "  - For multi-GPU training:  sbatch job_multi_gpu.sub"
