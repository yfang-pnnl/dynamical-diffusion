#!/bin/csh
# PyTorch 2.0 Upgrade Script
# This will create a new environment with PyTorch 2.0+

echo "=== PyTorch 2.0 Upgrade ==="
echo ""
echo "Step 1: Backup current environment"
conda list --export > dydiff_old_packages.txt

echo ""
echo "Step 2: Remove old environment"
conda deactivate
conda env remove -n dydiff -y

echo ""
echo "Step 3: Create new environment with PyTorch 2.0"
conda env create -f environment.yaml

echo ""
echo "Step 4: Activate and verify"
conda activate dydiff
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'SDP available: {hasattr(torch.nn.functional, \"scaled_dot_product_attention\")}')"

echo ""
echo "=== Upgrade Complete ==="
echo "Expected performance improvement: 2-3x faster attention"
echo ""
echo "To use the new environment:"
echo "  conda activate dydiff"
echo ""
echo "To test training with 4 GPUs (recommended):"
echo "  1. Edit job_simple.sub: #SBATCH --gres=gpu:4"
echo "  2. Update command: --n_gpu 4"
echo "  3. Submit: sbatch job_simple.sub"
