#!/bin/bash
#
# QUICK START SCRIPT for Setting Up Custom PFLOTRAN Data
# 
# This script guides you through the setup process for training the dynamical 
# diffusion model on your custom PFLOTRAN simulations.
#
# Usage: bash setup_quick_start.sh
#

set -e  # Exit on error

echo "========================================"
echo "Dynamical Diffusion - Custom Data Setup"
echo "========================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Ask user for data directories
echo -e "${YELLOW}Step 1: Specify Your Data Directories${NC}"
echo ""
echo "Where are your raw concentration *.npy files located?"
read -p "  Concentration directory: " CONC_DIR

echo ""
echo "Where should processed data be saved?"
read -p "  Output directory: " OUTPUT_DIR

echo ""
read -p "Do you have permeability fields saved? (y/n): " has_perm
read -p "Do you have porosity fields saved? (y/n): " has_poro

PERM_ARGS=""
PORO_ARGS=""

if [[ $has_perm == "y" ]]; then
    read -p "  Path to permeability file: " perm_path
    PERM_ARGS="--perm_values $perm_path"
fi

if [[ $has_poro == "y" ]]; then
    read -p "  Path to porosity file: " poro_path
    PORO_ARGS="--poro_values $poro_path"
fi

# Step 2: Ask for data parameters
echo ""
echo -e "${YELLOW}Step 2: Confirm Data Parameters${NC}"
echo ""
read -p "Number of realizations (default 300): " n_real
n_real=${n_real:-300}

read -p "Number of timesteps (default 240): " n_time
n_time=${n_time:-240}

read -p "Grid dimension nx (default 28): " nx
nx=${nx:-28}

read -p "Grid dimension nz (default 40): " nz
nz=${nz:-40}

echo ""
echo -e "${GREEN}Summary:${NC}"
echo "  Realizations: $n_real"
echo "  Timesteps: $n_time"
echo "  Grid: $nx × $nz"
echo ""

# Step 3: Prepare data
echo -e "${YELLOW}Step 3: Preparing Data${NC}"
echo ""

mkdir -p "$OUTPUT_DIR/splits"

python prepare_custom_data.py \
  --conc_dir "$CONC_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --n_realizations "$n_real" \
  --n_timesteps "$n_time" \
  --nx "$nx" \
  --nz "$nz" \
  $PERM_ARGS $PORO_ARGS

echo -e "${GREEN}✓ Data preparation complete${NC}"
echo ""

# Step 4: Create splits
echo -e "${YELLOW}Step 4: Creating Train/Val/Test Splits${NC}"
echo ""

python create_data_splits.py \
  --output_dir "$OUTPUT_DIR/splits" \
  --n_realizations "$n_real" \
  --train_ratio 0.80 \
  --val_ratio 0.16 \
  --seed 42

echo -e "${GREEN}✓ Splits created${NC}"
echo ""

# Step 5: Create config file
echo -e "${YELLOW}Step 5: Creating Configuration File${NC}"
echo ""

CONFIG_FILE="core/models/custom_pflotran_${n_real}r.yaml"

cat > "$CONFIG_FILE" << EOF
# Auto-generated config for custom PFLOTRAN data
# Data directory: $OUTPUT_DIR

data:
  perm_path: $OUTPUT_DIR/perm.npy
  poro_path: $OUTPUT_DIR/poro.npy
  conc_path: $OUTPUT_DIR/conc.npy
  train_idx_path: $OUTPUT_DIR/splits/train_idx.npy
  val_idx_path: $OUTPUT_DIR/splits/val_idx.npy
  test_idx_path: $OUTPUT_DIR/splits/test_idx.npy

  pred_length: 20
  input_length: 20
  total_length: 40
  t_keep: $n_time
  dt: 1

model:
  target: dydiff.dydiff_pflotran.DynamicalLDMForPFLOTRANWithStaticCondition
  params:
    total_length: 40
    input_length: 20
    x_channels: 1
    z_channels: 3
    static_latent_channels: 8

    timesteps: 1000
    beta_schedule: cosine
    linear_start: 0.00085
    linear_end: 0.0120

    gamma_schedule: "cosine-0.5"
    linear_start_gamma: 0.
    linear_end_gamma: 0.

    first_stage_key: "image"
    first_stage_key_prev: "prev"
    cond_stage_key: "cond"
    conditioning_key: "concat-video-mask-1st-static"
    cond_stage_config:
      target: torch.nn.Identity

    validate_kwargs:
      ddim: True
      ddim_steps: 50
      ddim_eta: 0.

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        ckpt_path: null
        monitor: "val/rec_loss"
        embed_dim: 3
        lossconfig:
          target: torch.nn.Identity
        ddconfig:
          double_z: True
          z_channels: 3
          resolution: $nz
          in_channels: 1
          out_ch: 1
          ch: 128
          ch_mult: [1, 2, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0

    unet_config:
      target: sgm.modules.diffusionmodules.video_model.VideoUNet
      params:
        num_video_frames: 0
        in_channels: 0
        out_channels: 0
        model_channels: 64
        attention_resolutions: [4, 2, 1]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_heads: 4
        transformer_depth: 1
        spatial_transformer_attn_type: softmax
        extra_ff_mix_layer: True
        merge_strategy: learned
        video_kernel_size: [3, 1, 1]

training:
  max_iterations: 200000
  model_attrs:
    learning_rate: 1e-4
  batch_size: 16
  num_workers: 8
  accumulate_grad_batches: 1
  logger:
    save_dir: ./logs/custom_pflotran
    logger_freq: 5000
    checkpoint_freq: 10000

ckpt_path: null
EOF

echo -e "${GREEN}✓ Configuration created: $CONFIG_FILE${NC}"
echo ""

# Step 6: Summary and next steps
echo "========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Your data is ready at:"
echo "  $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Activate your environment:"
echo "   conda activate dydiff"
echo ""
echo "2. Start training:"
echo "   cd /qfs/projects/dl_calibration/d3m045/dynamical-diffusion"
echo "   python core/train_pflotran_dydiff.py \\"
echo "     --config_file $CONFIG_FILE \\"
echo "     --n_gpu 1"
echo ""
echo "3. Monitor training:"
echo "   Logs will be saved to: ./logs/custom_pflotran"
echo ""
echo "For more details, see: SETUP_CUSTOM_DATA.md"
echo ""
