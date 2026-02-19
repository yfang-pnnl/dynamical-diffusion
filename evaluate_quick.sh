#!/bin/bash
# Quick evaluation and visualization script for PFLOTRAN model checkpoints

echo "======================================================"
echo "  PFLOTRAN Model Evaluation & Visualization"
echo "======================================================"
echo ""

# Find available checkpoints
echo "Available checkpoints in logs/custom_pflotran/pflotran_identity/:"
find logs/custom_pflotran/pflotran_identity -name "*.ckpt" -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Default checkpoint (use the latest/best one)
CHECKPOINT="logs/custom_pflotran/pflotran_identity/epoch=15-step=6000.ckpt"

# Allow user to override
if [ ! -z "$1" ]; then
    CHECKPOINT="$1"
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Usage: ./evaluate_quick.sh [checkpoint_path]"
    echo "Example: ./evaluate_quick.sh logs/custom_pflotran/pflotran_identity/epoch=10-step=4000.ckpt"
    exit 1
fi

# Configuration
CONFIG="core/models/pflotran_identity_eval.yaml"
NUM_SAMPLES=10
NUM_CASES=5
NUM_VIS=5

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Ensemble samples: $NUM_SAMPLES"
echo "  Test cases: $NUM_CASES"
echo "  Visualizations: $NUM_VIS"
echo ""

# Run evaluation
echo "======================================================"
echo "Step 1: Generating predictions"
echo "======================================================"
echo ""

python core/evaluation/evaluate_pflotran.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --num_samples $NUM_SAMPLES \
    --num_cases $NUM_CASES \
    --visualize \
    --num_vis $NUM_VIS

if [ $? -ne 0 ]; then
    echo "Error: Prediction generation failed"
    exit 1
fi

echo ""
echo "======================================================"
echo "Step 2: Creating additional visualizations"
echo "======================================================"
echo ""

# Extract step number from checkpoint name
STEP=$(echo "$CHECKPOINT" | grep -oP 'step=\K[0-9]+')
if [ -z "$STEP" ]; then
    STEP="unknown"
fi

OUTPUT_DIR="logs/custom_pflotran/pflotran_identity/predictions_step${STEP}"

# Create visualizations for multiple cases
for CASE in $(seq 0 $((NUM_VIS-1))); do
    echo "Visualizing case $CASE..."
    python core/evaluation/visualize_pflotran.py \
        --output_dir "$OUTPUT_DIR" \
        --case_id $CASE \
        --input_length 20 \
        --all_plots
done

echo ""
echo "======================================================"
echo "  Evaluation Complete!"
echo "======================================================"
echo ""
echo "Results saved to:"
echo "  Predictions: $OUTPUT_DIR"
echo "  Visualizations: $OUTPUT_DIR/visualizations/"
echo ""
echo "View results:"
echo "  ls -lh $OUTPUT_DIR/"
echo "  ls -lh $OUTPUT_DIR/visualizations/"
echo ""
echo "Next steps:"
echo "  1. View visualizations:"
echo "     firefox $OUTPUT_DIR/visualizations/*.png &"
echo ""
echo "  2. Compute metrics (if you have stats file):"
echo "     python core/evaluation/plot_pflotran_metrics.py --help"
echo ""
