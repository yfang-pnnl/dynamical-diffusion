#!/bin/bash
# Generate comprehensive visualizations for trained model

OUTPUT_DIR="logs/custom_pflotran/pflotran_identity/output_for_evaluation_0"
CASE_ID=0

echo "=== Creating Spatial Visualizations ==="
python core/evaluation/visualize_pflotran.py \
    --output_dir $OUTPUT_DIR \
    --case_id $CASE_ID \
    --all_plots

echo ""
echo "=== Creating Time Series Plot ==="
python plot_timeseries.py \
    --output_dir $OUTPUT_DIR \
    --case_idx $CASE_ID \
    --z 19 \
    --x 10

echo ""
echo "=== All visualizations complete! ==="
echo "Spatial plots: $OUTPUT_DIR/visualizations/"
echo "Time series: case_${CASE_ID}_timeseries_z19_x10.png"
