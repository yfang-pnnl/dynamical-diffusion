#!/bin/csh
# Script to plot time series at specific location

source /people/d3m045/.cshrc_vs
conda activate dydiff

python plot_timeseries.py \
  --output_dir logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0 \
  --case_id 0 \
  --z 19 \
  --x 10

echo ""
echo "Plot saved to: logs/custom_pflotran/pflotran_identity_eval/output_for_evaluation_0/visualizations/case_0_timeseries_z19_x10.png"
