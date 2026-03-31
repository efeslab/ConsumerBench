#!/bin/bash
# Usage: ./run_experiment.sh <config_file> <results_dir>
set -e

CONFIG_FILE=$1
RESULTS_DIR=$2

cd /home/yilegu/ConsumerBench/ConsumerBench

# Kill any existing llama-server
pkill -f llama-server 2>/dev/null || true
sleep 2

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate consumerbench

# Create results directory
mkdir -p "$RESULTS_DIR"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps

echo "=== Running experiment ==="
echo "Config: $CONFIG_FILE"
echo "Results: $RESULTS_DIR"
echo "Start time: $(date)"

python3 -u src/scripts/run_consumerbench.py \
    --config "$CONFIG_FILE" \
    --results "$RESULTS_DIR"

echo "End time: $(date)"
echo "=== Experiment complete ==="
