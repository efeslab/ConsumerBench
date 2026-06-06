#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

server_dir=$1
listen_port=$2
api_port=$3
model=$4
device=$5
mps=$6

# Activate the Python environment. Prefer the conda "whisper" env if conda is
# available; otherwise fall back to the repo-local .cb virtualenv (which has
# faster-whisper installed).
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

# Wrapper script for diffusion model application to use NVIDIA MPS

# Set environment variables for MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps

# Resource limits (optional): Adjust as needed for your workload
# This example reserves approximately 40% of GPU resources for the diffusion model
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${mps}

# Set environment variable to ensure the application uses MPS
# The actual GPU device number should be set to match your configuration
export CUDA_VISIBLE_DEVICES=0

if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q '^whisper\s'; then
    conda activate whisper
fi
cd ${server_dir}/whisper_streaming

# --buffer_trimming_sec 8 keeps the rolling audio buffer (and thus the encoder's
# peak GPU memory) bounded so it doesn't OOM when sharing the GPU with other apps.
python3 -u whisper_online_server.py --host 127.0.0.1 --port ${api_port} --device ${device} -l DEBUG --min-chunk-size 2.0 --buffer_trimming_sec 8 --warmup-file ${server_dir}/whisper-earnings21/4320211_chunk_001.wav --model ${model} &



SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
# echo $pid
