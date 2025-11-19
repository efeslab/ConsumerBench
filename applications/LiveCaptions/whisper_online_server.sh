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

source ~/miniconda3/etc/profile.d/conda.sh

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

conda activate consumerbench
cd ${server_dir}/whisper_streaming

python3 -u whisper_online_server.py --host 127.0.0.1 --port ${api_port} --device ${device} -l DEBUG --min-chunk-size 2.0 --warmup-file ${server_dir}/whisper-earnings21/10-sec-chunks/4320211_chunk_001.wav &



SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
# echo $pid
