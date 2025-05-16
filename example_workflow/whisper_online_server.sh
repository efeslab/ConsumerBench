#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

listen_port=$1
api_port=$2
model=$3
device=$4
mps=$5

source ~/anaconda3/etc/profile.d/conda.sh

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

conda activate whisper
cd /home/cc/applications/whisper_streaming
# nsys profile --capture-range=nvtx --nvtx-capture="Main" --trace=cuda,nvtx,osrt --cuda-memory-usage=true --stats=true --force-overwrite=true --python-backtrace=cuda  --cudabacktrace=true python3 -u whisper_online_server.py --host 127.0.0.1 --port ${api_port} --device ${device} -l DEBUG --min-chunk-size 2.0 --warmup-file /home/cc/datasets/whisper-earnings21/harvard_chunk_001.wav &
# nsys profile --capture-range=nvtx --nvtx-capture="Main" --trace=cuda,nvtx,osrt --stats=true --force-overwrite=true --python-backtrace=cuda  --cudabacktrace=true
# nsys profile --capture-range=nvtx --nvtx-capture="Main" --trace=cuda,nvtx,osrt --cuda-memory-usage=true --gpu-metrics-devices=all --stats=true --force-overwrite=true
python3 -u whisper_online_server.py --host 127.0.0.1 --port ${api_port} --device ${device} -l DEBUG --min-chunk-size 2.0 --warmup-file /home/cc/datasets/whisper-earnings21/harvard_chunk_001.wav &

# ncu --target-processes all python3 -u whisper_online_server.py --host 127.0.0.1 --port ${api_port} --device ${device} -l DEBUG --min-chunk-size 2.0 --warmup-file /home/cc/datasets/whisper-earnings21/harvard_chunk_001.wav &
# ncu \
#   --set speedometer \
#   --replay-mode application \
#   --target-processes all \
#   --export /home/cc/os-llm/results/profile.ncu-rep \

SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
# echo $pid
