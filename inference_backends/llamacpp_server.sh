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

# source ~/anaconda3/etc/profile.d/conda.sh

# if api port is been listening, do nothing
# pid=$(lsof -t -i :$api_port)
# if [ -n "$pid" ]; then
#     echo "Port $api_port is already in use by process $pid. Exiting."
#     exit 0
# fi

# conda activate llamacpp
cd $server_dir

# if device is cpu, use cpu version
if [ "$device" == "gpu" ]; then

    # Wrapper script for LLM server to use NVIDIA MPS

    # Set environment variables for MPS
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps

    # Resource limits (optional): Adjust as needed for your workload
    # This example reserves approximately 60% of GPU resources for the LLM server
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${mps}

    # Set environment variable to ensure the application uses MPS
    # The actual GPU device number should be set to match your configuration
    export CUDA_VISIBLE_DEVICES=0

    # nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --gpu-metrics-devices=all --stats=true --force-overwrite=true --python-backtrace=cuda --cudabacktrace=true build/bin/llama-server --port ${api_port} -m ${model} -ngl 99 --parallel 8 -c 4096 -nkvo &
    # nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --gpu-metrics-devices=all --stats=true --force-overwrite=true --python-backtrace=cuda --cudabacktrace=true build/bin/llama-server --port ${api_port} -m ${model} -ngl 99 --parallel 8 -c 4096 &
    # nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --gpu-metrics-devices=all --stats=true --force-overwrite=true --python-backtrace=cuda
    # build/bin/llama-server --port ${api_port} -m ${model} -ngl 99 --parallel 8 -c 128000 -nkvo &
    # Paper config: near-max model context (128000) with KV cache offloaded to CPU
    # RAM (-nkvo / --no-kv-offload). Offloading KV to host makes every token slower
    # (PCIe transfers), which is what drives the chatbot TTFT/TPOT SLO misses and the
    # long deep-research time in the "Greedy GPU allocation" figure.
    stdbuf -oL -eL build/bin/llama-server --port ${api_port} -m ${model} -ngl 99 --parallel 8 -c 128000 -nkvo &
    # GPU-resident KV variant (faster, but does not reproduce the paper's contention):
    # stdbuf -oL -eL build/bin/llama-server --port ${api_port} -m ${model} -ngl 99 --parallel 8 -c 131072 &
else
    export CUDA_VISIBLE_DEVICES=""
    stdbuf -oL -eL build/bin/llama-server --port ${api_port} -m ${model} -ngl 0 --parallel 8 -c 4096 &
fi

export CUDA_VISIBLE_DEVICES=0
SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
# echo $pid
