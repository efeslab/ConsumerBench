#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

listen_port=$1
api_port=$2
model=$3

source ~/anaconda3/etc/profile.d/conda.sh

conda activate llamacpp
cd /home/cc/llama.cpp
stdbuf -oL -eL build/bin/llama-server --port ${api_port} -m ${model}
SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
# echo $pid
