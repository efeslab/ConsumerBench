#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

listen_port=$1
api_port=$2
model=$3

source ~/anaconda3/etc/profile.d/conda.sh

conda activate textgen
cd /home/cc/applications/text-generation-webui
stdbuf -oL -eL python one_click.py --api --listen-port=${listen_port} --api-port=${api_port} --model=${model} &
SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
# echo $pid
