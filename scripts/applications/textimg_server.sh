#!/bin/bash

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding


source ~/anaconda3/etc/profile.d/conda.sh

cd /home/cc/applications/stable-diffusion-webui
source venv/bin/activate

stdbuf -oL -eL  python webui.py --nowebui --port=$1 --ckpt=$2
SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
# echo $pid
