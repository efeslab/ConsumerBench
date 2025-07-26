#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

api_port=$1
wav_file_path=$2

source ~/anaconda3/etc/profile.d/conda.sh

conda activate whisper

cd /home/cc/applications/whisper_streaming
stdbuf -oL -eL python3 /home/cc/applications/whisper_streaming/generate_raw_realtime.py ${wav_file_path} --port ${api_port}
