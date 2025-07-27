#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

set -x

source ~/anaconda3/etc/profile.d/conda.sh

conda activate deepresearch
cd $1

export HF_TOKEN=<your-hf-token>
export SERPAPI_API_KEY=<your-serpapi-api-key>
export SERPER_API_KEY=<your-serper-api-key>

# stdbuf -oL -eL 
python3 run.py --port "$2" --model-id "$3" "$4"