#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

set -x

source ~/anaconda3/etc/profile.d/conda.sh

conda activate deepresearch
cd $1

export HF_TOKEN=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
export SERPAPI_API_KEY=be7c2935c265041d83992142e94df46219eaf23fa709b7405320114c21453de6
export SERPER_API_KEY=8064131348ecef7f3bbd5b666acf1e50bec4905a

# stdbuf -oL -eL 
python3 run.py --port "$2" --model-id "$3" "$4"