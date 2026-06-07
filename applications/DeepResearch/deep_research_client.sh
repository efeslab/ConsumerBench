#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

set -x

# Resolve repo base (this script lives in <repo>/applications/DeepResearch/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate deepresearch
cd $1

# Load API keys (HF_TOKEN, SERPAPI_API_KEY, SERPER_API_KEY) from the repo .env
set -a
. "$REPO_DIR/.env"
set +a

# stdbuf -oL -eL
python3 run.py --port "$2" --model-id "$3" "$4"