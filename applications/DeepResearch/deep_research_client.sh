#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

set -x

# Activate the Python environment. Prefer the conda "deepresearch" env if conda
# is available; otherwise fall back to the repo-local .cb virtualenv.
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q '^deepresearch\s'; then
    conda activate deepresearch
fi
cd $1

# API keys are read from the environment / a .env file in the run dir (loaded by
# run.py via python-dotenv). Set them in the calling environment or in
# applications/DeepResearch/smolagents/examples/open_deep_research/.env

# stdbuf -oL -eL
python3 run.py --port "$2" --model-id "$3" "$4"