#!/bin/bash

# This app runs on port 5000

# Force unbuffered output for all commands in this script
export PYTHONUNBUFFERED=1    # Python-specific: disable buffering
export PYTHONIOENCODING=utf-8  # Ensure proper encoding

api_port=$1
wav_file_path=$2
app_dir=$3

# Activate the Python environment. Prefer the conda "whisper" env if available;
# otherwise fall back to the already-active repo-local .cb virtualenv.
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q '^whisper\s'; then
    conda activate whisper
fi

# Resolve the wav path to absolute BEFORE cd'ing, so a relative client_command_file
# (resolved against the repo root) still works from the whisper_streaming dir.
if [ -f "${wav_file_path}" ]; then
    wav_file_path="$(readlink -f "${wav_file_path}")"
fi

cd ${app_dir}/whisper_streaming
stdbuf -oL -eL python3 generate_raw_realtime.py ${wav_file_path} --port ${api_port}
