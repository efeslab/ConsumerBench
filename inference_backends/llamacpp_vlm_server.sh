#!/bin/bash

# Standalone launcher for a llama.cpp VLM server (text model + mmproj vision
# projector). Mirrors llamacpp_server.sh but adds --mmproj.
#
# -c is the total context shared across --parallel slots, so each slot gets
# c/2. A single video_understanding call sends every sampled frame as image
# tokens at once and easily runs to ~41K tokens, so a per-slot floor well
# above that is needed (same c/parallel gotcha as llamacpp_server.sh).

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

server_dir=$1
listen_port=$2
api_port=$3
model=$4
device=$5
mps=$6
mmproj=$7
ctx=${8:-131072}

cd $server_dir

if [ "$device" == "gpu" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${mps}
    export CUDA_VISIBLE_DEVICES=0

    stdbuf -oL -eL build/bin/llama-server --port ${api_port} -m ${model} --mmproj ${mmproj} \
        -ngl 99 --parallel 2 -c ${ctx} --jinja &
else
    export CUDA_VISIBLE_DEVICES=""
    stdbuf -oL -eL build/bin/llama-server --port ${api_port} -m ${model} --mmproj ${mmproj} \
        -ngl 0 --parallel 2 -c ${ctx} --jinja &
fi

export CUDA_VISIBLE_DEVICES=0
SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
