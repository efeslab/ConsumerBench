#!/bin/bash

# Launcher for the LiveVideo VLM server: a llama.cpp server hosting a vision
# model, i.e. the text weights plus an mmproj projector that turns images into
# embeddings the text model can attend over.
#
# This is a near-copy of inference_backends/llamacpp_vlm_server.sh with two
# deliberate differences:
#
#   - --parallel is 1 by default. -c is the *total* context split evenly across
#     slots, so with 2 slots each request only gets half. LiveVideo drives one
#     stream from one client, so giving the whole context to a single slot is
#     both simpler to reason about and strictly more headroom per chunk.
#   - The context size is a required argument rather than a default, because
#     the token cost of a chunk scales with frames_per_chunk x frame_height and
#     is the number most likely to need tuning.

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

server_dir=$1
api_port=$2
model=$3
mmproj=$4
device=$5
mps=$6
ctx=$7
parallel=${8:-1}

cd $server_dir

if [ "$device" == "gpu" ]; then
    # Cap this server's share of GPU SMs so it can coexist with another model
    # (e.g. a Chatbot's llama-server) on the same GPU.
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${mps}
    export CUDA_VISIBLE_DEVICES=0

    stdbuf -oL -eL build/bin/llama-server --port ${api_port} -m ${model} --mmproj ${mmproj} \
        -ngl 99 --parallel ${parallel} -c ${ctx} --jinja &
else
    export CUDA_VISIBLE_DEVICES=""
    stdbuf -oL -eL build/bin/llama-server --port ${api_port} -m ${model} --mmproj ${mmproj} \
        -ngl 0 --parallel ${parallel} -c ${ctx} --jinja &
fi

export CUDA_VISIBLE_DEVICES=0
SERVER_PID=$!

echo "SERVER_PID=$SERVER_PID"
