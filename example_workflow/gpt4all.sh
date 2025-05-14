#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpt4all

python3 /home/rohan/os-llm/gpt4all/gpt4all-bindings/python/benchmarks/benchmark_latency.py --input-len $1 --output-len $2 --num-iters-warmup 0 --num-iters 1