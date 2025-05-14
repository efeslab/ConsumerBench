#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate share4video

python3 /home/rohan/os-llm/ShareGPT4Video/run.py --model-path Lin-Chen/sharegpt4video-8b --video "/home/rohan/os-llm/ShareGPT4Video/examples/yoga.mp4" --query "Describe this video in detail."