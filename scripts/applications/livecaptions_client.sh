#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate livecaptions
python3 /home/rohan/os-llm/LiveCaptions/subprojects/april-asr/bindings/python/april_asr/example.py /home/rohan/os-llm/LiveCaptions/april-english-dev-01110_en.april /home/rohan/os-llm/LiveCaptions/harvard_processed.wav