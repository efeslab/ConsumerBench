#!/bin/bash

# This script monitors CPU usage and logs it to a file.
# Usage: ./get_cpu_usage.sh <Results Directory>
# Check if the results directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <Results Directory>"
    exit 1
fi
# Check if the results directory exists
RESULTS_DIR=$1
if [ ! -d $RESULTS_DIR ]; then
    echo "Results directory not found!"
    exit 1
fi

nohup sudo pcm-memory /csv 0.05 2>/dev/null > ${RESULTS_DIR}/memory-bw.csv
