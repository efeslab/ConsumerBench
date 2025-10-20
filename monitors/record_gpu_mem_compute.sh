#!/bin/bash

# This script monitors CPU usage and logs it to a file.
# Usage: ./get_cpu_usage.sh <results_directory>
# Check if the results directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <results_directory>"
    exit 1
fi
# Check if the results directory exists
if [ ! -d $1 ]; then
    echo "Results directory not found!"
    exit 1
fi

dcgmi dmon -e 1002,1003,1005 -d 50 | ts '[%Y-%m-%d %H:%M:%.S]' >> $1/gpu_utilization.log
