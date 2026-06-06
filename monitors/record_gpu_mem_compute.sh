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

# Field order matters: dcgm_plotter_gpu_compute.py parses SMACT/SMOCC/DRAMA by
# fixed column position (parts[4],[5],[6]), so the original three fields MUST stay
# first. Clock/throttle fields are appended after them and consumed by the new
# clock-throttle plotter:
#   1002 SMACT  (reserved SMs)        1003 SMOCC (occupied SMs)   1005 DRAMA (dram active)
#   100  SMCLK  (achieved SM clock)   113  SMMAX (max SM clock)
#   112  DVCCTR (current clock throttle-reason bitmask)
#   246  TAPCV  (app-clock violation, ns) -- time the GPU ran below requested app clock
dcgmi dmon -e 1002,1003,1005,100,113,112,246 -d 50 | ts '[%Y-%m-%d %H:%M:%.S]' >> $1/gpu_utilization.log
