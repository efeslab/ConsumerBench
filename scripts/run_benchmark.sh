#!/bin/bash

# This script runs the benchmark for the LLM model using the specified parameters.
# Usage: ./run_benchmark.sh <config_file>
# Example: ./run_benchmark.sh /home/cc/os-llm/configs/config.txt
# Check if the config file is provided

source ~/anaconda3/etc/profile.d/conda.sh
conda activate benchmark

SCRIPTS_DIR=`readlink -f $(dirname "$0")`
BASE_DIR=`readlink -f $SCRIPTS_DIR/..`

if [ $# -ne 2 ]; then
    echo "Usage: $0 <config_file> <nsight(0/1)>"
    exit 1
fi
# Check if the config file exists
if [ ! -f $1 ]; then
    echo "Config file not found!"
    exit 1
fi
# Load the configuration file
CONFIG_FILE=`readlink -f $1`
NSIGHT=$2

# Set environment variables for MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps

# Set environment variable to ensure the application uses MPS
# The actual GPU device number should be set to match your configuration
export CUDA_VISIBLE_DEVICES=0

# Create a config.json file from the config.txt file
cur_day=$(date +%b%d_%Y)
cur_time=$(date +%H_%M_%S)
start_time=$(date +%Y-%m-%d_%H:%M:%S)
RESULT_DIR_NAME=$(python3 ${SCRIPTS_DIR}/yml_to_json.py $CONFIG_FILE ${BASE_DIR}/results ${cur_day} ${cur_time})
RESULTS_DIR="${BASE_DIR}/results/${cur_day}/${RESULT_DIR_NAME}${cur_time}"

# create a results directory based on the current date and time
mkdir -p $RESULTS_DIR

# Read the config file
cat ${RESULTS_DIR}/config.json | jq

# Run the benchmark
# Check if nsight is enabled
if [ $NSIGHT -eq 1 ]; then
    echo "Running with Nsight Systems profiling..."
    #nsys profile --cuda-memory-usage=true --force-overwrite=true --trace=cuda,nvtx,osrt --gpu-metrics-devices=all --event-sample=system-wide --cpu-core-events='1' --event-sampling-interval=200 --cpuctxsw=system-wide -o memory_report --export=sqlite python3 -u /home/cc/os-llm/benchmark_v2.py --benchmark all --config $CONFIG_FILE --results $RESULTS_DIR
    nsys profile --delay=50 --duration=60 --cuda-memory-usage=true --force-overwrite=true --trace=cuda,nvtx,osrt --gpu-metrics-devices=all -o memory_report --export=sqlite python3 -u ${BASE_DIR}/benchmark_v2.py --benchmark workflow --config $CONFIG_FILE --results $RESULTS_DIR --start_time $start_time

    if [ $? -ne 0 ]; then
        echo "Benchmark failed!"
    fi

else
    echo "Running without Nsight Systems profiling..."

    # Get CPU utilization
    ${SCRIPTS_DIR}/get_cpu_usage.sh ${RESULTS_DIR} &
    sleep 1
    cpu_usage_pid=`pgrep "get_cpu_usage"`

    # Get CPU Memory bandwidth
    tmux new-session -d "sudo pcm-memory 0.05 -s -csv=${RESULTS_DIR}/memory-bw.csv"
    # Check if the command is running
    while ! pgrep "pcm-memory" > /dev/null; do
        sleep 1
    done
    pcm_memory_pid=`pgrep "pcm-memory"`

    # Get GPU compute and mem utilization
    ${SCRIPTS_DIR}/record_gpu_mem_compute.sh ${RESULTS_DIR} &
    sleep 1
    gpu_utilization_pid=`pgrep "dcgmi"`

    # Get power utilization
    tmux new-session -d sudo python3 ${SCRIPTS_DIR}/record_power_usage.py -o ${RESULTS_DIR}/power_data.csv -s ${start_time}
    power_pid=`pgrep -fo record_power`

    echo "--------------------------------"
    echo "Result directory":
    echo "$RESULTS_DIR"
    echo "--------------------------------"
    python3 -u ${BASE_DIR}/benchmark_v2.py --benchmark workflow --config $CONFIG_FILE --results $RESULTS_DIR --start_time $start_time
    # Check if the benchmark was successful
    if [ $? -ne 0 ]; then
        echo "Benchmark failed!"
    fi

    sleep 2
    # Kill the background processes
    kill -9 $cpu_usage_pid
    sudo kill -9 $pcm_memory_pid
    kill -9 $gpu_utilization_pid
    sudo kill -9 $power_pid

    sleep 2
fi

if [ $NSIGHT -eq 1 ]; then
    # Move the generated report to the results directory
    mv memory_report.sqlite $RESULTS_DIR/memory_report.sqlite
    mv memory_report.nsys-rep $RESULTS_DIR/memory_report.nsys-rep
    python3 ${SCRIPTS_DIR}/process_nsight_cpu.py $RESULTS_DIR/memory_report.sqlite $RESULTS_DIR
    python3 ${SCRIPTS_DIR}/process_nsight_gpu.py $RESULTS_DIR/memory_report.sqlite $RESULTS_DIR
    rm $RESULTS_DIR/memory_report.sqlite
    rm $RESULTS_DIR/memory_report.nsys-rep
fi

echo "start_time: $start_time"

# Create plots
python3 ${SCRIPTS_DIR}/plot_cpu_compute_usage.py --input_file_cpu ${RESULTS_DIR}/cpu_usage.log --input_file_mem ${RESULTS_DIR}/memory-bw.csv -o ${RESULTS_DIR}/cpu_compute_usage.png -s $start_time
python3 ${SCRIPTS_DIR}/plot_cpu_mem_usage.py --input_file_cpu ${RESULTS_DIR}/cpu_usage.log --input_file_mem ${RESULTS_DIR}/memory-bw.csv -o ${RESULTS_DIR}/cpu_mem_usage.png -s $start_time
python3 ${SCRIPTS_DIR}/dcgm_plotter_gpu_compute.py ${RESULTS_DIR}/gpu_utilization.log -o ${RESULTS_DIR}/gpu_compute_usage.png -s $start_time
python3 ${SCRIPTS_DIR}/dcgm_plotter_gpu_mem.py ${RESULTS_DIR}/gpu_utilization.log -o ${RESULTS_DIR}/gpu_mem_usage.png -s $start_time
python3 ${SCRIPTS_DIR}/plot_power_usage.py --input ${RESULTS_DIR}/power_data.csv -o ${RESULTS_DIR}/power_usage.png

IFS=$'\n'
perf_logs=($(find "$RESULTS_DIR" -type f -name "*_perf.log"))
unset IFS  # Reset IFS when done
for i in "${!perf_logs[@]}"
do
    python3 ${SCRIPTS_DIR}/parse-results-chatbot.py "${perf_logs[$i]}"
    python3 ${SCRIPTS_DIR}/parse-results-deepresearch.py "${perf_logs[$i]}"
    python3 ${SCRIPTS_DIR}/parse-results-imagegen.py "${perf_logs[$i]}"
    python3 ${SCRIPTS_DIR}/parse-results-whisper.py "${perf_logs[$i]}"
    python3 ${SCRIPTS_DIR}/parse-results-chatbot-log.py "${perf_logs[$i]}"
    python3 ${SCRIPTS_DIR}/parse-results-deepresearch-log.py "${perf_logs[$i]}"
    python3 ${SCRIPTS_DIR}/parse-results-imagegen-log.py "${perf_logs[$i]}"
    python3 ${SCRIPTS_DIR}/parse-results-whisper-log.py "${perf_logs[$i]}"
done

python3 /home/cc/os-llm/scripts/overall_benchmark_output.py ${RESULTS_DIR}/overall_perf.log
    
# Remove csv files
# sudo rm ${RESULTS_DIR}/cpu_usage.log
# sudo rm ${RESULTS_DIR}/memory-bw.csv
# sudo rm ${RESULTS_DIR}/gpu_utilization.log
# sudo rm ${RESULTS_DIR}/cpu_memory_util.csv
# sudo rm ${RESULTS_DIR}/gpu_memory_util.csv
# sudo rm ${RESULTS_DIR}/power_data.csv

echo "Benchmark completed successfully!"
echo "Results are saved in $RESULTS_DIR"
