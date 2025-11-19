#!/bin/bash

# This script runs the benchmark for the LLM model using the specified parameters.
# Usage: ./run_benchmark.sh <config_file>
# Example: ./run_benchmark.sh ConsumerBench/configs/workflow_chatbot.yml 0
# Check if the config file is provided

source ~/archiconda3/etc/profile.d/conda.sh

# Change to your conda environment name
conda activate consumerbench

SCRIPTS_DIR=`readlink -f $(dirname "$0")`
SCRIPTS_DIR=$SCRIPTS_DIR/../monitors
BASE_DIR=`readlink -f $SCRIPTS_DIR/..`
PLOT_SCRIPTS_DIR=$BASE_DIR/scripts/result_processing

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi
# Check if the config file exists
if [ ! -f $1 ]; then
    echo "Config file not found!"
    exit 1
fi
# Load the configuration file
CONFIG_FILE=`readlink -f $1`

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



tmux new-session -d "sudo -E $(which python3) ${SCRIPTS_DIR}/jetson_logger.py --file ${RESULTS_DIR}/jetson_logger.csv"
jetson_logger_pid=`pgrep -fo jetson_logger.py`
echo "--------------------------------"
echo "Result directory":
echo "$RESULTS_DIR"
echo "--------------------------------"
python3 -u ${BASE_DIR}/src/scripts/run_consumerbench.py --config $CONFIG_FILE --results $RESULTS_DIR
# Check if the benchmark was successful
if [ $? -ne 0 ]; then
    echo "Benchmark failed!"
fi
sleep 2
# Kill the background processes
kill -9 $jetson_logger_pid
sleep 2


echo "start_time: $start_time"


echo "Benchmark completed successfully!"
echo "Results are saved in $RESULTS_DIR"
