#!/bin/bash

SCRIPTS_DIR=`readlink -f $(dirname "$0")`
BASE_DIR=`readlink -f $SCRIPTS_DIR/..`

# Run single chatbot
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_file_chatbot_small.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_file_chatbot_cpu_small.txt 0

# Run single image generation
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_file_imagegen.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_file_imagegen_cpu.txt 0

# Run single live captions
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_file_live_captions.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_file_live_captions_cpu.txt 0

# Run live captions with others
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_file_live_captions_imagegen.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_file_live_captions_chatbot.txt 0

# Run all hybrid
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_all_gpu_cpu.txt 0

# Run all in GPU
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_all_gpu.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/concurrent_run_config_all_gpu_mps.txt 0

# Run researcher 2 with 100% mps
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_researcher_2.yml 0

# Run researcher 2 with 33% mps
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_researcher_2_mps.yml 0