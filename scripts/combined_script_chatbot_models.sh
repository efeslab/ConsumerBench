#!/bin/bash

SCRIPTS_DIR=`readlink -f $(dirname "$0")`
BASE_DIR=`readlink -f $SCRIPTS_DIR/..`

# Run single chatbot
# ${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_file_chatbot_small_quant_llama.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_file_chatbot_small_qwen3b.txt 0
${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_file_chatbot_cpu_small_quant_llama.txt 0
${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_file_chatbot_cpu_small.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_file_chatbot_cpu_small_qwen3b.txt 0

# Run all in GPU
# ${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_all_gpu_quant_llama.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_all_gpu_qwen3b.txt 0

# Chatbot with Deep Research
# ${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_chatbot_deep_research_quant_llama.txt 0
# ${SCRIPTS_DIR}/run_benchmark.sh /home/cc/os-llm/configs/concurrent_run_config_chatbot_deep_research_qwen3b.txt 0