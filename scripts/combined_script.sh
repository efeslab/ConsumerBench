#!/bin/bash

SCRIPTS_DIR=`readlink -f $(dirname "$0")`
BASE_DIR=`readlink -f $SCRIPTS_DIR/..`

# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_chatbot.yml 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_chatbot_cpu.yml 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_live_captions.yml 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_live_captions_cpu.yml 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_imagegen.yml 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_imagegen_cpu.yml 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_chatbot_imagegen_live_captions.yml 0
# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_chatbot_imagegen_live_captions_mps.yml 0

# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0
${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_content_creation_mps.yml 0

# ${SCRIPTS_DIR}/run_benchmark.sh ${BASE_DIR}/configs/workflow_chatbot_deep_research.yml 0
