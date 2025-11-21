#!/bin/bash

# Get the normalized latency comparison for two setups

folder_path_1=$1
folder_path_2=$2
result_folder_path=$3
# current script directory
consumerbench_path=$(dirname $(dirname $(dirname $(readlink -f $0))))

for folder_path in $folder_path_1 $folder_path_2; do
    # preprocess the results
    python3 $consumerbench_path/scripts/result_processing/parse-results-chatbot-log.py $folder_path/task_chat1_u0_perf.log

    python3 $consumerbench_path/scripts/result_processing/parse-results-imagegen-log.py $folder_path/task_imagegen1_u0_perf.log

    python3 $consumerbench_path/scripts/result_processing/parse-results-whisper-log.py $folder_path/task_lv_u0_perf.log

done

# plot the results
mkdir -p $result_folder_path
python3 $consumerbench_path/scripts/result_processing/plot_bar_plot.py --gpu_folder_path $folder_path_1 --cpu_folder_path $folder_path_2 --save_path $result_folder_path