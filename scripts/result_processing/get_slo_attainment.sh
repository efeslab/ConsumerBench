#!/bin/bash

# Get the SLO attainment for the given folder path

folder_path=$1
# current script directory
consumerbench_path=$(dirname $(dirname $(dirname $(readlink -f $0))))

# preprocess the results
python3 $consumerbench_path/scripts/result_processing/parse-results-chatbot-log.py $folder_path/task_chat1_u0_perf.log

python3 $consumerbench_path/scripts/result_processing/parse-results-imagegen-log.py $folder_path/task_imagegen1_u0_perf.log

python3 $consumerbench_path/scripts/result_processing/parse-results-whisper-log.py $folder_path/task_lv_u0_perf.log

# plot the results
mkdir -p $folder_path/processed
python3 $consumerbench_path/scripts/result_processing/plot_bar_plot.py --gpu_folder_path $folder_path --cpu_folder_path $folder_path --save_path $folder_path/processed