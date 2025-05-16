#!/bin/bash

cpu_usage_pid=`pgrep "get_cpu_usage"`
pcm_memory_pid=`pgrep "pcm-memory"`
gpu_utilization_pid=`pgrep "dcgmi"`
power_pid=`pgrep -fo record_power`

kill -9 $cpu_usage_pid
sudo kill -9 $pcm_memory_pid
kill -9 $gpu_utilization_pid
sudo kill -9 $power_pid