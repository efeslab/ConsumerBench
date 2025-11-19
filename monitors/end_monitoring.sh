#!/bin/bash

cpu_usage_pid=`pgrep "get_cpu_usage"`
pcm_memory_pid=`pgrep "pcm-memory"`
gpu_utilization_pid=`pgrep "dcgmi"`
power_pid=`pgrep -fo record_power`
echo "CPU usage monitor PID: $cpu_usage_pid"
echo "PCM memory monitor PID: $pcm_memory_pid"
echo "GPU utilization monitor PID: $gpu_utilization_pid"
echo "Power monitor PID: $power_pid"

echo "Killing CPU usage monitor"
kill -9 $cpu_usage_pid
echo "Killing PCM memory monitor"
sudo kill -9 $pcm_memory_pid
echo "Killing GPU utilization monitor"
kill -9 $gpu_utilization_pid
echo "Killing power monitor"
sudo kill -9 $power_pid