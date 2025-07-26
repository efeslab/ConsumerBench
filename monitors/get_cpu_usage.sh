#!/bin/bash

# This script monitors CPU usage and logs it to a file.
# Usage: ./get_cpu_usage.sh <Results Directory>
# Example: ./get_cpu_usage.sh /home/cc/os-llm/results
# Check if the results directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <Results Directory>"
    exit 1
fi
# Check if the results directory exists
RESULTS_DIR=$1
if [ ! -d $RESULTS_DIR ]; then
    echo "Results directory not found!"
    exit 1
fi

get_cpu_usage() {
  # Read /proc/stat
  local cpu_line=$(cat /proc/stat | grep "^cpu ")
  local user=$(echo $cpu_line | awk '{print $2}')
  local nice=$(echo $cpu_line | awk '{print $3}')
  local system=$(echo $cpu_line | awk '{print $4}')
  local idle=$(echo $cpu_line | awk '{print $5}')

  # Calculate total
  local total=$((user + nice + system + idle))
  echo "$user $nice $system $idle $total"
}

previous_data=$(get_cpu_usage)
previous_time=$(date +%s%N)

while true; do
  # Sleep for a millisecond
  sleep 0.05

  # Get current time and CPU data
  current_time=$(date +%s%N)
  current_data=$(get_cpu_usage)

  # Extract values
  prev_user=$(echo $previous_data | awk '{print $1}')
  prev_nice=$(echo $previous_data | awk '{print $2}')
  prev_system=$(echo $previous_data | awk '{print $3}')
  prev_idle=$(echo $previous_data | awk '{print $4}')
  prev_total=$(echo $previous_data | awk '{print $5}')

  curr_user=$(echo $current_data | awk '{print $1}')
  curr_nice=$(echo $current_data | awk '{print $2}')
  curr_system=$(echo $current_data | awk '{print $3}')
  curr_idle=$(echo $current_data | awk '{print $4}')
  curr_total=$(echo $current_data | awk '{print $5}')

  # Calculate deltas
  delta_total=$((curr_total - prev_total))
  delta_idle=$((curr_idle - prev_idle))

  # Calculate CPU usage percentage
  if [ $delta_total -gt 0 ]; then
    cpu_usage=$(( 100 * (delta_total - delta_idle) / delta_total ))
    elapsed_ms=$(( (current_time - previous_time) / 1000000 ))
    echo "$(date +%Y-%m-%d_%H:%M:%S.%N) | CPU: ${cpu_usage}" >> $RESULTS_DIR/cpu_usage.log
  fi

  # Update previous values
  previous_data=$current_data
  previous_time=$current_time
done
