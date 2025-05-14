import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re
from datetime import datetime
import argparse

def parse_mem_output(file_path, start_time = None):
    """Parse pci-memory output file with timestamps and extract metrics."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize data structures
    timestamps = []
    memory_bandwidth = []

    # Parse each line
    for line in lines:
        if 'SKT0' in line or 'Date' in line:
            # Skip header lines but catch lines with actual data
            continue
        # Skip header lines but catch lines with actual data
        # Extract timestamp
        #remove the blank lines
        if not line.strip():
            continue

        # Split the line into parts
        parts = line.split(',')
        timestamp_str = f'{parts[0]} {parts[1]}'
        # print(timestamp_str)
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            # Skip lines with invalid timestamps
            continue
        
        # Store the first timestamp to calculate relative time
        if not start_time:
            start_time = timestamp
        
        
        # Extract metrics, assuming the format matches what you provided
        try:
            mem_util = float(parts[-4])
            mem_util_perc = mem_util / 100000.0 * 100
            # print(cpu_util)
            
            # Calculate seconds since start
            seconds = (timestamp - start_time).total_seconds()
            # print(seconds)
            
            # Store the data
            timestamps.append(seconds)
            memory_bandwidth.append(mem_util_perc)  # As specified
            # memory_bandwidth.append(drama * 100)        # As specified
        except (ValueError, IndexError):
            # Skip lines with parsing errors
            continue

    return timestamps, memory_bandwidth


def parse_cpu_output(file_path, start_time = None):
    """Parse DCGM output file with timestamps and extract metrics."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize data structures
    timestamps = []
    cpu_throughput = []
    memory_bandwidth = []

    # Parse each line
    for line in lines:
        # Skip header lines but catch lines with actual data
        # Extract timestamp
        #remove the blank lines
        if not line.strip():
            continue

        timestamp_match = re.search(r'(.*?) \|', line)
        if not timestamp_match:
            continue
            
        timestamp_str = timestamp_match.group(1)
        try:
            timestamp = datetime.strptime(timestamp_str, '%H:%M:%S.%f')
            # print(timestamp)
            
            # Store the first timestamp to calculate relative time
            if not start_time:
                start_time = timestamp
            
            # Split the line into parts
            parts = line.split()
            
            # Extract metrics, assuming the format matches what you provided
            try:
                cpu_util = float(parts[-1])
                # print(cpu_util)
                
                # Calculate seconds since start
                seconds = (timestamp - start_time).total_seconds()
                # print(seconds)
                
                # Store the data
                timestamps.append(seconds)
                cpu_throughput.append(cpu_util)  # As specified
                # memory_bandwidth.append(drama * 100)        # As specified
            except (ValueError, IndexError):
                # Skip lines with parsing errors
                continue
        except ValueError:
            # Skip lines with invalid timestamps
            continue

    return timestamps, cpu_throughput

def create_plots(timestamps_cpu, cpu_throughput, timestamps_mem, memory_bandwidth, output_file=None):
    """Create the two subplots with the parsed data."""
    fig, (ax2) = plt.subplots(1, 1, figsize=(12, 5))
    
    # Style settings
    plt.style.use('ggplot')
    
    # Colors
    throughput_color = '#4CAF50'  # Green
    memory_color = '#2196F3'      # Blue
        
    # Plot Memory Bandwidth
    ax2.fill_between(timestamps_mem, memory_bandwidth, color=memory_color, linewidth=2)
    ax2.set_title('CPU Memory Bandwidth', fontsize=16)
    ax2.set_ylabel('Bandwidth (%)', fontsize=14)
    ax2.set_xlabel('Time (seconds)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 100)  # Set y-axis limits to 0-100%
    
    # Find average and max for annotations
    avg_memory = np.mean(memory_bandwidth)
    max_memory = np.max(memory_bandwidth)
    ax2.axhline(y=avg_memory, color='black', linestyle='--', alpha=0.7, 
                label=f'Avg: {avg_memory:.2f}')
    ax2.axhline(y=max_memory, color='black', linestyle='--', alpha=0.7, 
                label=f'Max: {avg_memory:.2f}')
    
    # Add text annotation for max and average
    ax2.text(timestamps_mem[-1] * 0.02, max_memory * 0.95, 
             f'Max: {max_memory:.2f}', fontsize=12)
    ax2.text(timestamps_mem[-1] * 0.02, avg_memory * 1.05, 
             f'Avg: {avg_memory:.2f}', fontsize=12)
    
    # Format both axes
    # for ax in [ax1, ax2]:
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
        
    # Add overall title
    # plt.suptitle('CPU Memory Performance Metrics Over Time', fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Parse stat output and create performance plots.')
    parser.add_argument('-o', '--output', help='Path to save the output plot file (optional)')
    parser.add_argument('--input_file_cpu', help='Path to the stat output file', required=True)
    parser.add_argument('--input_file_mem', help='Path to the stat output file', required=True)
    parser.add_argument('-s', '--start_time', help='Start time for relative timestamps (optional)')
    args = parser.parse_args()

    if args.start_time:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S')    

    # Parse the data
    timestamps_cpu, cpu_throughput = parse_cpu_output(args.input_file_cpu, start_time)
    timestamps_mem, memory_bandwidth = parse_mem_output(args.input_file_mem, start_time)
    
    if not timestamps_mem:
        print("No valid data found in the input file!")
        return
    
    # Create and display/save the plots
    create_plots(timestamps_cpu, cpu_throughput, timestamps_mem, memory_bandwidth, args.output)

if __name__ == "__main__":
    main()