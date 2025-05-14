import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re
from datetime import datetime
import argparse

def parse_dcgm_output(file_path, start_time = None):
    """Parse DCGM output file with timestamps and extract metrics."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize data structures
    timestamps = []
    sm_active = []
    sm_occupied = []
    memory_bandwidth = []

    # Parse each line
    for line in lines:
        # Skip header lines but catch lines with actual data
        if '[' in line and 'GPU' in line and len(line.split()) >= 5:
            # Extract timestamp
            timestamp_match = re.search(r'\[(.*?)\]', line)
            if not timestamp_match:
                continue
                
            timestamp_str = timestamp_match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                
                # Store the first timestamp to calculate relative time
                if not start_time:
                    start_time = timestamp
                
                # Split the line into parts
                parts = line.split()
                
                # Extract metrics, assuming the format matches what you provided
                if len(parts) >= 5:
                    try:
                        smact = float(parts[4])
                        smocc = float(parts[5])
                        drama = float(parts[6])
                        
                        # Calculate seconds since start
                        seconds = (timestamp - start_time).total_seconds()
                        
                        # Store the data
                        timestamps.append(seconds)
                        sm_active.append(smact * 100)
                        sm_occupied.append(smocc * 100)  # As specified
                        # gpu_throughput.append(smact * smocc * 100)  # As specified
                        memory_bandwidth.append(drama * 100)        # As specified
                    except (ValueError, IndexError):
                        # Skip lines with parsing errors
                        continue
            except ValueError:
                # Skip lines with invalid timestamps
                continue

    return timestamps, sm_active, sm_occupied, memory_bandwidth

def create_plots(timestamps, sm_active, sm_occupied, memory_bandwidth, output_file=None):
    """Create the two subplots with the parsed data."""
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 5))
    
    # Style settings
    plt.style.use('ggplot')
    
    # Colors
    sm_active_color = '#666A6D'  # Orange
    sm_occupied_color = '#FF9800'  # Red
    throughput_color = '#4CAF50'  # Green
    memory_color = '#2196F3'      # Blue
    
    # Plot GPU Throughput
    ax1.fill_between(timestamps, sm_active, color=sm_active_color, linewidth=2)
    ax1.fill_between(timestamps, sm_occupied, color=sm_occupied_color, linewidth=2)
    ax1.set_title('GPU Compute Throughput', fontsize=16)
    # ax1.set_ylabel('Throughput % (SMACT × SMOCC × 100)', fontsize=14)
    ax1.set_ylabel('Throughput % (SMACT, SMOCC)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 100)  # Set y-axis limits to 0-100%
    
    # Find average and max for annotations
    gpu_throughput = np.array(sm_active) * np.array(sm_occupied) / 100
    avg_throughput = np.mean(gpu_throughput)
    max_throughput = np.max(gpu_throughput)
    ax1.axhline(y=avg_throughput, color='black', linestyle='--', alpha=0.7, 
                label=f'Avg: {avg_throughput:.2f}')
    ax1.axhline(y=max_throughput, color='black', linestyle='--', alpha=0.7, 
                label=f'Max: {max_throughput:.2f}')
    
    # Add text annotation for max and average
    ax1.text(timestamps[-1] * 0.02, max_throughput * 0.95, 
             f'Max: {max_throughput:.2f}', fontsize=12)
    ax1.text(timestamps[-1] * 0.02, avg_throughput * 1.05, 
             f'Avg: {avg_throughput:.2f}', fontsize=12)
        
    # Format both axes
    # for ax in [ax1, ax2]:
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
        
    # Add overall title
    # plt.suptitle('GPU Compute Throughput Over Time', fontsize=18, y=0.98)
    
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
    parser = argparse.ArgumentParser(description='Parse DCGM output and create performance plots.')
    parser.add_argument('input_file', help='Path to the DCGM output file')
    parser.add_argument('-s', '--start_time', help='Start time for relative timestamps (optional)')
    parser.add_argument('-o', '--output', help='Path to save the output plot file (optional)')
    args = parser.parse_args()
    
    if args.start_time:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S')    

    # Parse the data
    timestamps, sm_active, sm_occupied, memory_bandwidth = parse_dcgm_output(args.input_file, start_time)
    
    if not timestamps:
        print("No valid data found in the input file!")
        return
    
    # Create and display/save the plots
    create_plots(timestamps, sm_active, sm_occupied, memory_bandwidth, args.output)

if __name__ == "__main__":
    main()