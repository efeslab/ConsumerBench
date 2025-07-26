import csv
import re
import ast
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def parse_results_from_file(file_path, generate_plot=True):
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # Check if the file contains the app_type for chatbot        
        pattern = r"app_type: live_captions"
        match = re.search(pattern, file_content)
        if not match:
            return True
        
        # Extract the results part using regular expressions
        pattern = r"Task .* results:\s*(.*)"
        match = re.search(pattern, file_content)
        
        if not match:
            raise ValueError(f"Could not find 'Task live_captions results:' in the file {file_path}")
    
        # Parse the extracted string as a Python list
        results_list = ast.literal_eval(match.group(1))
    
        # The first element is -1 and the last is True, so we skip those
        # The remaining elements are dictionaries with the metrics
        metrics_dicts = results_list[1:-1]

        # get directory of file
        output_dir = file_path.split("/")[:-1]
        if len(output_dir) > 0:
            output_dir = "/".join(output_dir)

        base_filename = file_path.split("/")[-1].split(".")[0]
        output_filename = f'{output_dir}/{base_filename}.csv'
        base_filename = f'{output_dir}/{base_filename}'
    
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['request_idx', 'time'])
        
            # Write data rows
            num_requests = 1
            for i, metrics in enumerate(metrics_dicts, 1):
                for chunk_idx, time in metrics.items():
                    if chunk_idx.startswith('processing'):
                        chunk_idx = int(chunk_idx.split('_')[2]) + 1 + ((i - 1) * (len(metrics_dicts[0]) - 1))                        
                        writer.writerow([num_requests, time])
                        num_requests += 1
                # print(metrics)
    
        print(f"Successfully created {output_filename} with {len(metrics_dicts)} requests")

        # Generate plot if requested
        if generate_plot:
            create_plot(metrics_dicts, base_filename)

        return True

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return False
    except ValueError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def create_plot(metrics_dicts, base_filename):
    # Extract data
    request_nums = []
    time_values = []
    request_idx = 1
    for i, metrics in enumerate(metrics_dicts, 1):
        for chunk_idx, time in metrics.items():
            if chunk_idx.startswith('processing'):
                request_nums.append(request_idx)
                time_values.append(time)
                request_idx += 1
    
    # Define SLO values
    time_slo = 2.0  
    
    # Set style and colors
    import seaborn as sns

    # Use seaborn's native styling functions
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Color palette
    compliant_color = '#4CAF50'  # Green
    non_compliant_color = '#F44336'  # Red
    slo_line_color = '#FF9800'  # Orange
    
    # Create figure and axes
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Whisper Performance Metrics', fontsize=20, fontweight='bold', y=0.98)
    
    # Add a subtle background color
    fig.patch.set_facecolor('#F5F5F5')

    
    # TTFT Plot
    ttft_colors = [compliant_color if val <= time_slo else non_compliant_color for val in time_values]
    print("plotting...")
    bars1 = ax1.bar(
        request_nums, 
        time_values, 
        color=ttft_colors,
        alpha=0.85,
        width=0.7,
        edgecolor='white',
        linewidth=1
    )
    
    # SLO line with shading
    ax1.axhline(y=time_slo, color=slo_line_color, linestyle='-', linewidth=2.5, 
               label=f'SLO Threshold: {time_slo}s')
    ax1.fill_between(
        [0, len(request_nums) + 1], 
        0, time_slo, 
        color=slo_line_color, 
        alpha=0.1
    )
    
    # Formatting TTFT plot
    ax1.set_title('Time to Process', fontsize=16, pad=15)
    ax1.set_ylabel('Seconds', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlim(0.25, len(request_nums) + 0.75)
    ax1.set_ylim(0, max(time_values) * 1.1)  # Add 10% headroom
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add legend
    ax1.legend(fontsize=12, loc='upper right')
    
    # Add summary stats as text
    time_compliance = sum(1 for v in time_values if v <= time_slo) / len(time_values) * 100
    
    summary_text = (
        f"Summary Statistics:\n"
        f"Time - Avg: {np.mean(time_values):.2f}s, Max: {max(time_values):.2f}s, SLO Compliance: {time_compliance:.1f}%\n"
    )
    
    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.12)
    
    # Save the plot with high DPI for better quality
    plot_filename = f'{base_filename}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Successfully created enhanced plot: {plot_filename}")
    
    # Show the plot (optional, comment out if running in a non-interactive environment)
    # plt.show()


if __name__ == "__main__":
    # Check if a file path was provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        parse_results_from_file(file_path)
    else:
        print("Usage: python script_name.py <path_to_results_file>")
        print("Example: python parse_results.py log_file.txt")