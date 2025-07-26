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
        pattern = r"app_type: chatbot"
        match = re.search(pattern, file_content)
        if not match:
            return True
        
        # Extract the results part using regular expressions
        pattern = r"Task .* results:\s*(.*)"
        match = re.search(pattern, file_content)
        
        if not match:
            raise ValueError(f"Could not find 'Task chatbot results:' in the file {file_path}")
        
        # Parse the extracted string as a Python list
        results_list = ast.literal_eval(match.group(1))
        
        # The first element is None and the last is True, so we skip those
        # The remaining elements are dictionaries with the metrics
        metrics_dicts = results_list[1:-1]
        
        # get directory of file
        output_dir = file_path.split("/")[:-1]
        if len(output_dir) > 0:
            output_dir = "/".join(output_dir)

        base_filename = file_path.split("/")[-1].split(".")[0]
        output_filename = f'{output_dir}/{base_filename}.csv'
        base_filename = f'{output_dir}/{base_filename}'
        
        # Write to CSV
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['request_num', 'ttft', 'tpot', 'itl'])
            
            # Write data rows
            for i, metrics in enumerate(metrics_dicts, 1):
                writer.writerow([i, metrics['ttft'], metrics['tpot'], metrics['itl']])
        
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
    request_nums = list(range(1, len(metrics_dicts) + 1))
    ttft_values = [metrics['ttft'] for metrics in metrics_dicts]
    tpot_values = [metrics['tpot'] for metrics in metrics_dicts]
    
    # Define SLO values
    ttft_slo = 1.0  # 1 second for ttft
    tpot_slo = 0.2  # 0.2 seconds for tpot
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Chatbot Performance Metrics', fontsize=20, fontweight='bold', y=0.98)
    
    # Add a subtle background color
    fig.patch.set_facecolor('#F5F5F5')
    
    # TTFT Plot
    ttft_colors = [compliant_color if val <= ttft_slo else non_compliant_color for val in ttft_values]
    bars1 = ax1.bar(
        request_nums, 
        ttft_values, 
        color=ttft_colors,
        alpha=0.85,
        width=0.7,
        edgecolor='white',
        linewidth=1
    )
    
    # SLO line with shading
    ax1.axhline(y=ttft_slo, color=slo_line_color, linestyle='-', linewidth=2.5, 
               label=f'SLO Threshold: {ttft_slo}s')
    ax1.fill_between(
        [0, len(request_nums) + 1], 
        0, ttft_slo, 
        color=slo_line_color, 
        alpha=0.1
    )
    
    # Formatting TTFT plot
    ax1.set_title('Time to First Token (TTFT)', fontsize=16, pad=15)
    ax1.set_ylabel('Seconds', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlim(0.25, len(request_nums) + 0.75)
    ax1.set_ylim(0, max(ttft_values) * 1.1)  # Add 10% headroom
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add data labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height + max(ttft_values) * 0.01,
            f'{height:.2f}s',
            ha='center', va='bottom', 
            fontsize=10, rotation=0,
            fontweight='bold'
        )
    
    # TPOT Plot
    tpot_colors = [compliant_color if val <= tpot_slo else non_compliant_color for val in tpot_values]
    bars2 = ax2.bar(
        request_nums, 
        tpot_values, 
        color=tpot_colors,
        alpha=0.85,
        width=0.7,
        edgecolor='white',
        linewidth=1
    )
    
    # SLO line with shading
    ax2.axhline(y=tpot_slo, color=slo_line_color, linestyle='-', linewidth=2.5, 
               label=f'SLO Threshold: {tpot_slo}s')
    ax2.fill_between(
        [0, len(request_nums) + 1], 
        0, tpot_slo, 
        color=slo_line_color, 
        alpha=0.1
    )
    
    # Formatting TPOT plot
    ax2.set_title('Time Per Output Token (TPOT)', fontsize=16, pad=15)
    ax2.set_xlabel('Request Number', fontsize=14)
    ax2.set_ylabel('Seconds', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlim(0.25, len(request_nums) + 0.75)
    ax2.set_ylim(0, max(tpot_values) * 1.2)  # Add 20% headroom
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add data labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            height + max(tpot_values) * 0.02,
            f'{height:.3f}s',
            ha='center', va='bottom', 
            fontsize=10, rotation=0,
            fontweight='bold'
        )
    
    # Add legend
    ax1.legend(fontsize=12, loc='upper right')
    ax2.legend(fontsize=12, loc='upper right')
    
    # Add summary stats as text
    ttft_compliance = sum(1 for v in ttft_values if v <= ttft_slo) / len(ttft_values) * 100
    tpot_compliance = sum(1 for v in tpot_values if v <= tpot_slo) / len(tpot_values) * 100
    
    summary_text = (
        f"Summary Statistics:\n"
        f"TTFT - Avg: {np.mean(ttft_values):.2f}s, Max: {max(ttft_values):.2f}s, SLO Compliance: {ttft_compliance:.1f}%\n"
        f"TPOT - Avg: {np.mean(tpot_values):.3f}s, Max: {max(tpot_values):.3f}s, SLO Compliance: {tpot_compliance:.1f}%"
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