import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import re
import sys
import os
import csv
import pandas as pd

# Global font size
plt.rcParams.update({'font.size': 14})


def parse_benchmark_file(filename):
    """
    Parse benchmark log file and extract task information.
    Expected format: INFO:root:Task taskname: start_time - end_time
    """
    tasks = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Use regex to extract task name, start time, and end time
                match = re.match(r'INFO:root:Task\s+([^:]+):\s+(\d+\.\d+)\s+-\s+(\d+\.\d+)', line)
                if match:
                    task_name = match.group(1)
                    start_time = float(match.group(2))
                    end_time = float(match.group(3))

                    # Skip the "start" task
                    if task_name != "start":
                        tasks.append((task_name, start_time, end_time))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    return tasks

def get_slo_status_for_task(task_name, csv_directory="."):
    """
    Analyze the CSV file for a task and determine SLO status percentages.
    Returns tuple: (percentage_met_slo, percentage_missed_slo)
    """
    # Map task names to their CSV files and SLOs
    task_configs = {
        "Analysis (deep_research)_u0": {
            "csv_format": "request_num,total_time",
            "slo_column": "total_time",
            "slo_value": 10000  # seconds
        },
        "Brainstorm (chatbot)_u0": {
            "csv_format": "request_num,ttft,tpot,itl",
            "slo_column": ["ttft", "tpot"],
            "slo_value": [1, 0.25]  # seconds
        },
        "Preparing Outline (chatbot)_u0": {
            "csv_format": "request_num,ttft,tpot,itl",
            "slo_column": ["ttft", "tpot"],
            "slo_value": [1, 0.25]  # seconds
        },
        "Creating Cover Art (imagegen)_u0": {
            "csv_format": "request_num,total_time",
            "slo_column": "total_time",
            "slo_value": 28  # seconds
        },
        "Generating Captions (live_captions)_u0": {
            "csv_format": "request_idx,time",
            "slo_column": "time",
            "slo_value": 2  # seconds
        }
    }

    # Clean task name if needed - remove 'Task ' prefix if present
    if task_name.startswith("Task "):
        clean_task_name = f"{task_name[5:]}"
    else:
        clean_task_name = f"{task_name}"

    # Get configuration for this task
    if clean_task_name not in task_configs:
        print(f"Warning: No CSV configuration found for task '{clean_task_name}'. Using default 100% SLO met.")
        return (100.0, 0.0)

    config = task_configs[clean_task_name]

    # Construct csv filename based on task name
    csv_filename = os.path.join(csv_directory, f"task_{clean_task_name}_perf.csv")

    # Check if CSV file exists
    if not os.path.exists(csv_filename):
        print(f"Warning: CSV file '{csv_filename}' not found. Using default 100% SLO met.")
        return (100.0, 0.0)

    try:
        # Read the CSV file
        df = pd.read_csv(csv_filename, header=None)

        # Parse column names from csv_format
        columns = config["csv_format"].split(",")
        # set these columns as the column names

        df.columns = columns

        # skipt the first row
        df = df.iloc[1:]

        # Check SLO status
        total_requests = len(df)

        if total_requests == 0:
            return (100.0, 0.0)

        met_slo_count = 0
        latency_ttft = []
        latency_tpot = []
        latency_req = []
        slo_ttft = 0
        slo_tpot = 0
        slo_req_col = []
        slo_req = 0
        ttft_done = False

        # For tasks with multiple SLO columns/values (like chatbot tasks)
        if isinstance(config["slo_column"], list):
            # A request meets the SLO only if ALL conditions are met
            for row_idx, row in df.iterrows():
                all_slos_met = True
                for col, slo_val in zip(config["slo_column"], config["slo_value"]):
                    col_value = float(row[col])
                    if not ttft_done:
                        latency_ttft.append(col_value)
                        slo_ttft = slo_val
                    else:
                        latency_tpot.append(col_value)
                        slo_tpot = slo_val
                    if col_value > slo_val:
                        all_slos_met = False
                        break
                if all_slos_met:
                    met_slo_count += 1
                ttft_done = True
        else:
            # For tasks with a single SLO column/value
            df[config["slo_column"]] = df[config["slo_column"]].astype(float)
            met_slo_count = sum(df[config["slo_column"]] <= config["slo_value"])
            latency_req = df[config["slo_column"]].tolist()
            slo_req = config["slo_value"]

        percent_met = (met_slo_count / total_requests) * 100
        percent_missed = 100 - percent_met

        return (percent_met, percent_missed, latency_ttft, latency_tpot, latency_req, slo_ttft, slo_tpot, slo_req)

    except Exception as e:
        print(f"Error processing CSV for '{clean_task_name}': {e}")
        return (100.0, 0.0)  # Default to 100% met if there's an error

def create_gantt_chart_with_slo(data, csv_directory=".", output_filename="task_benchmark_gantt_slo.pdf", show_legend=True):
    """
    Create a Gantt chart from the task data with integrated SLO status within each task bar.

    Args:
        data: List of tuples (task_name, start_time, end_time)
        csv_directory: Directory containing the CSV files
        output_filename: Name of the output PDF file
    """
    # Sort tasks by start time
    data.sort(key=lambda x: x[1])

    # Extract task information
    task_names = [task[0] for task in data]
    start_times = [task[1] for task in data]
    end_times = [task[2] for task in data]
    durations = [end - start for start, end in zip(start_times, end_times)]

    # Get SLO status for each task
    slo_statuses = [get_slo_status_for_task(task_name, csv_directory) for task_name in task_names]

    # Create a figure with a specific size
    plt.figure(figsize=(10, 4))

    # Define colors
    slo_met_color = '#70AD47'  # Green for SLO met
    slo_missed_color = '#E74C3C'  # Red for SLO missed

    # Create a list to store legend handles and labels
    legend_elements = []

    x_min = min(start_times) - 2
    x_max = 2200 + 2

    # Draw stacked bars for each task
    for i, (task, start, end, slo_status) in enumerate(zip(task_names, start_times, end_times, slo_statuses)):
        duration = end - start

        # Calculate the width of each segment based on SLO percentages
        met_width = duration * (slo_status[0] / 100)
        missed_width = duration * (slo_status[1] / 100)
        latency_ttft = slo_status[2]
        latency_tpot = slo_status[3]
        latency_req = slo_status[4]
        slo_ttft = slo_status[5]
        slo_tpot = slo_status[6]
        slo_req = slo_status[7]
        print(f"TTFT: {slo_ttft}, TPOT: {slo_tpot}, REQ: {slo_req}")
        if len(latency_ttft) > 0:
            mean_latency_ttft = np.mean(latency_ttft)
            mean_latency_tpot = np.mean(latency_tpot)
        else:
            mean_latency_req = np.mean(latency_req)

        # Draw the SLO met portion (green part)
        met_bar = plt.barh(
            y=i,
            width=met_width,
            left=start,
            height=0.7,
            color=slo_met_color,
            alpha=0.9,
            edgecolor='black'
        )

        # Draw the SLO missed portion (red part)
        missed_bar = plt.barh(
            y=i,
            width=missed_width,
            left=start + met_width,
            height=0.7,
            color=slo_missed_color,
            alpha=0.9,
            edgecolor='black'
        )

        # Only add to legend once
        if i == 0:
            legend_elements.append(met_bar)
            legend_elements.append(missed_bar)

        # Add SLO percentage text labels inside the bars
        # Only add text if percentage is large enough to fit text
        # if slo_status[0] > 10 and met_width > 3:
        #     plt.text(
        #         x=start + (met_width / 2),
        #         y=i,
        #         s=f"{int(slo_status[0])}%",
        #         ha='center',
        #         va='center',
        #         fontsize=9,
        #         fontweight='bold',
        #         color='black'
        #     )

        # if slo_status[1] > 10 and missed_width > 3:
        #     plt.text(
        #         x=start + met_width + (missed_width / 2),
        #         y=i,
        #         s=f"{int(slo_status[1])}%",
        #         ha='center',
        #         va='center',
        #         fontsize=9,
        #         fontweight='bold',
        #         color='black'
        #     )

        # # Add duration text label above each bar
        # plt.text(
        #     x=start + (duration / 2),
        #     y=i + 0.3,  # Position above the bar
        #     s=f"{duration:.1f}s",
        #     ha='center',
        #     va='bottom',
        #     fontsize=10,
        #     fontweight='bold',
        #     color='black'
        # )

        space_right = x_max - end
        text_space_needed = 600  # Adjust this value based on your font size and figure size

        # Prepare text to show with conditional formatting
        if len(latency_ttft) > 0:
            # For chatbot tasks with TTFT and TPOT metrics
            if mean_latency_ttft > slo_ttft:
                # TTFT exceeds SLO - should be red
                ttft_color = 'red'
                ttft_normal = f"TTFT: {mean_latency_ttft:.1f} sec/req "
                ttft_bold = f"({mean_latency_ttft/slo_ttft:.1f}x SLO)"
            else:
                # TTFT within SLO - should be green
                ttft_color = 'green'
                ttft_normal = f"TTFT: {mean_latency_ttft:.1f} sec/req"
                ttft_bold = ""

            if mean_latency_tpot > slo_tpot:
                # TPOT exceeds SLO - should be red
                tpot_color = 'red'
                tpot_normal = f"\nTPOT: {mean_latency_tpot:.1f} sec/req "
                tpot_bold = f"({mean_latency_tpot/slo_tpot:.1f}x SLO)"
            else:
                # TPOT within SLO - should be green
                tpot_color = 'green'
                tpot_normal = f"\nTPOT: {mean_latency_tpot:.1f} sec/req"
                tpot_bold = ""
        else:
            # For non-chatbot tasks with single request latency
            if mean_latency_req > slo_req:
                # Latency exceeds SLO - should be red
                req_color = 'red'
                req_normal = f"{mean_latency_req:.2f} sec/req "
                req_bold = f"({mean_latency_req/slo_req:.1f}x SLO)"
            else:
                # Latency within SLO - should be green
                req_color = 'green'
                req_normal = f"{mean_latency_req:.2f} sec/req"
                req_bold = ""


        if len(latency_ttft) > 0:
            text_to_show = f"TTFT: {mean_latency_ttft:.1f} sec/req"
            if mean_latency_ttft > slo_ttft:
                text_to_show += f" ({mean_latency_ttft/slo_ttft:.1f}x SLO)\n"
            if mean_latency_tpot > slo_tpot:
                text_to_show += f"TPOT: {mean_latency_tpot:.1f} sec/req"
                text_to_show += f" ({mean_latency_tpot/slo_tpot:.1f}x SLO)"
                color = slo_missed_color
            else:
                color = slo_met_color

            if mean_latency_tpot > slo_tpot*5:
                weight = 'bold'
            else:
                weight = 'normal'

            if mean_latency_ttft > slo_ttft*5:
                weight = 'bold'
            else:
                weight = 'normal'

        else:
            text_to_show = f"{mean_latency_req:.2f} sec/req"
            if mean_latency_req > slo_req:
                text_to_show += f" ({mean_latency_req/slo_req:.1f}x SLO)"
                color = slo_missed_color
            else:
                color = slo_met_color

            if mean_latency_req > slo_req*5:
                weight = 'bold'
            else:
                weight = 'normal'

        # Check if there's enough space to the right
        if space_right >= text_space_needed:
            plt.text(
                x=end + 5,  # Small offset from the end of the bar
                y=i,
                s=text_to_show,
                ha='left',
                va='center',
                # fontsize=10,
                fontweight=weight,
                color=color
            )
        else:
            # Not enough space on right, check if there's space on the left
            if start - x_min >= text_space_needed:
                plt.text(
                    x=start - 5,  # Small offset from the start of the bar
                    y=i,
                    s=text_to_show,
                    ha='right',
                    va='center',
                    # fontsize=10,
                    fontweight=weight,
                    color=color
                )
            else:
                # Not enough space on either side, place it on top of the bar
                plt.text(
                    x=start + (duration / 2),
                    y=i + 0.3,  # Position above the bar
                    s=text_to_show,
                    ha='center',
                    va='bottom',
                    # fontsize=10,
                    fontweight=weight,
                    color=color
                )

    # Set task names as y-tick labels
    # remove everything after "_" in task_names
    task_names = [name.split("(")[0] + "\n(" + name.split("(")[1] for name in task_names]
    task_names = [name.split(")")[0] + ")" for name in task_names]
    # have a new line before "(" in task_names
    plt.yticks(range(len(task_names)), task_names)

    # Customize the plot
    # plt.title('Task Benchmark Timeline with Integrated SLO Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time (seconds)')
    # plt.ylabel('Tasks', fontsize=12)

    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Customize x-axis ticks
    plt.xlim(x_min, x_max)  # Add some padding
    # plt.xticks(fontsize=10)

    # Determine appropriate tick spacing based on total time range
    time_range = max(end_times) - min(start_times)
    if time_range > 1000:
        major_tick = 200
        minor_tick = 50
    elif time_range > 400:
        major_tick = 100
        minor_tick = 20
    elif time_range > 200:
        major_tick = 50
        minor_tick = 10
    elif time_range > 100:
        major_tick = 20
        minor_tick = 5
    elif time_range > 50:
        major_tick = 10
        minor_tick = 2
    else:
        major_tick = 5
        minor_tick = 1

    plt.gca().xaxis.set_major_locator(MultipleLocator(major_tick))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(minor_tick))

    # Create custom legend
    legend_labels = ['SLO Met', 'SLO Missed']
    if show_legend:
        plt.legend(legend_elements, legend_labels, loc='upper center',
                bbox_to_anchor=(0.14, 0.98), ncol=1, frameon=True)

    # Adjust layout
    plt.tight_layout()

    # Save as PDF
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Gantt chart with integrated SLO analysis saved as '{output_filename}'")

    # Display the plot
    plt.show()

def main():
    """Main function to run the script."""
    # Check if a filename was provided as a command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = os.path.splitext(input_file)[0] + "_gantt_slo.pdf"

        # Get the directory of the input file to look for CSV files
        csv_directory = os.path.dirname(input_file) or "."
    else:
        # Default input file
        input_file = "overall_perf.log"
        output_file = "task_benchmark_gantt_slo.pdf"
        csv_directory = "."

    # Parse the benchmark file
    data = parse_benchmark_file(input_file)

    if not data:
        print("No valid task data found in the file.")
        sys.exit(1)

    # Create the Gantt chart with SLO analysis
    create_gantt_chart_with_slo(data, csv_directory, output_file, show_legend=True)

if __name__ == "__main__":
    main()