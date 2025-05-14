import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import re
import sys
import os

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

def create_gantt_chart(data, output_filename="task_benchmark_gantt.pdf"):
    """
    Create a Gantt chart from the task data and save it as a PDF.
    
    Args:
        data: List of tuples (task_name, start_time, end_time)
        output_filename: Name of the output PDF file
    """
    # Sort tasks by start time
    data.sort(key=lambda x: x[1])
    
    # Extract task information
    task_names = [task[0] for task in data]
    start_times = [task[1] for task in data]
    end_times = [task[2] for task in data]
    durations = [end - start for start, end in zip(start_times, end_times)]
    
    # Create a figure with a specific size
    plt.figure(figsize=(12, 6))
    
    # Define colors for each task - using a colorblind-friendly palette
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47', '#264478', '#9E480E']
    if len(task_names) > len(colors):
        # If we have more tasks than colors, cycle through the colors again
        colors = colors * (len(task_names) // len(colors) + 1)
    
    # Create horizontal bars
    bars = plt.barh(
        y=task_names,
        width=durations,
        left=start_times,
        height=0.6,
        color=colors[:len(task_names)],
        alpha=0.8,
        edgecolor='black'
    )
    
    # Customize the plot
    plt.title('Task Benchmark Timeline', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Tasks', fontsize=12)
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Customize x-axis ticks
    plt.xlim(min(start_times) - 2, max(end_times) * 1.05)  # Add some padding
    plt.xticks(fontsize=10)
    
    # Determine appropriate tick spacing based on total time range
    time_range = max(end_times) - min(start_times)
    if time_range > 200:
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
    
    # Add task durations as text labels
    for i, (task, start, end) in enumerate(zip(task_names, start_times, end_times)):
        duration = end - start
        # Only add text if duration is long enough to fit text
        if duration > 3:
            plt.text(
                x=(start + end) / 2,
                y=i,
                s=f"{duration:.1f}s",
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='black'
            )
        
        # Add start and end times at the edges of each bar
        # plt.text(
        #     x=start - 0.5, 
        #     y=i,
        #     s=f"{start:.1f}",
        #     ha='right',
        #     va='center',
        #     fontsize=8,
        #     color='black'
        # )
        
        # plt.text(
        #     x=end + 0.5, 
        #     y=i,
        #     s=f"{end:.1f}",
        #     ha='left',
        #     va='center',
        #     fontsize=8,
        #     color='black'
        # )
    
    # Add a legend
    legend_patches = [mpatches.Patch(color=color, label=task) 
                      for task, color in zip(task_names, colors[:len(task_names)])]
    plt.legend(handles=legend_patches, loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(task_names)))
    
    # Identify and annotate parallel tasks
    # Group tasks that overlap in time
    task_groups = []
    for i, (task1, start1, end1) in enumerate(data):
        overlaps = []
        for j, (task2, start2, end2) in enumerate(data):
            if i != j:  # Don't compare a task with itself
                # Check if tasks overlap
                if (start1 <= end2 and end1 >= start2):
                    overlaps.append(j)
        if overlaps:
            task_groups.append([i] + overlaps)
    
    # Remove duplicate groups and sort
    unique_groups = []
    for group in task_groups:
        sorted_group = sorted(group)
        if sorted_group not in unique_groups:
            unique_groups.append(sorted_group)
    
    # Annotate parallel task groups
    for group in unique_groups:
        if len(group) > 1:  # Only annotate if there are actually parallel tasks
            # Find the common time range for the group
            group_start = max(start_times[i] for i in group)
            group_end = min(end_times[i] for i in group)
            
            # Only annotate if there's a significant overlap
            if group_end - group_start > 1:
                # Find the average y position for the annotation
                y_pos = sum(group) / len(group)
                
                # # Add annotation
                # plt.annotate('', 
                #              xy=(group_start, y_pos), 
                #              xytext=(group_end, y_pos),
                #              arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
                # plt.text((group_start + group_end) / 2, y_pos + 0.2, 
                #          'Parallel Tasks', ha='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Gantt chart saved as '{output_filename}'")
    
    # Display the plot
    plt.show()

def main():
    """Main function to run the script."""
    # Check if a filename was provided as a command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = os.path.splitext(input_file)[0] + "_gantt.pdf"
    else:
        # Default input file
        input_file = "benchmark_log.txt"
        output_file = "task_benchmark_gantt.pdf"
        
    # Parse the benchmark file
    data = parse_benchmark_file(input_file)
    
    if not data:
        print("No valid task data found in the file.")
        sys.exit(1)
    
    # Create the Gantt chart
    create_gantt_chart(data, output_file)

if __name__ == "__main__":
    main()