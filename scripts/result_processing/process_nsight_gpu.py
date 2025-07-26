from datetime import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_gpu_metrics(sqlite_file, results_dir):
    """
    Extract and plot GPU metrics from Nsight Systems SQLite database
    
    Parameters:
    -----------
    sqlite_file : str
        Path to the SQLite file exported from Nsight Systems
    """
    # Connect to the database
    conn = sqlite3.connect(sqlite_file)

    # print all tables
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print("Tables in the SQLite database:")
    print(tables.to_string())

    # cpu_info = pd.read_sql_query("SELECT * FROM CPU_INFO", conn)
    
    # First, check the structure of the GPU_METRICS table
    try:
        table_info = pd.read_sql_query("PRAGMA table_info(GPU_METRICS)", conn)
        print("GPU_METRICS table structure:")
        print(table_info)
        
        # Get a sample of data to understand its format
        sample = pd.read_sql_query("SELECT * FROM GPU_METRICS LIMIT 5", conn)
        print("\nSample data from GPU_METRICS:")
        print(sample)
        
        # Extract metrics related to SMs Active and DRAM Bandwidth
        # First, find the relevant column indices
        target_info = pd.read_sql_query("SELECT * FROM TARGET_INFO_GPU_METRICS", conn)
        print("\nMetrics info from TARGET_INFO_GPU_METRICS:")
        print(target_info)
        
        # Find indices for our metrics of interest
        sm_active_idx = None
        dram_read_idx = None
        dram_write_idx = None
        
        for idx, row in target_info.iterrows():
            name = row.get('metricName', '')
            print(f"Checking metric: {name}")
            if isinstance(name, str):
                if "SMs Active" in name:
                    sm_active_idx = idx
                elif "DRAM Read Bandwidth" in name:
                    dram_read_idx = idx
                elif "DRAM Write Bandwidth" in name:
                    dram_write_idx = idx
        
        print(f"\nIndices found: SM Active: {sm_active_idx}, DRAM Read: {dram_read_idx}, DRAM Write: {dram_write_idx}")
        
        # Query all GPU metrics data
        gpu_metrics = pd.read_sql_query("SELECT * FROM GPU_METRICS", conn)

        # peek one record
        if not gpu_metrics.empty:
            print("\nSample GPU metrics data:")
            print(gpu_metrics.head(1))

        print(f"\nLoaded {len(gpu_metrics)} GPU metric records")
        
        # Assuming the data has a timestamp column and values columns
        timestamp_col = 'timestamp' if 'timestamp' in gpu_metrics.columns else 'startTime'
        
        # Check if we need to normalize timestamps
        if timestamp_col in gpu_metrics.columns:
            # Convert timestamps to seconds for readability
            base_time = gpu_metrics[timestamp_col].min()
            gpu_metrics['time_sec'] = (gpu_metrics[timestamp_col] - base_time) / 1_000_000_000  # Convert ns to seconds
        else:
            # Try to find an appropriate timestamp column
            time_cols = [col for col in gpu_metrics.columns if 'time' in col.lower()]
            if time_cols:
                timestamp_col = time_cols[0]
                base_time = gpu_metrics[timestamp_col].min()
                gpu_metrics['time_sec'] = (gpu_metrics[timestamp_col] - base_time) / 1_000_000_000
            else:
                # If no timestamp column found, create a sequential one
                gpu_metrics['time_sec'] = np.arange(len(gpu_metrics))

        # filter gpu_metrics by metricId
        sm_active_metric = gpu_metrics[gpu_metrics['metricId'] == sm_active_idx]
        dram_read_metric = gpu_metrics[gpu_metrics['metricId'] == dram_read_idx]
        dram_write_metric = gpu_metrics[gpu_metrics['metricId'] == dram_write_idx]

        print(f"\nFiltered metrics: SM Active: {len(sm_active_metric)}, DRAM Read: {len(dram_read_metric)}, DRAM Write: {len(dram_write_metric)}")
        
        # In the plot_gpu_metrics function, replace the plotting sections with this code:

        # Create a figure with two subplots with a nicer background color
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, facecolor='#f8f9fa')
        ax1.set_facecolor('#f0f1f5')
        ax2.set_facecolor('#f0f1f5')

        # Plot SMs Active with a nice blue color and transparency
        ax1.plot(sm_active_metric['time_sec'], sm_active_metric['value'], 
                color='#3498db', alpha=0.8, linewidth=2.5)
        ax1.set_title('GPU SMs Active Over Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('SMs Active [Throughput %]', fontsize=14)
        ax1.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')

        # Add statistics with nicer colors
        sm_mean = sm_active_metric['value'].mean()
        sm_max = sm_active_metric['value'].max()
        ax1.axhline(y=sm_mean, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=2, 
                    label=f'Mean: {sm_mean:.2f}%')
        ax1.axhline(y=sm_max, color='#27ae60', linestyle='--', alpha=0.6, linewidth=2, 
                    label=f'Max: {sm_max:.2f}%')
        ax1.legend(fontsize=12, facecolor='white', framealpha=0.9, edgecolor='#d4d4d4')
        ax1.set_ylim(0, 100)  # Set y-axis limits to 0-100%

        # Plot DRAM Bandwidth with nicer colors
        # Calculate and plot total bandwidth
        dram_total = []
        assert len(dram_read_metric) == len(dram_write_metric), "Length of read and write metrics do not match"
        for i in range(len(dram_read_metric)):
            dram_total.append(dram_read_metric['value'].iloc[i] + dram_write_metric['value'].iloc[i])
        dram_total = pd.Series(dram_total)

        # Plot the bandwidth lines with attractive colors and transparency
        ax2.plot(dram_read_metric['time_sec'], dram_total, 
                color='#8e44ad', alpha=0.7, linestyle='--', linewidth=2.5, 
                label='Total')
        ax2.plot(dram_read_metric['time_sec'], dram_read_metric['value'], 
                color='#2ecc71', alpha=0.7, linewidth=2.5, 
                label='Read')
        ax2.plot(dram_write_metric['time_sec'], dram_write_metric['value'], 
                color='#e74c3c', alpha=0.7, linewidth=2.5, 
                label='Write')

        ax2.set_title('GPU DRAM Bandwidth Over Time', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=14)
        ax2.set_ylabel('DRAM Bandwidth [Throughput %]', fontsize=14)
        ax2.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')
        ax2.set_ylim(0, 100)  # Set y-axis limits to 0-100%

        # Add statistics with a nicer text box
        read_mean = dram_read_metric['value'].mean()
        write_mean = dram_write_metric['value'].mean()
        total_mean = dram_total.mean()

        stats_text = (f"Mean Bandwidth:\n"
                    f"Read: {read_mean:.2f}%\n"
                    f"Write: {write_mean:.2f}%\n"
                    f"Total: {total_mean:.2f}%")

        ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8,
                        edgecolor='#d4d4d4'))

        ax2.legend(fontsize=12, loc='upper right', facecolor='white', 
                framealpha=0.9, edgecolor='#d4d4d4')
        
        # Adjust layout and save the plot
        plt.tight_layout()
        # run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{results_dir}/gpu_throughput.png"
        # output_file = Path(sqlite_file).with_stem(f"{Path(sqlite_file).stem}_gpu_utilization").with_suffix('.png')
        plt.savefig(output_file, dpi=300)
        # plt.show()
        
        print(f"Plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing SQLite file: {e}")
        exit(1)
    
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python plot_nsight_metrics.py <path_to_sqlite_file> <results-dir>")
        sys.exit(1)
    
    sqlite_file = sys.argv[1]
    results_dir = sys.argv[2]
    plot_gpu_metrics(sqlite_file, results_dir)