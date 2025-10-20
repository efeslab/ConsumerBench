from datetime import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_cpu_metrics(sqlite_file, results_dir):
    """
    Extract and plot CPU utilization metrics from Nsight Systems SQLite database
    
    Parameters:
    -----------
    sqlite_file : str
        Path to the SQLite file exported from Nsight Systems
    """
    # Connect to the database
    conn = sqlite3.connect(sqlite_file)

    # Print all tables
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print("Tables in the SQLite database:")
    print(tables.to_string())

    # First, try to find CPU core events if they were collected
    try:
        # Check if CPU core performance metrics exist
        cpu_tables = [table for table in tables['name'] if 'CPU' in table]
        print(f"\nFound CPU-related tables: {cpu_tables}")
        
        # Modified approach for COMPOSITE_EVENTS
        if 'COMPOSITE_EVENTS' in tables['name'].values:
            print("\nUsing COMPOSITE_EVENTS to calculate utilization")
            
            # Check table structure
            table_info = pd.read_sql_query("PRAGMA table_info(COMPOSITE_EVENTS)", conn)
            print("COMPOSITE_EVENTS table structure:")
            print(table_info)
            
            # Get composite events with CPU data
            composite_events = pd.read_sql_query("""
                SELECT start, cpu, cpuCycles
                FROM COMPOSITE_EVENTS
                WHERE cpu IS NOT NULL
                ORDER BY start
            """, conn)
            
            if composite_events.empty:
                raise ValueError("No CPU data found in COMPOSITE_EVENTS")
                
            # Convert timestamps to seconds from start
            base_time = composite_events['start'].min()
            composite_events['time_sec'] = (composite_events['start'] - base_time) / 1_000_000_000
            
            # Count the number of unique CPUs for normalizing
            num_cpus = composite_events['cpu'].nunique()
            print(f"\nNumber of CPU cores detected: {num_cpus}")
            
            # Based on the observation that cpuCycle=1 means CPU is active and 0 otherwise
            # We need to create time bins and calculate utilization in each bin
            # First, determine time range and create appropriate bins
            time_min = composite_events['time_sec'].min()
            time_max = composite_events['time_sec'].max()
            time_range = time_max - time_min
            
            # Create time bins (100 samples per second)
            bin_width = 0.001  # 10ms bins
            num_bins = int(time_range / bin_width) + 1
            time_bins = np.linspace(time_min, time_max, num_bins)
            
            # Create a dictionary to track CPU utilization in each bin
            bin_utilization = {i: 0 for i in range(len(time_bins)-1)}
            
            # Assign each event to a bin
            composite_events['bin_idx'] = np.digitize(composite_events['time_sec'], time_bins) - 1
            
            # Calculate active CPUs per bin
            unique_cpus_per_bin = composite_events.groupby('bin_idx')['cpu'].nunique()
            
            # Calculate utilization percentages
            # For each bin, count active CPUs and divide by total CPUs to get utilization
            for bin_idx in range(len(time_bins)-1):
                active_cpus = unique_cpus_per_bin.get(bin_idx, 0)
                # Calculate percentage: active_cpus / total_cpus * 100
                bin_utilization[bin_idx] = (active_cpus / num_cpus) * 100
            
            # Convert to arrays for plotting
            time_values = time_bins[:-1]  # Use left edges of bins
            utilization_values = np.array([bin_utilization[i] for i in range(len(time_bins)-1)])
            
            # Apply optional smoothing for better visualization (using rolling average)
            # window_size = 5  # Adjust based on desired smoothness
            # if len(utilization_values) > window_size:
            #     utilization_values = pd.Series(utilization_values).rolling(window=window_size, center=True).mean().fillna(0).values
            
        else:
            raise ValueError("No suitable CPU metrics tables found")
            
        # Create a figure for plotting
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='#f8f9fa')
        ax.set_facecolor('#f0f1f5')
        
        # Plot CPU utilization
        ax.plot(time_values, utilization_values, 
                color='#e74c3c', alpha=0.8, linewidth=2.5)
        
        # Fill the area between 0 and the utilization values
        ax.fill_between(time_values, 0, utilization_values, 
                color='#e74c3c', alpha=0.3)
        
        ax.set_title('CPU Utilization Over Time (All Cores Aggregated)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=14)
        ax.set_ylabel('CPU Utilization (%)', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')
        
        # Add statistics with nicer colors
        mean_util = np.mean(utilization_values)
        max_util = np.max(utilization_values)
        
        ax.axhline(y=mean_util, color='#3498db', linestyle='--', alpha=0.6, linewidth=2,
                   label=f'Mean: {mean_util:.2f}%')
        ax.axhline(y=max_util, color='#27ae60', linestyle='--', alpha=0.6, linewidth=2,
                   label=f'Max: {max_util:.2f}%')
        
        ax.legend(fontsize=12, facecolor='white', framealpha=0.9, edgecolor='#d4d4d4')
        ax.set_ylim(0, 100)  # Set y-axis limits to 0-100%
        
        # Adjust layout and save the plot
        plt.tight_layout()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{results_dir}/cpu_throughput.png"
        # output_file = Path(sqlite_file).with_stem(f"{Path(sqlite_file).stem}_cpu_utilization").with_suffix('.png')
        plt.savefig(output_file, dpi=300)
        # plt.show()
        
        print(f"Plot saved to {output_file}")
            
    except Exception as e:
        print(f"Error processing SQLite file for CPU metrics: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python plot_cpu_metrics.py <path_to_sqlite_file> <results_dir>")
        sys.exit(1)
    
    sqlite_file = sys.argv[1]
    results_dir = sys.argv[2]
    plot_cpu_metrics(sqlite_file, results_dir)