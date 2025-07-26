#!/usr/bin/env python3
"""
Power Consumption Visualization Script

This script creates beautiful visualizations from power consumption data collected
by the power monitoring script.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
from datetime import datetime
import os

def load_data(file_path):
    """Load and prepare the power consumption data."""
    # Load CSV data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert elapsed_time to float
    df['elapsed_time'] = df['elapsed_time'].astype(float)
    
    # Convert power columns to float
    for col in ['cpu_power', 'gpu_power', 'total_power']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    return df

def plot_power_over_time(df, output_file=None, dark_mode=False, smooth=0):
    """Create a beautiful plot of power consumption over time."""
    # Set the style
    if dark_mode:
        plt.style.use('dark_background')
        color_palette = ['#ff7f0e', '#2ca02c', '#d62728']  # Orange, Green, Red
        grid_color = '#555555'
        text_color = 'white'
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")
        color_palette = ['#ff7f0e', '#2ca02c', '#d62728']  # Orange, Green, Red  
        grid_color = '#cccccc'
        text_color = 'black'
    
    # Create figure and axes with appropriate size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Apply smoothing if requested
    if smooth > 0:
        df['cpu_power_smooth'] = df['cpu_power'].rolling(window=smooth, center=True).mean()
        df['gpu_power_smooth'] = df['gpu_power'].rolling(window=smooth, center=True).mean()
        df['total_power_smooth'] = df['total_power'].rolling(window=smooth, center=True).mean()
        
        # Plot smoothed data
        ax.plot(df['elapsed_time'], df['cpu_power_smooth'], linewidth=2.5, 
                label='CPU Power (W)', color=color_palette[0])
        ax.plot(df['elapsed_time'], df['gpu_power_smooth'], linewidth=2.5, 
                label='GPU Power (W)', color=color_palette[1])
        ax.plot(df['elapsed_time'], df['total_power_smooth'], linewidth=3, 
                label='Total Power (W)', color=color_palette[2])
        
        # Plot original data with lower alpha for reference
        ax.plot(df['elapsed_time'], df['cpu_power'], linewidth=0.8, alpha=0.2, color=color_palette[0])
        ax.plot(df['elapsed_time'], df['gpu_power'], linewidth=0.8, alpha=0.2, color=color_palette[1])
        ax.plot(df['elapsed_time'], df['total_power'], linewidth=0.8, alpha=0.2, color=color_palette[2])
    else:
        # Plot raw data
        ax.plot(df['elapsed_time'], df['cpu_power'], linewidth=2.5, 
                label='CPU Power (W)', color=color_palette[0])
        ax.plot(df['elapsed_time'], df['gpu_power'], linewidth=2.5, 
                label='GPU Power (W)', color=color_palette[1])
        ax.plot(df['elapsed_time'], df['total_power'], linewidth=3, 
                label='Total Power (W)', color=color_palette[2])
    
    # Calculate overall statistics
    cpu_avg = df['cpu_power'].mean()
    gpu_avg = df['gpu_power'].mean()
    total_avg = df['total_power'].mean()
    
    cpu_max = df['cpu_power'].max()
    gpu_max = df['gpu_power'].max()
    total_max = df['total_power'].max()
    
    # Add horizontal lines for averages
    ax.axhline(y=cpu_avg, color=color_palette[0], linestyle='--', alpha=0.5)
    ax.axhline(y=gpu_avg, color=color_palette[1], linestyle='--', alpha=0.5)
    ax.axhline(y=total_avg, color=color_palette[2], linestyle='--', alpha=0.5)
    
    # Add text annotations for averages
    ax.text(df['elapsed_time'].max() * 0.02, cpu_avg * 1.05, 
            f'CPU Avg: {cpu_avg:.2f}W', color=color_palette[0], fontweight='bold')
    ax.text(df['elapsed_time'].max() * 0.02, gpu_avg * 1.05, 
            f'GPU Avg: {gpu_avg:.2f}W', color=color_palette[1], fontweight='bold')
    ax.text(df['elapsed_time'].max() * 0.02, total_avg * 1.05, 
            f'Total Avg: {total_avg:.2f}W', color=color_palette[2], fontweight='bold')
    
    # Calculate peak areas
    peak_threshold = total_avg * 1.5
    peak_regions = df[df['total_power'] > peak_threshold]
    
    if not peak_regions.empty:
        # Highlight peak areas
        for idx, region in enumerate(np.split(peak_regions, np.where(np.diff(peak_regions.index) > 1)[0] + 1)):
            if not region.empty:
                ax.axvspan(region['elapsed_time'].iloc[0], region['elapsed_time'].iloc[-1], 
                           alpha=0.2, color='red', zorder=0)
    
    # Add titles and labels
    plt.title('Power Consumption Over Time', fontsize=24, pad=20, color=text_color, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=18, color=text_color, labelpad=10)
    plt.ylabel('Power (Watts)', fontsize=18, color=text_color, labelpad=10)
    
    # Add summary statistics
    stats_text = (f"Max CPU: {cpu_max:.2f}W | Max GPU: {gpu_max:.2f}W | Max Total: {total_max:.2f}W\n"
                 f"Avg CPU: {cpu_avg:.2f}W | Avg GPU: {gpu_avg:.2f}W | Avg Total: {total_avg:.2f}W")
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12, color=text_color)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
    
    # Customize ticks
    ax.tick_params(axis='both', colors=text_color, labelsize=12)
    
    # Format axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not dark_mode:
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
    
    # Add legend with custom styling
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.8, fontsize=12)
    legend.get_frame().set_edgecolor(grid_color)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Add timestamp
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.01, 0.01, f"Generated: {timestamp_str}", fontsize=8, color=text_color, alpha=0.7)
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig

def plot_power_distribution(df, output_file=None, dark_mode=False):
    """Create distribution plots of power consumption."""
    # Set the style
    if dark_mode:
        plt.style.use('dark_background')
        color_palette = ['#ff7f0e', '#2ca02c', '#d62728']  # Orange, Green, Red
        text_color = 'white'
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")
        color_palette = ['#ff7f0e', '#2ca02c', '#d62728']  # Orange, Green, Red
        text_color = 'black'
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Plot distributions
    sns.histplot(df['cpu_power'], kde=True, color=color_palette[0], ax=axes[0])
    sns.histplot(df['gpu_power'], kde=True, color=color_palette[1], ax=axes[1])
    sns.histplot(df['total_power'], kde=True, color=color_palette[2], ax=axes[2])
    
    # Set titles and labels
    axes[0].set_title('CPU Power Distribution', fontsize=16, color=text_color)
    axes[1].set_title('GPU Power Distribution', fontsize=16, color=text_color)
    axes[2].set_title('Total Power Distribution', fontsize=16, color=text_color)
    
    for ax in axes:
        ax.set_xlabel('Power (Watts)', color=text_color)
        ax.set_ylabel('Frequency', color=text_color)
        ax.tick_params(axis='both', colors=text_color)
    
    # Add overall title
    plt.suptitle('Power Consumption Distributions', fontsize=20, color=text_color, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {output_file}")
    
    return fig

def main():
    """Main function to parse arguments and generate visualizations."""
    parser = argparse.ArgumentParser(description='Visualize power consumption data.')
    parser.add_argument('-i', '--input', type=str, default='power_data.csv',
                        help='Input CSV file path (default: power_data.csv)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output image file path (default: power_plot_{timestamp}.png)')
    parser.add_argument('--dark', action='store_true',
                        help='Use dark mode for plots')
    parser.add_argument('--dist', action='store_true',
                        help='Create distribution plots in addition to time series')
    parser.add_argument('--smooth', type=int, default=0,
                        help='Smoothing window size (default: 0, no smoothing)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Generate default output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'power_plot_{timestamp}.png'
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Load data
    df = load_data(args.input)
    
    # Create time series plot
    fig1 = plot_power_over_time(df, args.output, args.dark, args.smooth)
    
    # Create distribution plots if requested
    if args.dist:
        dist_output = os.path.splitext(args.output)[0] + '_dist' + os.path.splitext(args.output)[1]
        fig2 = plot_power_distribution(df, dist_output, args.dark)
    
    # Show plots if requested
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()