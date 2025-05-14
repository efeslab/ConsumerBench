import statistics
import subprocess
import time
import matplotlib.pyplot as plt
import pandas as pd
import threading
import signal
import sys
import os
from datetime import datetime

class GpuMemoryMonitor:
    def __init__(self, gpu_id=0, interval=1, results_dir="./results"):
        self.gpu_id = gpu_id
        self.interval = interval
        self.results_dir = results_dir
        self.timestamps = []
        self.memory_used = []
        self.memory_used_cpu = []
        self.memory_total = []
        self.memory_total_cpu = []
        self.utilization_pct = []
        self.utilization_pct_cpu = []
        self.gpu_compute_throughput = []
        self.cpu_compute_throughput = []
        self.cpu_memory_bw = []
        self.gpu_memory_bw = []
        self.gpu_utilization = []
        self.running = False
        self.start_time = None
        
        # Create output directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate unique run ID based on timestamp
        # self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def flush_cache(self):
        """Flush the system cache to ensure accurate memory readings"""
        try:
            subprocess.check_call("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True)
            print("Cache flushed")
        except Exception as e:
            print(f"Error flushing cache: {e}")

    def get_cpu_compute_throughput(self):
        """Get CPU utilization using mpstat"""
        try:
            output = subprocess.check_output(
                "mpstat 1 1 | grep -A 5 '%usr' | tail -n 1 | awk '{print 100 - $12}'",
                shell=True,
                encoding='utf-8'
            )
            cpu_utilization = float(output.strip())
            return cpu_utilization
        except Exception as e:
            print(f"Error getting CPU utilization: {e}")
            return 0

    def get_cpu_memory_throughput(self):
        """Get CPU utilization using mpstat"""
        try:
            output = subprocess.check_output(
                "sudo pcm-memory -i=1 -s  | awk '{print $5}' | tail -2 | head -1",
                shell=True,
                encoding='utf-8'
            )
            cpu_utilization = float(output.strip())
            return cpu_utilization
        except Exception as e:
            print(f"Error getting CPU utilization: {e}")
            return 0

    def get_gpu_compute_memory_throughput(self):
        """Get GPU utilization using dcgmi"""
        try:
            output = subprocess.check_output(
                "dcgmi dmon -e 1002,1003,1005 -c 1 | tail -1 | awk '{print $3,$4,$5}'",
                shell=True,
                encoding='utf-8'
            )
            sm_active,sm_occ,memory_bw = map(float, output.strip().split(' '))
            compute_throughput = sm_active * sm_occ * 100
            memory_throughput = memory_bw * 100
            return compute_throughput, memory_throughput

        except Exception as e:
            print(f"Error getting GPU utilization: {e}")
            return 0

    def get_gpu_memory_usage(self):
        """Get GPU memory usage using nvidia-smi"""
        try:
            output = subprocess.check_output(
                [
                    'nvidia-smi',
                    f'--id={self.gpu_id}',
                    '--query-gpu=memory.used,memory.total',
                    '--format=csv,nounits,noheader'
                ], 
                encoding='utf-8'
            )
            memory_used, memory_total = map(int, output.strip().split(','))
            return memory_used, memory_total
        except Exception as e:
            print(f"Error getting GPU memory: {e}")
            return 0, 0
        
    def get_cpu_memory_usage(self):
        """Get CPU memory usage using free"""
        try:
            output = subprocess.check_output(
                "free -m | awk '/^Mem:/ {print $2,$4}'",
                shell=True,
                encoding='utf-8'
            )
            memory_total, memory_free = map(int, output.strip().split(' '))
            memory_used = memory_total - memory_free
            return memory_used, memory_total
        except Exception as e:
            print(f"Error getting CPU memory: {e}")
            return 0, 0
    
    def save_results_cpu(self):
        """Save monitoring results to CSV and generate plot"""
        if not self.timestamps:
            print("No data collected.")
            return
        
        # Create DataFrame with results
        results = pd.DataFrame({
            'timestamp': self.timestamps,
            'memory_used_mb_cpu': self.memory_used_cpu,
            'memory_total_mb_cpu': self.memory_total_cpu,
            'utilization_pct_cpu': self.utilization_pct_cpu
        })
        
        # Save to CSV
        csv_filename = os.path.join(self.results_dir, f"cpu_memory_util.csv")
        results.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.memory_used_cpu, 'b-', label='Memory Used (MB)')
        plt.plot(self.timestamps, self.memory_total_cpu, 'r--', label='Total Memory (MB)')
        plt.ylim(0, max(self.memory_total_cpu))
        
        plt.title(f'CPU Memory Usage')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        plt.legend()
        
        plot_filename = os.path.join(self.results_dir, f"cpu_memory_util.png")
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        return results

    def save_results_gpu(self):
        """Save monitoring results to CSV and generate plot"""
        if not self.timestamps:
            print("No data collected.")
            return
        
        # Create DataFrame with results
        results = pd.DataFrame({
            'timestamp': self.timestamps,
            'memory_used_mb': self.memory_used,
            'memory_total_mb': self.memory_total,
            'utilization_pct': self.utilization_pct
        })
        
        # Save to CSV
        csv_filename = os.path.join(self.results_dir, f"gpu_memory_util.csv")
        results.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.memory_used, 'b-', label='Memory Used (MB)')
        plt.plot(self.timestamps, self.memory_total, 'r--', label='Total Memory (MB)')
        plt.ylim(0, max(self.memory_total))
        
        plt.title(f'GPU {self.gpu_id} Memory Usage')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        plt.legend()
        
        plot_filename = os.path.join(self.results_dir, f"gpu_memory_util.png")
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        return results

    def save_results_cpu_compute_memory_throughput(self):
        """Save monitoring results to CSV and generate plot"""
        if not self.timestamps:
            print("No data collected.")
            return
        
        # Create DataFrame with results
        results = pd.DataFrame({
            'timestamp': self.timestamps,
            'cpu_compute_throughput': self.cpu_compute_throughput,
            'cpu_memory_bw': self.cpu_memory_bw
        })
        
        # Save to CSV
        csv_filename = os.path.join(self.results_dir, f"cpu_compute_memory_throughput.csv")
        results.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
        
        # Create a figure with two subplots with a nicer background color
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, facecolor='#f8f9fa')
        ax1.set_facecolor('#f0f1f5')
        ax2.set_facecolor('#f0f1f5')

        # Plot SMs Active with a nice blue color and transparency
        ax1.plot(self.timestamps, self.cpu_compute_throughput, 
                color='#3498db', alpha=0.8, linewidth=2.5)
        ax1.set_title('CPUs Active Over Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('CPUs Active [Throughput %]', fontsize=14)
        ax1.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')

        # Add statistics with nicer colors
        sm_mean = statistics.mean(self.cpu_compute_throughput)
        sm_max = max(self.cpu_compute_throughput)
        ax1.axhline(y=sm_mean, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=2, 
                    label=f'Mean: {sm_mean:.2f}%')
        ax1.axhline(y=sm_max, color='#27ae60', linestyle='--', alpha=0.6, linewidth=2, 
                    label=f'Max: {sm_max:.2f}%')
        ax1.legend(fontsize=12, facecolor='white', framealpha=0.9, edgecolor='#d4d4d4')
        ax1.set_ylim(0, 100)  # Set y-axis limits to 0-100%

        # Plot DRAM Bandwidth with nicer colors
        # Calculate and plot total bandwidth

        # Plot the bandwidth lines with attractive colors and transparency
        ax2.plot(self.timestamps, self.cpu_memory_bw, 
                color='#8e44ad', alpha=0.7, linewidth=2.5, 
                label='Total')

        ax2.set_title('CPU DRAM Bandwidth Over Time', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=14)
        ax2.set_ylabel('DRAM Bandwidth (MB/s)', fontsize=14)
        ax2.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')
        # ax2.set_ylim(0, 100)  # Set y-axis limits to 0-100%

        # Add statistics with a nicer text box
        total_mean = statistics.mean(self.cpu_memory_bw)

        stats_text = (f"Mean Bandwidth:\n"
                    f"Total: {total_mean:.2f} MB/s")

        ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8,
                        edgecolor='#d4d4d4'))

        ax2.legend(fontsize=12, loc='upper right', facecolor='white', 
                framealpha=0.9, edgecolor='#d4d4d4')
        
        # Adjust layout and save the plot
        plt.tight_layout()
        # run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.results_dir, f"cpu_compute_memory_throughput.png")
        # output_file = Path(sqlite_file).with_stem(f"{Path(sqlite_file).stem}_gpu_utilization").with_suffix('.png')
        plt.savefig(plot_filename, dpi=300)
        # plt.show()
        
        print(f"Plot saved to {plot_filename}")
        return results

    def save_results_gpu_compute_memory_throughput(self):
        """Save monitoring results to CSV and generate plot"""
        if not self.timestamps:
            print("No data collected.")
            return
        
        # Create DataFrame with results
        results = pd.DataFrame({
            'timestamp': self.timestamps,
            'gpu_compute_throughput': self.gpu_compute_throughput,
            'gpu_memory_bw': self.gpu_memory_bw
        })
        
        # Save to CSV
        csv_filename = os.path.join(self.results_dir, f"gpu_compute_memory_throughput.csv")
        results.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
        
        # Create a figure with two subplots with a nicer background color
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, facecolor='#f8f9fa')
        ax1.set_facecolor('#f0f1f5')
        ax2.set_facecolor('#f0f1f5')

        # Plot SMs Active with a nice blue color and transparency
        ax1.plot(self.timestamps, self.gpu_compute_throughput, 
                color='#3498db', alpha=0.8, linewidth=2.5)
        ax1.set_title('GPU SMs Active Over Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('SMs Active [Throughput %]', fontsize=14)
        ax1.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')

        # Add statistics with nicer colors
        sm_mean = statistics.mean(self.gpu_compute_throughput)
        sm_max = max(self.gpu_compute_throughput)
        ax1.axhline(y=sm_mean, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=2, 
                    label=f'Mean: {sm_mean:.2f}%')
        ax1.axhline(y=sm_max, color='#27ae60', linestyle='--', alpha=0.6, linewidth=2, 
                    label=f'Max: {sm_max:.2f}%')
        ax1.legend(fontsize=12, facecolor='white', framealpha=0.9, edgecolor='#d4d4d4')
        ax1.set_ylim(0, 100)  # Set y-axis limits to 0-100%

        # Plot DRAM Bandwidth with nicer colors
        # Calculate and plot total bandwidth

        # Plot the bandwidth lines with attractive colors and transparency
        ax2.plot(self.timestamps, self.gpu_memory_bw, 
                color='#8e44ad', alpha=0.7, linewidth=2.5, 
                label='Total')

        ax2.set_title('GPU DRAM Bandwidth Over Time', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=14)
        ax2.set_ylabel('DRAM Bandwidth [Throughput %]', fontsize=14)
        ax2.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')
        ax2.set_ylim(0, 100)  # Set y-axis limits to 0-100%

        # Add statistics with a nicer text box
        total_mean = statistics.mean(self.gpu_memory_bw)

        stats_text = (f"Mean Bandwidth:\n"
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
        plot_filename = os.path.join(self.results_dir, f"gpu_compute_memory_throughput.png")
        # output_file = Path(sqlite_file).with_stem(f"{Path(sqlite_file).stem}_gpu_utilization").with_suffix('.png')
        plt.savefig(plot_filename, dpi=300)
        # plt.show()
        
        print(f"Plot saved to {plot_filename}")
        return results

    def save_results(self):
        save_results_cpu = self.save_results_cpu()
        save_results_gpu = self.save_results_gpu()
        # save_results_gpu_util = self.save_results_gpu_compute_memory_throughput()
        # save_results_cpu_util = self.save_results_cpu_compute_memory_throughput()
        return save_results_cpu, save_results_gpu
        # return save_results_cpu, save_results_gpu, save_results_gpu_util, save_results_cpu_util
    
    def _monitoring_loop(self, duration=None):
        """Internal monitoring loop"""
        self.running = True
        self.start_time = time.time()

        self.flush_cache()
        
        try:
            while self.running:
                # Check if duration has elapsed
                if duration is not None and time.time() - self.start_time >= duration:
                    print(f"Reached monitoring duration of {duration} seconds")
                    self.running = False
                    break
                
                # Get current timestamp relative to start
                current_time = time.time() - self.start_time
                
                # Get memory usage
                used, total = self.get_gpu_memory_usage()
                utilization = (used / total * 100) if total > 0 else 0
                
                used_cpu, total_cpu = self.get_cpu_memory_usage()
                utilization_cpu = (used_cpu / total_cpu * 100) if total_cpu > 0 else 0

                # Get GPU utilization
                # gpu_compute_throughput, gpu_memory_bw = self.get_gpu_compute_memory_throughput()

                # Get CPU utilization
                # cpu_compute_throughput = self.get_cpu_compute_throughput()
                # cpu_memory_bw = self.get_cpu_memory_throughput()
    
                # Store data
                self.timestamps.append(current_time)
                self.memory_used.append(used)
                self.memory_total.append(total)
                self.utilization_pct.append(utilization)
                self.memory_used_cpu.append(used_cpu)
                self.memory_total_cpu.append(total_cpu)
                self.utilization_pct_cpu.append(utilization_cpu)
                # self.gpu_compute_throughput.append(gpu_compute_throughput)
                # self.gpu_memory_bw.append(gpu_memory_bw)
                # self.cpu_compute_throughput.append(cpu_compute_throughput)
                # self.cpu_memory_bw.append(cpu_memory_bw)

                # Print current status
                # print(f"[GPU] Time: {current_time:.1f}s | Memory: {used}/{total} MB ({utilization:.1f}%)", end="\r")
                # print(f"[CPU] Time: {current_time:.1f}s | Memory: {used_cpu}/{total_cpu} MB ({utilization_cpu:.1f}%)", end="\r")
                
                # Wait for next sample
                time.sleep(self.interval)
                
        except Exception as e:
            print(f"\nError during monitoring: {e}")
        
        finally:
            # Always save results when stopping
            print("\nFinishing monitoring, saving results...")
            return self.save_results()
    
    def start_monitoring_thread(self, duration=None):
        """
        Start monitoring in a separate thread
        
        Args:
            duration: Optional duration in seconds. If None, runs until stop_monitoring() is called.
        
        Returns:
            The thread object
        """
        print(f"Starting GPU {self.gpu_id} memory monitoring in a thread (sampling every {self.interval} seconds)...")
        print("Call stop_monitoring() to stop and save results")
        
        # Create and start thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(duration,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        return self.monitor_thread
    
    def stop_monitoring(self):
        """Stop the monitoring thread and save results"""
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.running = False
            self.monitor_thread.join()
            print("Monitoring stopped")
        else:
            print("No active monitoring thread to stop")
    
    def start_monitoring(self, duration=None):
        """
        Start monitoring in the current thread (use in main thread only)
        
        Args:
            duration: Optional duration in seconds. If None, runs until interrupted.
        """
        # Only set up signal handler if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            def signal_handler(sig, frame):
                print("\nStopping GPU memory monitoring...")
                self.running = False
            
            # Register signal handler for clean shutdown
            signal.signal(signal.SIGINT, signal_handler)
            
            print(f"Starting GPU {self.gpu_id} memory monitoring (sampling every {self.interval} seconds)...")
            print("Press Ctrl+C to stop monitoring and save results")
        else:
            print("Warning: Running in a non-main thread. Signal handling (Ctrl+C) won't work.")
            print(f"Starting GPU {self.gpu_id} memory monitoring (sampling every {self.interval} seconds)...")
        
        # Use the same monitoring loop
        return self._monitoring_loop(duration)


# Example usage
if __name__ == "__main__":
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description='Monitor GPU memory usage with nvidia-smi')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to monitor')
    parser.add_argument('--interval', type=float, default=0.5, help='Sampling interval in seconds')
    parser.add_argument('--duration', type=int, default=None, help='Monitoring duration in seconds (optional)')
    parser.add_argument('--output', type=str, default='./results', help='Output directory for logs and plots')
    args = parser.parse_args()
    
    # Create and start monitor
    monitor = GpuMemoryMonitor(
        gpu_id=args.gpu, 
        interval=args.interval,
        results_dir=args.output
    )
    
    # Start monitoring in the main thread
    monitor.start_monitoring(duration=args.duration)