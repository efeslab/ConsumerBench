import sys
import os
import argparse

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.SleepApplication.SleepApplication import SleepApplication
from applications.ImageGen.ImageGen import ImageGen
from applications.DeepResearch.DeepResearch import DeepResearch
from applications.Chatbot.Chatbot import Chatbot
from applications.LiveCaptions.LiveCaptions import LiveCaptions
from applications.MCPServer.MCPServer import MCPServer
from src.workflow import Workflow
import src.globals as globals

from monitors.memory_util import GpuMemoryMonitor

def main(args):
    """User workflow with ConsumerBench"""
    parser = argparse.ArgumentParser(description='User workflow with ConsumerBench')
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    parser.add_argument('--mcp_trace', type=str, help='Path to MCP trace JSONL file', default=None)
    parser.add_argument('--results', type=str, help='Path to save results', default=f"{repo_dir}/results")
    args = parser.parse_args()    
    config_file = args.config
    mcp_trace_file = args.mcp_trace
    print(f"=== Testing User Workflow with ConsumerBench ===\n")
    print(f"Using config file: {config_file}")

    # Initialize globals
    globals.set_start_time()
    globals.set_results_dir(f"{args.results}")
        
    # Create application instances
    sleepApplication1 = SleepApplication()
    imageGen = ImageGen()
    deepResearch = DeepResearch()
    chatbot = Chatbot()
    liveCaptions = LiveCaptions()
    mcpServer = MCPServer(mcp_trace_file=mcp_trace_file, config_file=config_file)
    
    # Create workflow from YAML
    workflow = Workflow(config_file)
    
    # Register applications
    workflow.register_application("SleepApplication", sleepApplication1)
    workflow.register_application("ImageGen", imageGen)
    workflow.register_application("DeepResearch", deepResearch)
    workflow.register_application("Chatbot", chatbot)
    workflow.register_application("LiveCaptions", liveCaptions)
    workflow.register_application("MCPServer", mcpServer)

    print("Registered applications:")
    for app_name, app in workflow.applications.items():
        print(f"  - {app_name}: {type(app).__name__}")
    print()
    
    # Load workflow configuration
    workflow.load_workflow_unit_config()
    print("Loaded workflow configuration:")
    for unit_name, unit_config in workflow.workflow_unit_map.items():
        print(f"  - {unit_name}: {unit_config['type']} (count: {unit_config['count']})")
        print(f"    Config: {unit_config['node_config']}")
    print()
    
    # Generate task queue
    workflow.generate_task_queue()
    print("Generated task queue:")
    for k, v in workflow.tasks_map_queue.items():
        print(f"Task group {k}:")
        for unit in v:
            print(f"  - {unit.type} (ID: {unit.id})")
            print(f"    Start node: {unit.node_start}")
            print(f"    End node: {unit.node_end}")
    print()
    
    # Generate benchmark
    bm = workflow.generate_benchmark()
    print("Benchmark generated successfully.")
    
    # Visualize the benchmark
    bm.visualize("complex_workflow_benchmark.png")
    print("Benchmark visualization saved to 'complex_workflow_benchmark.png'")
    
    #  Set up GPU memory monitoring
    gpu_monitor = GpuMemoryMonitor(gpu_id=0, interval=0.01, results_dir=args.results)
    import threading
    monitor_thread = threading.Thread(target=gpu_monitor.start_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Run the benchmark
    print("\n=== Running Benchmark ===")
    total_time = bm.run_concurrent()
    print(f"Total execution time: {total_time:.4f} seconds")
    
    # Display results
    print("\n=== Results ===")
    bm.display_results()

    # Stop GPU memory monitoring
    gpu_monitor.running = False
    monitor_thread.join()

if __name__ == "__main__":
    main(sys.argv[1:]) 