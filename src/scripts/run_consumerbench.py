import sys
import os
import argparse
import subprocess

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.SleepApplication.SleepApplication import SleepApplication
from applications.ImageGen.ImageGen import ImageGen
from applications.ImageGenServer.ImageGenServer import ImageGenServer
from applications.DeepResearch.DeepResearch import DeepResearch
from applications.Chatbot.Chatbot import Chatbot
from applications.ChatbotHF.ChatbotHF import ChatbotHF
from applications.DspyTool.DspyTool import DspyTool
from applications.LiveCaptions.LiveCaptions import LiveCaptions
from applications.LiveCaptionsHF.LiveCaptionsHF import LiveCaptionsHF
try:
    from applications.MCPServer.MCPServer import MCPServer
except (ImportError, Exception):
    MCPServer = None

try:
    from applications.Retriever.Retriever import Retriever
except (ImportError, Exception):
    Retriever = None

try:
    from applications.RetrieverServer.RetrieverServer import RetrieverServer
except (ImportError, Exception):
    RetrieverServer = None
from applications.DbBench.DbBench import DbBench
from applications.OSTool.OSTool import OsTool
from src.workflow import Workflow
import src.globals as globals
from src.tally import ensure_tally_runtime, release_tally_runtime, detect_tally_root

from monitors.memory_util import GpuMemoryMonitor


def get_scheduler(config_file):
    workflow = Workflow(config_file)
    return workflow.scheduler.lower() if workflow.scheduler else None


def ensure_tally_client(config_file):
    if os.environ.get("TALLY_CLIENT_WRAPPED") == "1":
        return False

    scheduler = get_scheduler(config_file)
    if scheduler not in {"tally", "tgs", "naive"}:
        return False

    print(f"Setting up tally scheduler={scheduler.upper()} before benchmark start")
    ensure_tally_runtime(scheduler, repo_dir)

    tally_root = detect_tally_root(repo_dir)
    start_client_script = os.path.join(tally_root, "scripts", "start_client.sh")
    if not os.path.exists(start_client_script):
        raise FileNotFoundError(f"Missing start_client.sh at {start_client_script}")

    env = os.environ.copy()
    env["TALLY_CLIENT_WRAPPED"] = "1"
    env["TALLY_RUNTIME_UP"] = "1"

    relaunch_cmd = [
        "bash",
        start_client_script,
        sys.executable,
        os.path.abspath(__file__),
    ] + sys.argv[1:]
    print("Relaunching ConsumerBench under Tally client context")
    try:
        rc = subprocess.run(relaunch_cmd, env=env).returncode
    finally:
        release_tally_runtime(scheduler, repo_dir)

    if rc != 0:
        raise RuntimeError(f"Benchmark run under Tally client failed with exit code {rc}")
    return True

def main(args):
    """User workflow with ConsumerBench"""
    parser = argparse.ArgumentParser(description='User workflow with ConsumerBench')
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    parser.add_argument('--mcp_trace', type=str, help='Path to MCP trace JSONL file', default=None)
    parser.add_argument('--results', type=str, help='Path to save results', default=f"{repo_dir}/results")
    args = parser.parse_args()    
    config_file = args.config
    mcp_trace_file = args.mcp_trace

    # Initialize global timing/results before any tally setup so tally logs are
    # written into the configured benchmark results directory.
    globals.set_start_time()
    globals.set_results_dir(f"{args.results}")

    if ensure_tally_client(config_file):
        return

    print(f"=== Testing User Workflow with ConsumerBench ===\n")
    print(f"Using config file: {config_file}")

    # Create application instances
    
    # TODO: Yile, fix application type and instance tied up
    sleepApplication1 = SleepApplication()
    imageGen = ImageGen()
    imageGenServer = ImageGenServer()
    deepResearch = DeepResearch()
    chatbot = Chatbot()
    chatbotHF = ChatbotHF()
    liveCaptions = LiveCaptions()
    liveCaptionsHF = LiveCaptionsHF()
    mcpServer = MCPServer(mcp_trace_file=mcp_trace_file, config_file=config_file) if MCPServer is not None else None
    retriever = Retriever() if Retriever is not None else None
    retrieverServer = RetrieverServer() if RetrieverServer is not None else None
    dspyTool = DspyTool()
    dbBench = DbBench()
    osTool = OsTool()

    # Create workflow from YAML
    workflow = Workflow(config_file)
    
    # Register applications
    workflow.register_application("SleepApplication", sleepApplication1)
    workflow.register_application("ImageGen", imageGen)
    workflow.register_application("ImageGenServer", imageGenServer)
    workflow.register_application("DeepResearch", deepResearch)
    workflow.register_application("Chatbot", chatbot)
    workflow.register_application("ChatbotHF", chatbotHF)
    workflow.register_application("LiveCaptions", liveCaptions)
    workflow.register_application("LiveCaptionsHF", liveCaptionsHF)
    if mcpServer is not None:
        workflow.register_application("MCPServer", mcpServer)
    if retriever is not None:
        workflow.register_application("Retriever", retriever)
    if retrieverServer is not None:
        workflow.register_application("RetrieverServer", retrieverServer)
    workflow.register_application("DspyTool", dspyTool)
    workflow.register_application("DbBench", dbBench)
    workflow.register_application("OsTool", osTool)

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

    valid_tally = workflow.scheduler is not None and workflow.scheduler.lower() in {"tally", "tgs", "naive"}
    tally_sched_name = workflow.scheduler.lower() if workflow.scheduler else None
    if valid_tally and os.environ.get("TALLY_RUNTIME_UP") != "1":
        print(f"Setting up tally scheduler={tally_sched_name.upper()} before benchmark start")
        ensure_tally_runtime(tally_sched_name, repo_dir)

    # Run the benchmark
    try:
        print("\n=== Running Benchmark ===")
        total_time = bm.run_concurrent()
        print(f"Total execution time: {total_time:.4f} seconds")

        # Display results
        print("\n=== Results ===")
        bm.display_results()
    finally:
        if valid_tally and os.environ.get("TALLY_RUNTIME_UP") != "1":
            release_tally_runtime(tally_sched_name, repo_dir)

        # Stop GPU memory monitoring
        gpu_monitor.running = False
        monitor_thread.join()

if __name__ == "__main__":
    main(sys.argv[1:]) 