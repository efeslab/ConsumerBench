import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import threading
import time
import subprocess
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Callable, List, Tuple, Any, Set, Optional
from datasets import load_dataset
import requests
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from diffusers import StableDiffusion3Pipeline
import sys
from datetime import datetime
import nvtx


# [ROHAN: We should remove these paths. All application stuff should be imported from applications/, datasets/, inference-backed/ respectively. ]
sys.path.append("/home/cc/os-llm/scripts")
sys.path.append("/home/cc/os-llm")
from memory_util import GpuMemoryMonitor
import globals

model_refcount = {}

# declare global variables
global_vars = {}
model_refcount_lock = threading.Lock()


class ExecutionNode:
    """Represents a specific execution node that contains a function pointer and arguments"""
    def __init__(self, 
                 node_id: str,
                 func: Callable,
                 func_args: Dict = None):
        """
        Initialize an execution node.
        
        Args:
            node_id: Unique identifier for this node
            func: Function pointer to be executed
            func_args: Arguments to pass to the function
        """
        self.node_id = node_id
        self.func = func
        self.func_args = func_args or {}
        
        # Timing metrics for this specific execution
        self.execution_time = 0
        self.result = None
        self.success = False
    
    def execute(self) -> Tuple[float, Any, bool]:
        """
        Execute the function with the provided arguments.
        
        Returns:
            Tuple of (execution_time, result, success)
        """
        start_time = time.time()
        try:
            self.result = self.func(**self.func_args)
            self.success = True if self.result is not None else False
        except Exception as e:
            print(f"Error executing node {self.node_id}: {e}")
            self.success = False
            self.result = None
        
        self.execution_time = time.time() - start_time
        return self.execution_time, self.result, self.success


class Task:
    """Represents a workflow task composed of execution nodes arranged in a DAG"""
    def __init__(self, task_id: str, task_type: str = "ephemeral", app_type: str = "chatbot"):
        """
        Initialize a task with a DAG of execution nodes.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Either "ephemeral" or "background"
                       - ephemeral: task is executed each time
                       - background: task persists across multiple executions
        """
        self.task_id = task_id
        self.dag = nx.DiGraph()
        self.node_map = {}  # Maps task-specific node ID -> ExecutionNode
        self.task_type = task_type
        self.is_set_up = False
        self.app_type = app_type
        self.server_pid = -1
        self.start_time = None
        self.end_time = None
        self.refs = 0
        self.task_lock = threading.Lock()
        self.total_time = 0
        self.results = []
    
    def add_node(self, node_id, func: Callable, func_args: Dict = None) -> str:
        """
        Add a node to the task's DAG.
        
        Args:
            node_id: Task-specific identifier for this node
            func: Function pointer to be executed
            func_args: Arguments to pass to the function
            
        Returns:
            Node ID
        """
        # Create the execution node
        node = ExecutionNode(node_id=node_id, func=func, func_args=func_args)
        
        # Store the node in the map
        self.node_map[node_id] = node
        
        # Add to the DAG
        self.dag.add_node(node_id)

        # Increment the node ID counter
        
        return node
    
    def add_edge(self, from_node_id: str, to_node_id: str):
        """
        Add a dependency edge between two nodes.
        
        Args:
            from_node_id: Task-specific ID of the source node
            to_node_id: Task-specific ID of the target node
        """
        # Check if nodes exist
        if from_node_id not in self.node_map:
            raise ValueError(f"Node {from_node_id} does not exist in task {self.task_id}")
        if to_node_id not in self.node_map:
            raise ValueError(f"Node {to_node_id} does not exist in task {self.task_id}")
        
        # Add the edge to the DAG
        self.dag.add_edge(from_node_id, to_node_id)
    
    def get_node(self, node_id: str) -> ExecutionNode:
        """
        Get a node by its ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            ExecutionNode object
        """
        if node_id not in self.node_map:
            raise ValueError(f"Node {node_id} does not exist in task {self.task_id}")
        
        return self.node_map[node_id]
    
    def get_node_map(self) -> Dict[str, ExecutionNode]:
        """Get the task's node map"""
        return self.node_map
    
    def get_dag(self):
        """Get the task's DAG"""
        return self.dag

    def validate(self):
        """Validate that the DAG is properly formed"""
        # Check if the DAG is acyclic
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError(f"Task {self.task_id} graph is not a Directed Acyclic Graph (DAG)")
        
        # Check if any nodes are isolated (have no connections)
        isolated_nodes = [n for n in self.dag.nodes if self.dag.degree(n) == 0]
        if isolated_nodes:
            print(f"Warning: Task {self.task_id} has isolated nodes: {isolated_nodes}")
    
    def reset_nodes(self):
        """Reset timing metrics for all nodes"""
        for node in self.node_map.values():
            node.execution_time = 0
            node.result = None
            node.success = False
            
    def update_total_time(self):
        """Update the total execution time for the task"""
        
        # TODO: may need to do CPM to get the total time if the graph is not a chain
        self.total_time = sum([node.execution_time for node in self.node_map.values()])
    
    def display_results(self):
        """Display task execution results"""
        print(f"\nTask {self.task_id} execution time: {self.total_time:.4f} seconds")
        print("\nExecution times for each node:")
        
        # Get nodes in topological order if possible
        try:
            ordered_nodes = list(nx.topological_sort(self.dag))
        except:
            ordered_nodes = list(self.node_map.keys())
            
        for node_id in ordered_nodes:
            node = self.node_map[node_id]
            print(f"Node {node.node_id}:")
            print(f"  Execution time: {node.execution_time:.4f} seconds")
            print(f"  Success: {node.success}")
            
        print(f"\nTask {self.task_id} results:")
        print(self.results)
    
    def write_results(self):
        """Write task execution results to a file"""
        results_dir = globals.get_results_dir()
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{results_dir}/task_{self.task_id}_perf.log"
        print(f"Writing results of task_{self.task_id} to {filename}")
        self.record_end_time()
        with open(filename, 'w') as f:
            f.write(f"app_type: {self.app_type}\n")
            f.write(f"task_id: {self.task_id}\n")
            f.write(f"start_time: {self.start_time}\n")
            f.write(f"end_time: {(datetime.now() - globals.start_time).total_seconds()}\n")
            f.write(f"Task {self.task_id} execution time: {self.total_time:.4f} seconds\n")
            f.write("\nExecution times for each node:\n")
            
            # Get nodes in topological order if possible
            try:
                ordered_nodes = list(nx.topological_sort(self.dag))
            except:
                ordered_nodes = list(self.node_map.keys())
                
            for node_id in ordered_nodes:
                node = self.node_map[node_id]
                f.write(f"Node {node.node_id}:\n")
                f.write(f"  Execution time: {node.execution_time:.4f} seconds\n")
                f.write(f"  Success: {node.success}\n")
                
            f.write(f"\nTask {self.task_id} results:\n")
            f.write(str(self.results))

    def visualize(self, output_filename=None):
        """Visualize the task DAG with execution times"""
        if output_filename is None:
            output_filename = f"task_{self.task_id}_visualization.png"
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.dag, seed=42)
        
        # Draw nodes with sizes proportional to execution time
        node_sizes = [self.node_map[node_id].execution_time * 500 + 300 for node_id in self.dag.nodes]
        nx.draw_networkx_nodes(self.dag, pos, node_size=node_sizes, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(self.dag, pos, arrowsize=20, width=1.5)
        
        # Draw labels with node information
        labels = {}
        for node_id in self.dag.nodes:
            node = self.node_map[node_id]
            labels[node_id] = f"{node.node_id}\n{node.execution_time:.2f}s"
        
        nx.draw_networkx_labels(self.dag, pos, labels=labels, font_size=10)
        
        plt.title(f"Task {self.task_id} DAG with Execution Times")
        plt.axis("off")
        plt.tight_layout()
        
        plt.savefig(output_filename)
        print(f"DAG visualization saved to {output_filename}")

    def record_start_time(self):
        """Record the start time of the task"""
        if self.start_time is None:
            self.start_time = (datetime.now() - globals.start_time).total_seconds()

    def record_end_time(self):
        """Record the start time of the task"""
        if self.end_time is None:
            self.end_time = (datetime.now() - globals.start_time).total_seconds()


class DAGScheduler:
    """Manages and runs benchmarks with multiple tasks"""
    def __init__(self, dag: nx.DiGraph, tasks: List[Task]):
        """
        Initialize a benchmark runner.
        
        Args:
            dag: Operation DAG for the benchmark  
        """
        self.dag = dag
        self.tasks = tasks
        self.node_map = {}  # Maps node ID -> ExecutionNode
        self.node_id_to_task = {}  # Maps node ID -> Task
        for task_id, task in self.tasks.items():
            node_map = task.get_node_map()
            for node_id, node in node_map.items():
                assert node_id not in self.node_map, f"Node ID {node_id} already exists in the DAG"
                self.node_map[node_id] = node
                self.node_id_to_task[node_id] = task
    
    def validate(self):
        """Validate the benchmark configuration"""
        
        # Check if the task dependencies form a DAG
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("The task dependencies graph is not a Directed Acyclic Graph (DAG)")
    
    def run_sequential(self):
        """Run all tasks sequentially based on dependencies if any"""
        self.validate()
        start_time = time.time()
        
        if self.dag.nodes:
            # Run tasks in topological order if dependencies exist
            topo_order = list(nx.topological_sort(self.dag))
            print(f"Topological order: {topo_order}")
            for node_id in topo_order:
                node = self.dag.nodes[node_id]
                print(f"\n=== Executing Node {node_id} ===")
                execution_time, result, success = self.node_map[node_id].execute()
                if not success:
                    raise ValueError(f"Node {node_id} failed to execute successfully")
                print(f"Node {node_id} completed in {execution_time:.4f} seconds")
                
        
        self.total_time = time.time() - start_time
        return self.total_time
    
    def run_concurrent(self):
        """Run tasks concurrently where possible based on dependencies"""
        self.validate()
        start_time = time.time()
        pending_tasks = [task for task in self.tasks]
        print(f"Pending tasks: {pending_tasks}")
        
        if not self.dag.nodes or len(self.dag.edges) == 0:
            # If no dependencies, run all nodes concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(node.execute): node_id
                        for node_id, node in self.node_map.items()}
                
                for future in concurrent.futures.as_completed(futures):
                    node_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error executing node {node_id}: {e}")
        else:
            completed = set()
            pending_futures = {}  # Maps Future objects to node_ids
            lock = threading.Lock()
            
            def can_execute(node_id):
                return all(pred in completed for pred in self.dag.predecessors(node_id))
            
            def execute_node(node_id):
                if node_id == "NULL":
                    return 0.0, None, True
                try:
                    result = self.node_map[node_id].execute()
                    return node_id, result
                except Exception as e:
                    print(f"Error executing node {node_id}: {e}")
                    raise
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # First, submit all tasks that can be executed immediately
                with lock:
                    executable = [node_id for node_id in self.dag.nodes 
                                if node_id not in completed and can_execute(node_id)]
                    
                    for node_id in executable:
                        task = self.node_id_to_task[node_id]
                        task.record_start_time()
                        future = executor.submit(execute_node, node_id)
                        pending_futures[future] = node_id
                        print(f"Initially executing node: {node_id}")
                
                # Continue until all nodes are completed
                while len(completed) < len(self.dag.nodes):
                    # Wait for any future to complete
                    if not pending_futures:
                        # No futures pending, but not all tasks complete - there might be a deadlock or cycle
                        time.sleep(0.1)
                        continue
                    
                    # Wait for the first future to complete
                    done, _ = concurrent.futures.wait(
                        pending_futures.keys(), 
                        return_when=concurrent.futures.FIRST_COMPLETED,
                        timeout=0.1
                    )
                    
                    if not done:
                        # Timeout occurred, but no futures completed yet
                        if len(pending_tasks) == 1 and \
                            ("deep" in pending_tasks[0] or "research" in pending_tasks[0] or "dr" in pending_tasks[0]):
                            print("Deep research task is pending, executing cleanup node")
                            task = self.tasks[pending_tasks[0]]
                            # execute the last node of deep research which is cleanup
                            executor.submit(execute_node, f"{pending_tasks[0]}_{len(task.node_map)-1}")
                            task.update_total_time()
                            print(f"Task {task.task_id} completed and result written to file.")
                            task.write_results()
                            print(f"Node {completed_node_id} completed with result: {result}")
                            pending_tasks.remove(task.task_id)
                            break
                        continue
                    
                    # Process completed futures and submit new tasks
                    for future in done:
                        node_id = pending_futures.pop(future)
                        
                        try:
                            completed_node_id, result = future.result()
                            
                            with lock:
                                completed.add(completed_node_id)
                                
                                if completed_node_id != "NULL":
                                    # get the task
                                    task = self.node_id_to_task[completed_node_id]
                                    task.results.append(result[1])
                                    # check if the number of nodes in the task is equal to the number of entries in result
                                    if len(task.node_map) == len(task.results):
                                        task.update_total_time()
                                        print(f"Task {task.task_id} has been completed and result written to file.")
                                        task.write_results()
                                        pending_tasks.remove(task.task_id)
                                    
                                print(f"Node {completed_node_id} completed with result: {result}")
                                
                                # Check for new executable nodes
                                new_executable = [node_id for node_id in self.dag.nodes 
                                                if node_id not in completed and 
                                                node_id not in [pending_futures[f] for f in pending_futures] and
                                                can_execute(node_id)]
                                
                                # Submit new tasks for execution
                                for new_node_id in new_executable:
                                    task = self.node_id_to_task[new_node_id]
                                    task.record_start_time()
                                    future = executor.submit(execute_node, new_node_id)
                                    pending_futures[future] = new_node_id
                                    print(f"Submitting new node: {new_node_id}")
                                    
                        except Exception as e:
                            print(f"Node {node_id} generated an exception: {e}")
                            # Mark as completed even if it failed, to avoid hanging
                            with lock:
                                completed.add(node_id)
        
        self.total_time = time.time() - start_time
        return self.total_time

    
    def display_results(self):
        """Display benchmark results"""
        # logging.info(f"\n=== Benchmark Summary ===")
        # logging.info(f"Total execution time: {self.total_time:.4f} seconds")
        
        for task_id, task in self.tasks.items():
            task.update_total_time()
            logging.info(f"Task {task_id}: {task.start_time:.4f} - {task.end_time:.4f}")
            task.display_results()
    
    def visualize(self, output_filename="benchmark_visualization.png"):
        """Visualize all tasks execution times"""
        # First, visualize each task's DAG
        for task_id, task in self.tasks.items():
            task.visualize()
        
        # If task dependencies exist, visualize them
        if self.dag.nodes and len(self.dag.edges) > 0:
            plt.figure(figsize=(12, 8))
            pos = nx.kamada_kawai_layout(self.dag)
            
            # Draw nodes with sizes proportional to task execution time
            node_sizes = [self.node_map[node_id].execution_time * 200 + 500 for node_id in self.dag.nodes]
            nx.draw_networkx_nodes(self.dag, pos, node_size=node_sizes, node_color="lightgreen")
            
            # Draw edges
            nx.draw_networkx_edges(self.dag, pos, arrowsize=20, width=1.5)
            
            # Draw labels with task information
            labels = {}
            for node_id in self.dag.nodes:
                node = self.node_map[node_id]
                labels[node_id] = f"{node.node_id}\n{node.execution_time:.2f}s"
            
            nx.draw_networkx_labels(self.dag, pos, labels=labels, font_size=10)
            
            plt.title("Task Dependencies with Execution Times")
            plt.axis("off")
            plt.tight_layout()
            
            plt.savefig(output_filename)
            print(f"Task dependencies visualization saved to {output_filename}")


# [ROHAN: are we using this?]
def parse_commands(filename: str):
    """Read commands from a file"""
    commands: list[str] = []

    with open(filename, 'r') as f:
        current_command = """"""
        for line in f:
            if line == "\n":
                continue

            if line.startswith("[COMMAND]"):
                if current_command != """""":
                    commands.append(current_command)
                    current_command = """"""
                    
                continue
            else:
                # remove last \n from line
                line = line.rstrip("\n\\")
                current_command += line

        # print(current_command)
        if current_command != """""":
            commands.append(str(current_command))

    return commands

def util_run_server_script_check_log(script_path: str, stdout_log_path: str, stderr_log_path: str, stderr_ready_patterns,
                              stdout_ready_patterns, listen_port, api_port, model, device="gpu", mps=100):
    """Run a script and check log files for startup indicators"""
    server_pid = -1
    max_wait = 120  # Maximum seconds to wait
    log_dir = os.path.join(globals.get_results_dir(), "server_logs")
    os.makedirs(log_dir, exist_ok=True)

    stdout_log = os.path.join(log_dir, f"{stdout_log_path}_{api_port}.log")
    stderr_log = os.path.join(log_dir, f"{stderr_log_path}_{api_port}.log")

    # Start the server process with log file redirection
    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
        process = subprocess.Popen(
            [script_path, str(listen_port), str(api_port), str(model), str(device), str(mps)],
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,  # Important for server processes
        )

    start_time = time.time()
    # Define patterns that indicate successful startup
    found_patterns = {}
    for pattern in stderr_ready_patterns:
        found_patterns[pattern] = False
    for pattern in stdout_ready_patterns:
        found_patterns[pattern] = False

    def check_logs():
        """Check log files for startup indicators"""
        nonlocal server_pid
        
        # Check stderr log
        for log in [stderr_log, stdout_log]:
            try:
                with open(log, 'r') as f:
                    content = f.read()
                    for pattern in stderr_ready_patterns:
                        if pattern in content and not found_patterns[pattern]:
                            found_patterns[pattern] = True
                            print(f"Server indicator found: {pattern}")
                    
                    for pattern in stdout_ready_patterns:
                        if pattern in content and not found_patterns[pattern]:
                            found_patterns[pattern] = True
                            print(f"Server indicator found: {pattern}")
                            
            except Exception as e:
                print(f"Error reading stderr log: {e}")
            
    # Wait for server to be ready using both methods
    while time.time() - start_time < max_wait:
        # Check log files for indicators
        check_logs()
        
        # If we found all log indicators, check if server is responding
        if all(found_patterns.values()):
            print("All server startup indicators found in logs")
            break
                    
        # Wait before checking again
        time.sleep(1)
    
    # Check if we timed out
    if time.time() - start_time >= max_wait:
        print("WARNING: Timed out waiting for server to be ready")
        # You might want to kill the process here
    
    return server_pid



# ====== LlamaCPP ========
# [ROHAN: no need to have this]
def setup_llamacpp_server(**kwargs):
    # return True

    server_pid = -1

    global model_refcount, model_refcount_lock
    api_port = kwargs.get('api_port', 8080)
    model = kwargs.get('model', "/home/cc/models/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-f16.gguf")
    device = kwargs.get('device', "gpu")
    mps = kwargs.get('mps', 100)

    # acquire the lock
    with model_refcount_lock:
        model_refcount["llama"] = model_refcount.get("llama", 0) + 1
        # check if the server is already running
        if model_refcount["llama"] > 1:
            print("Llama server is already running")
            return True        

        # def util_run_server_script_check_log(script_path: str, stdout_log_path: str, stderr_log_path: str, stderr_ready_patterns, stdout_ready_patterns, listen_port, api_port, model):
        print("Setting up llama.cpp server...")

        util_run_server_script_check_log(
            script_path="/home/cc/os-llm/example_workflow/llamacpp_server.sh",
            stdout_log_path="llamacpp_server_stdout",
            stderr_log_path="llamacpp_server_stderr",
            stderr_ready_patterns=["update_slots: all slots are idle"],
            stdout_ready_patterns=[],
            listen_port=api_port,
            api_port=api_port,
            model=model,
            device=device,
            mps=mps
        )

        # print("Pushing NVTX range 'Main'")
        # try:
        #     nvtx.push_range("Main")
        #     print("NVTX push successful")
        # except Exception as e:
        #     print(f"NVTX push error: {e}")
        #     sys.exit(1)



# [ROHAN: no need to have this]
def cleanup_llamacpp_server(**kwargs):
    # return True
    # print("Popping Main range")
    # try:
    #     nvtx.pop_range()
    #     print("Main NVTX pop successful")
    # except Exception as e:
    #     print(f"Main NVTX pop error: {e}")


    global model_refcount, model_refcount_lock

    api_port = kwargs.get('api_port', 8080)

    with model_refcount_lock:
        if "llama" in model_refcount:
            # check if the server is already running
            if model_refcount["llama"] > 0:
                model_refcount["llama"] -= 1
                if model_refcount["llama"] == 0:
                    print("Llama server is shutting down")
                    # kill the process
                    process = subprocess.Popen(
                        ["/home/cc/os-llm/example_workflow/cleanup.sh", str(api_port)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    process.wait()
                else:
                    print("Llama server is still running")

    return True


# ====== Deep Research ======
# [ROHAN: no need to have this]
def run_deep_research_dataset(api_port, model):
    global deep_research_prompts

    start_time = time.time()

    # select one random prompt from the dataset
    deep_research_prompt = random.sample(globals.deep_research_prompts, 1)
    log_dir = os.path.join(globals.get_results_dir(), "client_logs")
    os.makedirs(log_dir, exist_ok=True)
    stdout_log = os.path.join(log_dir, f"deep_research_client_stdout_{api_port}.log")
    stderr_log = os.path.join(log_dir, f"deep_research_client_stderr_{api_port}.log")

    # Start the server process with log file redirection
    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
        process = subprocess.Popen(
            ["/home/cc/os-llm/example_workflow/deep_research_client.sh", str(api_port), str(model), str(deep_research_prompt)],
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,  # Important for server processes
        )

        process.wait()


    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")
    result = {
        'total time': end_time - start_time
    }
    print(result)
    return result


# [ROHAN: no need to have this]
def run_deep_research(**kwargs):
    print("Running deep research (ephemeral app)...")

    api_port = kwargs.get('api_port', 8080)
    model = kwargs.get('model', "openai/meta-llama/Llama-3.1-8B-Instruct")

    result = run_deep_research_dataset(api_port, model)

    return result


# Define example functions for a simple benchmark

# ====== text2image ========
# [ROHAN: no need to have this]
def setup_imagegen(**kwargs):
    global global_vars

    model = kwargs.get('model', "/mnt/tmpfs/models/stable-diffusion-3.5-large")
    device = kwargs.get('device', "gpu")
    mps = kwargs.get('mps', 100)

    # print("Pushing NVTX range 'Main'")
    # try:
    #     nvtx.push_range("Main")
    #     print("NVTX push successful")
    # except Exception as e:
    #     print(f"NVTX push error: {e}")
    #     sys.exit(1)

    if device == "gpu":
        # Set environment variable for MPS
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
        global_vars['imagegen_pipeline'] = StableDiffusion3Pipeline.from_pretrained(
            model,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=torch.float16
        )
        global_vars['imagegen_pipeline'] = global_vars['imagegen_pipeline'].to("cuda")
    else:
        global_vars['imagegen_pipeline'] = StableDiffusion3Pipeline.from_pretrained(
            model,
            text_encoder_3=None,
            tokenizer_3=None
        )
        global_vars['imagegen_pipeline'] = global_vars['imagegen_pipeline'].to("cpu")


# [ROHAN: no need to have this]
def run_imagegen_prompt(prompt):
    global global_vars

    end_time = None

    nvtx.mark("[Imagegen request Start]")
    start_time = time.time()

    image = global_vars['imagegen_pipeline'](
        prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]

    end_time = time.time()
    nvtx.mark("[Imagegen request End]")

    result = {
        "total time": end_time - start_time,
    }
    print(result)
    return result

# With CUDA Graph

# def setup_imagegen(**kwargs):
#     global global_vars

#     model = kwargs.get('model', "/mnt/tmpfs/models/stable-diffusion-3.5-large")
#     device = kwargs.get('device', "gpu")
#     mps = kwargs.get('mps', 100)
#     fixed_prompt = globals.get_next_imagegen_prompt()

#     if device == "gpu":
#         os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
#         pipe = StableDiffusion3Pipeline.from_pretrained(
#             model,
#             text_encoder_3=None,
#             tokenizer_3=None,
#             torch_dtype=torch.float16
#         ).to("cuda")

#         # Warm-up and graph capture
#         # pipe(prompt=fixed_prompt, num_inference_steps=28, guidance_scale=3.5)  # warm-up

#         # torch.cuda.empty_cache()
#         # torch.cuda.synchronize()

#         # graph = torch.cuda.CUDAGraph()
#         # static_output = None

#         # with torch.cuda.graph(graph):
#         #     static_output = pipe(prompt=fixed_prompt, num_inference_steps=28, guidance_scale=3.5)

#         # Save all to global_vars
#         global_vars['imagegen_pipeline'] = pipe
#         # global_vars['imagegen_cuda_graph'] = graph
#         # global_vars['imagegen_output'] = static_output
#         global_vars['imagegen_prompt'] = fixed_prompt
#     else:
#         pipe = StableDiffusion3Pipeline.from_pretrained(
#             model,
#             text_encoder_3=None,
#             tokenizer_3=None
#         ).to("cpu")
#         global_vars['imagegen_pipeline'] = pipe

# def run_imagegen_prompt(prompt):
#     global global_vars

#     if (
#         'imagegen_cuda_graph' in global_vars
#         and prompt == global_vars.get('imagegen_prompt')
#     ):
#         nvtx.mark("[Imagegen request Start]")
#         torch.cuda.synchronize()
#         start_time = time.time()

#         global_vars['imagegen_cuda_graph'].replay()

#         torch.cuda.synchronize()
#         end_time = time.time()
#         nvtx.mark("[Imagegen request End]")

#         print({"total time": end_time - start_time})
#         return {
#             "total time": end_time - start_time,
#             "image": global_vars['imagegen_output'].images[0]
#         }

#     else:
#         # Fallback to normal pipeline execution
#         nvtx.mark("[Imagegen request Start]")
#         start_time = time.time()
#         image = global_vars['imagegen_pipeline'](
#             prompt,
#             num_inference_steps=28,
#             guidance_scale=3.5,
#         ).images[0]
#         end_time = time.time()
#         nvtx.mark("[Imagegen request End]")

#         print({"total time": end_time - start_time})
#         return {
#             "total time": end_time - start_time,
#             "image": image
#         }



# [ROHAN: no need to have this]
def run_imagegen_command_file(filename):
    # read the commands from the file
    commands = parse_commands(filename)
    # get a random command from the file
    command = random.choice(commands)

    # select one random prompt from the dataset
    result = run_imagegen_prompt(command)

    return result


# [ROHAN: no need to have this]
def run_imagegen_dataset():
    global imagegen_prompts

    # select one random prompt from the dataset
    imagegen_prompt = globals.get_next_imagegen_prompt()
    logging.info(f"Imagegen prompt: {imagegen_prompt}")
    result = run_imagegen_prompt(imagegen_prompt)

    return result


# [ROHAN: no need to have this]
def run_imagegen(**kwargs):
    print("Running imagegen")

    filename = kwargs.get('command_file', None)

    if filename is not None:
        result = run_imagegen_command_file(filename)
    else:
        result = run_imagegen_dataset()

    return result

# [ROHAN: no need to have this]
def cleanup_imagegen(**kwargs):
    # print("Popping Main range")
    # try:
    #     nvtx.pop_range()
    #     print("Main NVTX pop successful")
    # except Exception as e:
    #     print(f"Main NVTX pop error: {e}")
    return True


# ====== Nothing ========

# [ROHAN: no need to have this. Move it to applications/ if necessary]
def nothing_function(**kwargs):
    # This function does nothing
    return True

# [ROHAN: no need to have this. Move it to applications/ if necessary]
def sleep_function(**kwargs):
    # This function sleeps for 1 second
    time.sleep(60)
    return True

# ====== Live Captions =======
# [ROHAN: no need to have this]
def setup_whisper():
    global global_vars
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    global_vars["whisper_pipeline"] = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


# [ROHAN: no need to have this]
def run_whisper(**kwargs):
    global global_vars, livecaptions_prompts
    print("Running whisper (ephemeral app)...")

    pipe = global_vars["whisper_pipeline"]

    start_time = time.time()

    # select one random prompt from the dataset
    whisper_prompts = random.sample(livecaptions_prompts, 1)

    for prompt in whisper_prompts:
        # get the wav file from this prompt
        wav_file = prompt["audio"]

        pipe(prompt, return_timestamps="word")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")

    result = {
        'total time': end_time - start_time
    }
    print(result)
    return result

    
# [ROHAN: no need to have this]
def cleanup_whisper():
    # Do nothing
    return


# [ROHAN: no need to have this]
def setup_whisper_online(**kwargs):
    # python3 whisper_online_server.py --host 127.0.0.1 --port 5050
    # python3 /home/cc/applications/whisper_streaming/generate_raw_realtime.py /home/cc/datasets/whisper-earnings21/4320211_chunk_001.wav --port 5050
    server_pid = -1

    api_port = kwargs.get('api_port', 5000)
    device = kwargs.get('device', "gpu")
    mps = kwargs.get('mps', 100)

    print("Setting up whisper-online (ephemeral app)...")

    util_run_server_script_check_log(
        script_path="/home/cc/os-llm/example_workflow/whisper_online_server.sh",
        stdout_log_path="whisper_online_server_stdout",
        stderr_log_path="whisper_online_server_stderr",
        stderr_ready_patterns=["Listening on"],
        stdout_ready_patterns=[],
        listen_port=api_port,
        api_port=api_port,
        model=None,
        device=device,
        mps=mps
    )
    
    return server_pid


# [ROHAN: no need to have this]
def run_whisper_online_command_file(api_port, wav_file_path):
    print(f"Running whisper-online (ephemeral app) on {wav_file_path}...")
    end_time = None

    log_dir = os.path.join(globals.get_results_dir(), "client_logs")
    os.makedirs(log_dir, exist_ok=True)
    stdout_log = os.path.join(log_dir, f"whisper_online_stdout_{api_port}.log")
    stderr_log = os.path.join(log_dir, f"whisper_online_stderr_{api_port}.log")

    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
        process = subprocess.Popen(
            ["/home/cc/os-llm/example_workflow/whisper_online_client.sh", str(api_port), wav_file_path],
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )

    start_time = time.time()
    process.wait()
    end_time = time.time()
    result = {
        'total time': end_time - start_time
    }
    print(result)

    # Parse the stdout log to get the Processing Time
    with open(stdout_log, 'r') as f:
        chunk_idx = 0
        for line in f:
            if "Processing time" in line:
                processing_time = re.search(r"Processing time: (\d+\.\d+)", line)
                if processing_time:
                    processing_time = float(processing_time.group(1))
                    result[f'processing time_chunk_{chunk_idx}'] = processing_time
                    print(f"Processing Time: {processing_time:.4f} seconds")
                    chunk_idx += 1

    return result


# [ROHAN: no need to have this]
def run_whisper_online_dataset(api_port):
    # get a random file from /home/cc/datasets/whisper-earnings21
    directory = "/home/cc/datasets/whisper-earnings21"
    files = os.listdir(directory)
    wav_file = random.choice(files)
    wav_file_path = os.path.join(directory, wav_file)
    wav_file_path = "/home/cc/datasets/whisper-earnings21/4320211_chunk_040.wav"
    result = run_whisper_online_command_file(api_port, wav_file_path)
    return result


# [ROHAN: no need to have this]
def run_whisper_online(**kwargs):
    api_port = kwargs.get('api_port', 5050)
    wav_file_path = kwargs.get('command_file', None)
    if wav_file_path is not None:
        result = run_whisper_online_command_file(api_port, wav_file_path)
    else:
        result = run_whisper_online_dataset(api_port)
    return result


# [ROHAN: no need to have this]
def cleanup_whisper_online(**kwargs):
    api_port = kwargs.get('api_port', 5050)
    process = subprocess.Popen(
        ["/home/cc/os-llm/example_workflow/cleanup.sh", str(api_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait()
    return True


# ====== textgen ========
# [ROHAN: no need to have this]
def setup_textgen(**kwargs):
    server_pid = -1

    listen_port = kwargs.get('listen_port', 7860)
    api_port = kwargs.get('api_port', 5000)
    model = kwargs.get('model', "facebook_opt-1.3b")

    print("Setting up textgen (background app)...")

    util_run_server_script_check_log(
        script_path="/home/cc/os-llm/example_workflow/textgen_server.sh",
        stdout_log_path="textgen_server_stdout",
        stderr_log_path="textgen_server_stderr",
        stderr_ready_patterns=[],
        stdout_ready_patterns=["Running on local URL", "SERVER_PID="],
        listen_port=listen_port,
        api_port=api_port,
        model=model
    )
    
    return server_pid


# [ROHAN: no need to have this]
def run_textgen_command_file(filename, api_port):
    commands = parse_commands(filename)        

    ttft = None
    token_count = 0
    first_token_time = None
    end_time = None
    start_time = time.time()

    commands = random.sample(commands, 1)

    for command in commands:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # for line in iter(process.stderr.readline, ''):
            # print(f"Script output: {line.strip()}")
        # Read output to get the server PID
        for line in iter(process.stdout.readline, ''):
            if line:
                # print(f"Script output: {line.strip()}")
                current_time = time.time()
                if ttft is None:
                    ttft = current_time - start_time
                    first_token_time = current_time
                    print(f"Time to first token: {ttft:.4f} seconds")
                # print(f"Received token: {line.decode('utf-8')}")
                try:
                    data = json.loads(line.strip().replace("data: ", ""))
                    if data["choices"][0]["finish_reason"]:
                        token_count = data["usage"]["completion_tokens"]
                        break
                except json.JSONDecodeError as e:
                    continue
        end_time = time.time()        
        if process.errors != None:
            return False
    

    print(f"{end_time-first_token_time}, token counts: {token_count}")
    tpot = (end_time - first_token_time) / token_count if token_count > 0 else None    
    itl = (end_time - start_time) / token_count if token_count > 0 else None

    result = {
        "ttft": ttft,
        "tpot": tpot,
        "itl": itl,
    }
    print(result)
    return result

    # url = "http://127.0.0.1:5000/v1/completions"
    # headers = {"Content-Type": "application/json"}
    # payload = {
    #     "prompt": "Once upon a time there was a",
    #     "min_tokens": 200,
    #     "max_tokens": 200,
    #     "temperature": 1,
    #     "top_p": 0.9,
    #     "seed": 141293,
    #     "stream": True,
    # }


# [ROHAN: no need to have this]
def run_textgen_dataset(api_port):
    # TODO: Yile, use the following session to issue posts
    api_url = f"http://127.0.0.1:{api_port}/v1/completions"

    ttft = None
    start_time = time.time()

    # select one random prompt from the dataset
    # textgen_prompts = random.sample(globals.textgen_prompts, 1, seed=141293)
    textgen_prompts = [globals.get_next_textgen_prompt()]
    logging.info(f"Textgen prompt: {textgen_prompts}")
    # textgen_prompts = globals.textgen_prompts[:1]
    
    for prompt in textgen_prompts:
        payload = {
            "prompt": prompt,
            # "min_tokens": 200,
            # "max_tokens": 100,
            "max_tokens": 215,
            "temperature": 0,
            "top_p": 0.9,
            "seed": 141293,
            "stream": True
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            with requests.post(api_url, json=payload, headers=headers, stream=True) as response:
                if response.status_code != 200:
                    print("HTTP Error:", response.status_code, response.text)
                    return

                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        # print(f"Script output: {line.strip()}")
                        current_time = time.time()
                        if ttft is None:
                            ttft = current_time - start_time
                            first_token_time = current_time
                            print(f"Time to first token: {ttft:.4f} seconds")

                        try:
                            # Clean and parse the JSON
                            clean_line = line.strip().replace("data: ", "")
                            if clean_line == "[DONE]":
                                break

                            data = json.loads(clean_line)

                            # Exit if finish_reason appears
                            if data.get("choices") and data["choices"][0].get("finish_reason"):
                                token_count = data.get("usage", {}).get("completion_tokens")
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print("Request failed:", e)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")
    print(f"Completion tokens: {token_count}")

    print(f"{end_time-first_token_time}, token counts: {token_count}")
    tpot = (end_time - first_token_time) / token_count if token_count > 0 else None    
    itl = (end_time - start_time) / token_count if token_count > 0 else None

    result = {
        "ttft": ttft,
        "tpot": tpot,
        "itl": itl,
    }
    print(result)
    return result
    

# [ROHAN: no need to have this]
def run_textgen(**kwargs):
    print("Running textgen (background app)...")

    api_port = kwargs.get('api_port', 5000)
    filename = kwargs.get('command_file', None)

    if filename is not None:
        result = run_textgen_command_file(filename, api_port)
    else:
        nvtx.mark("[Chatbot request Start]")
        result = run_textgen_dataset(api_port)
        nvtx.mark("[Chatbot request End]")


    return result


# [ROHAN: no need to have this]
def cleanup_textgen(**kwargs):
    """Example function to cleanup textgen"""
    print("Cleaning up textgen app...")

    api_port = kwargs.get('api_port', 5000)
    process = subprocess.Popen(
        ["/home/cc/os-llm/example_workflow/cleanup.sh", str(api_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait()
    return True


# ====== gpt4all ========
# [ROHAN: no need to have this]
def run_gpt4all(**kwargs):
    """Example function to run GPT4All"""
    print("Running GPT4All app...")
    input_len = kwargs.get('input_len', 100)
    output_len = kwargs.get('output_len', 200)
    process = subprocess.Popen(
        ["/home/cc/os-llm/example_workflow/gpt4all.sh", str(input_len), str(output_len)],
        text=True,
    )
    process.wait()
    return True

# ====== gpt4all ========
# [ROHAN: no need to have this]
def run_share4video(**kwargs):
    """Example function to run share4video"""
    print("Running share4video app...")
    process = subprocess.Popen(
        ["/home/cc/os-llm/example_workflow/share4video.sh"],
        text=True,
    )
    process.wait()
    return True

# ====== livecaptions ========
# [ROHAN: no need to have this]
def run_livecaptions(**kwargs):
    """Example function to run livecaptions"""
    print("Running livecaptions app...")
    process = subprocess.Popen(
        ["/home/cc/os-llm/example_workflow/livecaptions_client.sh"],
        text=True,
    )
    process.wait()
    return True

# [ROHAN: no need to have this. Shadow can also be an application]
def shadow_function(**kwargs):
    return True

def create_task(task_id, app_type, num_requests = 1, listen_port = 5000,
                api_port = 5001, server_model = None, client_model = None,
                client_command_file = None, mps = 100,
                setup_func = None, run_func = None,
                cleanup_func = None, device = "gpu"):
    """Creates a task with the given parameters."""
    task = Task(task_id=task_id, task_type="emphemeral", app_type = app_type)
    num_nodes = num_requests

    # Setup node
    task.add_node(f"{task_id}_0", setup_func, {
        "listen_port": listen_port,
        "api_port": api_port,
        "model": server_model,
        "device": device,
        "mps": mps
    })

    for i in range(0, num_nodes):
        task.add_node(f"{task_id}_{i+1}", run_func, {
            "command_file": client_command_file,
            "api_port": api_port,
            "model": client_model
        })

    # Cleanup node
    task.add_node(f"{task_id}_{num_nodes+1}", cleanup_func, {
        "api_port": api_port
    })

    # Add edges
    for i in range(0, num_nodes+1):
        task.add_edge(f"{task_id}_{i}", f"{task_id}_{i + 1}")

    return task

# [ROHAN: no need to have this]
def create_textgen_benchmark_large():
    """Creates a textgen benchmark with Chatbot, Codebot, and Reasonbot using large model variants."""
    chatbot_task = create_task("chatbot", "chatbot", 5001, 5000, "Llama-3.1-8B-Instruct")
    codebot_task = create_task("codebot", "chatbot", 5003, 5002, "Qwen2.5-Coder-7B-Instruct")
    reasonbot_task = create_task("reasonbot", "chatbot", 5005, 5004, "DeepSeek-R1-Distill-Llama-8B")

    chatbot_dag = chatbot_task.get_dag()
    codebot_dag = codebot_task.get_dag()
    reasonbot_dag = reasonbot_task.get_dag()
    
    
    merged_dag = nx.compose_all([chatbot_dag, codebot_dag, reasonbot_dag])
    merged_dag.add_edge("chatbot_20", "codebot_0")
    merged_dag.add_edge("codebot_10", "reasonbot_0")
    
    benchmark = DAGScheduler(dag=merged_dag, tasks={"chatbot": chatbot_task, "codebot": codebot_task, "reasonbot": reasonbot_task})
        
    return benchmark

# [ROHAN: no need to have this]
def create_textgen_benchmark_small():
    """Creates a textgen benchmark with Chatbot, Codebot, and Reasonbot using small model variants."""
    chatbot_task = create_task("chatbot", "chatbot", 5001, 5000, "Llama-3.2-3B-Instruct", client_command_file="/home/cc/os-llm/example_workflow/textgen_client_1.log")
    codebot_task = create_task("codebot", "chatbot", 5003, 5002, "Qwen2.5-Coder-3B-Instruct", client_command_file="/home/cc/os-llm/example_workflow/textgen_client_2.log")
    reasonbot_task = create_task("reasonbot", "chatbot", 5005, 5004, "DeepSeek-R1-Distill-Qwen-7B", client_command_file="/home/cc/os-llm/example_workflow/textgen_client_3.log")

    chatbot_dag = chatbot_task.get_dag()
    codebot_dag = codebot_task.get_dag()
    reasonbot_dag = reasonbot_task.get_dag()
    
    
    merged_dag = nx.compose_all([chatbot_dag, codebot_dag, reasonbot_dag])
    merged_dag.add_edge("chatbot_10", "codebot_0")
    merged_dag.add_edge("codebot_10", "reasonbot_0")
    
    benchmark = DAGScheduler(dag=merged_dag, tasks={"chatbot": chatbot_task, "codebot": codebot_task, "reasonbot": reasonbot_task})
        
    return benchmark

# [ROHAN: no need to have this]
def create_textgen_benchmark_debug():
    """Creates a textgen benchmark with Chatbot, Codebot, and Reasonbot using small model variants."""
    chatbot_task = create_task("chatbot", "chatbot", 5001, 5000, "Llama-3.2-3B-Instruct", client_command_file="/home/cc/os-llm/example_workflow/textgen_client_1.log")
    
    chatbot_dag = chatbot_task.get_dag()
    
    
    benchmark = DAGScheduler(dag=chatbot_dag, tasks={"chatbot": chatbot_task})
        
    return benchmark

######################################


# [ROHAN: no need to have this]
def create_livecaptions_benchmark(num_requests):
    """Creates a textgen benchmark with Chatbot using small model variants."""
    # chatbot_task = create_task("chatbot", "chatbot", 5001, 5000, "Llama-3.2-3B-Instruct", "/home/cc/os-llm/example_workflow/textgen_client_1.log", num_requests)
    chatbot_task = create_task("chatbot", "chatbot", 5001, 5000, "", num_requests=num_requests)
    
    chatbot_dag = chatbot_task.get_dag()
    benchmark = DAGScheduler(dag=chatbot_dag, tasks={"chatbot": chatbot_task})
        
    return benchmark

# [ROHAN: no need to have this]
def create_whisper_online_benchmark(num_requests):
    """Creates a whisper benchmark with Chatbot using small model variants."""
    # chatbot_task = create_task("chatbot", 5001, 5000, "Llama-3.2-3B-Instruct", "/home/cc/os-llm/example_workflow/textgen_client_1.log", num_requests)
    # whisper_task = create_task("whisper-online", num_requests=num_requests)

    # python3 whisper_online_server.py --host 127.0.0.1 --port 5050
    # python3 /home/cc/applications/whisper_streaming/generate_raw_realtime.py /home/cc/datasets/whisper-earnings21/4320211_chunk_001.wav --port 5050

    whisper_task = create_task("whisper-online", "live_captions", api_port = 5050,
                                num_requests = num_requests,
                                setup_func = setup_whisper_online,
                                run_func = run_whisper_online,
                                cleanup_func = cleanup_whisper_online)


    whisper_dag = whisper_task.get_dag()
    benchmark = DAGScheduler(dag=whisper_dag, tasks={"whisper-online": whisper_task})
        
    return benchmark


# [ROHAN: no need to have this]
def create_whisper_benchmark(num_requests):
    """Creates a whisper benchmark with Chatbot using small model variants."""
    # chatbot_task = create_task("chatbot", 5001, 5000, "Llama-3.2-3B-Instruct", "/home/cc/os-llm/example_workflow/textgen_client_1.log", num_requests)
    whisper_task = create_whisper_task("whisper", num_requests=num_requests)
    
    whisper_dag = whisper_task.get_dag()
    benchmark = DAGScheduler(dag=whisper_dag, tasks={"whisper": whisper_task})
        
    return benchmark

# [ROHAN: no need to have this]
def create_imagegen_benchmark(num_requests):
    """Creates a textgen benchmark with Chatbot using small model variants."""
    # chatbot_task = create_task("chatbot", 5001, 5000, "Llama-3.2-3B-Instruct", "/home/cc/os-llm/example_workflow/textgen_client_1.log", num_requests)
    # imagegen_task = create_imagegen_task("imagegen", 7861, "stable-diffusion-3.5-large", num_requests=num_requests)
    imagegen_task = create_task("imagegen", "imagegen",
                                server_model = "/mnt/tmpfs/models/stable-diffusion-3.5-medium-turbo",
                                num_requests = num_requests,
                                setup_func = setup_imagegen,
                                run_func = run_imagegen,
                                cleanup_func = cleanup_imagegen)

    
    imagegen_dag = imagegen_task.get_dag()
    benchmark = DAGScheduler(dag=imagegen_dag, tasks={"imagegen": imagegen_task})
        
    return benchmark

# [ROHAN: no need to have this]
def create_deep_research_benchmark(num_requests):
    """Creates a textgen benchmark with Chatbot using small model variants."""
    # deep_research_task = create_deep_research_task("deep-research", 8080, "/home/cc/models/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-f16.gguf", "openai/meta-llama/Llama-3.1-8B-Instruct", num_requests=num_requests)
    deep_research_task = create_task("deep-research", "deep_research", api_port = 5000,
                                server_model = "/home/cc/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf",
                                client_model = "openai/meta-llama/Llama-3.2-3B-Instruct",
                                num_requests = num_requests,
                                setup_func = setup_llamacpp_server,
                                run_func = run_deep_research,
                                cleanup_func = cleanup_llamacpp_server)

    
    deep_research_dag = deep_research_task.get_dag()
    benchmark = DAGScheduler(dag=deep_research_dag, tasks={"deep-research": deep_research_task})
        
    return benchmark


# [ROHAN: no need to have this]
def create_chatbot_benchmark(num_requests):
    """Creates a textgen benchmark with Chatbot using small model variants."""
    # chatbot_task = create_task("chatbot", 5001, 5000, "Llama-3.2-3B-Instruct", "/home/cc/os-llm/example_workflow/textgen_client_1.log", num_requests)
    # chatbot_task = create_task("chatbot", 5001, 5000, "Llama-3.1-8B-Instruct", num_requests=num_requests)
    chatbot_task = create_task("chatbot", "chatbot", listen_port = 5001, api_port = 5000,
                                server_model = "Llama-3.2-3B-Instruct",
                                num_requests = num_requests,
                                setup_func = setup_textgen,
                                run_func = run_textgen,
                                cleanup_func = cleanup_textgen)
    
    chatbot_dag = chatbot_task.get_dag()
    benchmark = DAGScheduler(dag=chatbot_dag, tasks={"chatbot": chatbot_task})
        
    return benchmark


# [ROHAN: no need to have this]
def create_chatbot_llama_benchmark(num_requests):
    """Creates a textgen benchmark with Chatbot using small model variants."""
    # chatbot_task = create_task("chatbot", 5001, 5000, "Llama-3.2-3B-Instruct", "/home/cc/os-llm/example_workflow/textgen_client_1.log", num_requests)
    # chatbot_task = create_task("chatbot", 5001, 5000, "Llama-3.1-8B-Instruct", num_requests=num_requests)
    chatbot_task = create_task("chatbot", "chatbot", listen_port = 5001, api_port = 5000,
                                # server_model = "/home/cc/models/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-f16.gguf",
                                # client_model = "openai/meta-llama/Llama-3.1-8B-Instruct",
                                server_model = "/home/cc/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf",
                                client_model = "openai/meta-llama/Llama-3.2-3B-Instruct",
                                num_requests = num_requests,
                                setup_func = setup_llamacpp_server,
                                run_func = run_textgen,
                                cleanup_func = cleanup_llamacpp_server)
    
    chatbot_dag = chatbot_task.get_dag()
    benchmark = DAGScheduler(dag=chatbot_dag, tasks={"chatbot": chatbot_task})
        
    return benchmark


# [ROHAN: no need to have this]
def create_codebot_benchmark(num_requests):
    """Creates a textgen benchmark with Codebot using small model variants."""
    codebot_task = create_task("codebot", "codebot", listen_port = 5003, api_port = 5002,
                                server_model = "Qwen2.5-Coder-3B-Instruct",
                                client_command_file =  "/home/cc/os-llm/example_workflow/textgen_client_2.log",
                                num_requests = num_requests,
                                setup_func = setup_textgen,
                                run_func = run_textgen,
                                cleanup_func = cleanup_textgen)


    codebot_dag = codebot_task.get_dag()
    benchmark = DAGScheduler(dag=codebot_dag, tasks={"codebot": codebot_task})
        
    return benchmark

# [ROHAN: no need to have this]
def create_reasonbot_benchmark(num_requests):
    """Creates a textgen benchmark with Reasonbot using small model variants."""
    reasonbot_task = create_task("reasonbot", "chatbot", listen_port = 5005, api_port = 5004,
                                server_model = "DeepSeek-R1-Distill-Qwen-7B",
                                client_command_file = "/home/cc/os-llm/example_workflow/textgen_client_3.log",
                                num_requests = num_requests,
                                setup_func = setup_textgen,
                                run_func = run_textgen,
                                cleanup_func = cleanup_textgen)
    
    reasonbot_dag = reasonbot_task.get_dag()
    benchmark = DAGScheduler(dag=reasonbot_dag, tasks={"reasonbot": reasonbot_task})
        
    return benchmark

def parse_workflow(file_path, app_dicts):
    """
    Parse a workflow file and generate separate dictionaries for each application.
    
    Args:
        file_path (str): Path to the workflow file
        app_dicts (dict): Dictionary to store application configurations
        
    Returns:
        dict: Dictionary of application dictionaries
    """
    # app_dicts = {}
    workflow_started = False
    app_workflow = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip comments
            if line.startswith('#'):
                continue

            if not workflow_started:
                # Skip lines until we find the workflow start
                if line.startswith('Workflow'):
                    workflow_started = True
                continue
                
            # Check if the application is present in the dictionary
            if f"{line}_args" in app_dicts.keys():
                app_workflow.append(line)
            else:
                print(f"Warning: Application '{line}' not found in the dictionary. Skipping.")

    return app_workflow

def parse_config_file(file_path, app_dicts):
    """
    Parse a configuration file and generate separate dictionaries for each application.
    
    Args:
        file_path (str): Path to the configuration file
        
    Returns:
        dict: Dictionary of application dictionaries
    """
    # app_dicts = {}
    current_app = None
    workflow = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip comments
            if line.startswith('#'):
                continue

            if line.startswith('Workflow'):
                workflow = parse_workflow(file_path, app_dicts)
                break
                
            # Check if this is an application declaration (ends with colon)
            if line.endswith(':'):
                current_app = line.rstrip(':')
                app_name = f"{current_app}_args"
                # app_dicts[app_name] = {}
                continue
                
            # If we have a current application, parse its parameters
            if current_app and "=" in line:
                # Extract key and value
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Handle numeric values
                    if value.isdigit():
                        value = int(value)
                        
                    app_name = f"{current_app}_args"
                    app_dicts[app_name][key] = value
    
    # return app_dicts
    return app_dicts, workflow

def set_default_args(args = None):
    """Sets default arguments for the tasks. If args are not provided, it uses the default values."""
    config_args = {"chatbot_args": {}, "deep_research_args": {}, "imagegen_args": {}, "live_captions_args": {}, "sleep_args": {}}

    if args and args.config:
        # Load the config file and override default arguments
        # config_args = parse_config_file(args.config, config_args)
        config_args, workflow = parse_config_file(args.config, config_args)
    else:
        config_args["chatbot_args"] = {}
        config_args["deep_research_args"] = {}
        config_args["imagegen_args"] = {}
        config_args["live_captions_args"] = {}
        config_args["sleep_args"] = {}

        # Set default arguments for each task
        config_args["chatbot_args"]["num_requests"] = 10
        config_args["deep_research_args"]["num_requests"] = 2
        config_args["imagegen_args"]["num_requests"] = 10
        config_args["live_captions_args"]["num_requests"] = 2

        # Set default server and client models
        config_args["chatbot_args"]["server_model"] = "/home/cc/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf"
        config_args["chatbot_args"]["client_model"] = "openai/meta-llama/Llama-3.2-3B-Instruct"
        config_args["deep_research_args"]["server_model"] = "/home/cc/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf"
        config_args["deep_research_args"]["client_model"] = "openai/meta-llama/Llama-3.2-3B-Instruct"
        config_args["imagegen_args"]["server_model"] = "/mnt/tmpfs/models/stable-diffusion-3.5-medium-turbo"

        # Set default device
        config_args["chatbot_args"]["device"] = "gpu"
        config_args["deep_research_args"]["device"] = "gpu"
        config_args["imagegen_args"]["device"] = "gpu"
        config_args["live_captions_args"]["device"] = "gpu"

        # Set the mps value
        config_args["chatbot_args"]["mps"] = 100
        config_args["deep_research_args"]["mps"] = 100
        config_args["imagegen_args"]["mps"] = 100
        config_args["live_captions_args"]["mps"] = 100        

        workflow = []


    return config_args["chatbot_args"], config_args["deep_research_args"], config_args["imagegen_args"], config_args["live_captions_args"], config_args["sleep_args"], workflow
    # return config_args["chatbot_args"], config_args["deep_research_args"], config_args["imagegen_args"], config_args["live_captions_args"]

# Create a benchmark
def create_concurrent_benchmark(args):
    chatbot_args, deep_research_args, imagegen_args, live_captions_args, sleep_args, workflow = set_default_args(args)
    chatbot_task = None
    chatbot_dag = None
    deep_research_task = None
    deep_research_dag = None
    imagegen_task = None
    imagegen_dag = None
    whisper_task = None
    whisper_dag = None

    dag_compose = []

    # [ROHAN: Again, we only use workflows, so no need to check, and no names of applications in this file]
    if workflow != []:
        num_chatbot_requests = 0
        num_deep_research_requests = 0
        num_imagegen_requests = 0
        num_live_captions_requests = 0
        for app in workflow:
            if app == "chatbot":
                num_chatbot_requests += 1
            elif app == "deep-research":
                num_deep_research_requests += 1
            elif app == "imagegen":
                num_imagegen_requests += 1
            elif app == "live_captions":
                num_live_captions_requests += 1
            else:
                print(f"Warning: Application '{app}' not found in the dictionary. Skipping.")

        if num_chatbot_requests > 0:
            chatbot_args["num_requests"] = num_chatbot_requests
        if num_deep_research_requests > 0:
            deep_research_args["num_requests"] = num_deep_research_requests
        if num_imagegen_requests > 0:
            imagegen_args["num_requests"] = num_imagegen_requests
        if num_live_captions_requests > 0:
            live_captions_args["num_requests"] = num_live_captions_requests


    # [ROHAN: somehow we shouldn't mention applications in this file at all. all these applications should be a part of config, and here we only want generic functions, each application can override the generic functions that require application-specific inputs in the applications/ directory.]
    if (chatbot_args != {}):    
        chatbot_task = create_task("chatbot", "chatbot", listen_port = 5001, api_port = 5000,
                                    server_model = chatbot_args["server_model"],
                                    client_model = chatbot_args["client_model"],
                                    num_requests = chatbot_args["num_requests"],
                                    device = chatbot_args["device"],
                                    mps = chatbot_args["mps"],
                                    setup_func = setup_llamacpp_server,
                                    run_func = run_textgen,
                                    cleanup_func = cleanup_llamacpp_server)
        
        chatbot_dag = chatbot_task.get_dag()
        dag_compose.append(chatbot_dag)


    if (deep_research_args != {}):    
        deep_research_task = create_task("deep-research", "deep_research", api_port = 5000,
                                    server_model = deep_research_args["server_model"],
                                    client_model = deep_research_args["client_model"],
                                    num_requests = deep_research_args["num_requests"],
                                    device = deep_research_args["device"],
                                    mps = deep_research_args["mps"],
                                    setup_func = setup_llamacpp_server,
                                    run_func = run_deep_research,
                                    cleanup_func = cleanup_llamacpp_server)
        
        deep_research_dag = deep_research_task.get_dag()
        dag_compose.append(deep_research_dag)

    if (imagegen_args != {}):
        imagegen_task = create_task("imagegen", "imagegen", 
                                    server_model = imagegen_args["server_model"],
                                    num_requests = imagegen_args["num_requests"],
                                    device = imagegen_args["device"],
                                    mps = imagegen_args["mps"],
                                    setup_func = setup_imagegen,
                                    run_func = run_imagegen,
                                    cleanup_func = cleanup_imagegen)

        
        imagegen_dag = imagegen_task.get_dag()
        dag_compose.append(imagegen_dag)

    if (live_captions_args != {}):
        whisper_task = create_task("live_captions", "live_captions", api_port = 5050,
                                    num_requests = live_captions_args["num_requests"],
                                    device = live_captions_args["device"],
                                    mps = live_captions_args["mps"],
                                    setup_func = setup_whisper_online,
                                    run_func = run_whisper_online,
                                    cleanup_func = cleanup_whisper_online)

        whisper_dag = whisper_task.get_dag()
        dag_compose.append(whisper_dag)
    
    # create a shadow task
    shadow_task = create_task(task_id="shadow", app_type="shadow",
                        setup_func=shadow_function,
                        run_func=shadow_function,
                        cleanup_func=shadow_function)
    shadow_dag = shadow_task.get_dag()
    dag_compose.append(shadow_dag)

    # merged_dag = nx.compose_all([chatbot_dag])  
    # merged_dag = nx.compose_all([whisper_dag])  
    # merged_dag = nx.compose_all([chatbot_dag, deep_research_dag, shadow_dag])
    # merged_dag = nx.compose_all([imagegen_dag, whisper_dag, shadow_dag])

    merged_dag = nx.compose_all(dag_compose)
    task_set = {}

    if chatbot_dag is not None:
        merged_dag.add_edge("shadow_0", "chatbot_0")
        task_set["chatbot"] = chatbot_task
    if deep_research_dag is not None:
        merged_dag.add_edge("shadow_0", "deep-research_0")
        task_set["deep-research"] = deep_research_task
    if imagegen_dag is not None:
        merged_dag.add_edge("shadow_0", "imagegen_0")
        task_set["imagegen"] = imagegen_task
    if whisper_dag is not None:
        merged_dag.add_edge("shadow_0", "live_captions_0")
        task_set["live_captions"] = whisper_task
         
    task_set["shadow"] = shadow_task

    previous_app = None
    app_counters = {}
    app_counters["chatbot"] = 0
    app_counters["deep-research"] = 0
    app_counters["imagegen"] = 0
    app_counters["live_captions"] = 0
    for current_app in workflow:
        app_counters[current_app] += 1

        if previous_app is not None and current_app != previous_app:
            merged_dag.add_edge(f"{previous_app}_{app_counters[previous_app]}", f"{current_app}_{app_counters[current_app]}")
        
        previous_app = current_app


    # benchmark = DAGScheduler(dag=chatbot_dag, tasks={"chatbot": chatbot_task})
    # benchmark = DAGScheduler(dag=whisper_dag, tasks={"live-captions": whisper_task})
    # benchmark = DAGScheduler(dag=merged_dag, tasks={"chatbot": chatbot_task, "deep-research": deep_research_task, "shadow": shadow_task})
    # benchmark = DAGScheduler(dag=merged_dag, tasks={"live-captions": whisper_task, "imagegen": imagegen_task, "shadow": shadow_task})
    benchmark = DAGScheduler(dag=merged_dag, tasks=task_set)

    return benchmark


def main(args):
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    globals.set_results_dir(args.results)

    if args.start_time:
        globals.set_start_time(args.start_time)

    log_filename = f"overall_perf.log"
    log_path = os.path.join(args.results, log_filename)
    logging.basicConfig(filename=log_path, level=logging.INFO)

    # [ROHAN: Maybe lets just keep workflow? No other things needed. Basically no need to check workflow. We only support workflows, and workflows can have different applications running singularly within the yaml.]
    if args.benchmark == "chatbot":
        benchmark = create_chatbot_benchmark(args.num_requests)
        globals.load_textgen_dataset()
    elif args.benchmark == "chatbot-llama":
        benchmark = create_chatbot_llama_benchmark(args.num_requests)
        globals.load_textgen_dataset()
    elif args.benchmark == "codebot":
        benchmark = create_codebot_benchmark(args.num_requests)
    elif args.benchmark == "reasonbot":
        benchmark = create_reasonbot_benchmark(args.num_requests)
    elif args.benchmark == "whisper":
        benchmark = create_whisper_benchmark(args.num_requests)
        globals.load_livecaptions_dataset()
    elif args.benchmark == "whisper-online":
        benchmark = create_whisper_online_benchmark(args.num_requests)
    elif args.benchmark == "imagegen":
        benchmark = create_imagegen_benchmark(args.num_requests)
        globals.load_imagegen_dataset()
    elif args.benchmark == "deep-research":
        benchmark = create_deep_research_benchmark(args.num_requests)
        globals.load_deep_research_dataset()
    elif args.benchmark == "all":
        globals.load_textgen_dataset()
        globals.load_imagegen_dataset()
        globals.load_deep_research_dataset()
        benchmark = create_concurrent_benchmark(args)
    elif args.benchmark == "workflow":
        from workflow import Workflow

        globals.load_textgen_dataset()
        globals.load_imagegen_dataset()
        globals.load_deep_research_dataset()

        workflow = Workflow(args.config)
        workflow.load_workflow_unit_config()
        workflow.generate_task_queue()
        benchmark = workflow.generate_benchmark()

    benchmark.visualize()
    # logging.info("\n=== Running Concurrent Benchmark ===")

    # [ROHAN: monitoring can be in a separate directory as well to keep it modular]
    monitor = GpuMemoryMonitor(gpu_id=0, interval=0.01, results_dir=args.results)
    import threading
    monitor_thread = threading.Thread(target=monitor.start_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()

    benchmark.run_concurrent()
    benchmark.display_results()
    
    monitor.running = False
    monitor_thread.join()
    # logging.info("\n=== Benchmark Completed ===")

def get_parser():
    parser = argparse.ArgumentParser()
    # [ROHAN: remove --benchmark flag from here. Everything should be through config.yaml]
    parser.add_argument('--benchmark', type=str, help="Name of the application (chatbot, chatbot-llama, codebot, reasonbot, whisper, imagegen, deep-research, whisper-online, concurrent, all)", required=True)
    # [ROHAN: remove --num_requests flag from here. Everything should be through config.yaml]
    parser.add_argument('--num_requests', type=int, help="Number of requests", required=False, default=10)
    parser.add_argument('--config', type=str, help="Path to the config file", required=False)
    # [ROHAN: can get rid of start_time] 
    parser.add_argument('--start_time', type=str, help="Start time for the benchmark", required=False, default=0)
    parser.add_argument('--results', type=str, help="Path to the results directory", required=True)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)