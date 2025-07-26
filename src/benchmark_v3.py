import argparse
import concurrent.futures
import logging
import os
import threading
import time
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Callable, List, Tuple, Any, Set, Optional
import sys
from datetime import datetime

# [ROHAN: We should remove these paths. All application stuff should be imported from applications/, datasets/, inference-backed/ respectively. ]
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitors.memory_util import GpuMemoryMonitor
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

######################################


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


def main(args):
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    globals.set_results_dir(args.results)

    log_filename = f"overall_perf.log"
    log_path = os.path.join(args.results, log_filename)
    logging.basicConfig(filename=log_path, level=logging.INFO)

    # [ROHAN: Maybe lets just keep workflow? No other things needed. Basically no need to check workflow. We only support workflows, and workflows can have different applications running singularly within the yaml.]
    from workflow import Workflow
    globals.load_textgen_dataset()
    globals.load_imagegen_dataset()
    globals.load_deep_research_dataset()
    globals.set_start_time()
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
    parser.add_argument('--config', type=str, help="Path to the config file", required=True)
    parser.add_argument('--results', type=str, help="Path to the results directory", required=True)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)