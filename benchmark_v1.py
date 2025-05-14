import concurrent.futures
import json
import os
import re
import signal
import threading
import time
import subprocess
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Callable, List, Tuple, Any, Set, Optional

class ExecutionNode:
    """Represents a specific execution of an application with arguments"""
    def __init__(self, 
                 node_id: str, 
                 app_name: str, 
                 setup_args: Dict = None, 
                 run_args: Dict = None, 
                 finish_args: Dict = None):
        """
        Initialize an application node.
        
        Args:
            node_id: Unique identifier for this node
            app_name: Name of the application to execute
            setup_args: Arguments to pass to the setup function
            run_args: Arguments to pass to the run function
            finish_args: Arguments to pass to the finish function
        """
        self.node_id = node_id
        self.app_name = app_name
        self.setup_args = setup_args or {}
        self.run_args = run_args or {}
        self.finish_args = finish_args or {}
        
        # Timing metrics for this specific execution
        self.setup_time = 0
        self.run_time = 0
        self.finish_time = 0
        self.total_time = 0
        self.result = None
        self.success = False

    def get_total_time(self):
        """Get the total execution time for this node"""
        return self.setup_time + self.run_time + self.finish_time


class Application:
    def __init__(self, name: str, 
                 setup_fn: Callable[..., Any], 
                 run_fn: Callable[..., Any], 
                 finish_fn: Callable[..., bool],
                 app_type: str = "ephemeral"):
        """
        Initialize an application with setup, run, and finish functions.
        
        Args:
            name: Unique identifier for the application
            setup_fn: Function to prepare the environment, can accept arguments
            run_fn: Function to run the application, can accept arguments
            finish_fn: Function to validate the output, can accept arguments
            app_type: Either "ephemeral" or "background"
                      - ephemeral: setup, run, finish executed for each node
                      - background: setup executed only once across all nodes
        """
        self.name = name
        self.setup_fn = setup_fn
        self.run_fn = run_fn
        self.finish_fn = finish_fn
        self.app_type = app_type
        self.is_set_up = False
        self.server_pid = -1
        self.refs = 0
        self.app_lock = threading.Lock()
    
    def setup(self, args: Dict = None) -> Tuple[float, Any]:
        """
        Execute setup if needed based on application type.
        
        Args:
            args: Arguments to pass to the setup function
        
        Returns:
            Tuple of (setup_time, server_pid)
        """
        args = args or {}
        self.app_lock.acquire()
        
        if self.app_type == "ephemeral" or not self.is_set_up:
            start = time.time()
            self.server_pid = self.setup_fn(**args)
            setup_time = time.time() - start
            self.is_set_up = True
            print(f"Setup complete for {self.name}")
        else:
            # Skip setup for background applications that are already set up
            print(f"Skipping setup for background application {self.name} (already set up)")
            setup_time = 0

        self.refs += 1
        self.app_lock.release()
        return setup_time, self.server_pid
    
    def run(self, args: Dict = None) -> Tuple[float, Any]:
        """
        Run the application with the given arguments.
        
        Args:
            args: Arguments to pass to the run function
        
        Returns:
            Tuple of (run_time, result)
        """
        args = args or {}
        start = time.time()
        result = self.run_fn(**args)

        run_time = time.time() - start
        return run_time, result
    
    def finish(self, result: Any, args: Dict = None) -> Tuple[bool, float]:
        """
        Finish the application execution and validate the result.
        
        Args:
            result: The result from the run function
            args: Arguments to pass to the finish function
        
        Returns:
            Tuple of (success, finish_time)
        """
        args = args or {}
        self.app_lock.acquire()
        
        self.refs -= 1
        if self.app_type == "ephemeral" or self.refs == 0 or args.get('force', True):
            if self.server_pid != -1:
                try:
                    os.kill(self.server_pid, signal.SIGINT)
                    print(f"Sent SIGINT to process {self.server_pid}")
                except ProcessLookupError:
                    print(f"Process {self.server_pid} not found")
                self.is_set_up = False
                self.server_pid = -1

        start = time.time()
        finish_args = dict(args)
        finish_args['result'] = result  # Add result to the arguments
        self.finish_fn(**finish_args)
        finish_time = time.time() - start
        
        self.app_lock.release()
        return finish_time


class DAGBenchmark:
    def __init__(self, dag: nx.DiGraph, applications: Dict[str, Application], nodes: Dict[str, ExecutionNode]):
        """
        Initialize a DAG benchmark.
        
        Args:
            dag: A directed acyclic graph where nodes represent application executions
            applications: Dictionary mapping application names to Application objects
            nodes: Dictionary mapping node IDs to ExecutionNode objects
        """
        self.dag = dag
        self.applications = applications
        self.nodes = nodes
        self.total_time = 0
        
        # Validate that all node IDs in the DAG exist in the nodes dictionary
        for node_id in self.dag.nodes:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} in DAG not found in nodes dictionary")
    
    def validate_dag(self):
        """Validate that the DAG is properly formed"""
        # Check if the DAG is acyclic
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("The graph is not a Directed Acyclic Graph (DAG)")
        
        # Validate all nodes have corresponding applications
        for node_id in self.dag.nodes:
            app_name = self.nodes[node_id].app_name
            if app_name not in self.applications:
                raise ValueError(f"Node {node_id} references application '{app_name}' which does not exist")
    
    def reset_nodes(self):
        """Reset timing metrics for all nodes"""
        for node in self.nodes.values():
            node.setup_time = 0
            node.run_time = 0
            node.finish_time = 0
            node.total_time = 0
            node.result = None
            node.success = False
    
    def finish_background_apps(self):
        """Finish all background applications"""
        for app in self.applications.values():
            if app.app_type == "background" and app.is_set_up:
                # Empty finish call just to clean up resources
                app.finish(None, {'force': True})
    
    def run_sequential(self):
        """Run nodes sequentially based on topological ordering"""
        self.validate_dag()
        self.reset_nodes()
        
        start_time = time.time()
        topo_order = list(nx.topological_sort(self.dag))
        
        for node_id in topo_order:
            node = self.nodes[node_id]
            app = self.applications[node.app_name]
            
            print(f"Running node {node_id} with application {node.app_name}...")
            
            # Execute setup
            node.setup_time, _ = app.setup(node.setup_args)
            
            # Execute run
            node.run_time, node.success = app.run(node.run_args)
            
            # Execute finish for ephemeral apps
            if app.app_type == "ephemeral":
                node.finish_time = app.finish(node.result, node.finish_args)
                
                if not node.success:
                    raise ValueError(f"Node {node_id} with application {node.app_name} failed to produce expected output")
            
            # Calculate total time for this node
            node.total_time = node.get_total_time()
        
        # Clean up background applications
        self.finish_background_apps()
        
        self.total_time = time.time() - start_time
        return self.total_time
    
    def run_concurrent(self):
        """Run nodes concurrently where possible based on dependencies"""
        self.validate_dag()
        self.reset_nodes()
        
        start_time = time.time()
        
        # Track completed nodes
        completed = {node_id: False for node_id in self.dag.nodes}
        executing = {node_id: False for node_id in self.dag.nodes}
        
        # Check if all predecessors of a node are completed
        def can_execute(node_id):
            for pred in self.dag.predecessors(node_id):
                if not completed[pred]:
                    return False
            return True
        
        # Execute a single node
        def execute_node(node_id):
            node = self.nodes[node_id]
            app = self.applications[node.app_name]
            
            print(f"Executing node {node_id} with application {node.app_name}...")
            
            # Execute setup
            node.setup_time, _ = app.setup(node.setup_args)
            
            # Execute run
            node.run_time, node.success = app.run(node.run_args)
            
            # Execute finish for ephemeral apps
            if app.app_type == "ephemeral":
                node.finish_time = app.finish(node.result, node.finish_args)
                
                if not node.success:
                    raise ValueError(f"Node {node_id} with application {node.app_name} failed to produce expected output")
            
            # Calculate total time for this node
            node.total_time = node.get_total_time()
            
            completed[node_id] = True
            return node_id
        
        threading_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Continue until all nodes are processed
            while not all(completed.values()):
                # Find nodes that can be executed (all predecessors completed)
                threading_lock.acquire()
                executable = [node_id for node_id in self.dag.nodes 
                             if not executing[node_id] and not completed[node_id] and can_execute(node_id)]
                
                if not executable:
                    # Wait a bit and check again
                    threading_lock.release()
                    time.sleep(0.1)
                    continue
                
                # Mark nodes as executing
                for node_id in executable:
                    executing[node_id] = True
                threading_lock.release()
                
                # Execute nodes concurrently
                futures = {executor.submit(execute_node, node_id): node_id for node_id in executable}
                
                # Wait for at least one to complete
                concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                # Process completed futures
                for future in [f for f in futures if f.done()]:
                    try:
                        completed_node_id = future.result()
                        print(f"Completed node {completed_node_id}")
                    except Exception as e:
                        print(f"Error executing node: {e}")
                        raise
        
        # Clean up background applications
        self.finish_background_apps()
        
        self.total_time = time.time() - start_time
        return self.total_time
    
    def display_results(self):
        """Display benchmark results"""
        print(f"\nTotal execution time: {self.total_time:.4f} seconds")
        print("\nExecution times for each node:")
        
        for node_id, node in self.nodes.items():
            app = self.applications[node.app_name]
            
            print(f"Node {node_id} (Application: {node.app_name}):")
            print(f"  Setup time:  {node.setup_time:.4f} seconds")
            print(f"  Run time:    {node.run_time:.4f} seconds")
            print(f"  Finish time: {node.finish_time:.4f} seconds")
            print(f"  Total time:  {node.total_time:.4f} seconds")
            print(f"  Success:     {node.success}")
    
    def visualize_dag(self, output_filename="dag_visualization.png"):
        """Visualize the DAG with execution times"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.dag, seed=42)
        
        # Get node colors based on application type
        node_colors = []
        for node_id in self.dag.nodes:
            app_name = self.nodes[node_id].app_name
            app = self.applications[app_name]
            if app.app_type == "background":
                node_colors.append("lightgreen")
            else:
                node_colors.append("lightblue")
        
        # Draw nodes with sizes proportional to execution time
        node_sizes = [self.nodes[node_id].total_time * 500 + 300 for node_id in self.dag.nodes]
        nx.draw_networkx_nodes(self.dag, pos, node_size=node_sizes, node_color=node_colors)
        
        # Draw edges
        nx.draw_networkx_edges(self.dag, pos, arrowsize=20, width=1.5)
        
        # Draw labels with node information
        labels = {}
        for node_id in self.dag.nodes:
            node = self.nodes[node_id]
            labels[node_id] = f"{node_id}\n({node.app_name})\n{node.total_time:.2f}s"
        
        nx.draw_networkx_labels(self.dag, pos, labels=labels, font_size=10)
        
        plt.title("Application DAG with Execution Times")
        plt.axis("off")
        plt.tight_layout()
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Ephemeral App'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Background App')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(output_filename)
        print(f"DAG visualization saved to {output_filename}")


def parse_commands(filename: str):
    # Read a file in which every request starts with [COMMAND], and the next few lines are the command which should be appended to a string, until the next command
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

# Example of creating a benchmark with nodes that have specific arguments
def create_example_benchmark():
    # Example functions for each application
    def app1_setup(**kwargs):
        print("Setting up App1...")
        
        # Method 1: Use log files (most reliable)
        log_dir = "/home/rohan/os-llm/osllm-benchmark/server_logs"
        os.makedirs(log_dir, exist_ok=True)
        stdout_log = os.path.join(log_dir, "server_stdout.log")
        stderr_log = os.path.join(log_dir, "server_stderr.log")
        
        # Start the server process with log file redirection
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            process = subprocess.Popen(
                ["/home/rohan/os-llm/osllm-benchmark/example_workflow/textimg_server.sh"],
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,  # Important for server processes
            )
        
        server_pid = -1
        max_wait = 120  # Maximum seconds to wait
        start_time = time.time()
        
        # Define patterns that indicate successful startup
        stderr_ready_pattern = "Uvicorn running on"
        stdout_ready_patterns = ["Model loaded", "SERVER_PID="]
        
        # Track what we've found
        found_patterns = {
            stderr_ready_pattern: False,
            stdout_ready_patterns[0]: False,
            stdout_ready_patterns[1]: False
        }
        
        def check_logs():
            """Check log files for startup indicators"""
            nonlocal server_pid
            
            # Check stderr log
            try:
                with open(stderr_log, 'r') as f:
                    stderr_content = f.read()
                    if stderr_ready_pattern in stderr_content and not found_patterns[stderr_ready_pattern]:
                        found_patterns[stderr_ready_pattern] = True
                        print(f"Server indicator found: {stderr_ready_pattern}")
            except Exception as e:
                print(f"Error reading stderr log: {e}")
            
            # Check stdout log
            try:
                with open(stdout_log, 'r') as f:
                    for line in f:
                        # Check for PID
                        if stdout_ready_patterns[1] in line and not found_patterns[stdout_ready_patterns[1]]:
                            match = re.search(r'SERVER_PID=(\d+)', line)
                            if match:
                                server_pid = int(match.group(1))
                                found_patterns[stdout_ready_patterns[1]] = True
                                print(f"Found server PID: {server_pid}")
                        
                        # Check for model loaded
                        if stdout_ready_patterns[0] in line and not found_patterns[stdout_ready_patterns[0]]:
                            found_patterns[stdout_ready_patterns[0]] = True
                            print(f"Server indicator found: {stdout_ready_patterns[0]}")
            except Exception as e:
                print(f"Error reading stdout log: {e}")
        

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
    
    def app1_run(**kwargs):
        print("Running App1...")
        # read filename from kwargs
        filename = kwargs.get('command_file', None)
        if filename is None:
            print("No command file provided")
            return False

        commands = parse_commands(filename)        
        for command in commands:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            for line in iter(process.stderr.readline, ''):
                print(f"Script output: {line.strip()}")

            # Read output to get the server PID
            for line in iter(process.stdout.readline, ''):
                print(f"Script output: {line.strip()}")

            if process.errors != None:
                return False

        return True
    
    def app1_finish(**kwargs):
        print("Finishing App1...")
        time.sleep(0.3)  # Simulate finish time
    
    def app2_setup(**kwargs):
        print("Setting up App2...")
        time.sleep(0.3)  # Simulate setup time
        return -1
    
    def app2_run(**kwargs):
        print("Running App2...")
        process = subprocess.Popen(
            ["/home/rohan/os-llm/osllm-benchmark/example_workflow/livecaptions_client.sh"],
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            text=True,
            bufsize=1            
        )
        process.wait()
        return process.errors == None
    
    def app2_finish(**kwargs):
        print("Finishing App2...")
        time.sleep(0.5)  # Simulate finish time
    
    def app3_setup(**kwargs):
        server_pid = -1
        print("Setting up App3 (background app)...")
        # execute "background_setup.sh" in the background

        log_dir = "/home/rohan/os-llm/osllm-benchmark/server_logs"
        os.makedirs(log_dir, exist_ok=True)
        stdout_log = os.path.join(log_dir, "app3_server_stdout.log")
        stderr_log = os.path.join(log_dir, "app3_server_stderr.log")

        # Start the server process with log file redirection
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            process = subprocess.Popen(
                ["/home/rohan/os-llm/osllm-benchmark/example_workflow/textgen_server.sh"],
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,  # Important for server processes
            )

        max_wait = 120  # Maximum seconds to wait
        start_time = time.time()
        
        # Define patterns that indicate successful startup
        stdout_ready_patterns = ["Running on local URL", "SERVER_PID="]
        
        # Track what we've found
        found_patterns = {
            stdout_ready_patterns[0]: False,
            stdout_ready_patterns[1]: False
        }
        
        def check_logs():
            """Check log files for startup indicators"""
            nonlocal server_pid
            
            # Check stdout log
            try:
                with open(stdout_log, 'r') as f:
                    for line in f:
                        # Check for PID
                        if stdout_ready_patterns[1] in line and not found_patterns[stdout_ready_patterns[1]]:
                            match = re.search(r'SERVER_PID=(\d+)', line)
                            if match:
                                server_pid = int(match.group(1))
                                found_patterns[stdout_ready_patterns[1]] = True
                                print(f"Found server PID: {server_pid}")
                        
                        # Check for model loaded
                        if stdout_ready_patterns[0] in line and not found_patterns[stdout_ready_patterns[0]]:
                            found_patterns[stdout_ready_patterns[0]] = True
                            print(f"Server indicator found: {stdout_ready_patterns[0]}")
            except Exception as e:
                print(f"Error reading stdout log: {e}")
        

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

    def app3_run(**kwargs):
        print("Running App3 (background app)...")
        # read filename from kwargs
        filename = kwargs.get('command_file', None)
        if filename is None:
            print("No command file provided")
            return False
        
        commands = parse_commands(filename)
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
            #     print(f"Script output: {line.strip()}")

            # Read output to get the server PID
            for line in iter(process.stdout.readline, ''):
                print(f"Script output: {line.strip()}")

            if process.errors != None:
                return False

        return True
    
    def app3_finish(**kwargs):
        print("Finishing App3 (background app)...")    
        time.sleep(0.4)  # Simulate finish time
    
    # Create Application objects
    app1 = Application("App1", app1_setup, app1_run, app1_finish, app_type="ephemeral")
    app2 = Application("App2", app2_setup, app2_run, app2_finish, app_type="ephemeral")
    app3 = Application("App3", app3_setup, app3_run, app3_finish, app_type="background")
    
    # Create a dictionary mapping application names to Application objects
    applications = {
        "App1": app1,
        "App2": app2,
        "App3": app3
    }
    
    # Create ExecutionNode objects for specific executions
    node1 = ExecutionNode(
        node_id="Task1", 
        app_name="App1",
        setup_args={},
        run_args={"command_file": "/home/rohan/os-llm/osllm-benchmark/example_workflow/textimg_client.log"},
        finish_args={}
    )
    
    node2 = ExecutionNode(
        node_id="Task2", 
        app_name="App2",
        setup_args={},
        run_args={},
        finish_args={}
    )
    
    node3 = ExecutionNode(
        node_id="Task3", 
        app_name="App3",
        setup_args={},
        run_args={"command_file": "/home/rohan/os-llm/osllm-benchmark/example_workflow/textgen_client_node3.log"},
        finish_args={}
    )
    
    node4 = ExecutionNode(
        node_id="Task4", 
        app_name="App3",
        run_args={"command_file": "/home/rohan/os-llm/osllm-benchmark/example_workflow/textgen_client_node4.log"},
        finish_args={'result': "App3 result"}
    )
    
    # Create nodes dictionary
    nodes = {
        "Task1": node1,
        "Task2": node2,
        "Task3": node3,
        "Task4": node4
    }
    
    # Create DAG
    dag = nx.DiGraph()
    
    # Add nodes to DAG
    for node_id in nodes:
        dag.add_node(node_id)
    
    # Add edges to represent dependencies
    dag.add_edge("Task1", "Task2")  # Task2 depends on Task1
    dag.add_edge("Task1", "Task3")  # Task3 depends on Task1
    dag.add_edge("Task2", "Task4")  # Task4 depends on Task2
    
    return dag, applications, nodes


def main():
    # Create example benchmark
    dag, applications, nodes = create_example_benchmark()
    
    # Create benchmark object
    benchmark = DAGBenchmark(dag, applications, nodes)
    
    # Run sequential benchmark
    print("\n=== Running Sequential Benchmark ===")
    seq_time = benchmark.run_sequential()
    benchmark.display_results()
    benchmark.visualize_dag("sequential_dag.png")
    
    # Create a new benchmark instance for concurrent run
    benchmark = DAGBenchmark(dag, applications, nodes)
    
    # Run concurrent benchmark
    print("\n=== Running Concurrent Benchmark ===")
    con_time = benchmark.run_concurrent()
    benchmark.display_results()
    benchmark.visualize_dag("concurrent_dag.png")
    
    # Compare results
    print("\n=== Benchmark Comparison ===")
    print(f"Sequential execution time: {seq_time:.4f} seconds")
    print(f"Concurrent execution time: {con_time:.4f} seconds")
    if seq_time > con_time:
        speedup = seq_time / con_time
        print(f"Speedup with concurrent execution: {speedup:.2f}x")
    else:
        ratio = con_time / seq_time
        print(f"Sequential execution was {ratio:.2f}x faster")


if __name__ == "__main__":
    main()