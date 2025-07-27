from collections import deque
import yaml
import sys
import os

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

from applications.application import Application
from src.benchmark_v3 import DAGScheduler, Task
import networkx as nx

# Helper functions to wrap Application methods for ExecutionNode
def create_setup_wrapper(application: Application):
    """Create a wrapper function for application setup"""
    def setup_wrapper(**kwargs):
        return application.run_setup(**kwargs)
    return setup_wrapper

def create_run_wrapper(application: Application):
    """Create a wrapper function for application run"""
    def run_wrapper(**kwargs):
        return application.run_application(**kwargs)
    return run_wrapper

def create_cleanup_wrapper(application: Application):
    """Create a wrapper function for application cleanup"""
    def cleanup_wrapper(**kwargs):
        return application.run_cleanup(**kwargs)
    return cleanup_wrapper

class WorkflowUnit:
    def __init__(self, type: str, task: Task, node_start: str, node_end: str):
        self.type = type
        self.task = task
        self.node_start = node_start
        self.node_end = node_end
        self.id = task.task_id


class Workflow:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.workflow_config = self.load_yaml()
        self.tasks_map_queue = {}
        self.workflow_unit_map = {}
        self.applications = {}

    def load_yaml(self):
        # Load the YAML file and remove comments
        cleaned_yaml = self._remove_config_comments(self.yaml_file)
        return yaml.safe_load(cleaned_yaml)
    
    def register_application(self, application_name: str, application: Application):
        """Register an application instance with the workflow"""
        self.applications[application_name] = application
        application.load_dataset()
    
    def load_workflow_unit_config(self):
        """Load workflow unit configuration from YAML"""
        for k, v in self.workflow_config.items():
            if k == "workflows":
                continue

            app_type = v["type"]
            application = self.applications[app_type]
            default_config = application.get_default_config()
            node_config = {k: val for k, val in v.items() if k != "type"}
            node_config = {**default_config, **node_config}
            
            # Store the configuration
            self.workflow_unit_map[k] = {
                "type": app_type,
                "node_config": node_config,
                "count": 0
            }

        # Count how many times each unit is used in workflows
        workflows = self.workflow_config.get("workflows", {})
        for k, v in workflows.items():
            unit_id = v["uses"]
            if unit_id in self.workflow_unit_map:
                self.workflow_unit_map[unit_id]["count"] += 1

    def generate_task_queue(self):
        """Generate a task queue based on the workflow unit map."""
        for k, v in self.workflow_unit_map.items():
            if v["count"] == 0:
                continue
                
            count = v["count"]
            app_type = v["type"]
            node_config = v["node_config"]

            # Check if the application type is registered
            if app_type not in self.applications:
                raise ValueError(f"Application type '{app_type}' not registered. Please register it using register_application()")

            self.tasks_map_queue[k] = deque()
            for i in range(count):
                task_id = f"{k}_u{i}"
                task, start_node, end_node = self._generate_application_task_group(
                    task_id=task_id,
                    app_type=app_type,
                    node_config=node_config
                )
                self.tasks_map_queue[k].append(WorkflowUnit(app_type, task, start_node, end_node))

    def _generate_application_task_group(self, task_id: str, app_type: str, node_config: dict):
        """Generate a task group using an Application instance"""
        task = Task(task_id=task_id, task_type="ephemeral", app_type=app_type)
        
        # Get the registered application
        application = self.applications[app_type]
        
        # Update application config with YAML config
        application.add_config(node_config)
        
        # Get number of requests (default to 1)
        num_requests = node_config.get("num_requests", 1)
        
        start_node = f"{task_id}_0"
        end_node = f"{task_id}_{num_requests + 1}"

        # Create setup node
        setup_wrapper = create_setup_wrapper(application)
        task.add_node(start_node, setup_wrapper, node_config)

        # Create run nodes
        for i in range(1, num_requests + 1):
            run_wrapper = create_run_wrapper(application)
            task.add_node(f"{task_id}_{i}", run_wrapper, node_config)

        # Create cleanup node
        cleanup_wrapper = create_cleanup_wrapper(application)
        task.add_node(f"{task_id}_{num_requests+1}", cleanup_wrapper, node_config)

        # Add edges: setup -> run_1 -> run_2 -> ... -> run_n -> cleanup
        for i in range(num_requests + 1):
            task.add_edge(f"{task_id}_{i}", f"{task_id}_{i+1}")

        return (task, start_node, end_node)
        
    def _remove_config_comments(self, file_path) -> str:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            # Remove comments and empty lines
            cleaned_lines = [line for line in lines if not line.strip().startswith('#') and line.strip() != '']
            
            return ''.join(cleaned_lines)
        
    def generate_benchmark(self):
        """Generate a benchmark based on the workflow."""
        workflow = self.workflow_config.get("workflows", {})
        # Make sure we've already called load_workflow_unit_config() and generate_task_queue()
        
        task_sets = {}
        dag_list = []

        # 1) Create a dummy "start" task so that any unit with no deps hooks to it
        start_task, start_node, _ = self._generate_dummy_task_group("start")
        dag_list.append(start_task.get_dag())
        task_sets[start_task.task_id] = start_task

        # 2) Pull one WorkflowUnit per workflow-entry & stash its DAG + Task
        units = {}
        for unit_id, unit_conf in workflow.items():
            uses = unit_conf["uses"]
            if uses not in self.tasks_map_queue:
                raise ValueError(f"Task group '{uses}' not found in queue.")
            wf_unit: WorkflowUnit = self.tasks_map_queue[uses].popleft()

            dag_list.append(wf_unit.task.get_dag())
            task_sets[wf_unit.task.task_id] = wf_unit.task

            units[unit_id] = {
                "unit":        wf_unit,
                "dependencies": unit_conf.get("depend_on", [])
            }

        # 3) Compose all the sub‑DAGs into one big graph
        merged_dag = nx.compose_all(dag_list)

        # 4) Wire edges:
        #    - No-dep units hook from start_node
        #    - Otherwise, from each dep's end_node → this unit's start_node
        for unit_id, info in units.items():
            wfu   = info["unit"]
            deps  = info["dependencies"]

            if not deps:
                merged_dag.add_edge(start_node, wfu.node_start)
            else:
                for dep_id in deps:
                    if dep_id not in units:
                        raise ValueError(f"Unknown dependency '{dep_id}' for '{unit_id}'")
                    dep_wfu = units[dep_id]["unit"]
                    merged_dag.add_edge(dep_wfu.node_end, wfu.node_start)

        # 5) Finally, hand it off to your scheduler
        return DAGScheduler(merged_dag, task_sets)
    
    def _generate_dummy_task_group(self, task_id: str):
        """Generate a dummy task group for the start node"""
        def dummy_function(**kwargs):
            return {"status": "dummy_complete"}
        
        task = Task(task_id=task_id, task_type="ephemeral", app_type="dummy")
        start_node = f"{task_id}_0"
        end_node = f"{task_id}_1"
        
        task.add_node(start_node, dummy_function, {})
        task.add_node(end_node, dummy_function, {})
        task.add_edge(start_node, end_node)
        
        return (task, start_node, end_node)