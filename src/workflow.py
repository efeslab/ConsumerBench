from collections import deque
import yaml
import sys

sys.path.append("/home/cc/os-llm")


from benchmark_v3 import DAGScheduler, Task
# [ROHAN: This file also should not have explicit application names. But if we absolutely need it, we can require that users add a line here for their application maybe. ]
from handles import (
    # chatbot func
    setup_llamacpp_server,
    run_textgen,
    cleanup_llamacpp_server,
    # deep_research func
    # also use setup_llamacpp_server,
    run_deep_research,
    # also use cleanup_llamacpp_server,
    # imagegen func
    setup_imagegen,
    run_imagegen,
    cleanup_imagegen,
    # live_captions func
    setup_whisper_online,
    run_whisper_online,
    cleanup_whisper_online,
    # sleep func
    nothing_function,
    sleep_function,

    # shadow func
    shadow_function
)

import networkx as nx

NODE_TYPE_CONFIG_MAPPING_INDEX = {
    "chatbot":       0,
    "deep_research": 1,
    "imagegen":      2,
    "live_captions":  3,
    "sleep":         4
}

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

    def load_yaml(self):
        # Load the YAML file and remove comments
        cleaned_yaml = self._remove_config_comments(self.yaml_file)
        return yaml.safe_load(cleaned_yaml)
    
    def load_workflow_unit_config(self):
        for k, v in self.workflow_config.items():
            if k == "workflows":
                continue

            type = v["type"]
            node_config = {k: val for k, val in v.items() if k != "type"}
            final_node_config = self.load_node_config(node_config, type)
            self.workflow_unit_map[k] = {
                "type": type,
                "node_config": final_node_config,
                "count": 0
            }

        workflows = self.workflow_config.get("workflows", {})
        for k, v in workflows.items():
            id = v["uses"]
            self.workflow_unit_map[id]["count"] += 1

    def generate_task_queue(self):
        """Generate a task queue based on the workflow unit map."""
        for k, v in self.workflow_unit_map.items():
            if v["count"] == 0:
                continue
            count = v["count"]
            type = v["type"]
            node_config = v["node_config"]

            self.tasks_map_queue[k] = deque()
            for i in range(count):
                task_id = f"{k}_u{i}"
                task, start_node, end_node = self._generate_task_group(
                    task_id=task_id,
                    app_type=type,
                    num_requests=node_config.get("num_requests", 1),
                    listen_port=node_config.get("listen_port", None),
                    api_port=node_config.get("api_port", None),
                    server_model=node_config.get("server_model", None),
                    client_model=node_config.get("client_model", None),
                    client_command_file=node_config.get("client_command_file", None),
                    mps=node_config.get("mps", 100),
                    setup_func=node_config.get("setup_func", None),
                    run_func=node_config.get("run_func", None),
                    cleanup_func=node_config.get("cleanup_func", None),
                    device=node_config.get("device", "gpu")
                )
                self.tasks_map_queue[k].append(WorkflowUnit(type, task, start_node, end_node))


    # [ROHAN: here as well, can we move out the application-specific stuff to applications/ directory? I am not sure how to automatically get the node_type and stuff, maybe there is a way? Or again, if it is impossible or too much of a hassle, we can maybe ask users to add their application in this function].
    def load_node_config(self, node_config, node_type):
        default_args = set_default_args()
        node_defaults_args = default_args[NODE_TYPE_CONFIG_MAPPING_INDEX[node_type]]
        if node_type == "chatbot":
            node_defaults_args = {
                **node_defaults_args,
                "setup_func": setup_llamacpp_server,
                "run_func": run_textgen,
                "cleanup_func": cleanup_llamacpp_server,
                "listen_port": 5001,
                "api_port": 5000,
            }
        elif node_type == "deep_research":
            node_defaults_args = {
                **node_defaults_args,
                "setup_func": setup_llamacpp_server,
                "run_func": run_deep_research,
                "cleanup_func": cleanup_llamacpp_server,
                "api_port": 5000,
            }
        elif node_type == "imagegen":
            node_defaults_args = {
                **node_defaults_args,
                "setup_func": setup_imagegen,
                "run_func": run_imagegen,
                "cleanup_func": cleanup_imagegen,
            }
        elif node_type == "live_captions":
            node_defaults_args = {
                **node_defaults_args,
                "setup_func": setup_whisper_online,
                "run_func": run_whisper_online,
                "cleanup_func": cleanup_whisper_online,
                "api_port": 5050,
            }
        elif node_type == "sleep":
            node_defaults_args = {
                **node_defaults_args,
                "setup_func": nothing_function,
                "run_func": sleep_function,
                "cleanup_func": nothing_function,
            }

        return {**node_defaults_args, **node_config}
        
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
        start_task, start_node, _ = self._generate_task_group(
            "start", "start",
            setup_func=shadow_function, run_func=shadow_function, cleanup_func=shadow_function
        )
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
        
        
    def _generate_task_group(self, task_id, app_type, num_requests = 1, listen_port = 5000,
                api_port = 5001, server_model = None, client_model = None,
                client_command_file = None, mps = 100,
                setup_func = None, run_func = None,
                cleanup_func = None, device = "gpu"):
        task = Task(task_id=task_id, task_type="emphemeral", app_type=app_type)
        
        start_node = f"{task_id}_{0}"
        end_node = f"{task_id}_{num_requests + 1}"

        task.add_node(start_node, setup_func, {
            "listen_port": listen_port,
            "api_port": api_port,
            "model": server_model,
            "device": device,
            "mps": mps
        })

        # [ROHAN: maybe we don't need command_file and so on]
        for i in range(1, num_requests + 1):
            task.add_node(f"{task_id}_{i}", run_func, {
                "command_file": client_command_file,
                "api_port": api_port,
                "model": client_model
            })

        task.add_node(f"{task_id}_{num_requests+1}", cleanup_func, {
            "api_port": api_port
        })

        for i in range(num_requests + 1):
            task.add_edge(f"{task_id}_{i}", f"{task_id}_{i+1}")

        return (task, start_node, end_node)
    


# [We don't need main method here. We are currently also just calling workflow.py from within benchmark_v2.py. This was just added for testing.]
# if __name__ == "__main__":
#     workflow = Workflow("/home/cc/os-llm/configs/workflow_test.yml")
#     workflow.load_workflow_unit_config()
#     workflow.generate_task_queue()
#     for k, v in workflow.tasks_map_queue.items():
#         print(f"Task group {k}:")
#         for unit in v:
#             print(f"  - {unit.type} (ID: {unit.id})")
#             print(f"    Start node: {unit.node_start}")
#             print(f"    End node: {unit.node_end}")
#     bm = workflow.generate_benchmark()
#     print("Benchmark generated successfully.")
#     bm.visualize()
#     print("Benchmark visualization generated successfully.")