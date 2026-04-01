import json
import os
import sys
import time
from typing import Any, Dict, List
from .Interaction import Container

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application


class OsTool(Application):
    """
    Application to run OS interaction tasks.
    Config should include:
      - tool: str, tool identifier (e.g. "bash")
      - data_file: str, path to the JSONL task file
      - num_runs: int, how many times to run each task
    """

    def __init__(self):
        super().__init__()
        self.tool = None
        self.num_runs = 1
        self.container = None
        self.dataset = []

    def run_setup(self, *args, **kwargs):
        print("OsBench setup")
        # self.tool = kwargs.get("tool", self.config["tool"])
        # self.num_runs = kwargs.get("num_runs", self.config["num_runs"])
        # data_file = kwargs.get("data_file", self.config["data_file"])

        tool_kwargs = self.config.get("tool_kwargs", {}) or self.config
    
        self.tool = kwargs.get("tool", tool_kwargs.get("tool"))
        self.num_runs = kwargs.get("num_runs", tool_kwargs.get("num_runs", 1))
        data_file = kwargs.get("data_file", tool_kwargs.get("data_file"))

        with open(data_file) as f:
            content = f.read()
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # fall back to JSONL (one object per line)
                data = [json.loads(line.strip()) for line in content.splitlines() if line.strip()]
        print("data file read")

        for entry in data:
            ans = entry.get("label", "")
            inp = entry
            self.dataset.append((inp, ans))

        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("OsBench cleanup")
        self.tool = None
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        if not self.dataset:
            self.run_setup(*args, **kwargs)

        entry, expected = self.dataset[0]
        image = entry.get("create", {}).get("local", "default")
        container = Container(f"local-os/{image}")
        self.container = container
        print(entry)

        init_cmd = entry.get("create", {}).get("init", "")
        if init_cmd:
            print(f"[init] {init_cmd}")
            container.execute(init_cmd)

        task_cmd = entry.get("task", "")
        print(f"[task] {task_cmd}")
        response = container.execute(task_cmd)
        print(response)

        eval_cmd = entry.get("evaluation", {}).get("eval", "")
        if eval_cmd:
            eval_response = container.execute(eval_cmd)
            print(f"[eval] {eval_response}")

        self.container.__del__()
        return {"status": "tool_complete", "result": response}

    def load_dataset(self, *args, **kwargs):
        print("OsBench loading dataset")
        return {"status": "dataset_loaded"}

    # def get_default_config(self) -> Dict[str, Any]:
    #     return {
    #         "tool": "bash",
    #         "num_runs": 1,
    #         "data_file": "data_ostool.jsonl",
    #     }