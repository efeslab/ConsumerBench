## Add DeepResearch class here
import time
from typing import Any, Dict
import sys
import os
import subprocess

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application
import src.utils as utils
import src.globals as globals

class DeepResearch(Application):
    def __init__(self):
        super().__init__()
        self.deep_research_prompts = []

    def run_setup(self, *args, **kwargs):
        print("DeepResearch setup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        model = kwargs.get('model', self.get_default_config()['server_model'])
        device = kwargs.get('device', self.get_default_config()['device'])
        mps = kwargs.get('mps', self.get_default_config()['mps'])
        llama_cpp_path = kwargs.get('llamacpp_path', self.get_default_config()['llamacpp_path'])

        utils.util_run_server_script_check_log(
            script_path=f"{repo_dir}/scripts/inference_backends/llamacpp_server.sh",
            server_dir=f"{llama_cpp_path}",
            stdout_log_path=f"{globals.get_results_dir()}/llamacpp_server_stdout",
            stderr_log_path=f"{globals.get_results_dir()}/llamacpp_server_stderr",
            stderr_ready_patterns=["update_slots: all slots are idle"],
            stdout_ready_patterns=[],
            listen_port=api_port,
            api_port=api_port,
            model=model,
            device=device,
            mps=mps
        )

        print(f"DeepResearch setup complete")

        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("DeepResearch cleanup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        process = subprocess.Popen(
                        [f"{repo_dir}/scripts/cleanup.sh", str(api_port)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
        process.wait()
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print(f"DeepResearch application")
        deep_research_prompt = self.deep_research_prompts.pop(0)
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        model = kwargs.get('model', self.get_default_config()['client_model'])

        stdout_log = os.path.join(globals.get_results_dir(), f"deep_research_client_stdout_{api_port}.log")
        stderr_log = os.path.join(globals.get_results_dir(), f"deep_research_client_stderr_{api_port}.log")

        start_time = time.time()
        # Start the server process with log file redirection
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            process = subprocess.Popen(
                [f"{repo_dir}/scripts/applications/deep_research_client.sh", f"{repo_dir}/applications/DeepResearch/smolagents/examples/open_deep_research", str(api_port), str(model), str(deep_research_prompt)],
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,  # Important for server processes
            )
            process.wait()
        
        return {"status": "deep_research_complete", "total time": time.time() - start_time}

    def load_dataset(self):
        """Load the deep research dataset"""
        self.deep_research_prompts.append("What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?")

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "server_model": f"{repo_dir}/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf",
            "device": "gpu",
            "mps": 100,
            "api_port": 8080,
            "llamacpp_path": f"{repo_dir}/inference_backends/llama.cpp",
            "client_model": f"openai/meta-llama/Llama-3.2-3B-Instruct"
        }
    