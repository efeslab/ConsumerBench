## Add DeepResearch class here
import sys
import os
import subprocess
import threading

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import src.utils as utils
import src.globals as globals

class LlamaCpp:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LlamaCpp, cls).__new__(cls)
                    cls.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return

        self.refcount = 0
        self.lock = threading.Lock()
        self.__initialized = True

    def launch_backend(self, *args, **kwargs):
        print("Launching LlamaCpp backend")
        api_port = kwargs.get('api_port', 8080)
        model = kwargs.get('model', f"{repo_dir}/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf")
        device = kwargs.get('device', "gpu")
        mps = kwargs.get('mps', 100)
        llama_cpp_path = kwargs.get('llamacpp_path', f"{repo_dir}/inference_backends/llama.cpp")

        with self.lock:
            self.refcount += 1
            if self.refcount > 1:
                print("LlamaCpp backend already running")
                return {"status": "backend_already_running"}

            utils.util_run_server_script_check_log(
                script_path=f"{repo_dir}/inference_backends/llamacpp_server.sh",
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

            print(f"LlamaCpp backend launched")
            return {"status": "backend_launched"}

    def cleanup_backend(self, *args, **kwargs):
        print("Cleaning up LlamaCpp backend")
        api_port = kwargs.get('api_port', 8080)
        with self.lock:
            self.refcount -= 1
            if self.refcount == 0:
                print("Cleaning up LlamaCpp backend")
                process = subprocess.Popen(
                    [f"{repo_dir}/scripts/cleanup.sh", str(api_port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                process.wait()
                return {"status": "backend_cleaned_up"}
            else:
                print("LlamaCpp backend still running")
                return {"status": "backend_still_running"}
    