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

        self.servers = {}  # port -> refcount
        self.ready_events = {}  # port -> threading.Event signalled once the server is ready
        self.lock = threading.Lock()
        self.__initialized = True

    def __deepcopy__(self, memo):
        # Singleton: deepcopy should return the same instance
        return self

    def launch_backend(self, *args, **kwargs):
        print("Launching LlamaCpp backend")
        api_port = kwargs.get('api_port', 8080)
        model = kwargs.get('model', f"{repo_dir}/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf")
        # llamacpp_server.sh does `cd $server_dir` before launching llama-server, so a
        # relative model path (e.g. from a YAML "model: models/...") would resolve
        # against the llama.cpp dir and fail to load. Resolve relative paths against
        # the repo root here so configs can use repo-relative paths interchangeably.
        if not os.path.isabs(model):
            model = os.path.join(repo_dir, model)
        device = kwargs.get('device', "gpu")
        mps = kwargs.get('mps', 100)
        llama_cpp_path = kwargs.get('llamacpp_path', f"{repo_dir}/inference_backends/llama.cpp")

        # Only hold the lock briefly for refcount check; release before blocking launch
        # so that servers on different ports can start concurrently.
        with self.lock:
            if api_port not in self.servers:
                self.servers[api_port] = 0
                self.ready_events[api_port] = threading.Event()
            self.servers[api_port] += 1
            ready_event = self.ready_events[api_port]
            is_first_launcher = self.servers[api_port] == 1
            if not is_first_launcher:
                # Server is being (or has been) launched by another concurrent
                # workflow node. Wait for it to become ready before returning so
                # callers don't fire requests against a still-loading model
                # (which returns HTTP 503 "Loading model").
                print(f"LlamaCpp backend already starting/running on port {api_port}; waiting for readiness")
                ready_event.wait()
                print(f"LlamaCpp backend ready on port {api_port}")
                return {"status": "backend_already_running"}

        # Launch server without holding the lock (blocking wait for readiness).
        # Always signal the ready event afterwards (even on failure) so concurrent
        # launchers waiting on this port can never block forever.
        try:
            utils.util_run_server_script_check_log(
                script_path=f"{repo_dir}/inference_backends/llamacpp_server.sh",
                server_dir=f"{llama_cpp_path}",
                stdout_log_path=f"llamacpp_server_stdout",
                stderr_log_path=f"llamacpp_server_stderr",
                stderr_ready_patterns=["update_slots: all slots are idle"],
                stdout_ready_patterns=[],
                listen_port=api_port,
                api_port=api_port,
                model=model,
                device=device,
                mps=mps
            )
        finally:
            # Unblock any concurrent launchers waiting on this port.
            ready_event.set()

        print(f"LlamaCpp backend launched on port {api_port}")
        return {"status": "backend_launched"}

    def cleanup_backend(self, *args, **kwargs):
        print("Cleaning up LlamaCpp backend")
        api_port = kwargs.get('api_port', 8080)
        with self.lock:
            if api_port not in self.servers:
                print(f"No LlamaCpp backend on port {api_port}")
                return {"status": "no_backend"}
            self.servers[api_port] -= 1
            if self.servers[api_port] == 0:
                del self.servers[api_port]
                self.ready_events.pop(api_port, None)
                print(f"Cleaning up LlamaCpp backend on port {api_port}")
                process = subprocess.Popen(
                    [f"{repo_dir}/scripts/cleanup.sh", str(api_port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                process.wait()
                return {"status": "backend_cleaned_up"}
            else:
                print(f"LlamaCpp backend still running on port {api_port} (refcount={self.servers[api_port]})")
                return {"status": "backend_still_running"}
    