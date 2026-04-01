import json
import os
import signal
import sys
import time
import subprocess
import urllib.request
from typing import Any, Dict

from datasets import load_dataset
import src.globals as globals

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application

class ChatbotHF(Application):
    def __init__(self):
        super().__init__()
        self.chatbot_prompts = []
        self.server_process = None
        self.server_stdout = None
        self.server_stderr = None
        self._setup_scheduler = None
        self.api_port = None

    def _stop_server_process(self):
        if not self.server_process:
            return

        try:
            if self.server_process.poll() is None:
                # Kill the entire process group because tally wrapping may add
                # intermediate shell processes.
                os.killpg(self.server_process.pid, signal.SIGTERM)
                self.server_process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.server_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            self.server_process = None

    def _read_log_tail(self, path, max_lines=40):
        if not path or not os.path.exists(path):
            return ""
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:])
        except Exception:
            return ""

    def _wait_server_ready(self, timeout_secs=120):
        if self.api_port is None:
            raise RuntimeError("API port is not set")

        deadline = time.time() + timeout_secs
        health_url = f"http://127.0.0.1:{self.api_port}/health"
        while time.time() < deadline:
            if self.server_process and self.server_process.poll() is not None:
                stderr_tail = self._read_log_tail(
                    os.path.join(globals.get_results_dir(), f"chatbothf_server_{self.api_port}_stderr.log")
                )
                raise RuntimeError(
                    f"ChatbotHF server process exited early on port {self.api_port} "
                    f"(rc={self.server_process.returncode}).\n"
                    f"stderr tail:\n{stderr_tail}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(1)

        stderr_tail = self._read_log_tail(
            os.path.join(globals.get_results_dir(), f"chatbothf_server_{self.api_port}_stderr.log")
        )
        raise TimeoutError(
            f"ChatbotHF server did not become healthy on port {self.api_port}.\n"
            f"stderr tail:\n{stderr_tail}"
        )

    def _post_json(self, url, payload, timeout=120):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)

    def run_setup(self, *args, **kwargs):
        print("Chatbot (HuggingFace) setup")

        model_id = kwargs.get("model", self.get_default_config()["model"])
        device = kwargs.get("device", self.get_default_config()["device"])
        api_port = int(kwargs.get("api_port", self.get_default_config()["api_port"]))
        mps = int(kwargs.get("mps", self.get_default_config()["mps"]))
        torch_dtype_name = "float16"
        trust_remote_code = False
        hf_token = os.environ.get("HF_TOKEN")
        scheduler = kwargs.get("scheduler", self.get_default_config()["scheduler"])
        priority = int(kwargs.get("priority", self.get_default_config()["priority"]))

        scheduler = scheduler.lower() if isinstance(scheduler, str) else None
        if scheduler not in {None, "naive", "tally", "tgs"}:
            raise ValueError(f"Unsupported scheduler for ChatbotHF: {scheduler}")

        if isinstance(model_id, str) and model_id.lower().endswith(".gguf"):
            raise ValueError(
                "ChatbotHF expects a Hugging Face model id or directory, not a GGUF file. "
                "Use Chatbot for GGUF/llama.cpp backends."
            )

        self._setup_scheduler = scheduler
        self.api_port = api_port

        server_script = os.path.join(repo_dir, "applications", "ChatbotHF", "backend.py")
        server_cmd = [
            sys.executable,
            "-u",
            server_script,
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
            "--model",
            model_id,
            "--device",
            device,
            "--torch-dtype",
            str(torch_dtype_name),
        ]
        if trust_remote_code:
            server_cmd.append("--trust-remote-code")
        if hf_token:
            server_cmd.extend(["--hf-token", hf_token])

        env = os.environ.copy()
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
        env["PRIORITY"] = str(priority)
        launch_cmd = server_cmd

        results_dir = globals.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        self.server_stdout = open(os.path.join(results_dir, f"chatbothf_server_{api_port}_stdout.log"), "w")
        self.server_stderr = open(os.path.join(results_dir, f"chatbothf_server_{api_port}_stderr.log"), "w")

        self.server_process = subprocess.Popen(
            launch_cmd,
            stdout=self.server_stdout,
            stderr=self.server_stderr,
            text=True,
            env=env,
            start_new_session=True,
        )

        try:
            self._wait_server_ready(timeout_secs=360)
        except Exception:
            self._stop_server_process()

            stderr_path = os.path.join(results_dir, f"chatbothf_server_{api_port}_stderr.log")
            stderr_tail = self._read_log_tail(stderr_path)
            raise RuntimeError(
                f"ChatbotHF setup failed on port {api_port}. "
                f"stderr log: {stderr_path}\n"
                f"stderr tail:\n{stderr_tail}"
            )

        print("Chatbot (HuggingFace) setup complete")
        return {
            "status": "setup_complete",
            "config": self.config,
            "api_port": api_port,
            "scheduler": scheduler,
            "priority": priority,
            "mps": mps,
        }

    def run_cleanup(self, *args, **kwargs):
        print("Chatbot (HuggingFace) cleanup")

        self._stop_server_process()

        if self.server_stdout:
            self.server_stdout.close()
            self.server_stdout = None
        if self.server_stderr:
            self.server_stderr.close()
            self.server_stderr = None

        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print("Chatbot (HuggingFace) application")

        if self.server_process is None or self.server_process.poll() is not None:
            raise RuntimeError("Server is not running. Call run_setup before run_application.")
        if not self.chatbot_prompts:
            raise RuntimeError("No prompts available. Call load_dataset before run_application.")

        chatbot_prompt = self.chatbot_prompts.pop(0)
        max_new_tokens = 215
        temperature = 0
        top_p = 0.9
        seed = 141293

        api_url = f"http://127.0.0.1:{self.api_port}/generate"
        response = self._post_json(
            api_url,
            {
                "prompt": chatbot_prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            },
            timeout=300,
        )

        if "error" in response:
            raise RuntimeError(f"ChatbotHF server generation failed: {response['error']}")

        return {
            "status": "chatbot_complete",
            "ttft": response.get("ttft"),
            "tpot": response.get("tpot"),
            "itl": response.get("itl"),
            "total time": response.get("total_time"),
            "token_count": response.get("token_count"),
        }

    def load_dataset(self, *args, **kwargs):
        """Load the chatbot dataset"""
        mcp_trace = kwargs.get("mcp_trace_json", None)
        if mcp_trace is not None:
            with open(mcp_trace, "r") as f:
                trace_json = json.load(f)
            for section_name, section_data in trace_json.items():
                if section_name == "text_generate":
                    for _, call_data in section_data.items():
                        prompt = call_data.get("prompt", None)
                        if prompt is not None:
                            self.chatbot_prompts.append(prompt)
        else:
            try:
                ds_textgen = load_dataset("lmsys/lmsys-chat-1m")
                ds_textgen = ds_textgen["train"]
                ds_textgen = ds_textgen.shuffle(seed=42)
                ds_textgen = ds_textgen.select(range(0, 100))
                for item in ds_textgen:
                    self.chatbot_prompts.append(item["conversation"][0]["content"])
            except Exception:
                fallback = [
                    "Explain the concept of machine learning in simple terms.",
                    "Write a Python function to sort a list of numbers.",
                    "What are the main differences between TCP and UDP?",
                    "Describe the architecture of a transformer neural network.",
                    "How does garbage collection work in Java?",
                    "Explain the CAP theorem in distributed systems.",
                    "Write a SQL query to find duplicate rows in a table.",
                    "What is the difference between a process and a thread?",
                    "Explain how HTTPS encryption works step by step.",
                    "Describe the benefits and drawbacks of microservices architecture.",
                ]
                self.chatbot_prompts = fallback * 10

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "device": "gpu",
            "api_port": 8010,
            "mps": 100,
            "scheduler": None,
            "priority": 1,
        }



