## Add DeepResearch class here
import time
from typing import Any, Dict
import sys
import os
from datasets import load_dataset
import subprocess
import requests
import json

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application
import src.utils as utils
import src.globals as globals
from inference_backends.Llamacpp import LlamaCpp

class Chatbot(Application):
    def __init__(self):
        super().__init__()
        self.chatbot_prompts = []
        self.backend = LlamaCpp()

    def run_setup(self, *args, **kwargs):
        print("Chatbot setup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        model = kwargs.get('model', self.get_default_config()['model'])
        device = kwargs.get('device', self.get_default_config()['device'])
        mps = kwargs.get('mps', self.get_default_config()['mps'])
        llamacpp_path = kwargs.get('llamacpp_path', self.get_default_config()['llamacpp_path'])

        self.backend.launch_backend(api_port=api_port, model=model, device=device, mps=mps, llamacpp_path=llamacpp_path)
        print(f"Chatbot setup complete")

        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("Chatbot cleanup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])

        self.backend.cleanup_backend(api_port=api_port)
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print(f"Chatbot application")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])

        chatbot_prompt = self.chatbot_prompts.pop(0)
        chatbot_prompts = [chatbot_prompt]

        api_url = f"http://127.0.0.1:{api_port}/v1/completions"

        ttft = None

        start_time = time.time()

        for prompt in chatbot_prompts:
            payload = {
                "prompt": prompt,
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

        return {"status": "chatbot_complete", "ttft": end_time - start_time, "tpot": tpot, "itl": itl}

    def load_dataset(self, *args, **kwargs):
        """Load the chatbot dataset"""
        mcp_trace = kwargs.get('mcp_trace_json', None)
        if mcp_trace is not None:
            trace_json = json.loads(open(mcp_trace, 'r').read())
            for section_name, section_data in trace_json.items():
                if section_name == "text_generate":
                    for call_id, call_data in section_data.items():
                        prompt = call_data.get('prompt', None)
                        if prompt is not None:
                            self.chatbot_prompts.append(prompt)
        else:
            ds_textgen = load_dataset("lmsys/lmsys-chat-1m")
            ds_textgen = ds_textgen["train"]
            ds_textgen = ds_textgen.shuffle(seed=42)
            ds_textgen = ds_textgen.select(range(0, 100))
            for item in ds_textgen:
                self.chatbot_prompts.append(item['conversation'][0]['content'])

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": f"{repo_dir}/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf",
            "device": "gpu",
            "mps": 100,
            "api_port": 8080,
            "llamacpp_path": f"{repo_dir}/inference_backends/llama.cpp",
            "dataset": f"lmsys/lmsys-chat-1m"
        }
    