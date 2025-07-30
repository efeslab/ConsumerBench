## Add LiveCaptions class here
import time
from typing import Any, Dict
import sys
import os
import subprocess
import re

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application
import src.utils as utils
import src.globals as globals

class LiveCaptions(Application):
    def __init__(self):
        super().__init__()
        self.live_captions_paths = []

    def run_setup(self, *args, **kwargs):
        print("LiveCaptions setup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        device = kwargs.get('device', self.get_default_config()['device'])
        mps = kwargs.get('mps', self.get_default_config()['mps'])

        utils.util_run_server_script_check_log(
            script_path=f"{repo_dir}/applications/LiveCaptions/whisper_online_server.sh",
            server_dir=f"{repo_dir}/applications/LiveCaptions",
            stdout_log_path=f"{globals.get_results_dir()}/whisper_online_server_stdout",
            stderr_log_path=f"{globals.get_results_dir()}/whisper_online_server_stderr",
            stderr_ready_patterns=["Listening on"],
            stdout_ready_patterns=[],
            listen_port=api_port,
            api_port=api_port,
            model=None,
            device=device,
            mps=mps
        )

        print(f"LiveCaptions setup complete")

        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("LiveCaptions cleanup")
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
        print(f"LiveCaptions application")
        live_captions_path = kwargs.get('client_command_file', self.get_default_config()['client_command_file'])
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])

        stdout_log = os.path.join(globals.get_results_dir(), f"live_captions_client_stdout_{api_port}.log")
        stderr_log = os.path.join(globals.get_results_dir(), f"live_captions_client_stderr_{api_port}.log")

        # Start the server process with log file redirection
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            process = subprocess.Popen(
                [f"{repo_dir}/applications/LiveCaptions/whisper_online_client.sh", str(api_port), str(live_captions_path), f"{repo_dir}/applications/LiveCaptions"],
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,  # Important for server processes
            )

        start_time = time.time()
        process.wait()
        end_time = time.time()
        result = {}

        # Parse the stdout log to get the Processing Time
        with open(stdout_log, 'r') as f:
            chunk_idx = 0
            for line in f:
                if "Processing time" in line:
                    processing_time = re.search(r"Processing time: (\d+\.\d+)", line)
                    if processing_time:
                        processing_time = float(processing_time.group(1))
                        result[f'processing time_chunk_{chunk_idx}'] = processing_time
                        print(f"Processing Time: {processing_time:.4f} seconds")
                        chunk_idx += 1

        result["total time"] = end_time - start_time
        result["status"] = "live_captions_complete"

        return result

    def load_dataset(self, *args, **kwargs):
        """Load the live captions dataset"""

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "device": "gpu",
            "mps": 100,
            "api_port": 5000,
            "client_command_file": f"{repo_dir}/applications/LiveCaptions/whisper-earnings21/4320211_chunk_001.wav"
        }
    