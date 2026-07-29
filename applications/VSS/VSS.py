## Add VSS class here
## Video Search and Summarization (NVIDIA VSS) application.
##
## This is the video agent only. The LLM runs through the shared LlamaCpp
## backend — the same one Chatbot and DeepResearch use — rather than being
## managed here. Implements README step 5 (launch VSS), step 6 (VLM readiness),
## step 7 (upload a video) and step 8 (summarize).
##
## Assumes it is running on the DGX Spark itself: the hardware profile is fixed,
## and VSS's agent container shares the host network namespace, so it reaches
## llama-server over loopback.
import time
from typing import Any, Dict
import sys
import os
import subprocess

import requests

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application
from inference_backends.Llamacpp import LlamaCpp

# Fixed setup — not part of the workflow config. Change these here if you need
# to deviate from the DGX Spark deployment this application targets.
DEVICE = "gpu"
VSS_PROFILE = "base"
VSS_HARDWARE = "DGX-SPARK"
# Caps the VLM's KV-cache so it can share the GPU with llama-server.
VLM_ENV_FILE = os.path.expanduser("~/vlm-shared.env")
VLM_KVCACHE_PERCENT = 0.3
# Cosmos-Reason2-8B compiles a TensorRT-LLM engine on first boot (~8-9 min).
READINESS_MAX_WAIT = 900
# Request timeouts. Summarizing runs the VLM over every chunk and then loops
# through the agent's plan/tool cycle, so it is minutes, not seconds.
UPLOAD_TIMEOUT = 600
SUMMARIZE_TIMEOUT = 1800


class VSS(Application):
    def __init__(self):
        super().__init__()
        # Videos to process, populated by load_dataset().
        self.videos = []
        self.backend = LlamaCpp()
        # Name the LLM reports at /v1/models; resolved during setup.
        self.llm_model_name = None

    # ------------------------------------------------------------------ #
    # Step 5 (VSS stack) + Step 6 (VLM readiness)
    # ------------------------------------------------------------------ #
    def run_setup(self, *args, **kwargs):
        print("VSS setup")
        cfg = self.get_default_config()

        llm_port = kwargs.get('llm_port', cfg['llm_port'])
        llm_model = self._abspath(kwargs.get('llm_model', cfg['llm_model']))
        vlm_model = kwargs.get('vlm_model', cfg['vlm_model'])
        vlm_port = kwargs.get('vlm_port', cfg['vlm_port'])
        mps = kwargs.get('mps', cfg['mps'])
        llamacpp_path = self._abspath(kwargs.get('llamacpp_path', cfg['llamacpp_path']))
        vss_repo_dir = self._abspath(kwargs.get('vss_repo_dir', cfg['vss_repo_dir']))

        # ---- LLM: llama.cpp server via the shared backend ---------------
        # VSS's agent sends OpenAI tool_choice: "auto", which llama.cpp only
        # supports on the Jinja chat-template path — llamacpp_server.sh passes
        # --jinja for every application, so there is nothing to configure here.
        self.backend.launch_backend(
            api_port=llm_port,
            model=llm_model,
            device=DEVICE,
            mps=mps,
            llamacpp_path=llamacpp_path,
        )
        self.llm_model_name = self._discover_model_name(llm_port)
        print(f"LLM reports model name: {self.llm_model_name}")

        # ---- Step 5: launch VSS via dev-profile.sh ----------------------
        with open(VLM_ENV_FILE, 'w') as f:
            f.write(f"NIM_KVCACHE_PERCENT={VLM_KVCACHE_PERCENT}\n")

        env = os.environ.copy()
        # The agent container runs with network_mode: host, so it shares the
        # host's loopback and reaches llama-server at 127.0.0.1. The Spark's
        # routable IP does *not* work here: llama-server binds loopback only
        # unless started with --host, and the shared launcher doesn't pass it.
        env["LLM_ENDPOINT_URL"] = f"http://127.0.0.1:{llm_port}"
        vss_cmd = [
            "bash", os.path.join(vss_repo_dir, "deploy/docker/scripts/dev-profile.sh"),
            "up", "-p", VSS_PROFILE, "-H", VSS_HARDWARE,
            "--use-remote-llm",
            "--llm", self.llm_model_name,
            "--vlm", vlm_model,
            "--vlm-env-file", VLM_ENV_FILE,
        ]
        print(f"Launching VSS (vlm={vlm_model})")
        subprocess.run(vss_cmd, cwd=vss_repo_dir, env=env, check=True)

        # ---- Step 6: block until the VLM is ready -----------------------
        # The LLM's readiness is already handled by launch_backend, which waits
        # on the llama.cpp server log.
        self._wait_for_vlm(vlm_port, READINESS_MAX_WAIT)

        print("VSS setup complete")
        return {"status": "setup_complete", "config": self.config}

    @staticmethod
    def _abspath(path):
        """Resolve a config path against the repo root.

        Workflow configs write paths relative to the repo (e.g.
        models/.../model.gguf), but llamacpp_server.sh cds into the llama.cpp
        checkout before loading the model, so a relative path never resolves.
        """
        return path if os.path.isabs(path) else os.path.join(repo_dir, path)

    def _discover_model_name(self, llm_port):
        """Read the model name llama.cpp advertises, for VSS's --llm / requests."""
        resp = requests.get(f"http://localhost:{llm_port}/v1/models", timeout=10)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]

    def _wait_for_vlm(self, vlm_port, max_wait, interval=10):
        """Poll the VLM's /v1/health/ready until it returns 200.

        The VLM is a NIM container deployed by VSS itself, and Cosmos-Reason2-8B
        can take 8-9 minutes to compile on cold start.
        """
        vlm_url = f"http://localhost:{vlm_port}/v1/health/ready"
        elapsed = 0
        print("Waiting for the VLM to become ready...")
        while True:
            vlm_status = self._http_status(vlm_url)
            if vlm_status == 200:
                print("VLM is ready.")
                return
            if elapsed >= max_wait:
                raise RuntimeError(
                    f"Timed out after {max_wait}s waiting for VLM readiness "
                    f"(last status: {vlm_status})."
                )
            print(f"  ...VLM: {vlm_status} (waited {elapsed}s)")
            time.sleep(interval)
            elapsed += interval

    @staticmethod
    def _http_status(url):
        try:
            return requests.get(url, timeout=5).status_code
        except requests.RequestException:
            return None

    # ------------------------------------------------------------------ #
    # Step 7 (upload) + Step 8 (summarize)
    # ------------------------------------------------------------------ #
    def run_application(self, *args, **kwargs):
        print("VSS application")
        cfg = self.get_default_config()

        vss_port = kwargs.get('vss_port', cfg['vss_port'])
        prompt = kwargs.get('prompt', cfg['prompt'])

        video_path = self._abspath(self.videos.pop(0))
        vss_base = f"http://localhost:{vss_port}"
        filename = os.path.basename(video_path)

        start_time = time.time()

        # ---- Step 7: upload the video file ------------------------------
        # Uploading is a three-part handshake: ask the agent where to put the
        # video, POST the bytes to that VST endpoint, then tell the agent the
        # upload finished so it registers the clip as a sensor.
        url_resp = requests.post(
            f"{vss_base}/api/v1/videos", json={"filename": filename}, timeout=60
        )
        url_resp.raise_for_status()
        upload_url = url_resp.json()["url"]

        with open(video_path, "rb") as f:
            vst_resp = requests.post(
                upload_url,
                files={"file": (filename, f, "video/mp4")},
                timeout=UPLOAD_TIMEOUT,
            )
        vst_resp.raise_for_status()
        vst_info = vst_resp.json()
        sensor_id = vst_info["sensorId"]

        complete = requests.post(
            f"{vss_base}/api/v1/videos/{sensor_id}/complete",
            json=vst_info,
            timeout=UPLOAD_TIMEOUT,
        )
        complete.raise_for_status()
        # VST strips the extension; this is the name the agent knows it by.
        video_name = complete.json()["filename"]
        upload_time = time.time() - start_time
        print(f"Uploaded {video_path} -> {video_name} ({upload_time:.2f}s)")

        # ---- Step 8: summarize the uploaded video -----------------------
        # The agent picks its own tools from a natural-language message, so the
        # prompt has to name the clip — "this video" gives it nothing to resolve.
        if "this video" in prompt:
            input_message = prompt.replace("this video", f"the video {video_name}")
        else:
            input_message = f"{prompt} The video is {video_name}."

        summarize_start = time.time()
        summarize = requests.post(
            f"{vss_base}/generate",
            headers={"Content-Type": "application/json"},
            json={"input_message": input_message},
            timeout=SUMMARIZE_TIMEOUT,
        )
        summarize.raise_for_status()
        result = summarize.json()
        summarize_time = time.time() - summarize_start
        total_time = time.time() - start_time
        print(f"Summarize complete ({summarize_time:.2f}s)")

        return {
            "status": "vss_complete",
            "sensor_id": sensor_id,
            "video_name": video_name,
            "upload_time": upload_time,
            "summarize_time": summarize_time,
            "total_time": total_time,
            "response": result,
        }

    # ------------------------------------------------------------------ #
    # Tear down (README step 5 "Tear down" + release the LLM)
    # ------------------------------------------------------------------ #
    def run_cleanup(self, *args, **kwargs):
        print("VSS cleanup")
        cfg = self.get_default_config()
        vss_repo_dir = self._abspath(kwargs.get('vss_repo_dir', cfg['vss_repo_dir']))
        llm_port = kwargs.get('llm_port', cfg['llm_port'])

        subprocess.run(
            ["bash", os.path.join(vss_repo_dir, "deploy/docker/scripts/dev-profile.sh"), "down"],
            cwd=vss_repo_dir, check=False,
        )
        self.backend.cleanup_backend(api_port=llm_port)

        return {"status": "cleanup_complete"}

    def load_dataset(self, *args, **kwargs):
        """Load the list of videos to process."""
        print("VSS loading dataset")
        video_path = self.config.get("video_path", self.get_default_config()["video_path"])
        num_requests = self.config.get("num_requests", 1)
        self.videos = [video_path for _ in range(num_requests)]

    def get_default_config(self) -> Dict[str, Any]:
        return {
            # Models. llm_model is a GGUF path served by llama.cpp; any GGUF
            # works as long as its chat template supports tools, since VSS's
            # agent sends tool_choice: "auto". vlm_model is deployed by VSS.
            "llm_model": f"{repo_dir}/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf",
            "vlm_model": "nvidia/cosmos-reason2-8b",
            # Ports. llm_port is not 8080, so the video agent runs its own model
            # rather than landing on a Chatbot's shared server.
            "llm_port": 8081,
            "vlm_port": 30082,
            "vss_port": 8000,
            # Share of GPU SMs given to llama-server, leaving the rest for the VLM.
            "mps": 50,
            # Repo directories.
            "llamacpp_path": f"{repo_dir}/inference_backends/llama.cpp",
            "vss_repo_dir": f"{repo_dir}/applications/VSS/video-search-and-summarization",
            # Workload.
            "video_path": f"{repo_dir}/applications/VSS/my_video.mp4",
            "prompt": "Summarize what happens in this video.",
            "num_requests": 1,
        }
