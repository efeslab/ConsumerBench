# VSS on DGX Spark — Local Setup Guide

Fully local deployment of NVIDIA VSS (Video Search and Summarization) on a DGX Spark, using
**llama.cpp** (`llama-server`) as the LLM backend, running as a plain process on the same machine.

No remote GPU or cloud inference is used — everything runs on-device.

---

## Architecture overview

- **LLM**: a GGUF model served by `llama-server` from ConsumerBench's `inference_backends/llama.cpp`
  checkout, running standalone on the Spark (no container)
- **VLM**: `Cosmos-Reason2-8B`, deployed automatically by VSS itself as a NIM container (no manual
  container needed)
- **VSS Agent**: orchestrates VLM + LLM + embeddings + CV pipeline, exposes REST API + Web UI
- Both models share the Spark's single GB10 GPU and 128GB unified memory — memory budgeting
  between them is the main operational concern (see Troubleshooting).

---

## 1. Prerequisites

- DGX Spark (ARM64 / Blackwell GB10 GPU) with DGX OS, Docker, and NVIDIA Container Toolkit installed
- Verify the NVIDIA Docker runtime is registered:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If this fails with `unknown or invalid runtime name: nvidia`, install/configure the toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 2. Set up API keys

Two distinct NVIDIA keys may come up — for this fully-local setup, you only need the first one.

| Key | Purpose | Needed here? |
|---|---|---|
| `NGC_API_KEY` | Authenticates Docker pulls from `nvcr.io` and lets NIM containers download model weights | **Yes** — VSS's own images and the Cosmos-Reason2-8B VLM still come from `nvcr.io`, even though the LLM no longer does |
| `NVIDIA_API_KEY` | Used for NVIDIA's *hosted cloud* inference (`integrate.api.nvidia.com`) | No — not used in a local-only setup |

Get an NGC API key at [ngc.nvidia.com](https://ngc.nvidia.com) → Setup → API Keys.

```bash
export NGC_API_KEY='your_ngc_api_key'
export NGC_CLI_API_KEY="$NGC_API_KEY"   # dev-profile.sh expects this name

# Authenticate Docker against the NVIDIA registry
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

---

## 3. Start the cache cleaner (recommended)

DGX Spark's unified memory architecture (UMA) can show memory pressure even within nominal
capacity. This script helps offset that during heavy container operations.

```bash
sudo tee /usr/local/bin/sys-cache-cleaner.sh << 'EOF'
#!/bin/bash
set -e
echo 0 | tee /proc/sys/vm/nr_hugepages
echo "Starting cache cleaner — Ctrl+C to stop"
while true; do
  sync && echo 3 | tee /proc/sys/vm/drop_caches > /dev/null
  sleep 3
done
EOF
sudo chmod +x /usr/local/bin/sys-cache-cleaner.sh

# Run in a separate terminal (or append & to background it)
sudo sh /usr/local/bin/sys-cache-cleaner.sh
```

---

## 4. Start the LLM (llama.cpp)

Build llama.cpp with CUDA support first, if you haven't already (from the ConsumerBench root):

```bash
cd inference_backends/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Then start the server with ConsumerBench's shared launcher, which passes `--jinja` for every
application:

```bash
cd /path/to/ConsumerBench

inference_backends/llamacpp_server.sh \
  inference_backends/llama.cpp \
  8081 8081 \
  models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf \
  gpu 50
```

Or invoke `llama-server` directly if you'd rather not use the wrapper:

```bash
cd inference_backends/llama.cpp
build/bin/llama-server \
  --port 8080 \
  -m /path/to/model.gguf \
  -ngl 99 --parallel 8 -c 4096 \
  --jinja
```

Under ConsumerBench this step is automatic — `VSS.run_setup` calls the shared `LlamaCpp` backend.
See "Running under ConsumerBench" below.

**Notes:**
- `--jinja` — **required**, and passed unconditionally by `inference_backends/llamacpp_server.sh`.
  VSS's agent uses OpenAI-style tool/function calling internally and
  sends `tool_choice: "auto"` on requests. llama.cpp only supports tools on its Jinja
  chat-template path: without this flag it falls back to a hardcoded legacy template that has no
  notion of a `tools` field, and the server rejects requests carrying `tools`/`tool_choice`
  outright. It's the llama.cpp counterpart to NIM's `--enable-auto-tool-choice` +
  `--tool-call-parser` pair — one flag instead of two, because llama.cpp infers the tool-call
  format from the template rather than being told which parser to use.
- **The GGUF must have a tool-capable chat template.** Because the tool-call format comes from
  the model's own embedded `tokenizer.chat_template`, `--jinja` only helps if that template
  actually renders tools. Some quantized builds strip or simplify it — Llama-3.2-3B-Instruct
  GGUFs in particular vary between publishers. If yours lacks tool support, supply a correct
  template with `--chat-template-file` in `llamacpp_server.sh` rather than
  assuming any GGUF will work. Verify with the tool-calling curl in Troubleshooting before
  running a full VSS workload.
- `-ngl 99` offloads all layers to the GPU. Drop it (or set `device=cpu` in the wrapper, which
  uses `-ngl 0`) to run on CPU and leave the whole GPU to the VLM.
- `mps 50` sets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, capping the LLM's share of GPU SMs so it can
  coexist with VSS's Cosmos-Reason2-8B VLM. This is the llama.cpp analogue of NIM's
  `--gpu-memory-utilization`, but note it partitions *compute*, not memory — llama.cpp's memory
  footprint is governed by the model size, `-ngl`, and `-c` instead. Shrink `-c` or use a smaller
  quant if you hit OOM.
- `-c 4096` is the shared launcher's fixed context size. VSS agent requests typically use well
  under 10–20K tokens, so this is usually fine; raise it in `llamacpp_server.sh` if you hit
  truncation.

### Verify the LLM is healthy

```bash
# llama.cpp's readiness endpoint is /health, not NIM's /v1/health/ready
curl -s http://localhost:8080/health
curl -s http://localhost:8080/v1/models | jq
```

`/health` returns `{"status":"ok"}` once slots are free. In the server log, the line to wait for
is `update_slots: all slots are idle`.

Note the exact model name string returned by `/v1/models` — you'll need it verbatim in later
steps. (`VSS.run_setup` reads it from this endpoint automatically.)

---

## 5. Launch VSS

Clone the repo (first time only):

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
cd video-search-and-summarization
```

Get your Spark's IP (`localhost` inside VSS's containers refers to the container, not the host
where `llama-server` is running):

```bash
hostname -I
```

### Limit the VLM's KV-cache (recommended when sharing the GPU with the LLM)

VSS deploys Cosmos-Reason2-8B locally by default, and by default it will try to reserve a large
chunk of GPU memory for its KV-cache — enough to collide with the separately-running llama.cpp
LLM server from Step 4. Cap it via a `--vlm-env-file`.

**Note:** `--vlm-env-file` expects a path to an actual file, not an inline `KEY=VALUE` string.

```bash
cat > ~/vlm-shared.env << 'EOF'
NIM_KVCACHE_PERCENT=0.3
EOF
```

Adjust the value based on available headroom — start conservative (0.3–0.4) and raise it only if
`nvidia-smi` shows spare memory after both models are running. If the VLM fails to start with a
memory-related error at a very low value, you may also need `NIM_RELAX_MEM_CONSTRAINTS=1` in the
same file to override its default minimum-memory check.

Launch VSS, pointing it at your local llama.cpp server and passing the VLM env file:

```bash
export LLM_ENDPOINT_URL=http://<SPARK_IP>:8080

bash deploy/docker/scripts/dev-profile.sh up -p base -H DGX-SPARK \
  --use-remote-llm \
  --llm <exact_model_name_from_/v1/models> \
  --vlm-env-file ~/vlm-shared.env
```

This will take several minutes on cold start — Cosmos-Reason2-8B specifically compiles a
TensorRT-LLM engine on first boot (~8–9 minutes is normal on DGX Spark). Watch progress with
`docker ps` / `docker logs` if it appears stalled.

### Verify VSS is up

```bash
curl -I http://localhost:3000
```

Open `http://localhost:3000` (or `http://<SPARK_IP>:3000`) in a browser to confirm the Agent UI loads.

### Tear down

```bash
bash deploy/docker/scripts/dev-profile.sh down
```

⚠️ This destroys processed video data and analysis results.

---

## 6. Health check — confirm LLM and VLM are both ready before making any calls

Both the LLM (your standalone llama.cpp server) and the VLM (Cosmos-Reason2-8B, deployed by VSS
itself) expose a health endpoint — but **at different paths**: llama.cpp serves `/health`, while
the VLM, still a NIM container, serves `/v1/health/ready`. Ports are `8080` for the LLM and
`30082` for the VLM in the base profile. **Always check both before uploading a video or issuing a
summarize/chat request** — an unready VLM in particular is easy to mistake for a hung upload,
since Cosmos-Reason2-8B can take 8–9 minutes to finish compiling on cold start.

### One-off check

```bash
echo "LLM:" && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/health
echo "VLM:" && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:30082/v1/health/ready
```

Both should return `200`. Anything else (connection refused, `000`, `5xx`) means that backend
isn't ready yet — wait and recheck rather than proceeding.

### Polling script (blocks until both are ready, or times out)

```bash
#!/bin/bash
LLM_URL="http://localhost:8080/health"
VLM_URL="http://localhost:30082/v1/health/ready"
MAX_WAIT=900   # seconds (15 min — generous for Cosmos-Reason2 cold start)
INTERVAL=10
elapsed=0

check_ready() {
  curl -s -o /dev/null -w "%{http_code}" "$1"
}

echo "Waiting for LLM and VLM to become ready..."
while true; do
  llm_status=$(check_ready "$LLM_URL")
  vlm_status=$(check_ready "$VLM_URL")

  if [[ "$llm_status" == "200" && "$vlm_status" == "200" ]]; then
    echo "✅ Both LLM and VLM are ready."
    break
  fi

  if (( elapsed >= MAX_WAIT )); then
    echo "❌ Timed out after ${MAX_WAIT}s waiting for readiness (LLM: $llm_status, VLM: $vlm_status)."
    echo "Check the llama.cpp server log and 'docker logs' on the VLM before retrying."
    exit 1
  fi

  echo "  ...LLM: $llm_status, VLM: $vlm_status (waited ${elapsed}s)"
  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
done
```

Save this as `wait_for_ready.sh`, `chmod +x wait_for_ready.sh`, and run it after launching VSS
(Step 5) and before Step 7 (upload video). Only proceed once it prints the ✅ success line.

---

## 7. Upload a video file

The `base` profile deployed in Step 5 runs the **agent** service (`vss-agent`), a different
service from VSS's older "video-summarization" (VIA) REST API — the two expose different
endpoints, and only the agent's are reachable here. The agent's port is `8000` by default (confirm
your actual port via VSS startup logs / `dev-profile.sh` output). Uploading is a three-step
handshake, mirroring what the Agent UI itself does:

1. `POST /api/v1/videos` with the filename → returns a VST upload URL.
2. `POST` the raw file bytes to that URL → returns VST's `sensorId` (and other metadata).
3. `POST /api/v1/videos/{sensor_id}/complete` with that metadata → finalizes ingest and returns
   the `filename` the agent will use to refer to the clip.

### curl

```bash
FILENAME="my_video.mp4"

# 1. Get the upload URL
UPLOAD_URL=$(curl -s -X POST http://localhost:8000/api/v1/videos \
  -H "Content-Type: application/json" \
  -d "{\"filename\": \"${FILENAME}\"}" | jq -r '.url')

# 2. Upload the file bytes
VST_RESPONSE=$(curl -s -X POST "$UPLOAD_URL" -F "file=@${FILENAME};type=video/mp4")
SENSOR_ID=$(echo "$VST_RESPONSE" | jq -r '.sensorId')

# 3. Complete the upload (forward VST's response verbatim)
curl -s -X POST "http://localhost:8000/api/v1/videos/${SENSOR_ID}/complete" \
  -H "Content-Type: application/json" \
  -d "$VST_RESPONSE"
```

### Python

```python
import requests

VSS_BASE = "http://localhost:8000"
video_path = "my_video.mp4"
filename = "my_video.mp4"

# 1. Get the upload URL
url_resp = requests.post(f"{VSS_BASE}/api/v1/videos", json={"filename": filename}, timeout=60)
url_resp.raise_for_status()
upload_url = url_resp.json()["url"]

# 2. Upload the file bytes
with open(video_path, "rb") as f:
    vst_resp = requests.post(upload_url, files={"file": (filename, f, "video/mp4")}, timeout=600)
vst_resp.raise_for_status()
vst_info = vst_resp.json()
sensor_id = vst_info["sensorId"]

# 3. Complete the upload (forward VST's response verbatim)
complete = requests.post(
    f"{VSS_BASE}/api/v1/videos/{sensor_id}/complete", json=vst_info, timeout=600
)
complete.raise_for_status()
video_name = complete.json()["filename"]
print(sensor_id, video_name)
```

Save the returned `filename` (what the agent calls the clip internally) — you'll refer to it by
name, not by ID, in the chat message below.

---

## 8. Run an example request

The agent doesn't expose separate `/summarize` and `/chat/completions` REST endpoints — it's a
conversational agent (built on the NeMo Agent Toolkit) that picks its own tools from a natural-
language message. Both summarization and Q&A go through the same chat endpoint; the only
difference is what you ask for, and you must name the clip in the message since the agent has
nothing else to resolve "this video" against.

### Option A — Chat (agent REST API, curl)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_message": "Summarize what happens in the video my_video.mp4."
  }'
```

Swap the message for a question (`"What is happening in the video my_video.mp4?"`) to do Q&A
instead of summarization — same endpoint. A streaming variant is available at
`/generate/stream` (server-sent events).

### Option B — Direct LLM sanity check (OpenAI SDK, bypasses VSS entirely)

Useful for testing the raw llama.cpp server independent of VSS.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",    # your local llama-server, NOT NVIDIA's cloud
    api_key="not-needed"                     # llama-server doesn't enforce this by default
)

completion = client.chat.completions.create(
    model="<exact_model_name_from_/v1/models>",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    temperature=0.2,
    max_tokens=64,
    stream=False
)

print(completion.choices[0].message.content)
```

> ⚠️ Do not point `base_url` at `https://integrate.api.nvidia.com` — that routes to NVIDIA's
> hosted cloud infrastructure, not your local llama.cpp server, defeating the point of this setup.

### Getting exact request shapes for any endpoint

The agent service is a FastAPI app, so its live OpenAPI docs (generated from your running
instance) are the source of truth for the full route list: `http://<VSS_API_ENDPOINT>/docs`.

> Note: the repo also ships a `via_client_cli.py` (`services/video-summarization/src/`) with a
> `summarize --print-curl-command` mode, but it targets the older `video-summarization` (VIA)
> service's `/files` / `/summarize` API — a different service, run under a different profile
> (`bp_developer_lvs_2d`), not the `base` profile used here. It won't produce working requests
> against the agent deployed in Step 5.

---

## Running under ConsumerBench

Steps 4–8 above are what `VSS.py` automates. The class is the **video agent only** — it launches
its LLM through the shared `LlamaCpp` backend (`inference_backends/Llamacpp.py`), the same one
Chatbot and DeepResearch use, rather than managing a server itself. Its `llm_port` defaults to
`8081` so the video agent runs its own model, separate from a chatbot's server on `8080`.

It assumes it is running on the Spark: the hardware profile is fixed at `DGX-SPARK`. There is no
`hostname -I` / IP-discovery step and no `host_ip` to configure — `vss-agent`'s compose service
runs with `network_mode: host`, so it shares the Spark's loopback interface directly, and
`run_setup` points `LLM_ENDPOINT_URL` at `http://127.0.0.1:{llm_port}` unconditionally.

```bash
# VSS alone
python src/scripts/run_consumerbench.py --config configs/workflow_vss.yml

# VSS contending with a chatbot on the same GPU
python src/scripts/run_consumerbench.py --config configs/workflow_chatbot_vss.yml

# VSS with the VLM hosted by a second llama.cpp server instead of the Cosmos-Reason2-8B NIM
python src/scripts/run_consumerbench.py --config configs/workflow_vss_remote_vlm.yml
```

### Optional: host the VLM with llama.cpp instead of Cosmos-Reason2-8B

By default `run_setup` deploys VSS's own Cosmos-Reason2-8B NIM as the VLM (Step 5's normal path,
including its ~8–9 minute cold-start TensorRT-LLM compile). Setting `use_remote_vlm: true` in the
workflow config switches to a second, ConsumerBench-managed llama.cpp server instead — the same
pattern as the LLM, but with `--mmproj` for vision, launched via
`inference_backends/llamacpp_vlm_server.sh` and passed to `dev-profile.sh up` as `--use-remote-vlm
--vlm <name>` instead of `--vlm <name> --vlm-env-file <file>`. This skips the NIM's cold-start
compile entirely, at the cost of running a second local model process.

This changes several things silently if you're comparing against the manual walkthrough above:
- `vlm_port` stops meaning the NIM's health port (`30082`) and instead means the llama.cpp VLM
  server's port.
- Readiness is checked at `/health` (llama.cpp) rather than `/v1/health/ready` (NIM).
- `--vlm-env-file` / `NIM_KVCACHE_PERCENT` don't apply — `dev-profile.sh` rejects that flag when
  the VLM is remote — so GPU-memory tuning for the VLM instead goes through the llama.cpp flags
  below (`vlm_mps`, `vlm_ctx`).

What each lifecycle method does:

| Method | Work |
|---|---|
| `run_setup` | `LlamaCpp.launch_backend`, read the model name from `/v1/models`, optionally launch a llama.cpp VLM server (`use_remote_vlm`), `dev-profile.sh up`, wait for the VLM |
| `run_application` | one video upload (the `/api/v1/videos` handshake) + one `/generate` chat request (repeated `num_requests` times) |
| `run_cleanup` | `dev-profile.sh down`, `LlamaCpp.cleanup_backend`, and (if `use_remote_vlm`) kill the VLM llama-server by port |

Tool calling needs no configuration: `llamacpp_server.sh` passes `--jinja` for every application.

---

## Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `unknown or invalid runtime name: nvidia` | NVIDIA Container Toolkit not registered with Docker | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `Access Denied` on image pull | Not logged into `nvcr.io`, or wrong image name | `docker login nvcr.io`; verify exact name via `ngc registry image list` |
| `exec format error` | Pulled an x86_64 image on ARM64 Spark | Find the `sbsa`/`arm64` tag, or confirm multi-arch support |
| Container stuck on "waiting..." | Normal cold-start compilation (esp. Cosmos-Reason2, ~8–9 min) | Check `docker logs -f` for active progress before assuming it's hung |
| `ValueError: Free memory ... less than desired GPU memory utilization` (VLM) | Two models competing for Spark's shared unified memory | Lower `NIM_KVCACHE_PERCENT` for the VLM, and/or shrink the LLM's `-c` / use a smaller quant / drop `-ngl` |
| llama.cpp CUDA OOM at load | GGUF + KV-cache don't fit alongside the VLM | Lower `-c`, use a smaller quantization, or run the LLM on CPU (`-ngl 0`) |
| VSS can't reach `LLM_ENDPOINT_URL` (manual walkthrough, Step 5) | Used `localhost` from inside a container that isn't on the host network | Use the Spark's actual IP (`hostname -I`), not `localhost`. (Under ConsumerBench this is a non-issue: `vss-agent` runs with `network_mode: host`, and `VSS.py` points `LLM_ENDPOINT_URL` at `127.0.0.1` directly.) |
| `--vlm-env-file` / `--llm-env-file` not applying settings | Passed inline `KEY=VALUE` instead of a file path | Write settings to an actual `.env` file and pass its path |
| Server error to the effect that `tools` / `tool_choice` requires `--jinja` (often surfaces on video upload/summarize) | VSS's agent sends `tool_choice: "auto"`, but llama-server was started without `--jinja` | Restart `llama-server` with `--jinja` |
| `--jinja` is set but tool calls still fail, or the model emits raw JSON instead of a parsed `tool_calls` field | The GGUF's embedded chat template has no tool support | Test with the tool-calling curl below; if it fails, pass a tool-capable template via `--chat-template-file`, or use a GGUF whose template renders tools |

### Tool-calling smoke test

Run this against the LLM directly before trusting it through VSS. A working setup returns a
`tool_calls` array in `choices[0].message`; a broken one either errors, or returns the call as raw
text in `content`.

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<exact_model_name_from_/v1/models>",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }' | jq '.choices[0].message'
```

---

## Key parameters reference

**LLM tuning (llama.cpp flags):**
- `--jinja` — required for tool calling; see Step 4
- `-c N` — context window; the main lever on the LLM's KV-cache memory footprint
- `-ngl N` — layers offloaded to GPU (`99` = all, `0` = CPU-only)
- `--parallel N` — concurrent request slots
- `--chat-template-file PATH` — override the GGUF's embedded chat template
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` — caps the LLM's share of GPU SMs (set from the wrapper's `mps` arg)

**Workflow-config keys (`configs/workflow_vss.yml`):**
| Key | Default | Meaning |
|---|---|---|
| `llm_model` | `models/Llama-3.2-3B-Instruct-GGUF/...f16.gguf` | GGUF path served by llama.cpp; its chat template must support tools |
| `vlm_model` | `nvidia/cosmos-reason2-8b` | VLM deployed by VSS (`--vlm`); ignored when `use_remote_vlm` is set |
| `llm_port` | `8081` | llama-server port — not `8080`, so VSS runs its own model |
| `vlm_port` | `30082` | VLM's readiness-check port. Means the NIM's `/v1/health/ready` port by default; means the llama.cpp VLM server's port when `use_remote_vlm` is set |
| `vss_port` | `8000` | VSS agent's REST API |
| `mps` | `50` | LLM's share of GPU SMs; the rest is left for the VLM |
| `llamacpp_path` | `inference_backends/llama.cpp` | llama.cpp checkout |
| `vss_repo_dir` | `applications/VSS/video-search-and-summarization` | VSS repo checkout |
| `use_remote_vlm` | `False` | if `true`, host the VLM with a second llama.cpp server instead of the Cosmos-Reason2-8B NIM — see "Optional: host the VLM with llama.cpp" above |
| `vlm_llamacpp_model` | `models/Qwen3-VL-8B-Instruct-GGUF/...Q4_K_M.gguf` | GGUF path for the remote VLM (only used when `use_remote_vlm` is set) |
| `vlm_mmproj` | `models/Qwen3-VL-8B-Instruct-GGUF/mmproj-...F16.gguf` | vision projector for the remote VLM |
| `vlm_mps` | `50` | remote VLM's share of GPU SMs |
| `vlm_ctx` | `131072` | total context across the remote VLM server's 2 parallel slots (`--parallel 2`), so each slot effectively gets half; a single `video_understanding` tool call can run to ~41K tokens of image tokens, so keep this well above 2× that |
| `video_path`, `prompt`, `num_requests` | — | the workload |

Everything else is fixed at the top of `VSS.py` — `DEVICE`, `VSS_PROFILE`, `VSS_HARDWARE`,
`VLM_ENV_FILE`, `VLM_KVCACHE_PERCENT`, `READINESS_MAX_WAIT`, `UPLOAD_TIMEOUT`, `SUMMARIZE_TIMEOUT`,
`UPLOAD_READY_MAX_WAIT`. Edit those constants if you need to deviate from the DGX Spark deployment.

**Memory-tuning env vars (VLM, when it's still a NIM container — i.e. `use_remote_vlm` unset):**
- `NIM_KVCACHE_PERCENT` — fraction of GPU memory reserved for KV cache (default 0.9, or 0.75 on DGX Spark). Passed via `--vlm-env-file`, since the VLM is managed by `dev-profile.sh` rather than being a container you launch directly.
- `NIM_RELAX_MEM_CONSTRAINTS=1` — required if pushing memory settings below NIM's expected minimum

**Passing settings to each model:**
- **LLM** (llama.cpp, your own process): command-line flags, or the wrapper env vars above
- **VLM, NIM mode** (managed by `dev-profile.sh`): must go in a real file passed via `--vlm-env-file /path/to/file.env` — inline `KEY=VALUE` strings are not accepted and will silently fail or error
- **VLM, `use_remote_vlm` mode**: the same llama.cpp flags as the LLM (`vlm_mps`, `vlm_ctx`), since it's just another llama-server process

**Agent `/generate` and `/generate/stream` request body:**
- `input_message` — the only field. A natural-language instruction; name the clip by its uploaded
  `filename` so the agent's tool-selection has something to resolve (e.g. `"Summarize the video
  my_video.mp4."`) — `VSS.run_application` does this substitution automatically for any prompt
  containing the literal phrase `"this video"`.

**`/api/v1/videos` upload-handshake params:**
- `POST /api/v1/videos` — body: `{"filename": "..."}`; returns `{"url": "<VST upload URL>"}`
- `POST <that url>` — multipart file upload; returns VST's response, including `sensorId`
- `POST /api/v1/videos/{sensor_id}/complete` — body: VST's response forwarded verbatim; returns
  `{"filename": "...", "sensor_id": "...", ...}` — the `filename` is what you reference in chat
  messages