# LiveVideo — real-time video understanding on a bare llama.cpp VLM

LiveVideo treats a video file as if it were a live camera feed. It walks the video in
fixed-length chunks, pulls a handful of frames out of each chunk, and sends those frames to a
vision model as one chat request. The question it answers is simple:

> **Can the model describe an N-second chunk in less than N seconds?**

If yes, the stream is sustainable. If no, work arrives faster than it is finished and the
backlog grows without bound — which is exactly what the `lag` column in the output shows.

## How this differs from VSS

| | VSS | LiveVideo |
|---|---|---|
| Input | a finished file, uploaded | the same file, replayed at real-time speed |
| Who chunks it | the VSS agent, internally | this app, at a size you set |
| What runs | VSS containers + VST + agent + LLM + VLM | one `llama-server` process |
| What you measure | one end-to-end summarization time | per-chunk latency against a per-chunk deadline |

There is no agent, no video storage service, and no LLM writing a final summary. That makes
LiveVideo a measurement of **the vision model on this hardware**, not of a video-analytics
product. It is the cheap baseline you want before deciding whether a full streaming stack is
worth deploying.

## Prerequisites

**1. llama.cpp built with CUDA** (shared with the other applications):

```bash
cd inference_backends/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

**2. `ffmpeg` and `ffprobe` on `PATH`** — used to cut frames out of the video.

```bash
sudo apt-get install -y ffmpeg
```

**3. The Qwen3-VL-8B GGUF and its vision projector.** Both files are required: the `mmproj`
projector is what turns images into embeddings the text model can attend over. Without it
`llama-server` loads a text-only model and rejects every request that carries an image.

```bash
mkdir -p models/Qwen3-VL-8B-Instruct-GGUF && cd models/Qwen3-VL-8B-Instruct-GGUF
wget https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf
wget https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf
```

(These are the same two files `configs/workflow_vss_remote_vlm.yml` already points at, so if
you have run VSS with `use_remote_vlm: true` you have them.)

## Running

```bash
# LiveVideo alone
python src/scripts/run_consumerbench.py --config configs/workflow_livevideo.yml

# LiveVideo contending with a chatbot on the same GPU
python src/scripts/run_consumerbench.py --config configs/workflow_chatbot_livevideo.yml
```

## What it does, step by step

`run_setup` starts one `llama-server` with the model and its projector, then polls `/health`
until it answers. `run_cleanup` kills it by port. Each `run_application` call is **one
streaming session**:

1. Work out the chunk list — `session_duration / chunk_duration` chunks, looping back to the
   start of the file when it runs out (`loop_video`).
2. Send one untimed warm-up chunk, so the vision encoder's cold start doesn't land on chunk 0.
3. For each chunk:
   - Under `pacing: realtime`, wait until the moment the chunk would have finished arriving
     from a camera. If the previous chunk already overran, there is nothing to wait for and
     the run falls behind immediately.
   - `ffmpeg` cuts `frames_per_chunk` evenly spaced JPEGs out of that slice of video.
   - The frames go to `/v1/chat/completions` as one message: a text prompt followed by one
     `image_url` part per frame, base64-encoded.
   - Record how long the whole thing took.

## What it measures

Per chunk (written to `results/livevideo_<port>_session<N>_chunks.csv`):

| Column | Meaning |
|---|---|
| `extract_time` | seconds in ffmpeg |
| `vlm_time` | seconds in the VLM request |
| `processing_time` | the two together — **the number that must beat `chunk_duration`** |
| `real_time_factor` | `processing_time / chunk_duration`; below 1.0 is sustainable |
| `kept_up` | whether this chunk beat its deadline |
| `lag` | how far behind the live edge the caption landed. Under real-time pacing this **compounds**: a model at 1.5x budget falls a further half-chunk behind every chunk |
| `prompt_tokens` | image tokens dominate this; it is the main thing `frames_per_chunk` and `frame_height` control |

Per session, returned to the benchmark and written to `..._summary.json`: `chunks_kept_up`,
`kept_up_fraction`, `mean_real_time_factor`, mean/median/p95/max `processing_time`,
`max_lag`, `final_lag`. The captions themselves go to `..._captions.log` so you can check the
model is actually describing the video rather than emitting plausible filler.

## Tuning

The knobs that matter, in order:

- **`frames_per_chunk`** — the real cost dial. Every frame is hundreds of image tokens, and
  prefill over those tokens is usually what breaks real time. 8 frames of a 10s chunk means
  one frame per 1.25s of video.
- **`frame_height`** — token cost scales with area, so halving the height cuts tokens roughly
  fourfold. 448 is a reasonable floor for Qwen3-VL.
- **`chunk_duration`** — the deadline itself. Longer chunks give the model more time but make
  each caption cover more video, and lengthen the delay before anything is reported at all.
- **`mps`** — this server's share of GPU SMs, for sharing the GPU with another application.
- **`ctx`** — total context. With `parallel: 1` the whole of it belongs to each request. The
  defaults land near 4K tokens per chunk, so 131072 is deliberate headroom for raising frames
  or resolution; a request that exceeds the context fails outright rather than degrading.

**`pacing`** picks what you are measuring:

- `realtime` — waits for each chunk to "arrive". A slow model visibly falls behind, and `lag`
  grows across the session. Use this for anything about sustainability or contention.
- `asap` — processes chunks back to back. Measures the same per-chunk cost in far less wall
  clock, but cannot show a growing backlog. Use it to sweep `frames_per_chunk` quickly.

## Config reference

| Key | Default | Meaning |
|---|---|---|
| `model` | `models/Qwen3-VL-8B-Instruct-GGUF/...Q4_K_M.gguf` | VLM weights |
| `mmproj` | `models/Qwen3-VL-8B-Instruct-GGUF/mmproj-...F16.gguf` | vision projector; required |
| `llamacpp_path` | `inference_backends/llama.cpp` | llama.cpp checkout |
| `api_port` | `8083` | not 8080 (Chatbot), 8081 (VSS LLM) or 8082 (VSS remote VLM) |
| `device` / `mps` | `gpu` / `100` | placement and share of GPU SMs |
| `ctx` / `parallel` | `131072` / `1` | total context and slot count |
| `chunk_duration` | `10.0` | seconds of video per request — and the deadline |
| `frames_per_chunk` | `8` | frames sampled from each chunk |
| `frame_height` | `448` | frames scaled to this height, width follows aspect |
| `jpeg_quality` | `3` | ffmpeg `-q:v`, lower is better quality |
| `max_tokens` | `256` | caption length cap |
| `prompt` | "Describe what happens in this clip…" | asked of every chunk |
| `pacing` | `realtime` | `realtime` or `asap` |
| `session_duration` | `0` | seconds of feed per session; 0 = one pass over the file |
| `loop_video` | `True` | restart the file when it runs out |
| `warmup` | `True` | spend one untimed chunk on the encoder's cold start |
| `request_timeout` | `300` | per-request cap, so a wedged chunk fails instead of hanging |
| `video_path` | `applications/VSS/sample_video_1.mp4` | the source |
| `num_requests` | `1` | number of streaming sessions |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Setup raises "Timed out waiting for llama-server" | model still loading, or CUDA OOM | check `results/server_logs/livevideo_vlm_server_stderr_<port>.log` |
| Every chunk fails with an HTTP 400/500 | the server has no vision support | confirm `--mmproj` reached it — the log should mention the projector at load |
| Requests fail once frames or resolution go up | chunk exceeds the context | raise `ctx`, or lower `frames_per_chunk` / `frame_height` |
| `real_time_factor` far above 1.0 | too many image tokens per chunk | drop `frames_per_chunk` first, then `frame_height` |
| ffmpeg errors on every chunk | codec missing, or the path is not a video | test by hand: `ffprobe <video>` |
| Captions are generic and never mention what is on screen | the frames may be near-duplicates or black | inspect `..._captions.log` and raise `frames_per_chunk` |
