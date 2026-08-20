## Add LiveVideo class here
## Real-time video understanding against a bare llama.cpp VLM server.
##
## Where VSS uploads a finished file and asks an agent to summarize it,
## LiveVideo treats the video as a live feed: it walks the video in fixed
## chunk_duration-second chunks, pulls a handful of frames out of each chunk,
## and sends those frames to the VLM as one chat request. The question it
## answers is whether the model keeps up -- whether a chunk is described in
## less time than the chunk itself lasts.
##
## There is no VSS stack here, no agent, no video storage service: just ffmpeg
## and one llama.cpp server. That makes this a measurement of the vision model
## on this hardware, not of a video-analytics product.
import base64
import csv
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application
import src.globals as globals

# Fixed setup -- not part of the workflow config.
# llama.cpp reports readiness on /health once its slots are free. Loading an 8B
# VLM plus its projector off disk is the slow part; the vision encoder itself
# is warmed separately by the first (untimed) request.
READINESS_MAX_WAIT = 600
READINESS_INTERVAL = 5
# ffmpeg should never take anywhere near this long for a few seconds of video;
# the cap exists so a wedged decode fails the chunk instead of the session.
FFMPEG_TIMEOUT = 120


class LiveVideo(Application):
    def __init__(self):
        super().__init__()
        # Populated by load_dataset().
        self.video_path = None
        self.video_duration = None
        # Name the VLM reports at /v1/models; resolved during setup.
        self.model_name = None
        # Increments per run_application so per-session output files don't collide.
        self.session_idx = 0

    # ------------------------------------------------------------------ #
    # Setup: one llama.cpp server hosting the VLM
    # ------------------------------------------------------------------ #
    def run_setup(self, *args, **kwargs):
        print("LiveVideo setup")
        cfg = self.get_default_config()

        api_port = kwargs.get('api_port', cfg['api_port'])
        model = self._abspath(kwargs.get('model', cfg['model']))
        mmproj = self._abspath(kwargs.get('mmproj', cfg['mmproj']))
        device = kwargs.get('device', cfg['device'])
        mps = kwargs.get('mps', cfg['mps'])
        ctx = kwargs.get('ctx', cfg['ctx'])
        parallel = kwargs.get('parallel', cfg['parallel'])
        llamacpp_path = self._abspath(kwargs.get('llamacpp_path', cfg['llamacpp_path']))

        for label, path in (("model", model), ("mmproj", mmproj)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"LiveVideo {label} not found: {path}")
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("LiveVideo needs ffmpeg and ffprobe on PATH")

        log_dir = os.path.join(globals.get_results_dir(), "server_logs")
        os.makedirs(log_dir, exist_ok=True)
        stdout_log = os.path.join(log_dir, f"livevideo_vlm_server_stdout_{api_port}.log")
        stderr_log = os.path.join(log_dir, f"livevideo_vlm_server_stderr_{api_port}.log")

        # Launched directly rather than through util_run_server_script_check_log:
        # that helper passes a fixed argument list and scrapes logs for a ready
        # pattern, and this server needs two extra arguments (mmproj, ctx) plus
        # a much longer readiness window than the helper's 120s cap allows for
        # an 8B VLM. Polling /health is also a more direct readiness signal than
        # matching a log line.
        with open(stdout_log, 'w') as out, open(stderr_log, 'w') as err:
            subprocess.Popen(
                [
                    f"{repo_dir}/applications/LiveVideo/llamacpp_vlm_server.sh",
                    str(llamacpp_path), str(api_port), str(model), str(mmproj),
                    str(device), str(mps), str(ctx), str(parallel),
                ],
                stdout=out,
                stderr=err,
                start_new_session=True,
            )

        self._wait_for_server(api_port, READINESS_MAX_WAIT)
        self.model_name = self._discover_model_name(api_port)
        print(f"LiveVideo VLM reports model name: {self.model_name}")
        print(f"LiveVideo setup complete (ctx={ctx} across {parallel} slot(s))")

        return {"status": "setup_complete", "config": self.config}

    # ------------------------------------------------------------------ #
    # One streaming session
    # ------------------------------------------------------------------ #
    def run_application(self, *args, **kwargs):
        """Walk the video once as if it were a live feed, one chunk at a time.

        Each iteration is: wait until the chunk would have finished arriving,
        cut frames out of it, ask the VLM to describe it, and record how long
        that took relative to the chunk's own duration.
        """
        print("LiveVideo application")
        cfg = self.get_default_config()

        api_port = kwargs.get('api_port', cfg['api_port'])
        chunk_duration = float(kwargs.get('chunk_duration', cfg['chunk_duration']))
        frames_per_chunk = int(kwargs.get('frames_per_chunk', cfg['frames_per_chunk']))
        frame_height = int(kwargs.get('frame_height', cfg['frame_height']))
        jpeg_quality = int(kwargs.get('jpeg_quality', cfg['jpeg_quality']))
        prompt = kwargs.get('prompt', cfg['prompt'])
        max_tokens = int(kwargs.get('max_tokens', cfg['max_tokens']))
        session_duration = float(kwargs.get('session_duration', cfg['session_duration']))
        pacing = kwargs.get('pacing', cfg['pacing'])
        loop_video = bool(kwargs.get('loop_video', cfg['loop_video']))
        warmup = bool(kwargs.get('warmup', cfg['warmup']))
        request_timeout = float(kwargs.get('request_timeout', cfg['request_timeout']))

        if pacing not in ("realtime", "asap"):
            raise ValueError(f"pacing must be 'realtime' or 'asap', got {pacing!r}")
        if chunk_duration <= 0 or frames_per_chunk <= 0:
            raise ValueError("chunk_duration and frames_per_chunk must be positive")

        session = self.session_idx
        self.session_idx += 1
        api_url = f"http://127.0.0.1:{api_port}/v1/chat/completions"

        chunk_offsets = self._plan_chunks(chunk_duration, session_duration, loop_video)
        if not chunk_offsets:
            raise RuntimeError(
                f"No chunks to process: video is {self.video_duration:.1f}s and "
                f"chunk_duration is {chunk_duration}s"
            )
        print(
            f"LiveVideo session {session}: {len(chunk_offsets)} chunks of "
            f"{chunk_duration}s ({frames_per_chunk} frames each), pacing={pacing}"
        )

        # The first request also pays for the vision encoder's one-off setup,
        # which would otherwise land entirely on chunk 0 and skew the session.
        if warmup:
            print("  warming up the VLM with one untimed chunk...")
            with tempfile.TemporaryDirectory(prefix="livevideo_warmup_") as tmpdir:
                frames = self._extract_frames(
                    chunk_offsets[0], chunk_duration, frames_per_chunk,
                    frame_height, jpeg_quality, tmpdir,
                )
                self._call_vlm(api_url, frames, prompt, max_tokens, request_timeout)

        records: List[Dict[str, Any]] = []
        failures = 0
        session_start = time.time()

        for i, source_offset in enumerate(chunk_offsets):
            # The moment this chunk would have finished arriving from a live
            # camera. Under 'asap' pacing it is still recorded, so the lag
            # column shows what the same run would have looked like live.
            available_at = session_start + (i + 1) * chunk_duration
            if pacing == "realtime":
                sleep_for = available_at - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)

            started = time.time()
            record: Dict[str, Any] = {
                "chunk": i,
                "source_offset": round(source_offset, 3),
                "available_at": round(available_at - session_start, 3),
                "started_at": round(started - session_start, 3),
            }

            try:
                with tempfile.TemporaryDirectory(prefix="livevideo_") as tmpdir:
                    frames = self._extract_frames(
                        source_offset, chunk_duration, frames_per_chunk,
                        frame_height, jpeg_quality, tmpdir,
                    )
                    extracted = time.time()
                    caption, usage = self._call_vlm(
                        api_url, frames, prompt, max_tokens, request_timeout
                    )
                finished = time.time()

                processing_time = finished - started
                record.update({
                    "extract_time": round(extracted - started, 4),
                    "vlm_time": round(finished - extracted, 4),
                    "processing_time": round(processing_time, 4),
                    # The headline number: processing a chunk must cost less
                    # than the chunk's own duration, or the backlog grows.
                    "real_time_factor": round(processing_time / chunk_duration, 4),
                    "kept_up": processing_time < chunk_duration,
                    # How far behind the live edge the caption landed.
                    "lag": round(finished - available_at, 4),
                    "frames": len(frames),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "error": "",
                    "caption": caption,
                })
                print(
                    f"  chunk {i:3d} @{source_offset:7.1f}s  "
                    f"{processing_time:6.2f}s ({record['real_time_factor']:.2f}x budget)  "
                    f"lag {record['lag']:+7.2f}s  "
                    f"{'ok' if record['kept_up'] else 'BEHIND'}"
                )
            except Exception as e:
                failures += 1
                finished = time.time()
                record.update({
                    "extract_time": None,
                    "vlm_time": None,
                    "processing_time": round(finished - started, 4),
                    "real_time_factor": None,
                    "kept_up": False,
                    "lag": round(finished - available_at, 4),
                    "frames": 0,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "error": str(e),
                    "caption": "",
                })
                print(f"  chunk {i:3d} FAILED: {e}")

            records.append(record)

        session_time = time.time() - session_start
        result = self._summarize(records, chunk_duration, session_time, failures)
        self._write_session_files(records, result, api_port, session)

        mean_time = result.get("mean_processing_time")
        print(
            f"LiveVideo session {session} complete: "
            f"{result['chunks_kept_up']}/{result['chunks_total']} chunks kept up, "
            + (
                f"mean {mean_time}s per {chunk_duration}s chunk"
                if mean_time is not None
                else "no chunk completed"
            )
        )
        return result

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def run_cleanup(self, *args, **kwargs):
        print("LiveVideo cleanup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        subprocess.run(
            [f"{repo_dir}/scripts/cleanup.sh", str(api_port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        return {"status": "cleanup_complete"}

    def load_dataset(self, *args, **kwargs):
        """Resolve the video and measure how long it is."""
        print("LiveVideo loading dataset")
        cfg = self.get_default_config()
        video_path = self._abspath(self.config.get("video_path", cfg["video_path"]))
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"LiveVideo video not found: {video_path}")
        self.video_path = video_path
        self.video_duration = self._probe_duration(video_path)
        print(f"LiveVideo source: {video_path} ({self.video_duration:.1f}s)")

    # ------------------------------------------------------------------ #
    # Chunk planning
    # ------------------------------------------------------------------ #
    def _plan_chunks(self, chunk_duration, session_duration, loop_video) -> List[float]:
        """Return the source offset (seconds into the file) of each chunk.

        A session either runs for session_duration seconds of feed or, when
        that is 0, for one pass over the file. Looping restarts at the top
        whenever a whole chunk no longer fits before the end -- the seam is a
        visual discontinuity, not a timing one, so it does not affect what is
        being measured.
        """
        if session_duration > 0:
            n_chunks = int(session_duration // chunk_duration)
        else:
            n_chunks = int(self.video_duration // chunk_duration)

        offsets = []
        cursor = 0.0
        for _ in range(max(n_chunks, 0)):
            if cursor + chunk_duration > self.video_duration:
                if not loop_video:
                    break
                cursor = 0.0
            offsets.append(cursor)
            cursor += chunk_duration
        return offsets

    # ------------------------------------------------------------------ #
    # Frames in, caption out
    # ------------------------------------------------------------------ #
    def _extract_frames(self, start, duration, n_frames, height, quality, tmpdir) -> List[str]:
        """Cut n_frames evenly spaced JPEGs out of [start, start+duration).

        Frames are scaled to a fixed height because image tokens, not wall
        clock, are the binding constraint on a chunk: token cost scales with
        frames_per_chunk x resolution, and blowing past the server's context
        fails the request outright.
        """
        pattern = os.path.join(tmpdir, "frame_%03d.jpg")
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            # -ss before -i seeks by keyframe before decoding, which is what
            # keeps extraction cheap relative to the VLM call.
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", self.video_path,
            "-vf", f"fps={n_frames / duration},scale=-2:{height}",
            "-frames:v", str(n_frames), "-q:v", str(quality),
            pattern,
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=FFMPEG_TIMEOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[:400]}")

        paths = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
        if not paths:
            raise RuntimeError(f"ffmpeg produced no frames for offset {start:.1f}s")

        frames = []
        for path in paths:
            with open(path, "rb") as f:
                frames.append(base64.b64encode(f.read()).decode("ascii"))
        return frames

    def _call_vlm(self, api_url, frames, prompt, max_tokens, timeout) -> Tuple[str, Dict]:
        """Send one chunk's frames to llama.cpp as a single chat request.

        llama.cpp takes images, not video, so a chunk is expressed as an
        ordered set of stills in one user message -- the same shape VSS's own
        OpenAI-compatible path uses when it talks to a non-NIM backend.
        """
        content = [{"type": "text", "text": prompt}]
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            })

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        resp = requests.post(
            api_url, json=payload, headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        caption = body["choices"][0]["message"]["content"].strip()
        return caption, body.get("usage", {}) or {}

    # ------------------------------------------------------------------ #
    # Measurement bookkeeping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _summarize(records, chunk_duration, session_time, failures) -> Dict[str, Any]:
        ok = [r for r in records if not r["error"]]
        times = [r["processing_time"] for r in ok]
        kept_up = sum(1 for r in ok if r["kept_up"])

        result: Dict[str, Any] = {
            "status": "livevideo_complete",
            "chunks_total": len(records),
            "chunks_failed": failures,
            "chunks_kept_up": kept_up,
            "kept_up_fraction": round(kept_up / len(ok), 4) if ok else 0.0,
            "chunk_duration": chunk_duration,
            "session_time": round(session_time, 3),
            "final_lag": records[-1]["lag"] if records else None,
        }

        if times:
            result.update({
                "mean_processing_time": round(statistics.mean(times), 4),
                "median_processing_time": round(statistics.median(times), 4),
                "p95_processing_time": round(LiveVideo._percentile(times, 95), 4),
                "max_processing_time": round(max(times), 4),
                # Mean cost of a chunk as a multiple of its own duration.
                # Below 1.0 the stream is sustainable; above it, the backlog grows.
                "mean_real_time_factor": round(statistics.mean(times) / chunk_duration, 4),
                "mean_extract_time": round(
                    statistics.mean([r["extract_time"] for r in ok]), 4
                ),
                "mean_vlm_time": round(statistics.mean([r["vlm_time"] for r in ok]), 4),
                "max_lag": round(max(r["lag"] for r in ok), 4),
            })
            prompt_tokens = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
            if prompt_tokens:
                result["mean_prompt_tokens"] = round(statistics.mean(prompt_tokens), 1)
                result["max_prompt_tokens"] = max(prompt_tokens)

        # Per-chunk timings alongside the summary, matching how LiveCaptions
        # surfaces its own chunk timings in the returned result.
        for r in records:
            result[f"processing time_chunk_{r['chunk']}"] = r["processing_time"]

        return result

    @staticmethod
    def _percentile(values, pct) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (pct / 100) * (len(ordered) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return ordered[int(rank)]
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)

    @staticmethod
    def _write_session_files(records, summary, api_port, session):
        """Write the per-chunk table, the captions, and the summary to results."""
        results_dir = globals.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        prefix = os.path.join(results_dir, f"livevideo_{api_port}_session{session}")

        columns = [
            "chunk", "source_offset", "available_at", "started_at", "extract_time",
            "vlm_time", "processing_time", "real_time_factor", "kept_up", "lag",
            "frames", "prompt_tokens", "completion_tokens", "error",
        ]
        with open(f"{prefix}_chunks.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(record)

        with open(f"{prefix}_captions.log", "w") as f:
            for record in records:
                start = record["source_offset"]
                f.write(f"[{start:.1f}s] {record['caption'] or record['error']}\n")

        with open(f"{prefix}_summary.json", "w") as f:
            json.dump(
                {k: v for k, v in summary.items() if not k.startswith("processing time_chunk_")},
                f,
                indent=2,
            )

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _abspath(path):
        """Resolve a config path against the repo root.

        Workflow configs write paths relative to the repo, but the server
        script cds into the llama.cpp checkout before loading the model, so a
        relative path would never resolve.
        """
        return path if os.path.isabs(path) else os.path.join(repo_dir, path)

    @staticmethod
    def _probe_duration(video_path) -> float:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"ffprobe could not read {video_path}: {proc.stderr.strip()[:200]}")
        return float(proc.stdout.strip())

    @staticmethod
    def _wait_for_server(api_port, max_wait, interval=READINESS_INTERVAL):
        url = f"http://127.0.0.1:{api_port}/health"
        elapsed = 0
        print("Waiting for the VLM server to become ready...")
        while True:
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    print("VLM server is ready.")
                    return
            except requests.RequestException:
                pass
            if elapsed >= max_wait:
                raise RuntimeError(
                    f"Timed out after {max_wait}s waiting for llama-server on port {api_port}. "
                    f"Check the server log in {globals.get_results_dir()}/server_logs."
                )
            time.sleep(interval)
            elapsed += interval

    @staticmethod
    def _discover_model_name(api_port) -> str:
        resp = requests.get(f"http://127.0.0.1:{api_port}/v1/models", timeout=10)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]

    def get_default_config(self) -> Dict[str, Any]:
        return {
            # --- Model: Qwen3-VL-8B, the VLM already exercised by VSS's
            # use_remote_vlm path, served by llama.cpp. mmproj is the vision
            # projector; without it llama-server loads a text-only model and
            # rejects images.
            "model": f"{repo_dir}/models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            "mmproj": f"{repo_dir}/models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
            "llamacpp_path": f"{repo_dir}/inference_backends/llama.cpp",
            "device": "gpu",
            "mps": 100,
            # Not 8080 (Chatbot), 8081 (VSS LLM) or 8082 (VSS remote VLM), so
            # LiveVideo can run alongside any of them.
            "api_port": 8083,
            # --- Token budget. ctx is the total context; with parallel=1 the
            # whole of it is available to each chunk. A chunk costs roughly
            # frames_per_chunk x (frame_height/28) x (frame_width/28) tokens,
            # so the defaults below land near 4K -- but raising frames or
            # resolution scales that fast, which is why the headroom is here.
            "ctx": 131072,
            "parallel": 1,
            # --- Streaming workload.
            # How much feed each request covers, and the deadline it must beat.
            "chunk_duration": 10.0,
            # The main cost dial: more frames means more image tokens, which
            # means more prefill, which is what usually breaks real time.
            "frames_per_chunk": 8,
            "frame_height": 448,
            "jpeg_quality": 3,
            "max_tokens": 256,
            # 'realtime' waits for each chunk to "arrive" before processing it,
            # so a slow model visibly falls behind the feed. 'asap' processes
            # back to back, which measures the same per-chunk cost in less
            # wall-clock time but cannot show a growing backlog.
            "pacing": "realtime",
            # Seconds of feed per session; 0 means one pass over the file.
            "session_duration": 0,
            # Restart the file when it runs out, so a session can outlast it.
            "loop_video": True,
            # Spend one untimed chunk on the vision encoder's cold start.
            "warmup": True,
            "request_timeout": 300,
            "video_path": f"{repo_dir}/applications/VSS/my_video.mp4",
            "prompt": (
                "Describe what happens in this clip in two or three sentences. "
                "Mention any people, vehicles, or notable actions."
            ),
            # One run_application call = one streaming session.
            "num_requests": 1,
        }
