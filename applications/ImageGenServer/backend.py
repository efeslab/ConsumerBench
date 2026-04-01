import argparse
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from diffusers import StableDiffusion3Pipeline, AutoPipelineForText2Image
import torch


class ImageGenServer:
    def __init__(self, model, device, mps=100):
        self.model = model
        self.device = self._resolve_device(device)
        self.mps = mps

        if self.device == "cuda":
            # Only set MPS percentage when explicitly enabled.
            # For tally/tgs workflows we avoid overriding scheduler behavior.
            if self.mps and int(self.mps) > 0:
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.mps)

            try:
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    model,
                    text_encoder_3=None,
                    tokenizer_3=None,
                    torch_dtype=torch.float16
                )
            except (ValueError, OSError):
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    model,
                    torch_dtype=torch.float16,
                    variant="fp16" if os.path.isdir(model) else None,
                )

            # Reduce runtime VRAM spikes when co-running with other GPU apps.
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing("max")
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            if hasattr(self.pipeline, "enable_vae_tiling"):
                self.pipeline.enable_vae_tiling()

            # Prefer CPU offload for better coexistence with ChatbotHF.
            # If accelerate/offload is unavailable, fall back to full CUDA placement.
            try:
                self.pipeline.enable_model_cpu_offload()
                logging.info("ImageGenServer enabled model CPU offload")
            except Exception as e:
                logging.warning("Failed to enable model CPU offload, falling back to .to('cuda'): %s", e)
                self.pipeline = self.pipeline.to("cuda")
        else:
            try:
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    model,
                    text_encoder_3=None,
                    tokenizer_3=None
                )
            except (ValueError, OSError):
                self.pipeline = AutoPipelineForText2Image.from_pretrained(model)
            self.pipeline = self.pipeline.to("cpu")

    def _resolve_device(self, device):
        if device == "gpu":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device in ["cuda", "cpu", "mps"]:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt, num_inference_steps=28, guidance_scale=3.5, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        start_time = time.time()
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    _ = self.pipeline(
                        prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    ).images[0]
            else:
                _ = self.pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logging.warning("CUDA cache cleanup failed: %s", e)

        return {
            "status": "image_gen_complete",
            "total_time": time.time() - start_time,
        }


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status_code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._write_json(200, {"status": "ok"})
            else:
                self._write_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                self._write_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))

                prompt = payload.get("prompt", "")
                if not prompt:
                    self._write_json(400, {"error": "prompt is required"})
                    return

                response = app.generate(
                    prompt=prompt,
                    num_inference_steps=int(payload.get("num_inference_steps", 28)),
                    guidance_scale=float(payload.get("guidance_scale", 3.5)),
                    seed=payload.get("seed", None),
                )
                self._write_json(200, response)
            except Exception as e:
                self._write_json(500, {"error": str(e)})

        def log_message(self, format, *args):
            return

    return Handler


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[imagegen_server] %(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--mps", type=int, default=100)
    args = parser.parse_args()

    app = ImageGenServer(
        model=args.model,
        device=args.device,
        mps=args.mps,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.serve_forever()


if __name__ == "__main__":
    main()
