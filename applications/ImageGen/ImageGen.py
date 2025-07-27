## Add ImageGen class here
import time
from typing import Any, Dict
import sys
import os
from diffusers import StableDiffusion3Pipeline
import torch
from datasets import load_dataset

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application

class ImageGen(Application):
    def __init__(self):
        super().__init__()
        self.imagegen_pipeline = None
        self.imagegen_prompts = []

    def run_setup(self, *args, **kwargs):
        print("ImageGen setup")
        model = kwargs.get('model', self.get_default_config()['model'])
        device = kwargs.get('device', self.get_default_config()['device'])
        mps = kwargs.get('mps', self.get_default_config()['mps'])

        if device == "gpu":
            # Set environment variable for MPS
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
            self.imagegen_pipeline = StableDiffusion3Pipeline.from_pretrained(
                model,
                text_encoder_3=None,
                tokenizer_3=None,
                torch_dtype=torch.float16
            )
            self.imagegen_pipeline = self.imagegen_pipeline.to("cuda")
        else:
            self.imagegen_pipeline = StableDiffusion3Pipeline.from_pretrained(
                model,
                text_encoder_3=None,
                tokenizer_3=None
            )
            self.imagegen_pipeline = self.imagegen_pipeline.to("cpu")
            return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("ImageGen cleanup")
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        imagegen_prompt = self.imagegen_prompts.pop(0)
        start_time = time.time()
        _ = self.imagegen_pipeline(imagegen_prompt, 
                                       num_inference_steps=28, 
                                       guidance_scale=3.5).images[0]
        return {"status": "image_gen_complete", "total time": time.time() - start_time}

    def load_dataset(self):
        """Load the image generation dataset"""
        ds_imagegen = load_dataset(self.config.get("dataset", self.get_default_config()['dataset']))
        ds_imagegen = ds_imagegen["train"]
        ds_imagegen = ds_imagegen.shuffle(seed=42)
        ds_imagegen = ds_imagegen.select(range(0, 100))
        for item in ds_imagegen:
            self.imagegen_prompts.append(item['caption1'])

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": f"/mnt/tmpfs/models/stable-diffusion-3.5-medium-turbo",
            "device": "gpu",
            "mps": 100,
            "dataset": "sentence-transformers/coco-captions"
        }
    