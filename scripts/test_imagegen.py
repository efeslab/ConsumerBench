import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "/mnt/tmpfs/models/stable-diffusion-3.5-large", 
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16)
pipe = pipe.to("cuda")
# pipe = pipe.text_encoder.to("cpu")
# pipe = pipe.enable_model_cpu_offload()

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")
