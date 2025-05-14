import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "/home/cc/models/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

livecaptions_prompts = []
ds_livecaptions = ds = load_dataset("distil-whisper/earnings21")
ds_livecaptions = ds_livecaptions["test"]
ds_livecaptions = ds_livecaptions.shuffle(seed=2)
ds_livecaptions = ds_livecaptions.select(range(0, 1))
for item in ds_livecaptions:
    livecaptions_prompts.append(item['audio'])

# check the length of the audio
for item in livecaptions_prompts:
    print(item["array"].shape)

result = pipe(livecaptions_prompts[0], return_timestamps=True)
print(result["text"])
