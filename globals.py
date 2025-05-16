from datasets import load_dataset
import os
from datetime import datetime

results_dir = None  # Default value
# declare dataset variables
start_time = None
textgen_prompts = []
imagegen_prompts = []
livecaptions_prompts = []
deep_research_prompts = []

# You can add a function to update it
def set_results_dir(path):
    global results_dir
    results_dir = path

def get_results_dir():
    global results_dir
    if results_dir is None:
        raise ValueError("Results directory is not set. Please set it using set_results_dir()")
    return results_dir

def set_start_time(time):
    global start_time    
    start_time = datetime.strptime(time, '%Y-%m-%d_%H:%M:%S')

def load_deep_research_dataset():
    global deep_research_prompts
    # deep_research_prompts.append("Give me a summary of the latest research paper on AI.")
    deep_research_prompts.append("What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?")

def load_textgen_dataset():
    global textgen_prompts
    """Load the text generation dataset"""
    ds_textgen_lmsys = load_dataset("lmsys/lmsys-chat-1m")
    ds_textgen_lmsys = ds_textgen_lmsys["train"]
    ds_textgen_lmsys = ds_textgen_lmsys.shuffle(seed=42)
    ds_textgen_lmsys = ds_textgen_lmsys.select(range(0, 100))
    for item in ds_textgen_lmsys:
        textgen_prompts.append(item['conversation'][0]['content'])

def get_next_textgen_prompt():
    global textgen_prompts
    return textgen_prompts.pop(0)


def load_imagegen_dataset():
    global imagegen_prompts
    """Load the image generation dataset"""
    ds_imagegen_cococaptions = load_dataset("sentence-transformers/coco-captions")
    ds_imagegen_cococaptions = ds_imagegen_cococaptions["train"]
    ds_imagegen_cococaptions = ds_imagegen_cococaptions.shuffle(seed=42)
    ds_imagegen_cococaptions = ds_imagegen_cococaptions.select(range(0, 100))
    for item in ds_imagegen_cococaptions:
        imagegen_prompts.append(item['caption1'])

def get_next_imagegen_prompt():
    global imagegen_prompts
    return imagegen_prompts.pop(0)


def load_livecaptions_dataset():
    global livecaptions_prompts
    """Load the live captions dataset"""
    ds_livecaptions = ds = load_dataset("distil-whisper/earnings21")
    ds_livecaptions = ds_livecaptions["test"]
    # ds_livecaptions = ds_livecaptions.shuffle(seed=42)
    # ds_livecaptions = ds_livecaptions.select(range(0, 1))
    for item in ds_livecaptions:
        # livecaptions_prompts.append(item['audio'])
        # save the audio file
        audio_file = item['audio']
        audio_file_path = os.path.join("/home/cc/datasets/whisper-earnings21", audio_file)
