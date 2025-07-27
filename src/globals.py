from datasets import load_dataset
import os
import threading
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results_dir = None  # Default value
# declare dataset variables
textgen_prompts = []
imagegen_prompts = []
livecaptions_prompts = []
deep_research_prompts = []

start_time = None

model_refcount_lock = threading.Lock()
model_refcount = {}

def set_working_dir(path):
    global working_dir
    working_dir = path

def get_working_dir():
    global working_dir
    if working_dir is None:
        raise ValueError("Working directory is not set. Please set it using set_working_dir()")
    return working_dir

# You can add a function to update it
def set_results_dir(path):
    global results_dir
    results_dir = path

def get_results_dir():
    global results_dir
    if results_dir is None:
        raise ValueError("Results directory is not set. Please set it using set_results_dir()")
    return results_dir

def set_start_time():
    global start_time    
    start_time = datetime.now()

# [ROHAN: We should remove this explicit prompt]
def load_deep_research_dataset():
    global deep_research_prompts
    # deep_research_prompts.append("Give me a summary of the latest research paper on AI.")
    deep_research_prompts.append("What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?")

# [ROHAN: We should remove names of datasets if possible]
def load_textgen_dataset():
    global textgen_prompts
    """Load the text generation dataset"""
    ds_textgen_lmsys = load_dataset("lmsys/lmsys-chat-1m")
    ds_textgen_lmsys = ds_textgen_lmsys["train"]
    ds_textgen_lmsys = ds_textgen_lmsys.shuffle(seed=42)
    ds_textgen_lmsys = ds_textgen_lmsys.select(range(0, 100))
    for item in ds_textgen_lmsys:
        textgen_prompts.append(item['conversation'][0]['content'])

# [ROHAN: We should remove names of datasets if possible]
def get_next_textgen_prompt():
    global textgen_prompts
    return textgen_prompts.pop(0)


# [ROHAN: We should remove names of datasets if possible]
def load_imagegen_dataset():
    global imagegen_prompts
    """Load the image generation dataset"""
    ds_imagegen_cococaptions = load_dataset("sentence-transformers/coco-captions")
    ds_imagegen_cococaptions = ds_imagegen_cococaptions["train"]
    ds_imagegen_cococaptions = ds_imagegen_cococaptions.shuffle(seed=42)
    ds_imagegen_cococaptions = ds_imagegen_cococaptions.select(range(0, 100))
    for item in ds_imagegen_cococaptions:
        imagegen_prompts.append(item['caption1'])

# [ROHAN: We should remove names of datasets if possible]
def get_next_imagegen_prompt():
    global imagegen_prompts
    return imagegen_prompts.pop(0)


# [ROHAN: We should remove names of datasets if possible]
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
