import os
import sys
import logging
import time
import random
import subprocess
import json
import requests
import nvtx
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from diffusers import StableDiffusion3Pipeline

import globals
from utils import util_run_server_script_check_log, parse_commands


# ====== LlamaCPP ========
# [ no need to have this]
def setup_llamacpp_server(**kwargs):
    # return True

    server_pid = -1

    api_port = kwargs.get('api_port', 8080)
    model = kwargs.get('model', os.path.join(globals.project_dir, "models/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-f16.gguf"))
    device = kwargs.get('device', "gpu")
    mps = kwargs.get('mps', 100)

    # acquire the lock
    with globals.model_refcount_lock:
        globals.model_refcount["llama"] = globals.model_refcount.get("llama", 0) + 1
        # check if the server is already running
        if globals.model_refcount["llama"] > 1:
            print("Llama server is already running")
            return True        

        # def util_run_server_script_check_log(script_path: str, stdout_log_path: str, stderr_log_path: str, stderr_ready_patterns, stdout_ready_patterns, listen_port, api_port, model):
        print("Setting up llama.cpp server...")

        util_run_server_script_check_log(
            script_path=os.path.join(globals.project_dir, "scripts/inference_backends/llamacpp_server.sh"),
            stdout_log_path="llamacpp_server_stdout",
            stderr_log_path="llamacpp_server_stderr",
            stderr_ready_patterns=["update_slots: all slots are idle"],
            stdout_ready_patterns=[],
            listen_port=api_port,
            api_port=api_port,
            model=model,
            device=device,
            mps=mps
        )

        # print("Pushing NVTX range 'Main'")
        # try:
        #     nvtx.push_range("Main")
        #     print("NVTX push successful")
        # except Exception as e:
        #     print(f"NVTX push error: {e}")
        #     sys.exit(1)



# [ no need to have this]
def cleanup_llamacpp_server(**kwargs):
    # return True
    # print("Popping Main range")
    # try:
    #     nvtx.pop_range()
    #     print("Main NVTX pop successful")
    # except Exception as e:
    #     print(f"Main NVTX pop error: {e}")



    api_port = kwargs.get('api_port', 8080)

    with globals.model_refcount_lock:
        if "llama" in globals.model_refcount:
            # check if the server is already running
            if globals.model_refcount["llama"] > 0:
                globals.model_refcount["llama"] -= 1
                if globals.model_refcount["llama"] == 0:
                    print("Llama server is shutting down")
                    # kill the process
                    process = subprocess.Popen(
                        [os.path.join(globals.project_dir, "scripts/cleanup.sh"), str(api_port)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    process.wait()
                else:
                    print("Llama server is still running")

    return True



# ====== Deep Research ======
# [ no need to have this]
def run_deep_research_dataset(api_port, model):
    global deep_research_prompts

    start_time = time.time()

    # select one random prompt from the dataset
    deep_research_prompt = random.sample(globals.deep_research_prompts, 1)
    log_dir = os.path.join(globals.get_results_dir(), "client_logs")
    os.makedirs(log_dir, exist_ok=True)
    stdout_log = os.path.join(log_dir, f"deep_research_client_stdout_{api_port}.log")
    stderr_log = os.path.join(log_dir, f"deep_research_client_stderr_{api_port}.log")

    # Start the server process with log file redirection
    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
        process = subprocess.Popen(
            [os.path.join(globals.project_dir, "scripts/applications/deep_research_client.sh"), str(api_port), str(model), str(deep_research_prompt)],
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,  # Important for server processes
        )

        process.wait()


    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")
    result = {
        'total time': end_time - start_time
    }
    print(result)
    return result


# [ no need to have this]
def run_deep_research(**kwargs):
    print("Running deep research (ephemeral app)...")

    api_port = kwargs.get('api_port', 8080)
    model = kwargs.get('model', "openai/meta-llama/Llama-3.1-8B-Instruct")

    result = run_deep_research_dataset(api_port, model)

    return result


# Define example functions for a simple benchmark

# ====== text2image ========
# [ no need to have this]
def setup_imagegen(**kwargs):
    global global_vars

    model = kwargs.get('model', "/mnt/tmpfs/models/stable-diffusion-3.5-large")
    device = kwargs.get('device', "gpu")
    mps = kwargs.get('mps', 100)

    # print("Pushing NVTX range 'Main'")
    # try:
    #     nvtx.push_range("Main")
    #     print("NVTX push successful")
    # except Exception as e:
    #     print(f"NVTX push error: {e}")
    #     sys.exit(1)

    if device == "gpu":
        # Set environment variable for MPS
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
        global_vars['imagegen_pipeline'] = StableDiffusion3Pipeline.from_pretrained(
            model,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=torch.float16
        )
        global_vars['imagegen_pipeline'] = global_vars['imagegen_pipeline'].to("cuda")
    else:
        global_vars['imagegen_pipeline'] = StableDiffusion3Pipeline.from_pretrained(
            model,
            text_encoder_3=None,
            tokenizer_3=None
        )
        global_vars['imagegen_pipeline'] = global_vars['imagegen_pipeline'].to("cpu")


# [ no need to have this]
def run_imagegen_prompt(prompt):
    global global_vars

    end_time = None

    nvtx.mark("[Imagegen request Start]")
    start_time = time.time()

    image = global_vars['imagegen_pipeline'](
        prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]

    end_time = time.time()
    nvtx.mark("[Imagegen request End]")

    result = {
        "total time": end_time - start_time,
    }
    print(result)
    return result

# With CUDA Graph

# def setup_imagegen(**kwargs):
#     global global_vars

#     model = kwargs.get('model', "/mnt/tmpfs/models/stable-diffusion-3.5-large")
#     device = kwargs.get('device', "gpu")
#     mps = kwargs.get('mps', 100)
#     fixed_prompt = globals.get_next_imagegen_prompt()

#     if device == "gpu":
#         os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
#         pipe = StableDiffusion3Pipeline.from_pretrained(
#             model,
#             text_encoder_3=None,
#             tokenizer_3=None,
#             torch_dtype=torch.float16
#         ).to("cuda")

#         # Warm-up and graph capture
#         # pipe(prompt=fixed_prompt, num_inference_steps=28, guidance_scale=3.5)  # warm-up

#         # torch.cuda.empty_cache()
#         # torch.cuda.synchronize()

#         # graph = torch.cuda.CUDAGraph()
#         # static_output = None

#         # with torch.cuda.graph(graph):
#         #     static_output = pipe(prompt=fixed_prompt, num_inference_steps=28, guidance_scale=3.5)

#         # Save all to global_vars
#         global_vars['imagegen_pipeline'] = pipe
#         # global_vars['imagegen_cuda_graph'] = graph
#         # global_vars['imagegen_output'] = static_output
#         global_vars['imagegen_prompt'] = fixed_prompt
#     else:
#         pipe = StableDiffusion3Pipeline.from_pretrained(
#             model,
#             text_encoder_3=None,
#             tokenizer_3=None
#         ).to("cpu")
#         global_vars['imagegen_pipeline'] = pipe

# def run_imagegen_prompt(prompt):
#     global global_vars

#     if (
#         'imagegen_cuda_graph' in global_vars
#         and prompt == global_vars.get('imagegen_prompt')
#     ):
#         nvtx.mark("[Imagegen request Start]")
#         torch.cuda.synchronize()
#         start_time = time.time()

#         global_vars['imagegen_cuda_graph'].replay()

#         torch.cuda.synchronize()
#         end_time = time.time()
#         nvtx.mark("[Imagegen request End]")

#         print({"total time": end_time - start_time})
#         return {
#             "total time": end_time - start_time,
#             "image": global_vars['imagegen_output'].images[0]
#         }

#     else:
#         # Fallback to normal pipeline execution
#         nvtx.mark("[Imagegen request Start]")
#         start_time = time.time()
#         image = global_vars['imagegen_pipeline'](
#             prompt,
#             num_inference_steps=28,
#             guidance_scale=3.5,
#         ).images[0]
#         end_time = time.time()
#         nvtx.mark("[Imagegen request End]")

#         print({"total time": end_time - start_time})
#         return {
#             "total time": end_time - start_time,
#             "image": image
#         }



# [ no need to have this]
def run_imagegen_command_file(filename):
    # read the commands from the file
    commands = parse_commands(filename)
    # get a random command from the file
    command = random.choice(commands)

    # select one random prompt from the dataset
    result = run_imagegen_prompt(command)

    return result


# [ no need to have this]
def run_imagegen_dataset():
    global imagegen_prompts

    # select one random prompt from the dataset
    imagegen_prompt = globals.get_next_imagegen_prompt()
    logging.info(f"Imagegen prompt: {imagegen_prompt}")
    result = run_imagegen_prompt(imagegen_prompt)

    return result


# [ no need to have this]
def run_imagegen(**kwargs):
    print("Running imagegen")

    filename = kwargs.get('command_file', None)

    if filename is not None:
        result = run_imagegen_command_file(filename)
    else:
        result = run_imagegen_dataset()

    return result

# [ no need to have this]
def cleanup_imagegen(**kwargs):
    # print("Popping Main range")
    # try:
    #     nvtx.pop_range()
    #     print("Main NVTX pop successful")
    # except Exception as e:
    #     print(f"Main NVTX pop error: {e}")
    return True


# ====== Nothing ========

# [ no need to have this. Move it to applications/ if necessary]
def nothing_function(**kwargs):
    # This function does nothing
    return True

# [ no need to have this. Move it to applications/ if necessary]
def sleep_function(**kwargs):
    # This function sleeps for 1 second
    time.sleep(60)
    return True

# ====== Live Captions =======
# [ no need to have this]
def setup_whisper():
    global global_vars
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    global_vars["whisper_pipeline"] = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


# [ no need to have this]
def run_whisper(**kwargs):
    global global_vars, livecaptions_prompts
    print("Running whisper (ephemeral app)...")

    pipe = global_vars["whisper_pipeline"]

    start_time = time.time()

    # select one random prompt from the dataset
    whisper_prompts = random.sample(livecaptions_prompts, 1)

    for prompt in whisper_prompts:
        # get the wav file from this prompt
        wav_file = prompt["audio"]

        pipe(prompt, return_timestamps="word")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")

    result = {
        'total time': end_time - start_time
    }
    print(result)
    return result

    
# [ no need to have this]
def cleanup_whisper():
    # Do nothing
    return


# [ no need to have this]
def setup_whisper_online(**kwargs):
    server_pid = -1

    api_port = kwargs.get('api_port', 5000)
    device = kwargs.get('device', "gpu")
    mps = kwargs.get('mps', 100)

    print("Setting up whisper-online (ephemeral app)...")

    util_run_server_script_check_log(
        script_path=os.path.join(globals.project_dir, "scripts/applications/whisper_online_server.sh"),
        stdout_log_path="whisper_online_server_stdout",
        stderr_log_path="whisper_online_server_stderr",
        stderr_ready_patterns=["Listening on"],
        stdout_ready_patterns=[],
        listen_port=api_port,
        api_port=api_port,
        model=None,
        device=device,
        mps=mps
    )
    
    return server_pid


# [ no need to have this]
def run_whisper_online_command_file(api_port, wav_file_path):
    print(f"Running whisper-online (ephemeral app) on {wav_file_path}...")
    end_time = None

    log_dir = os.path.join(globals.get_results_dir(), "client_logs")
    os.makedirs(log_dir, exist_ok=True)
    stdout_log = os.path.join(log_dir, f"whisper_online_stdout_{api_port}.log")
    stderr_log = os.path.join(log_dir, f"whisper_online_stderr_{api_port}.log")

    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
        process = subprocess.Popen(
            [os.path.join(globals.project_dir, "scripts/applications/whisper_online_client.sh"), str(api_port), wav_file_path],
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )

    start_time = time.time()
    process.wait()
    end_time = time.time()
    result = {
        'total time': end_time - start_time
    }
    print(result)

    # Parse the stdout log to get the Processing Time
    with open(stdout_log, 'r') as f:
        chunk_idx = 0
        for line in f:
            if "Processing time" in line:
                processing_time = re.search(r"Processing time: (\d+\.\d+)", line)
                if processing_time:
                    processing_time = float(processing_time.group(1))
                    result[f'processing time_chunk_{chunk_idx}'] = processing_time
                    print(f"Processing Time: {processing_time:.4f} seconds")
                    chunk_idx += 1

    return result


# [ no need to have this]
def run_whisper_online_dataset(api_port):
    # get a random file from datasets/whisper-earnings21
    directory = os.path.join(globals.project_dir, "datasets/whisper-earnings21")
    files = os.listdir(directory)
    wav_file = random.choice(files)
    wav_file_path = os.path.join(directory, wav_file)
    wav_file_path = os.path.join(globals.project_dir, "datasets/whisper-earnings21/4320211_chunk_040.wav")
    result = run_whisper_online_command_file(api_port, wav_file_path)
    return result


# [ no need to have this]
def run_whisper_online(**kwargs):
    api_port = kwargs.get('api_port', 5050)
    wav_file_path = kwargs.get('command_file', None)
    if wav_file_path is not None:
        result = run_whisper_online_command_file(api_port, wav_file_path)
    else:
        result = run_whisper_online_dataset(api_port)
    return result


# [ no need to have this]
def cleanup_whisper_online(**kwargs):
    api_port = kwargs.get('api_port', 5050)
    process = subprocess.Popen(
        [os.path.join(globals.project_dir, "scripts/cleanup.sh"), str(api_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait()
    return True


# ====== textgen ========
# [ no need to have this]
def setup_textgen(**kwargs):
    server_pid = -1

    listen_port = kwargs.get('listen_port', 7860)
    api_port = kwargs.get('api_port', 5000)
    model = kwargs.get('model', "facebook_opt-1.3b")

    print("Setting up textgen (background app)...")

    util_run_server_script_check_log(
        script_path=os.path.join(globals.project_dir, "scripts/applications/textgen_server.sh"),
        stdout_log_path="textgen_server_stdout",
        stderr_log_path="textgen_server_stderr",
        stderr_ready_patterns=[],
        stdout_ready_patterns=["Running on local URL", "SERVER_PID="],
        listen_port=listen_port,
        api_port=api_port,
        model=model
    )
    
    return server_pid


# [ no need to have this]
def run_textgen_command_file(filename, api_port):
    commands = parse_commands(filename)        

    ttft = None
    token_count = 0
    first_token_time = None
    end_time = None
    start_time = time.time()

    commands = random.sample(commands, 1)

    for command in commands:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # for line in iter(process.stderr.readline, ''):
            # print(f"Script output: {line.strip()}")
        # Read output to get the server PID
        for line in iter(process.stdout.readline, ''):
            if line:
                # print(f"Script output: {line.strip()}")
                current_time = time.time()
                if ttft is None:
                    ttft = current_time - start_time
                    first_token_time = current_time
                    print(f"Time to first token: {ttft:.4f} seconds")
                # print(f"Received token: {line.decode('utf-8')}")
                try:
                    data = json.loads(line.strip().replace("data: ", ""))
                    if data["choices"][0]["finish_reason"]:
                        token_count = data["usage"]["completion_tokens"]
                        break
                except json.JSONDecodeError as e:
                    continue
        end_time = time.time()        
        if process.errors != None:
            return False
    

    print(f"{end_time-first_token_time}, token counts: {token_count}")
    tpot = (end_time - first_token_time) / token_count if token_count > 0 else None    
    itl = (end_time - start_time) / token_count if token_count > 0 else None

    result = {
        "ttft": ttft,
        "tpot": tpot,
        "itl": itl,
    }
    print(result)
    return result

    # url = "http://127.0.0.1:5000/v1/completions"
    # headers = {"Content-Type": "application/json"}
    # payload = {
    #     "prompt": "Once upon a time there was a",
    #     "min_tokens": 200,
    #     "max_tokens": 200,
    #     "temperature": 1,
    #     "top_p": 0.9,
    #     "seed": 141293,
    #     "stream": True,
    # }


# [ no need to have this]
def run_textgen_dataset(api_port):
    # TODO: use the following session to issue posts
    api_url = f"http://127.0.0.1:{api_port}/v1/completions"

    ttft = None
    token_count = 0
    first_token_time = None
    end_time = None
    start_time = time.time()

    # select one random prompt from the dataset
    # textgen_prompts = random.sample(globals.textgen_prompts, 1, seed=141293)
    textgen_prompts = [globals.get_next_textgen_prompt()]
    logging.info(f"Textgen prompt: {textgen_prompts}")
    # textgen_prompts = globals.textgen_prompts[:1]
    
    for prompt in textgen_prompts:
        payload = {
            "prompt": prompt,
            # "min_tokens": 200,
            # "max_tokens": 100,
            "max_tokens": 215,
            "temperature": 0,
            "top_p": 0.9,
            "seed": 141293,
            "stream": True
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            with requests.post(api_url, json=payload, headers=headers, stream=True) as response:
                if response.status_code != 200:
                    print("HTTP Error:", response.status_code, response.text)
                    return

                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        # print(f"Script output: {line.strip()}")
                        current_time = time.time()
                        if ttft is None:
                            ttft = current_time - start_time
                            first_token_time = current_time
                            print(f"Time to first token: {ttft:.4f} seconds")

                        try:
                            # Clean and parse the JSON
                            clean_line = line.strip().replace("data: ", "")
                            if clean_line == "[DONE]":
                                break

                            data = json.loads(clean_line)

                            # Exit if finish_reason appears
                            if data.get("choices") and data["choices"][0].get("finish_reason"):
                                token_count = data.get("usage", {}).get("completion_tokens")
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print("Request failed:", e)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")
    print(f"Completion tokens: {token_count}")

    print(f"{end_time-first_token_time}, token counts: {token_count}")
    tpot = (end_time - first_token_time) / token_count if token_count > 0 else None    
    itl = (end_time - start_time) / token_count if token_count > 0 else None

    result = {
        "ttft": ttft,
        "tpot": tpot,
        "itl": itl,
    }
    print(result)
    return result
    

# [ no need to have this]
def run_textgen(**kwargs):
    print("Running textgen (background app)...")

    api_port = kwargs.get('api_port', 5000)
    filename = kwargs.get('command_file', None)

    if filename is not None:
        result = run_textgen_command_file(filename, api_port)
    else:
        nvtx.mark("[Chatbot request Start]")
        result = run_textgen_dataset(api_port)
        nvtx.mark("[Chatbot request End]")


    return result


# [ no need to have this]
def cleanup_textgen(**kwargs):
    """Example function to cleanup textgen"""
    print("Cleaning up textgen app...")

    api_port = kwargs.get('api_port', 5000)
    process = subprocess.Popen(
        [os.path.join(globals.project_dir, "scripts/cleanup.sh"), str(api_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait()
    return True

# ====== livecaptions ========
# [ no need to have this]
def run_livecaptions(**kwargs):
    """Example function to run livecaptions"""
    print("Running livecaptions app...")
    process = subprocess.Popen(
        [os.path.join(globals.project_dir, "scripts/applications/livecaptions_client.sh")],
        text=True,
    )
    process.wait()
    return True

# [ no need to have this. Shadow can also be an application]
def shadow_function(**kwargs):
    return True