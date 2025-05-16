## Overview

ConsumerBench is a a comprehensive benchmarking framework that evaluates the runtime performance of user-defined GenAI applications under realistic conditions on end-user devices.

## Setup and Installation

### Python Environment

ConsumerBench is a python-based framework.
It is necessary for the users to install required python packages before using Argos.
We recommend installing packages in a conda virtual environment.
Follow the below steps for setup.

```
cd ConsumerBench
conda create -n consumerbench python=3.10
conda activate consumerbench
pip install -r requirements.txt
```


## Repo Structure

```
├── ConsumerBench
│   ├── applications			  # Repos for applications supported
│   ├── configs			  # Example user configurations / workflows
│   ├── scripts		  # Scripts for processing and plotting results
│   ├── benchmark_v2.py		      # Core benchmark code
│   ├── workflow.py		      # Define Workflow class
│   ├── globals.py		  	  # Define globals shared between applications
```

## Supported Applications

### Chatbot
A text-to-text generation application for chat and Q&A. It features a minimal frontend and a local backend that mimics the OpenAI API, powered by llama.cpp for efficient CPU-GPU co-execution.
The local minimal frontend is included in `benchmark_v2.py` that accepts user prompts and sends HTTP requests with the OpenAI API to a backend llama.cpp server

### DeepResearch
An agent-based reasoning tool for complex, multi-step fact gathering and synthesis. Built on open-deep-research and served via LiteLLM, it interacts with a local model through llama.cpp.
The agent is in `applications/smolagents`, which accepts a user prompt and sends HTTP requests with the OpenAI API to a backend llama.cpp server.

### ImageGen
A text-to-image generation app utilizing stable-diffusion-webui in API mode. Tailored for edge deployment.
The path to the application is: `applications/stable-diffusion-webui`, which loads the model and generates images based on user prompts.

### LiveCaptions
An audio-to-text transcription app for both real-time and offline scenarios. It uses a Whisper-based backend over HTTP. 
The backend model and the front-end is located in `applications/whisper_streaming`. The (`generate_raw_realtime.py`) implements the front-end that sends audio chunks to the backend server (`whisper_online_server.py`)

## Getting Started

To run an example workflow for ConsumerBench, you can start by running the following command

```
python3 -u benchmark_v2.py --benchmark workflow --config configs/workflow_imagegen.yml
```

## Installing Applications

Please follow the respective application READMEs for their installations. All the applications use the standard installation flags, with the exception of llama.cpp, for which the CMAKE flags are: `-DGGML_CUDA=ON -DGGML_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES="75"`

## Executing the benchmarks along with system-level metrics

### Running the benchmark

The following script can be used to run benchmarks with system-level metrics. The script loads the datasets mentioned in the paper. 
```
./scripts/run_benchmark.sh configs/workflow_imagegen.yml 0
```

This script uses the following tools for system-level metrics:
1. GPU compute / memory bandwith utilization: Data Center GPU Management (DCGM)
2. CPU compute utilization: `stat` utility
3. CPU memory bandwidth utilization: `pcm-memory` utility
4. GPU power consumption: `NVML` utility
5. CPU power consumption: `RAPL` utility


### Gathering results

The script creates a results directory with a timestamp under `results`, and automatically generates plots for the applications. 
All the plots are generated as PDF files in the results directory. 
The current SLOs configured are the SLOs used in the paper. In order to change SLOs, you can modify the following files:
1. Chatbot: `scripts/parse-results-chatbot-log.py`
2. DeepResearch: `scripts/parse-results-deepresearch-log.py`
3. ImageGen: `scripts/parse-results-imagegen-log.py`
4. LiveCaptions: `scripts/parse-results-whisper-log.py`

## Config files used in the paper

Following are the config files used in the paper. Please run the experiments to gather application-level metrics and system-level metrics in the following way:
```
./scripts/run_benchmark.sh <config-file> 0
```

The config files are:
1. Exclusive GPU execution:
    1. Chatbot: `configs/workflow_chatbot.yml`
    2. LiveCaptions: `configs/workflow_whisper.yml`
    3. ImageGen: `configs/workflow_imagegen.yml`
2. Exclusive CPU execution: Change the "device" in the previous workflows from "gpu" to "cpu" and re-run. 
3. Concurrent Execution:
    1. Greedy resource allocation: `configs/workflow_chatbot_imagegen_livecaptions.yml`
    2. GPU partitioning: `configs/workflow_chatbot_imagegen_livecaptions_mps.yml`
6. Model sharing using Inference Server (change `example_workflow/llamacpp_server.sh` to spawn llama.cpp with -c 128000 -nkvo for Chatbot-KVCache-CPU)
    1. Config file: `configs/workflow_chatbot_deepresearch.yml`
6. Running an end-to-end user workflow:
    1. Greedy resource allocation: `configs/workflow_content_creation.yml`
    2. GPU partitioning: `configs/workflow_content_creation_mps.yml`
