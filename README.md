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

### DeepResearch
An agent-based reasoning tool for complex, multi-step fact gathering and synthesis. Built on open-deep-research and served via LiteLLM, it interacts with a local model through llama.cpp.

### ImageGen
A text-to-image generation app utilizing stable-diffusion-webui in API mode. Tailored for edge deployment.

### LiveCaptions
An audio-to-text transcription app for both real-time and offline scenarios. It uses a Whisper-based backend over HTTP. 


## Getting Started

To run an example workflow for ConsumerBench, you can start by running the following command

```
python3 -u benchmark_v2.py --benchmark workflow --config configs/workflow_imagegen.yml
```
