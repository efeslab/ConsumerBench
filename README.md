# ConsumerBench

## 📑 Overview

ConsumerBench is a comprehensive benchmarking framework that evaluates the runtime performance of user-defined GenAI applications under realistic conditions on end-user devices.

## 🚀 Benchmark Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ConsumerBench.git
cd ConsumerBench

# Set up environment
conda create -n consumerbench python=3.10
conda activate consumerbench
pip install -r requirements.txt
```
### Install applications
Follow instructions mentioned in `applications/`

### Adding config
Add your own yml workflow in `configs/`

### Running benchmark
Run the benchmark using the command
```
python src/scripts/run_consumerbench.py --config <path-to-config>
```

## 🔧 Hardware / System Requirements
The benchmark has been tested on the following hardware:
- Setup 1:
  - CPU: Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz
  - GPU: NVIDIA RTX 6000  
  - System Memory: 32GB
  - CPU cores: 12
- Setup 2:
  - Macbook Pro M1
  - Unified Memory: 32GB

## 📋 Repository Structure

```
ConsumerBench/
├── src/                    # Source code
├── inference_backends/     # Inference backends
├── models/                 # GenAI models
├── applications/           # Applications
├── configs/                # Example user configurations & workflows
└── scripts/                # Result processing and plotting scripts
```

## 🧩 Current Supported Applications

### 💬 Chatbot
Text-to-text generation for chat and Q&A with:
- Local backend mimicking OpenAI API
- Powered by llama.cpp for efficient CPU-GPU co-execution
- Located in `applications/Chatbot`

### 🔍 DeepResearch
Agent-based reasoning for complex fact gathering:
- Built on open-deep-research framework
- Served via LiteLLM
- Located in `applications/DeepResearch`

### 🖼️ ImageGen
Text-to-image generation optimized for edge devices:
- Utilizes stable-diffusion-webui in API mode
- Located in `applications/ImageGen`

### 🎙️ LiveCaptions
Audio-to-text transcription for real-time and offline use:
- Whisper-based backend over HTTP
- Located in `applications/LiveCaptions`


## System Metrics Collection
Run the script:
```bash
./scripts/run_benchmark.sh configs/workflow_imagegen.yml 0
```

This script collects:
1. **GPU metrics** - Compute/memory bandwidth (DCGM)
2. **CPU utilization** - Via `stat` utility
3. **CPU memory bandwidth** - Via `pcm-memory` utility
4. **GPU power** - Via `NVML` utility
5. **CPU power** - Via `RAPL` utility

### Results Analysis

Results are saved in the `results` directory with timestamps. PDF plots are automatically generated.

To modify Service Level Objectives (SLOs):
- Chatbot: [`scripts/result_processing/parse-results-chatbot-log.py`](scripts/result_processing/parse-results-chatbot-log.py)
- DeepResearch: [`scripts/result_processing/parse-results-deepresearch-log.py`](scripts/result_processing/parse-results-deepresearch-log.py)
- ImageGen: [`scripts/result_processing/parse-results-imagegen-log.py`](scripts/result_processing/parse-results-imagegen-log.py)
- LiveCaptions: [`scripts/result_processing/parse-results-whisper-log.py`](scripts/result_processing/parse-results-whisper-log.py)

## 📝 Experiment Configurations

### Exclusive Execution
| Application | Config |
|-------------|--------|
| Chatbot | [`configs/workflow_chatbot.yml`](configs/workflow_chatbot.yml) |
| LiveCaptions | [`configs/workflow_live_captions.yml`](configs/workflow_live_captions.yml) |
| ImageGen | [`configs/workflow_imagegen.yml`](configs/workflow_imagegen.yml) |

> **CPU-only:** Change `device` from "gpu" to "cpu" in the configs.

### Concurrent Execution
- **Greedy allocation:** [`configs/workflow_chatbot_imagegen_live_captions.yml`](configs/workflow_chatbot_imagegen_live_captions.yml)
- **GPU partitioning:** [`configs/workflow_chatbot_imagegen_live_captions_mps.yml`](configs/workflow_chatbot_imagegen_live_captions_mps.yml)

### Model Sharing (Inference Server)
- **Config:** [`configs/workflow_chatbot_deep_research.yml`](configs/workflow_chatbot_deep_research.yml)
- Edit [`example_workflow/llamacpp_server.sh`](example_workflow/llamacpp_server.sh) to add `-c 128000 -nkvo` for Chatbot-KVCache-CPU

### End-to-End User Workflow
- **Greedy allocation:** [`configs/workflow_content_creation.yml`](configs/workflow_content_creation.yml)
- **GPU partitioning:** [`configs/workflow_content_creation_mps.yml`](configs/workflow_content_creation_mps.yml)
