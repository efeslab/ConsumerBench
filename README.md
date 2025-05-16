# ConsumerBench

## 📑 Overview

ConsumerBench is a comprehensive benchmarking framework that evaluates the runtime performance of user-defined GenAI applications under realistic conditions on end-user devices.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/ConsumerBench.git
cd ConsumerBench

# Set up environment
conda create -n consumerbench python=3.10
conda activate consumerbench
pip install -r requirements.txt

# Run a sample benchmark
python3 benchmark_v2.py --benchmark workflow --config configs/workflow_imagegen.yml
```

## 📋 Repository Structure

```
ConsumerBench/
├── applications/           # Application repositories
├── configs/                # Example user configurations & workflows
├── scripts/                # Result processing and plotting scripts
├── benchmark_v2.py         # Core benchmark code
├── workflow.py             # Workflow class definition
└── globals.py              # Shared global variables
```

## 🧩 Supported Applications

### 💬 Chatbot
Text-to-text generation for chat and Q&A with:
- Minimal frontend in `benchmark_v2.py`
- Local backend mimicking OpenAI API
- Powered by llama.cpp for efficient CPU-GPU co-execution

### 🔍 DeepResearch
Agent-based reasoning for complex fact gathering:
- Built on open-deep-research framework
- Served via LiteLLM
- Located in `applications/smolagents`

### 🖼️ ImageGen
Text-to-image generation optimized for edge devices:
- Utilizes stable-diffusion-webui in API mode
- Located in `applications/stable-diffusion-webui`

### 🎙️ LiveCaptions
Audio-to-text transcription for real-time and offline use:
- Whisper-based backend over HTTP
- Front-end: `applications/whisper_streaming/generate_raw_realtime.py`
- Back-end: `applications/whisper_streaming/whisper_online_server.py`

## ⚙️ Installation & Setup

### Application Installation

Follow the README in each application directory for specific installation instructions. 

> **Note:** For llama.cpp, use these CMAKE flags:
> ```
> -DGGML_CUDA=ON -DGGML_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES="75"
> ```

## 📊 Running Benchmarks

### Basic Benchmark

```bash
python3 benchmark_v2.py --benchmark workflow --config configs/workflow_imagegen.yml
```

### Comprehensive Benchmark with System Metrics

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
- Chatbot: [`scripts/parse-results-chatbot-log.py`](https://github.com/efeslab/ConsumerBench/blob/master/scripts/parse-results-chatbot-log.py)
- DeepResearch: [`scripts/parse-results-deepresearch-log.py`](https://github.com/efeslab/ConsumerBench/blob/master/scripts/parse-results-deepresearch-log.py)
- ImageGen: [`scripts/parse-results-imagegen-log.py`](https://github.com/efeslab/ConsumerBench/blob/master/scripts/parse-results-imagegen-log.py)
- LiveCaptions: [`scripts/parse-results-whisper-log.py`](https://github.com/efeslab/ConsumerBench/blob/master/scripts/parse-results-whisper-log.py)

## 📝 Experiment Configurations

### Exclusive Execution
| Application | Config |
|-------------|--------|
| Chatbot | [`configs/workflow_chatbot.yml`](https://github.com/efeslab/ConsumerBench/blob/master/configs/workflow_chatbot.yml) |
| LiveCaptions | [`configs/workflow_live_captions.yml`](https://github.com/efeslab/ConsumerBench/blob/master/configs/workflow_live_captions.yml) |
| ImageGen | [`configs/workflow_imagegen.yml`](https://github.com/efeslab/ConsumerBench/blob/master/configs/workflow_imagegen.yml) |

> **CPU-only:** Change `device` from "gpu" to "cpu" in the configs.

### Concurrent Execution
- **Greedy allocation:** [`configs/workflow_chatbot_imagegen_live_captions.yml`](https://github.com/efeslab/ConsumerBench/blob/master/configs/workflow_chatbot_imagegen_live_captions.yml)
- **GPU partitioning:** [`configs/workflow_chatbot_imagegen_live_captions_mps.yml`](https://github.com/efeslab/ConsumerBench/blob/master/configs/workflow_chatbot_imagegen_live_captions_mps.yml)

### Model Sharing (Inference Server)
- **Config:** [`configs/workflow_chatbot_deep_research.yml`](https://github.com/efeslab/ConsumerBench/blob/master/configs/workflow_chatbot_deep_research.yml)
- Edit [`example_workflow/llamacpp_server.sh`](https://github.com/efeslab/ConsumerBench/blob/master/example_workflow/llamacpp_server.sh) to add `-c 128000 -nkvo` for Chatbot-KVCache-CPU

### End-to-End User Workflow
- **Greedy allocation:** [`configs/workflow_content_creation.yml`](https://github.com/efeslab/ConsumerBench/blob/master/configs/workflow_content_creation.yml)
- **GPU partitioning:** [`configs/workflow_content_creation_mps.yml`](https://github.com/efeslab/ConsumerBench/blob/master/configs/workflow_content_creation_mps.yml)
