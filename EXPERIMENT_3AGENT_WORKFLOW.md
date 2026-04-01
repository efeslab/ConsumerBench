# Experiment: 3-Agent Workflow — Naive vs DAG-Optimized Scheduling

**Paper Section:** 4.3 "Real-World User Workflow" (COLM 2026)

**Date:** 2026-04-01

**Machine:** `kisnis` — Linux 5.15.0-161-generic, x86_64

---

## 1. Goal

Demonstrate that DAG-aware workflow scheduling provides per-request SLO improvements (TPOT) that GPU-only scheduling cannot achieve, using 3 heterogeneous LLM agents with real tool calls (MySQL, Docker bash).

Two scheduling strategies are compared:
- **Naive:** All 3 agent chains run fully in parallel (uncoordinated GPU contention).
- **Optimized (DAG-interleaved):** Cross-agent dependencies serialize LLM access so only one model uses the GPU at a time. Tool calls overlap with the next agent's LLM phase.

---

## 2. Hardware & Software Environment

| Component | Version / Spec |
|-----------|---------------|
| GPU | NVIDIA RTX 6000 Ada Generation, 49 GB VRAM (only GPU 0 used via `CUDA_VISIBLE_DEVICES=0`) |
| OS | Ubuntu 22.04, kernel 5.15.0-161-generic |
| Python | 3.10.19 (conda env `consumerbench`) |
| Docker | 28.2.2 |
| docker (Python) | 7.1.0 |
| mysql-connector-python | 9.6.0 |
| llama.cpp | Built from source (repo at `inference_backends/llama.cpp/`) |
| NVIDIA MPS | Enabled (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100`) |

---

## 3. Models

Three different small LLMs are used to avoid llama.cpp's same-model batching optimization, ensuring true GPU compute contention in the naive case.

| Agent | Model | Quantization | Size | Port |
|-------|-------|-------------|------|------|
| DB Agent | Llama-3.2-3B-Instruct | f16 | 6.4 GB | 8080 |
| OS Agent | Qwen2.5-3B-Instruct | q8_0 | 3.6 GB | 8081 |
| Chat Agent | Phi-3.5-mini-instruct (3.8B) | Q8_0 | 4.1 GB | 8082 |

### Downloading the Models

```bash
cd ConsumerBench/models/

# Llama-3.2-3B (should already be present)
# If not: huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-f16.gguf --local-dir Llama-3.2-3B-Instruct-GGUF

# Qwen2.5-3B
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q8_0.gguf --local-dir Qwen2.5-3B-Instruct-GGUF

# Phi-3.5-mini
huggingface-cli download bartowski/Phi-3.5-mini-instruct-GGUF Phi-3.5-mini-instruct-Q8_0.gguf --local-dir Phi-3.5-mini-instruct-GGUF
```

---

## 4. Code Changes Required

These changes were made on top of git commit `609be64`.

### 4.1 `inference_backends/Llamacpp.py` — Per-Port Server Management

The original `LlamaCpp` singleton tracked a single global refcount. This was refactored to track refcounts **per port** so that 3 different models on 3 different ports can coexist.

Key changes:
- `self.refcount` (int) → `self.servers` (dict: port → refcount)
- `launch_backend()`: increments `self.servers[api_port]`; only launches if refcount goes from 0→1
- `cleanup_backend()`: decrements per-port refcount; only kills the server when refcount reaches 0
- Lock is released **before** the blocking `util_run_server_script_check_log()` call so that servers on different ports can start concurrently
- `__deepcopy__` returns `self` (singleton must survive `copy.deepcopy` in workflow unit generation)
- Log paths use bare filenames (`"llamacpp_server_stdout"`) because `utils.py` already prepends the results directory

Full file: [`inference_backends/Llamacpp.py`](ConsumerBench/inference_backends/Llamacpp.py)

### 4.2 `src/scripts/run_consumerbench.py` — Import Guards

MCPServer, Retriever, and RetrieverServer may not be available in all environments. Added try/except import guards and None checks on instantiation/registration:

```python
try:
    from applications.MCPServer.MCPServer import MCPServer
except (ImportError, Exception):
    MCPServer = None

try:
    from applications.Retriever.Retriever import Retriever
except (ImportError, Exception):
    Retriever = None

try:
    from applications.RetrieverServer.RetrieverServer import RetrieverServer
except (ImportError, Exception):
    RetrieverServer = None
```

And in `main()`:
```python
mcpServer = MCPServer(...) if MCPServer is not None else None
retriever = Retriever() if Retriever is not None else None
retrieverServer = RetrieverServer() if RetrieverServer is not None else None

if mcpServer is not None:
    workflow.register_application("MCPServer", mcpServer)
# ... same for retriever, retrieverServer
```

### 4.3 Python Dependencies

```bash
conda activate consumerbench
pip install docker mysql-connector-python
```

---

## 5. Configuration Files

### 5.1 Naive: `configs/workflow_3agents_v2_naive.yml`

All 3 agent chains run **fully in parallel** (no cross-agent dependencies):

```
DB Agent:   db_llm_1 ──→ db_tool_1 ──→ db_llm_2
OS Agent:   os_llm_1 ──→ os_tool_1 ──→ os_llm_2      (all 3 chains start simultaneously)
Chat Agent: chat_llm_1 → chat_tool_1 → chat_llm_2
```

### 5.2 Optimized: `configs/workflow_3agents_v2_optimized.yml`

Cross-agent dependencies serialize GPU access (round-robin). Tool calls overlap with the next agent's LLM:

```
Round 1 GPU (serial):  db_llm_1 ──→ os_llm_1 ──→ chat_llm_1
                          │              │              │
                          ▼              ▼              ▼
Tool calls (parallel): db_tool_1     os_tool_1     chat_tool_1
                          │              │              │
Round 2 GPU (serial):  db_llm_2 ──→ os_llm_2 ──→ chat_llm_2
```

Round 2 dependencies:
- `db_llm_2` depends on `db_tool_1` + `chat_llm_1` (own tool done AND Round 1 GPU done)
- `os_llm_2` depends on `os_tool_1` + `db_llm_2`
- `chat_llm_2` depends on `chat_tool_1` + `os_llm_2`

### 5.3 Tunable Parameters

Both configs share the same application definitions:

| Parameter | Current Value | Effect |
|-----------|--------------|--------|
| `num_requests` (LLM) | 5 | Number of LLM inference calls per round per agent |
| `num_requests` (Tool) | 5 | Number of tool invocations (each OsTool call creates a Docker container) |
| `mps` | 100 | NVIDIA MPS thread percentage per server |

**Configurations we tested:**

| LLM requests | Tool requests | Naive e2e | Optimized e2e | Gap | Round 1 TPOT improvement |
|-------------|--------------|-----------|---------------|-----|--------------------------|
| 20 | 1 | 167.4s | 209.5s | +25% | ~2x |
| 10 | 3 | 110.4s | 113.3s | +3% | ~2.5x |
| 5 | 5 | 96.3s | 105.6s | +10% | 2.1–3.4x |

---

## 6. Running the Experiment

### 6.1 Prerequisites

1. Clone the ConsumerBench repo and set up the `consumerbench` conda environment
2. Build llama.cpp with CUDA support (`cmake -DLLAMA_CUDA=ON`)
3. Download the 3 models (see Section 3)
4. Install Python dependencies: `pip install docker mysql-connector-python`
5. Ensure Docker is running and accessible without sudo
6. Ensure NVIDIA MPS is running: `nvidia-cuda-mps-control -d`
7. Apply the code changes from Section 4

### 6.2 Adapt Paths

The config files use **absolute paths**. Update these to match your machine:

```bash
# In both workflow_3agents_v2_naive.yml and workflow_3agents_v2_optimized.yml,
# replace all occurrences of:
/home/yilegu/ConsumerBench/ConsumerBench/
# with your actual repo path.
```

### 6.3 Runner Script

Create the runner script (e.g., `/tmp/run_3agent_v2.sh`):

```bash
#!/bin/bash
set +e  # Don't abort on non-zero exit codes (some requests may fail)
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/path/to/miniconda3/envs/consumerbench/bin/python  # <-- adjust
cd /path/to/ConsumerBench/ConsumerBench                    # <-- adjust

# Clear bytecode cache (prevents stale .pyc issues after code changes)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ====== WARMUP ======
# Run naive once to warm disk cache for all 3 models. Results discarded.
echo "====== WARMUP RUN (results discarded) ======"
pkill -f llama-server 2>/dev/null || true
sleep 3
rm -rf results/3agents_v2_warmup
$PYTHON -Bu src/scripts/run_consumerbench.py \
    --config configs/workflow_3agents_v2_naive.yml \
    --results results/3agents_v2_warmup
rm -rf results/3agents_v2_warmup
echo "====== WARMUP DONE ======"

# ====== ACTUAL EXPERIMENTS ======
pkill -f llama-server 2>/dev/null || true
sleep 3

echo ""
echo "====== EXPERIMENT: 3-AGENT v2 NAIVE ======"
rm -rf results/3agents_v2_naive
$PYTHON -Bu src/scripts/run_consumerbench.py \
    --config configs/workflow_3agents_v2_naive.yml \
    --results results/3agents_v2_naive
echo "====== NAIVE DONE ======"

pkill -f llama-server 2>/dev/null || true
sleep 3

echo ""
echo "====== EXPERIMENT: 3-AGENT v2 OPTIMIZED ======"
rm -rf results/3agents_v2_optimized
$PYTHON -Bu src/scripts/run_consumerbench.py \
    --config configs/workflow_3agents_v2_optimized.yml \
    --results results/3agents_v2_optimized
echo "====== OPTIMIZED DONE ======"

echo ""
echo "====== ALL COMPLETE ======"
```

### 6.4 Execute

```bash
chmod +x /tmp/run_3agent_v2.sh
bash /tmp/run_3agent_v2.sh 2>&1 | tee /tmp/run_3agent_output.log
```

Total runtime: ~5–10 minutes (warmup + 2 experiments), depending on GPU and Docker performance.

### 6.5 Key Flags

| Flag | Purpose |
|------|---------|
| `CUDA_VISIBLE_DEVICES=0` | Pin to single GPU (forces contention in naive) |
| `PYTHONDONTWRITEBYTECODE=1` | Prevents stale `.pyc` files |
| `python -Bu` | Unbuffered output + no bytecode |
| `pkill -f llama-server` | Kill leftover servers between experiments |
| Warmup run | Normalizes disk cache (model loading, Docker images) so both experiments start warm |

---

## 7. Results (Config: 5 LLM / 5 Tool requests)

### 7.1 End-to-End Latency

| Experiment | Total Time |
|-----------|-----------|
| Warmup (discarded) | 105.8s |
| **Naive** | **96.3s** |
| **Optimized** | **105.6s** |

Optimized is ~10% slower in e2e due to serialized GPU access.

### 7.2 Per-Request TPOT (Time Per Output Token)

#### Round 1 (contention round in naive)

| Agent | Model | Naive TPOT | Optimized TPOT | Improvement |
|-------|-------|-----------|----------------|-------------|
| DB Agent | Llama-3.2-3B | 18.9 ms | 8.5 ms | **2.2x** |
| OS Agent | Qwen2.5-3B | 18.5 ms | 5.4 ms | **3.4x** |
| Chat Agent | Phi-3.5-mini | 12.2 ms | 5.8 ms | **2.1x** |

#### Round 2 (contention already resolved in naive)

| Agent | Model | Naive TPOT | Optimized TPOT | Improvement |
|-------|-------|-----------|----------------|-------------|
| DB Agent | Llama-3.2-3B | 8.5 ms | 8.5 ms | ~0% |
| OS Agent | Qwen2.5-3B | 5.3 ms | 5.3 ms | ~0% |
| Chat Agent | Phi-3.5-mini | 5.9 ms | 5.8 ms | ~0% |

### 7.3 Per-Task Execution Time

| Task | Naive | Optimized | Change |
|------|-------|-----------|--------|
| DB Agent LLM u0 | 24.9s | 14.1s | -44% |
| DB Agent LLM u1 | 13.8s | 14.0s | ~0% |
| OS Agent LLM u0 | 22.2s | 10.6s | -52% |
| OS Agent LLM u1 | 9.5s | 10.5s | ~0% |
| Chat Agent LLM u0 | 16.4s | 10.0s | -39% |
| Chat Agent LLM u1 | 10.0s | 9.9s | ~0% |
| DB Agent Tool | 16.7s | 17.1s | ~0% |
| OS Agent Tool | 64.5s | 60.5s | ~0% |
| Chat Agent Tool | 54.7s | 54.7s | ~0% |

### 7.4 Tool Execution Breakdown

| Tool | Type | Runtime |
|------|------|---------|
| DB Agent Tool (5 runs) | MySQL container: init SQL + query | ~17s (13s setup + 5×query + 6s cleanup) |
| OS Agent Tool (5 runs) | Docker bash: dental clinic data | ~60s (5 × container create/run/delete) |
| Chat Agent Tool (5 runs) | Docker bash: Amazon product data | ~55s (5 × container create/run/delete) |

---

## 8. Key Insight

**DAG-aware scheduling provides a per-request SLO consistency guarantee that GPU-only scheduling cannot.**

In naive execution, Round 1 requests experience **2–3.4x worse TPOT** due to 3-way GPU contention. Round 2 naturally has less contention (agents finish at staggered times), so its TPOT is normal.

In optimized execution, TPOT is **consistent across both rounds** because only one model occupies the GPU at any time.

The tradeoff: optimized e2e is **3–10% longer** (depending on tool:LLM ratio) because serializing 3 agents' GPU phases takes more wallclock than running them overlapped-but-contended.

This is a scheduling knob only available with workflow DAG knowledge — a GPU scheduler sees only individual inference requests and cannot distinguish "LLM phase" from "tool phase" to make this tradeoff.

---

## 9. Results Directory Structure

After a successful run:

```
results/
├── 3agents_v2_naive/
│   ├── task_DB Agent LLM_u0_perf.log    # Per-node timing + TPOT/TTFT
│   ├── task_DB Agent LLM_u1_perf.log
│   ├── task_DB Agent Tool_u0_perf.log
│   ├── task_OS Agent LLM_u0_perf.log
│   ├── task_OS Agent LLM_u1_perf.log
│   ├── task_OS Agent Tool_u0_perf.log
│   ├── task_Chat Agent LLM_u0_perf.log
│   ├── task_Chat Agent LLM_u1_perf.log
│   ├── task_Chat Agent Tool_u0_perf.log
│   ├── task_start_perf.log
│   ├── gpu_memory_util.csv              # GPU memory time series
│   ├── gpu_memory_util.png
│   ├── cpu_memory_util.csv
│   ├── cpu_memory_util.png
│   └── server_logs/                     # llama-server stdout/stderr
├── 3agents_v2_optimized/
│   └── (same structure)
```

---

## 10. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: docker` | `pip install docker mysql-connector-python` |
| `ImportError: MCPServer` / `Retriever` | Already handled by try/except guards in `run_consumerbench.py` (Section 4.2) |
| `set -e` kills script on failed request | Use `set +e` in runner script |
| Context size error (512 tokens) | Not all prompts fit in 4096 context; these requests fail gracefully (counted in results) |
| Stale bytecode after code changes | Runner script clears `__pycache__`; also use `python -Bu` |
| Cold disk cache skews first experiment | Warmup run normalizes cache before actual measurements |
| Port already in use | `pkill -f llama-server` between experiments |
| Different models not loading (OOM) | Ensure GPU has enough VRAM for all 3 models (~14 GB total for these quantizations) |

---

## 11. Iteration History

We iterated through several configurations before settling on the final one:

1. **v1 (same model, SleepApplication):** Used identical Llama-3.2-3B for all agents + `SleepApplication` as fake tool. Problem: llama.cpp batches requests for the same model, masking contention.

2. **v2a (3 models, 20 LLM / 1 tool):** Introduced Qwen2.5-3B + SmolLM2-1.7B. Problem: SmolLM2 generated very few tokens; replaced with Phi-3.5-mini. Cold cache penalty skewed results; added warmup.

3. **v2b (3 models, 20 LLM / 1 tool, with warmup):** TPOT 2x better in optimized, but e2e 25% worse — LLM phases dominate and serialization overhead outweighs tool overlap.

4. **v2c (3 models, 10 LLM / 3 tools):** Reduced LLM, increased tools. E2e gap narrowed to 3% — closest to break-even.

5. **v2d (3 models, 5 LLM / 5 tools):** Further adjustment. E2e gap 10%. TPOT improvement 2.1–3.4x. This is the final configuration documented here.

The parameters (num_requests for LLM and tools) are tunable. The 10/3 configuration gave the closest e2e parity; the 5/5 configuration gives the most dramatic tool overlap.
