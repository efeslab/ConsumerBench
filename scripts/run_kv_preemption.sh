#!/bin/bash
# Launch a shared vLLM server (priority scheduling + recompute preemption, tight
# KV headroom), run the KV-preemption experiment in both conditions
# (isolated vs contended), then tear the server down. Per the experiment spec:
#   - shared vLLM for agent + chatbot
#   - --scheduling-policy priority, chatbot higher priority than agent
#   - constrained KV headroom (gpu_memory_utilization) + large max_model_len
#   - --preemption-mode recompute (show preemption hurts latency)
set -u
REPO=/home/cc/ConsumerBench
cd "$REPO"
OUT="${1:-$REPO/results_kv}"
GPU_UTIL="${2:-0.84}"
EXP_ARGS="${3:-}"
MAX_MODEL_LEN="${4:-16384}"
mkdir -p "$OUT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vllm
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$REPO/.env" | cut -d= -f2)

echo "[kv] launching shared vLLM server..."
VLLM_ATTENTION_BACKEND=XFORMERS nohup python -m vllm.entrypoints.openai.api_server \
  --model "$REPO/models/Qwen3-8B-HF" \
  --served-model-name Qwen3-8B \
  --port 8090 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 64 \
  --scheduling-policy priority \
  --preemption-mode recompute \
  --dtype float16 \
  > "$OUT/vllm_server.log" 2>&1 &

# wait for readiness
for i in $(seq 1 120); do
  grep -q "Application startup complete" "$OUT/vllm_server.log" 2>/dev/null && { echo "[kv] server up"; break; }
  sleep 3
done
grep -iE "reserved for KV Cache|Maximum concurrency" "$OUT/vllm_server.log" | head

# Condition 1: isolated (agent alone)
echo "[kv] === ISOLATED ==="
python scripts/kv_preemption_experiment.py --mode isolated --out "$OUT/isolated.json" $EXP_ARGS

sleep 5  # let server settle

# Condition 2: contended (agent + bursty high-priority chatbot)
echo "[kv] === CONTENDED ==="
python scripts/kv_preemption_experiment.py --mode contended --out "$OUT/contended.json" $EXP_ARGS

# preemption evidence
echo "[kv] preemption events logged by vLLM:"
grep -c "is preempted by PreemptionMode" "$OUT/vllm_server.log"

# teardown
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 3
echo "[kv] final GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "[kv] done. results in $OUT"
