#!/bin/bash
# Run one rebuttal experiment condition cleanly.
#
# Usage:
#   scripts/run_rebuttal_condition.sh <config> <results_dir> <regime> [mcp_trace]
#
#   regime = resident   -> -c 32768 --parallel 4            (GPU-resident KV; F3 preemption)
#          = offload    -> -c 65536 --parallel 4 -nkvo      (CPU KV offload;  F2 memory pressure)
#
# Handles: .env load, conda env, Qwen3 thinking-off flags, a ground-truth
# nvidia-smi memory sampler, and cleanup of the detached llama-server afterwards.
set -u
REPO=/home/cc/ConsumerBench
cd "$REPO"

CONFIG="$1"; RESULTS="$2"; REGIME="$3"; TRACE="${4:-}"

# The framework joins the results dir into server-log paths in a way that breaks
# for relative paths (produces results/server_logs/results/...). Force absolute.
case "$RESULTS" in
    /*) : ;;
    *) RESULTS="$REPO/$RESULTS" ;;
esac

set -a; . "$REPO/.env"; set +a
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate consumerbench

export LLAMA_PARALLEL=4
export LLAMA_EXTRA_ARGS='--reasoning-budget 0 --chat-template-kwargs {"enable_thinking":false}'
if [ "$REGIME" == "offload" ]; then
    export LLAMA_CTX=65536
    export LLAMA_NKVO=1
else
    export LLAMA_CTX=32768
    export LLAMA_NKVO=0
fi

rm -rf "$RESULTS"; mkdir -p "$RESULTS"

# Ground-truth GPU memory sampler (the in-framework monitor under-reports).
( while true; do
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
    sleep 0.5
  done ) > "$RESULTS/gpu_mem_groundtruth.csv" 2>/dev/null &
SAMPLER=$!

ARGS=(--config "$CONFIG" --results "$RESULTS")
if [ -n "$TRACE" ]; then ARGS+=(--mcp_trace "$TRACE"); fi

echo "[runner] regime=$REGIME ctx=$LLAMA_CTX nkvo=${LLAMA_NKVO} config=$CONFIG"
timeout 1800 python src/scripts/run_consumerbench.py "${ARGS[@]}"
RC=$?
echo "[runner] orchestrator exit=$RC"

kill "$SAMPLER" 2>/dev/null

# Clean up any detached llama-server (start_new_session=True escapes pkill -f).
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
    echo "[runner] killing leaked GPU pid $pid"; kill -9 "$pid" 2>/dev/null
done
sleep 3
echo "[runner] final GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "[runner] model loaded: $(grep -m1 'general.name' "$RESULTS"/llamacpp_server_stderr_8080.log 2>/dev/null | sed 's/.*= //')"
exit $RC
