#!/usr/bin/env python3
"""KV-cache preemption experiment: a long-running, large-KV coding agent on a
shared vLLM server, with and without bursty high-priority chatbot requests.

Shows that when bursty latency-sensitive chatbot requests (higher priority) share
the GPU with a sustained agent that holds a large KV cache, vLLM PREEMPTS the
agent's sequence (recompute mode) under KV-cache pressure, inflating the agent's
end-to-end latency vs. running in isolation. This is the KV-eviction mechanism
the paper does not evaluate (asymmetric: one large stateful consumer vs. bursty
short requests at 8B scale on a 24GB GPU).

Two conditions, identical agent workload:
  isolated : agent alone
  contended: agent + repeated bursts of high-priority chatbot requests

Usage:
  python scripts/kv_preemption_experiment.py --mode isolated  --out results_kv/iso.json
  python scripts/kv_preemption_experiment.py --mode contended --out results_kv/cont.json
"""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"

# vLLM priority scheduling: LOWER number = HIGHER priority.
AGENT_PRIORITY = 10   # background agent: low priority (preemptible)
CHAT_PRIORITY = 0     # latency-sensitive chatbot: high priority (preempts agent)

# Large repeated code context so the agent holds a big KV footprint (~12k tokens).
_CODE_BLOCK = "def process_item(item, config):\n    return item * config.factor + config.offset\n"

CODING_STEPS = [
    "Review this codebase and list 5 concrete performance problems.",
    "Rewrite process_item to be vectorized with numpy. Show full code.",
    "Add input validation and type hints throughout. Show full code.",
    "Write a pytest suite covering edge cases and concurrency. Show full code.",
    "There is a subtle off-by-one bug in batch handling. Find and fix it; explain.",
    "Refactor into a class-based pipeline with pluggable stages. Show full code.",
    "Add structured logging and error handling. Show full code.",
    "Write a benchmark script and summarize the final architecture.",
]

CHAT_PROMPTS = [
    "Write a comprehensive, detailed essay (at least 1200 words) explaining the CAP theorem with examples.",
    "Write a long, thorough explanation (at least 1200 words) of TCP vs UDP with real-world scenarios.",
    "Write a detailed step-by-step walkthrough (at least 1200 words) of how HTTPS/TLS works.",
    "Write an in-depth explanation (at least 1200 words) of the Raft consensus algorithm.",
    "Write a long detailed article (at least 1200 words) on B-trees and database indexing.",
    "Write a thorough explanation (at least 1200 words) of copy-on-write memory and fork().",
]


def _post(messages, max_tokens, priority, timeout=1200):
    payload = json.dumps({
        "model": "Qwen3-8B",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "priority": priority,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    return dt, d.get("usage", {})


def run_agent(num_steps, ctx_blocks, max_tokens):
    """Long-running coding agent: multi-step, growing conversation (large KV).
    Returns (total_latency, per_step_latencies, final_prompt_tokens)."""
    base_ctx = "Here is the codebase to work on:\n" + (_CODE_BLOCK * ctx_blocks) + "\n"
    messages = [{"role": "system", "content": "You are an expert software engineer. Always return complete code."}]
    step_lat = []
    last_prompt_tokens = 0
    t0 = time.time()
    for i in range(num_steps):
        instr = CODING_STEPS[i % len(CODING_STEPS)]
        content = (base_ctx + instr) if i == 0 else instr
        messages.append({"role": "user", "content": content})
        dt, usage = _post(messages, max_tokens, AGENT_PRIORITY)
        step_lat.append(dt)
        last_prompt_tokens = usage.get("prompt_tokens", last_prompt_tokens)
        # Append a synthetic assistant turn to keep context (KV) growing without
        # depending on the actual returned text length.
        messages.append({"role": "assistant", "content": "(done step %d)" % (i + 1)})
        print(f"[agent] step {i+1}/{num_steps}: {dt:.1f}s prompt_tokens~{usage.get('prompt_tokens',0)}", flush=True)
    return time.time() - t0, step_lat, last_prompt_tokens


def run_chatbot_bursts(stop_event, burst_size, gap_s, results, chat_max_tokens=500):
    """Bursty high-priority chatbot: repeated bursts until stop_event is set."""
    burst_id = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size * 2) as ex:
        while not stop_event.is_set():
            futs = []
            for j in range(burst_size):
                p = CHAT_PROMPTS[(burst_id + j) % len(CHAT_PROMPTS)]
                futs.append(ex.submit(_post, [{"role": "user", "content": p}], chat_max_tokens, CHAT_PRIORITY))
            for f in futs:
                try:
                    dt, _ = f.result()
                    results.append(dt)
                except Exception as e:
                    print(f"[chat] error: {e}", flush=True)
            burst_id += 1
            print(f"[chat] burst {burst_id} done ({burst_size} reqs), gap {gap_s}s", flush=True)
            stop_event.wait(gap_s)


def main():
    import threading
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["isolated", "contended"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--ctx-blocks", type=int, default=620)   # ~11k token context (large KV)
    ap.add_argument("--agent-max-tokens", type=int, default=1000)
    ap.add_argument("--burst-size", type=int, default=12)
    ap.add_argument("--burst-gap", type=float, default=1.0)
    ap.add_argument("--chat-max-tokens", type=int, default=500)
    args = ap.parse_args()

    chat_lat = []
    stop = threading.Event()
    chat_thread = None
    if args.mode == "contended":
        chat_thread = threading.Thread(target=run_chatbot_bursts,
                                       args=(stop, args.burst_size, args.burst_gap, chat_lat, args.chat_max_tokens),
                                       daemon=True)
        chat_thread.start()
        time.sleep(2)  # let first burst start so the agent meets contention early

    total, step_lat, prompt_tokens = run_agent(args.num_steps, args.ctx_blocks, args.agent_max_tokens)

    stop.set()
    if chat_thread:
        chat_thread.join(timeout=30)

    result = {
        "mode": args.mode,
        "agent_total_latency_s": total,
        "agent_step_latencies_s": step_lat,
        "agent_final_prompt_tokens": prompt_tokens,
        "num_steps": args.num_steps,
        "chat_requests": len(chat_lat),
        "chat_latencies_s": chat_lat,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n=== {args.mode.upper()} ===")
    print(f"agent total latency: {total:.1f}s over {args.num_steps} steps "
          f"(ctx ~{prompt_tokens} tokens)")
    if chat_lat:
        print(f"chatbot reqs: {len(chat_lat)}, mean {sum(chat_lat)/len(chat_lat):.1f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
