#!/usr/bin/env python3
"""Analyze the KV-cache preemption experiment: long-running coding agent latency
isolated vs. contended by bursty high-priority chatbot requests on a shared vLLM
server, plus the vLLM preemption events that explain the slowdown.

Usage:
  python scripts/analyze_kv_preemption.py --dir results_kv_preempt
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def count_preemptions(server_log):
    if not os.path.exists(server_log):
        return 0, []
    n = 0
    cum = []
    with open(server_log) as f:
        for line in f:
            if "is preempted by PreemptionMode" in line:
                n += 1
                m = re.search(r"total_num_cumulative_preemption=(\d+)", line)
                if m:
                    cum.append(int(m.group(1)))
    return n, cum


def peak_kv(server_log):
    if not os.path.exists(server_log):
        return None
    vals = [float(x) for x in re.findall(r"GPU KV cache usage: ([0-9.]+)%", open(server_log).read())]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    iso = load(args.dir, "isolated.json")
    cont = load(args.dir, "contended.json")
    npre, _ = count_preemptions(os.path.join(args.dir, "vllm_server.log"))
    pk = peak_kv(os.path.join(args.dir, "vllm_server.log"))

    print("=" * 70)
    print("KV-CACHE PREEMPTION: coding agent vs bursty chatbot (shared vLLM, Qwen3-8B, RTX 6000 24GB)")
    print("=" * 70)
    it = iso["agent_total_latency_s"]
    ct = cont["agent_total_latency_s"]
    print(f"\nAgent end-to-end latency:")
    print(f"  isolated : {it:.1f}s")
    print(f"  contended: {ct:.1f}s")
    print(f"  -> slowdown {ct/it:.2f}x  (+{ct-it:.1f}s) due to bursty high-priority chatbot")
    print(f"\nAgent per-step latency (s):")
    print(f"  isolated : {[round(x,1) for x in iso['agent_step_latencies_s']]}")
    print(f"  contended: {[round(x,1) for x in cont['agent_step_latencies_s']]}")
    print(f"\nChatbot requests served during contention: {cont['chat_requests']}")
    print(f"vLLM preemption events (RECOMPUTE) on the low-priority agent: {npre}")
    print(f"Peak GPU KV-cache usage: {pk:.1f}%")

    # Plot: per-step agent latency isolated vs contended + totals
    steps = list(range(1, len(iso["agent_step_latencies_s"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    w = 0.38
    ax1.bar([s - w/2 for s in steps], iso["agent_step_latencies_s"], w, label="isolated", color="#4C72B0")
    ax1.bar([s + w/2 for s in steps], cont["agent_step_latencies_s"], w, label="contended (+ bursty chatbot)", color="#C44E52")
    ax1.set_xlabel("agent step"); ax1.set_ylabel("step latency (s)")
    ax1.set_title("Per-step agent latency"); ax1.set_xticks(steps); ax1.legend()

    ax2.bar(["isolated", "contended"], [it, ct], color=["#4C72B0", "#C44E52"])
    ax2.set_ylabel("agent end-to-end latency (s)")
    ax2.set_title(f"Agent total latency  ({ct/it:.2f}x slower)\n{npre} KV preemptions, peak KV {pk:.0f}%")
    for i, v in enumerate([it, ct]):
        ax2.text(i, v + 2, f"{v:.0f}s", ha="center")
    fig.tight_layout()
    out = os.path.join(args.dir, "kv_preemption_analysis.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
