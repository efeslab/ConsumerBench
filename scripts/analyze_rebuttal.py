#!/usr/bin/env python3
"""Analyze the rebuttal asymmetric-contention experiment.

Compares a bursty, latency-sensitive Chatbot's SLO attainment when run alone vs.
colocated with a long-running DeepResearch agent on a shared Qwen3-8B server, and
the agent's end-to-end latency in both cases. Also overlays the GPU-memory
timeline so Chatbot SLO misses can be correlated with memory pressure.

Usage:
  python scripts/analyze_rebuttal.py \
      --chat-iso results_reb_chat_iso \
      --dr-iso   results_reb_dr_iso \
      --shared   results_reb_shared \
      --out      results_reb_shared/rebuttal_analysis
"""
import argparse
import ast
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SLOs from the paper (Table 1): Chatbot TTFT 1s, TPOT 0.25s.
TTFT_SLO = 1.0
TPOT_SLO = 0.25


def parse_chat_requests(results_dir):
    """Return list of {'ttft','tpot','itl'} dicts from a Chatbot task perf log."""
    logs = glob.glob(os.path.join(results_dir, "task_*hatbot*_perf.log")) or \
           glob.glob(os.path.join(results_dir, "task_*Chat*_perf.log"))
    if not logs:
        return []
    text = open(logs[0]).read()
    # The 'results:' section holds a Python list literal spanning to EOF.
    idx = text.rfind("results:")
    if idx == -1:
        return []
    list_str = text[idx + len("results:"):].strip()
    # Trim to the outermost [...] bracket.
    start = list_str.find("[")
    depth, end = 0, None
    for i, ch in enumerate(list_str[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        items = ast.literal_eval(list_str[start:end])
    except Exception:
        return []
    return [r for r in items if isinstance(r, dict) and r.get("status") == "chatbot_complete"]


def parse_agent_latency(results_dir):
    """Return DeepResearch agent run-node time (seconds), or None."""
    logs = glob.glob(os.path.join(results_dir, "task_*eep*esearch*_perf.log")) or \
           glob.glob(os.path.join(results_dir, "task_*Research*_perf.log"))
    if not logs:
        return None
    text = open(logs[0]).read()
    idx = text.rfind("results:")
    if idx != -1:
        seg = text[idx:]
        m = re.search(r"'total time': ([0-9.]+)", seg)
        if m:
            return float(m.group(1))
    return None


def slo_stats(reqs):
    n = len(reqs)
    if n == 0:
        return None
    ttfts = [r["ttft"] for r in reqs if r.get("ttft") is not None]
    tpots = [r["tpot"] for r in reqs if r.get("tpot") is not None]
    ttft_ok = sum(1 for t in ttfts if t <= TTFT_SLO)
    tpot_ok = sum(1 for t in tpots if t <= TPOT_SLO)
    both_ok = sum(1 for r in reqs
                  if r.get("ttft", 9e9) <= TTFT_SLO and r.get("tpot", 9e9) <= TPOT_SLO)
    return {
        "n": n,
        "ttft_mean": sum(ttfts) / len(ttfts),
        "ttft_p95": sorted(ttfts)[int(0.95 * (len(ttfts) - 1))],
        "ttft_max": max(ttfts),
        "tpot_mean": sum(tpots) / len(tpots),
        "tpot_p95": sorted(tpots)[int(0.95 * (len(tpots) - 1))],
        "tpot_max": max(tpots),
        "ttft_attain": 100.0 * ttft_ok / len(ttfts),
        "tpot_attain": 100.0 * tpot_ok / len(tpots),
        "both_attain": 100.0 * both_ok / n,
    }


def load_gpu_csv(results_dir):
    """Prefer the ground-truth nvidia-smi sampler (gpu_mem_groundtruth.csv,
    'epoch,MiB' with no header); fall back to the framework monitor."""
    gt = os.path.join(results_dir, "gpu_mem_groundtruth.csv")
    if os.path.exists(gt):
        ts, mem = [], []
        with open(gt) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        ts.append(float(parts[0]))
                        mem.append(float(parts[1]))
                    except ValueError:
                        pass
        if ts:
            t0 = ts[0]
            return [t - t0 for t in ts], mem
    path = os.path.join(results_dir, "gpu_memory_util.csv")
    if not os.path.exists(path):
        return [], []
    ts, mem = [], []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    ts.append(float(parts[0]))
                    mem.append(float(parts[1]))
                except ValueError:
                    pass
    return ts, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-iso", required=True)
    ap.add_argument("--dr-iso", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--out", default="rebuttal_analysis")
    args = ap.parse_args()

    chat_iso = slo_stats(parse_chat_requests(args.chat_iso))
    chat_shared = slo_stats(parse_chat_requests(args.shared))
    dr_iso = parse_agent_latency(args.dr_iso)
    dr_shared = parse_agent_latency(args.shared)

    print("=" * 70)
    print("REBUTTAL: asymmetric contention on shared Qwen3-8B (RTX 6000, 24GB)")
    print("=" * 70)
    print(f"\nSLOs: TTFT <= {TTFT_SLO}s, TPOT <= {TPOT_SLO}s\n")

    def show(label, s):
        if not s:
            print(f"{label}: NO DATA")
            return
        print(f"{label}  (n={s['n']})")
        print(f"  TTFT  mean={s['ttft_mean']:.3f}s  p95={s['ttft_p95']:.3f}s  max={s['ttft_max']:.3f}s  attain={s['ttft_attain']:.0f}%")
        print(f"  TPOT  mean={s['tpot_mean']:.4f}s p95={s['tpot_p95']:.4f}s max={s['tpot_max']:.4f}s attain={s['tpot_attain']:.0f}%")
        print(f"  BOTH-SLO attainment: {s['both_attain']:.0f}%")

    show("Chatbot ISOLATED", chat_iso)
    print()
    show("Chatbot + Agent (SHARED)", chat_shared)
    print(f"\nDeepResearch agent latency:  isolated={dr_iso}s  shared={dr_shared}s")
    if dr_iso and dr_shared:
        print(f"  -> agent slowdown under contention: {dr_shared / dr_iso:.2f}x "
              f"(+{dr_shared - dr_iso:.1f}s)")

    # --- Plot 1: SLO attainment + TPOT comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if chat_iso and chat_shared:
        conds = ["Isolated", "Shared\nw/ agent"]
        ttft_at = [chat_iso["ttft_attain"], chat_shared["ttft_attain"]]
        tpot_at = [chat_iso["tpot_attain"], chat_shared["tpot_attain"]]
        x = range(len(conds))
        w = 0.35
        ax1.bar([i - w / 2 for i in x], ttft_at, w, label="TTFT SLO", color="#4C72B0")
        ax1.bar([i + w / 2 for i in x], tpot_at, w, label="TPOT SLO", color="#DD8452")
        ax1.set_xticks(list(x)); ax1.set_xticklabels(conds)
        ax1.set_ylabel("SLO attainment (%)"); ax1.set_ylim(0, 105)
        ax1.set_title("Chatbot SLO attainment"); ax1.legend()

        ax2.bar([i - w / 2 for i in x], [chat_iso["tpot_mean"], chat_shared["tpot_mean"]],
                w, label="mean", color="#55A868")
        ax2.bar([i + w / 2 for i in x], [chat_iso["tpot_p95"], chat_shared["tpot_p95"]],
                w, label="p95", color="#C44E52")
        ax2.axhline(TPOT_SLO, ls="--", color="k", label=f"TPOT SLO ({TPOT_SLO}s)")
        ax2.set_xticks(list(x)); ax2.set_xticklabels(conds)
        ax2.set_ylabel("TPOT (s/token)"); ax2.set_title("Chatbot per-token latency")
        ax2.legend()
    fig.tight_layout()
    fig.savefig(args.out + "_slo.png", dpi=130)
    print(f"\nSaved {args.out}_slo.png")

    # --- Plot 2: GPU memory timeline (shared run) ---
    ts, mem = load_gpu_csv(args.shared)
    if ts:
        fig2, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, [m / 1024 for m in mem], color="#4C72B0")
        ax.axhline(24, ls="--", color="r", label="24GB GPU capacity")
        ax.set_xlabel("time (s)"); ax.set_ylabel("GPU memory used (GB)")
        ax.set_title("GPU memory during shared agent+Chatbot contention")
        ax.legend()
        fig2.tight_layout()
        fig2.savefig(args.out + "_mem.png", dpi=130)
        print(f"Saved {args.out}_mem.png  (peak {max(mem)/1024:.1f}GB)")


if __name__ == "__main__":
    main()
