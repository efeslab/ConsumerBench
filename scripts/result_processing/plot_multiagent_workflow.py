#!/usr/bin/env python3
"""Plot request-level latency breakdown for 2-agent workflows.

This script reads ConsumerBench task perf logs and creates a figure similar to the
provided timeline image:
- one lane per agent (Coding Agent, DB Agent)
- colored segments for each request-level phase (LLM vs Tool)
- summary box with per-agent average TPOT/TTFT

Expected inputs are result folders that contain either per-use logs like:
    task_Coding Agent LLM_u0_perf.log
    task_Coding Agent Tool_u0_perf.log
    task_DB Agent LLM_u0_perf.log
    task_DB Agent Tool_u0_perf.log

or shared-lifecycle logs like:
    task_Coding Agent LLM_perf.log
    task_Coding Agent Tool_perf.log
    task_DB Agent LLM_perf.log
    task_DB Agent Tool_perf.log

Usage:
    python scripts/result_processing/plot_2agent_latency_breakdown.py \
      --left-dir results/2agents_v2_naive \
      --right-dir results/2agents_v2_optimized \
      --output results/2agents_latency_breakdown.png
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
PHASE_COLORS = {
    "LLM": "#5B8FCB",
    "Tool": "#F0A444",
}

AGENT_ORDER = ["Coding Agent", "DB Agent"]
AGENT_LABELS = {
    "Coding Agent": "Coding Agent\n(Qwen3-8B)",
    "DB Agent": "DB Agent\n(Llama-3.2-3B)",
}
SUMMARY_EDGE_COLORS = ["#E53935", "#43A047"]
GRID_COLOR = "#C7CFD9"
TITLE_COLOR = "#111111"
LABEL_COLOR = "#222222"
SUMMARY_FONT = "DejaVu Sans Mono"
UI_FONT = "DejaVu Sans"


@dataclass
class Segment:
    agent: str
    phase: str
    start: float
    duration: float


@dataclass
class ScenarioData:
    title: str
    segments_by_agent: Dict[str, List[Segment]]
    ttft_by_agent: Dict[str, List[float]]
    tpot_by_agent: Dict[str, List[float]]


def _extract_first_float(text: str) -> Optional[float]:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def _parse_perf_log(perf_log: Path) -> Optional[dict]:
    raw = perf_log.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    task_id = None
    app_type = None
    start_time = None
    node_records: List[dict] = []

    in_nodes = False
    current_record = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("task_id:"):
            task_id = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("app_type:"):
            app_type = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("start_time:"):
            start_time = _extract_first_float(stripped)
        elif stripped == "Execution times for each node:":
            in_nodes = True
            continue
        elif stripped.startswith("Task ") and stripped.endswith("results:"):
            in_nodes = False

        if not in_nodes:
            continue

        if stripped.startswith("Node ") and stripped.endswith(":"):
            current_record = {"node_id": stripped[len("Node ") : -1]}
            node_records.append(current_record)
            continue

        if current_record is None:
            continue

        if stripped.startswith("Start time:"):
            current_record["start_time"] = _extract_first_float(stripped)
        elif stripped.startswith("End time:"):
            current_record["end_time"] = _extract_first_float(stripped)
        elif stripped.startswith("Execution time:"):
            duration = _extract_first_float(stripped)
            if duration is not None:
                current_record["execution_time"] = duration

    results_marker = "results:\n"
    marker_idx = raw.find(results_marker)
    results = []
    if marker_idx != -1:
        results_raw = raw[marker_idx + len(results_marker) :].strip()
        if results_raw:
            try:
                results = ast.literal_eval(results_raw)
            except (ValueError, SyntaxError):
                results = []

    if not task_id or not app_type or start_time is None:
        return None

    return {
        "task_id": task_id,
        "app_type": app_type,
        "start_time": float(start_time),
        "node_records": node_records,
        "results": results,
    }


def _infer_agent(task_id: str) -> Optional[str]:
    if task_id.startswith("Coding Agent"):
        return "Coding Agent"
    if task_id.startswith("DB Agent"):
        return "DB Agent"
    return None


def _infer_phase(app_type: str) -> Optional[str]:
    if app_type in {"Chatbot", "ChatbotHF"}:
        return "LLM"
    if app_type in {"DbBench", "OsTool", "SleepApplication"}:
        return "Tool"
    return None


def _request_segments_from_task(task: dict, agent: str, phase: str) -> List[Segment]:
    indexed: List[Tuple[int, dict]] = []
    for node_record in task["node_records"]:
        node_name = node_record.get("node_id", "")
        idx_match = re.search(r"_(\d+)$", node_name)
        if idx_match:
            indexed.append((int(idx_match.group(1)), node_record))

    if len(indexed) < 3:
        return []

    indexed.sort(key=lambda x: x[0])

    fallback_time = task["start_time"]
    segments: List[Segment] = []
    for pos, (_idx, node_record) in enumerate(indexed):
        duration = node_record.get("execution_time")
        if duration is None:
            continue

        start = node_record.get("start_time")
        if start is None:
            start = fallback_time
        fallback_time = max(fallback_time, start + duration)

        if pos == 0 or pos == len(indexed) - 1:
            continue

        segments.append(Segment(agent=agent, phase=phase, start=start, duration=duration))

    return segments


def _extract_llm_metrics(results: list) -> Tuple[List[float], List[float]]:
    ttft_values: List[float] = []
    tpot_values: List[float] = []

    if not isinstance(results, list):
        return ttft_values, tpot_values

    for item in results:
        if not isinstance(item, dict):
            continue

        ttft = item.get("ttft")
        tpot = item.get("tpot")

        if isinstance(ttft, (int, float)) and math.isfinite(ttft):
            ttft_values.append(float(ttft))
        if isinstance(tpot, (int, float)) and math.isfinite(tpot):
            tpot_values.append(float(tpot))

    return ttft_values, tpot_values


def load_scenario(results_dir: Path, title: str) -> ScenarioData:
    perf_logs = sorted(results_dir.glob("task_*_perf.log"))
    if not perf_logs:
        raise FileNotFoundError(f"No perf logs found in: {results_dir}")

    segments_by_agent: Dict[str, List[Segment]] = {agent: [] for agent in AGENT_ORDER}
    ttft_by_agent: Dict[str, List[float]] = {agent: [] for agent in AGENT_ORDER}
    tpot_by_agent: Dict[str, List[float]] = {agent: [] for agent in AGENT_ORDER}

    for perf_log in perf_logs:
        parsed = _parse_perf_log(perf_log)
        if not parsed:
            continue

        agent = _infer_agent(parsed["task_id"])
        phase = _infer_phase(parsed["app_type"])

        if not agent or not phase:
            continue

        segments = _request_segments_from_task(parsed, agent, phase)
        segments_by_agent[agent].extend(segments)

        if phase == "LLM":
            ttft_values, tpot_values = _extract_llm_metrics(parsed["results"])
            ttft_by_agent[agent].extend(ttft_values)
            tpot_by_agent[agent].extend(tpot_values)

    for agent in AGENT_ORDER:
        segments_by_agent[agent].sort(key=lambda s: s.start)

    return ScenarioData(
        title=title,
        segments_by_agent=segments_by_agent,
        ttft_by_agent=ttft_by_agent,
        tpot_by_agent=tpot_by_agent,
    )


def _format_mean_std(values_seconds: List[float]) -> str:
    if not values_seconds:
        return "n/a"

    values_ms = [v * 1000.0 for v in values_seconds]
    mu = mean(values_ms)
    sigma = pstdev(values_ms) if len(values_ms) > 1 else 0.0
    return f"{mu:.1f}±{sigma:.1f}ms"


def _plot_scenario(ax, scenario: ScenarioData, summary_edge_color: str):
    all_segments = [s for segs in scenario.segments_by_agent.values() for s in segs]
    if not all_segments:
        ax.set_title(f"{scenario.title} (no data)", fontsize=15, weight="bold")
        ax.axis("off")
        return

    t0 = min(seg.start for seg in all_segments)
    max_end = max(seg.start + seg.duration for seg in all_segments)
    total_span = max(max_end - t0, 1e-6)

    y_positions = {
        "Coding Agent": 12.5,
        "DB Agent": 1.5,
    }
    lane_height = 5.2

    for agent in AGENT_ORDER:
        for seg in scenario.segments_by_agent[agent]:
            x = seg.start - t0
            ax.broken_barh(
                [(x, seg.duration)],
                (y_positions[agent], lane_height),
                facecolors=PHASE_COLORS[seg.phase],
                edgecolors="none",
                linewidth=0,
            )

            # Label wide segments only to avoid clutter.
            if seg.duration > (total_span * 0.02):
                ax.text(
                    x + seg.duration / 2.0,
                    y_positions[agent] + lane_height / 2.0,
                    seg.phase,
                    color="white",
                    fontsize=8,
                    family=UI_FONT,
                    weight="bold",
                    ha="center",
                    va="center",
                )

    ax.set_title(scenario.title, fontsize=20, weight="bold", color=TITLE_COLOR, family=UI_FONT, pad=10)
    ax.set_xlim(0, max_end - t0 + (total_span * 0.03))
    ax.set_ylim(0, 19.5)
    ax.set_yticks([y_positions["Coding Agent"] + lane_height / 2.0, y_positions["DB Agent"] + lane_height / 2.0])
    ax.set_yticklabels([AGENT_LABELS[agent] for agent in AGENT_ORDER], fontsize=13, family=UI_FONT, color=LABEL_COLOR)
    ax.grid(axis="x", linestyle=":", color=GRID_COLOR, alpha=0.9, linewidth=1.0)
    ax.tick_params(axis="x", labelsize=11, colors=LABEL_COLOR)

    coding_summary = (
        "Coding Agent: "
        f"TPOT={_format_mean_std(scenario.tpot_by_agent['Coding Agent'])}, "
        f"TTFT={_format_mean_std(scenario.ttft_by_agent['Coding Agent'])}"
    )
    db_summary = (
        "DB Agent: "
        f"TPOT={_format_mean_std(scenario.tpot_by_agent['DB Agent'])}, "
        f"TTFT={_format_mean_std(scenario.ttft_by_agent['DB Agent'])}"
    )

    summary_text = f"{coding_summary}\n{db_summary}"

    ax.text(
        0.02,
        -0.18,
        summary_text,
        transform=ax.transAxes,
        fontsize=12,
        family=SUMMARY_FONT,
        color=summary_edge_color,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": summary_edge_color, "lw": 1.2},
        va="top",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 2-agent request latency breakdown from perf logs")
    parser.add_argument(
        "--left-dir",
        type=Path,
        required=True,
        help="Results directory for left subplot (e.g., naive/time-slicing)",
    )
    parser.add_argument(
        "--right-dir",
        type=Path,
        default=None,
        help="Results directory for right subplot (e.g., optimized/offline profiling)",
    )
    parser.add_argument(
        "--left-title",
        type=str,
        default="Time Slicing",
        help="Title for the left subplot",
    )
    parser.add_argument(
        "--right-title",
        type=str,
        default="Offline Profiling",
        help="Title for the right subplot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/2agents_latency_breakdown.png"),
        help="Output path for the generated figure",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    left = load_scenario(args.left_dir, args.left_title)

    scenarios = [left]
    edge_colors = [SUMMARY_EDGE_COLORS[0]]

    if args.right_dir is not None:
        right = load_scenario(args.right_dir, args.right_title)
        scenarios.append(right)
        edge_colors.append(SUMMARY_EDGE_COLORS[1])

    ncols = len(scenarios)
    fig, axes = plt.subplots(1, ncols, figsize=(7.2 * ncols, 2.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, scenario, edge_color in zip(axes, scenarios, edge_colors):
        _plot_scenario(ax, scenario, edge_color)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
