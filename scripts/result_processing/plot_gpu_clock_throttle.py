#!/usr/bin/env python3
"""Plot GPU SM clock frequency and throttling over time.

Added to answer reviewer gVNR's question: "to what extent does frequency throttling
under increased loads from multiple tasks affect results?". This visualizes the
achieved SM clock vs. the hardware max, with throttle-active regions shaded, so a
run's slowdown can be attributed to (or cleared of) GPU down-clocking as opposed to
scheduling / kernel contention.

Input: the gpu_utilization.log produced by monitors/record_gpu_mem_compute.sh, whose
columns are (after the leading "[timestamp] GPU 0"):
    parts[4]=SMACT  parts[5]=SMOCC  parts[6]=DRAMA
    parts[7]=SMCLK  parts[8]=SMMAX  parts[9]=DVCCTR(throttle bitmask)  parts[10]=TAPCV
This is a superset of what dcgm_plotter_gpu_compute.py reads; that script keeps
working because the first three fields are unchanged.

Usage:
    python3 plot_gpu_clock_throttle.py <gpu_utilization.log> -o out.png [-s START]
"""

import argparse
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# DVCCTR (field 112) throttle-reason bits. These are stable across driver versions.
# GpuIdle(0x1) and ApplicationsClocksSetting(0x2) are NOT performance-limiting, so we
# exclude them when deciding whether a sample is "throttled".
THROTTLE_BITS = {
    'SwPowerCap':   0x4,
    'HwSlowdown':   0x8,
    'SyncBoost':    0x10,
    'SwThermal':    0x20,
    'HwThermal':    0x40,
    'HwPowerBrake': 0x80,
}
PERF_LIMITING_MASK = 0
for _b in THROTTLE_BITS.values():
    PERF_LIMITING_MASK |= _b


def parse_clock_log(file_path, start_time=None):
    """Return (t, sm_clock, max_clock, throttled, reason_mask) lists."""
    t, sm_clock, max_clock, throttled, reason_mask = [], [], [], [], []
    with open(file_path, 'r') as fh:
        for line in fh:
            if '[' not in line or 'GPU' not in line:
                continue
            m = re.search(r'\[(.*?)\]', line)
            if not m:
                continue
            try:
                ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                continue
            parts = line.split()
            # Need at least up to the throttle bitmask (index 9). Older logs without
            # the appended clock fields are silently skipped.
            if len(parts) < 10:
                continue
            try:
                smclk = float(parts[7])
                smmax = float(parts[8])
                mask = int(float(parts[9]))
            except (ValueError, IndexError):
                continue
            if start_time is None:
                start_time = ts
            t.append((ts - start_time).total_seconds())
            sm_clock.append(smclk)
            max_clock.append(smmax)
            reason_mask.append(mask)
            throttled.append(bool(mask & PERF_LIMITING_MASK))
    return t, sm_clock, max_clock, throttled, reason_mask


def summarize(t, sm_clock, max_clock, throttled, reason_mask):
    """Print a short text summary (also reusable for the rebuttal table)."""
    if not t:
        print("No clock data found.")
        return None
    sm = np.array(sm_clock, dtype=float)
    mx = np.array(max_clock, dtype=float)
    thr = np.array(throttled, dtype=bool)
    # Only consider samples where the GPU is doing work (clock above idle ~300MHz).
    active = sm > 350
    sm_active = sm[active] if active.any() else sm
    pct_throttled = 100.0 * thr.sum() / len(thr)
    reasons = set()
    for mask in reason_mask:
        for label, bit in THROTTLE_BITS.items():
            if mask & bit:
                reasons.add(label)
    summary = {
        'samples': len(t),
        'mean_sm_clock': float(sm_active.mean()) if len(sm_active) else 0.0,
        'min_sm_clock': float(sm_active.min()) if len(sm_active) else 0.0,
        'max_sm_clock': float(sm.max()),
        'hw_max_clock': float(mx.max()),
        'pct_time_throttled': pct_throttled,
        'reasons': sorted(reasons),
    }
    print("=== GPU clock / throttle summary ===")
    print(f"  samples                 : {summary['samples']}")
    print(f"  hardware max SM clock   : {summary['hw_max_clock']:.0f} MHz")
    print(f"  mean SM clock (active)  : {summary['mean_sm_clock']:.0f} MHz "
          f"({100.0 * summary['mean_sm_clock'] / max(summary['hw_max_clock'], 1):.1f}% of max)")
    print(f"  min  SM clock (active)  : {summary['min_sm_clock']:.0f} MHz")
    print(f"  time perf-throttled     : {summary['pct_time_throttled']:.1f}%")
    print(f"  throttle reasons seen   : {', '.join(summary['reasons']) if summary['reasons'] else 'none'}")
    return summary


def create_plot(t, sm_clock, max_clock, throttled, output_file):
    # Clean white background with a light grey grid (no ggplot grey panel).
    plt.style.use('default')
    plt.rcParams.update({'font.size': 16, 'axes.facecolor': 'white',
                         'figure.facecolor': 'white'})
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))

    ax.plot(t, sm_clock, color='#2196F3', linewidth=2, label='Achieved SM clock')
    if max_clock and max(max_clock) > 0:
        ax.axhline(y=max(max_clock), color='#444444', linestyle='--', linewidth=1.5,
                   label=f'HW max ({max(max_clock):.0f} MHz)')

    # Shade contiguous throttled regions.
    any_throttle = False
    in_region = False
    region_start = None
    for i, is_thr in enumerate(throttled):
        if is_thr and not in_region:
            in_region = True
            region_start = t[i]
        elif not is_thr and in_region:
            in_region = False
            ax.axvspan(region_start, t[i], color='#e74c3c', alpha=0.18, linewidth=0)
            any_throttle = True
    if in_region:
        ax.axvspan(region_start, t[-1], color='#e74c3c', alpha=0.18, linewidth=0)
        any_throttle = True

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('SM Clock (MHz)')
    ax.grid(True, color='#cccccc', linestyle='--', linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    # Lighten the frame; drop the top/right spines for a cleaner look.
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#888888')
    # Horizontal legend centered above the axes. Build handles explicitly so the
    # throttle swatch (a 0.18-alpha span on the plot) shows up at a readable
    # opacity in the legend box.
    handles, labels = ax.get_legend_handles_labels()
    if any_throttle:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor='#e74c3c', alpha=0.45, linewidth=0))
        labels.append('Throttled')
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=len(handles), frameon=False, fontsize=13,
              columnspacing=1.8, handletextpad=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot GPU SM clock and throttling over time.')
    parser.add_argument('input_file', help='Path to the DCGM gpu_utilization.log')
    parser.add_argument('-s', '--start_time', help='Start time YYYY-MM-DD_HH:MM:SS (optional)')
    parser.add_argument('-o', '--output', help='Output plot path', default=None)
    parser.add_argument('--tmax', type=float, default=None,
                        help='Only plot/summarize samples up to this many seconds')
    args = parser.parse_args()

    start_time = None
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S')
        except ValueError:
            start_time = None

    t, sm_clock, max_clock, throttled, reason_mask = parse_clock_log(args.input_file, start_time)
    if not t:
        print("No valid clock data found (log may predate the clock/throttle fields).")
        return
    if args.tmax is not None:
        n = sum(1 for x in t if x <= args.tmax)
        t, sm_clock, max_clock, throttled, reason_mask = (
            t[:n], sm_clock[:n], max_clock[:n], throttled[:n], reason_mask[:n])
        print(f"(clipped to first {args.tmax:.0f}s: {n} samples)")
    summarize(t, sm_clock, max_clock, throttled, reason_mask)
    if args.output:
        create_plot(t, sm_clock, max_clock, throttled, args.output)


if __name__ == '__main__':
    main()
