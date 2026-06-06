#!/usr/bin/env python3
"""Compare GPU frequency-throttling across runs (isolated vs. concurrent).

Answers reviewer gVNR: "to what extent does frequency throttling under increased
loads from multiple tasks affect results?" by tabulating, for each run directory, the
achieved SM clock and the fraction of time the GPU was performance-throttled. The
intended comparison is the 3 isolated runs (chatbot / imagegen / live_captions)
against the concurrent run: if the concurrent run shows a markedly lower achieved
clock and more power-cap/thermal throttling, that quantifies the throttling
contribution to the concurrency slowdown. If the clock is comparable, the slowdown is
attributable to scheduling/kernel contention instead.

Usage:
    python3 compare_throttle.py LABEL=path/to/run_dir [LABEL2=path2 ...]
    # each path is a results dir containing gpu_utilization.log
"""

import sys
import os

# Reuse the parser/summary from the single-run plotter.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_gpu_clock_throttle import parse_clock_log, THROTTLE_BITS  # noqa: E402

import numpy as np  # noqa: E402


def analyze(run_dir):
    log = os.path.join(run_dir, 'gpu_utilization.log')
    if not os.path.exists(log):
        return None
    t, sm, mx, thr, masks = parse_clock_log(log)
    if not t:
        return None
    sm = np.array(sm, dtype=float)
    active = sm > 350  # ignore idle samples (~300 MHz)
    sm_active = sm[active] if active.any() else sm
    thr = np.array(thr, dtype=bool)
    reasons = set()
    for m in masks:
        for label, bit in THROTTLE_BITS.items():
            if m & bit:
                reasons.add(label)
    hw_max = max(mx) if mx else 0
    return {
        'mean_clock': float(sm_active.mean()) if len(sm_active) else 0.0,
        'min_clock': float(sm_active.min()) if len(sm_active) else 0.0,
        'p05_clock': float(np.percentile(sm_active, 5)) if len(sm_active) else 0.0,
        'hw_max': hw_max,
        'pct_of_max': 100.0 * float(sm_active.mean()) / hw_max if hw_max else 0.0,
        'pct_throttled': 100.0 * thr.sum() / len(thr),
        'reasons': sorted(reasons),
        'duration': t[-1],
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rows = []
    for arg in sys.argv[1:]:
        if '=' not in arg:
            print(f"skip (need LABEL=path): {arg}")
            continue
        label, path = arg.split('=', 1)
        res = analyze(path)
        if res is None:
            print(f"skip (no clock data): {label} -> {path}")
            continue
        rows.append((label, res))

    if not rows:
        print("No runs with clock data found.")
        sys.exit(1)

    hdr = (f"{'run':<16} {'dur(s)':>7} {'mean':>7} {'p05':>6} {'min':>6} "
           f"{'%max':>6} {'%throt':>7}  reasons")
    print(hdr)
    print('-' * len(hdr))
    for label, r in rows:
        print(f"{label:<16} {r['duration']:>7.0f} {r['mean_clock']:>7.0f} "
              f"{r['p05_clock']:>6.0f} {r['min_clock']:>6.0f} {r['pct_of_max']:>6.1f} "
              f"{r['pct_throttled']:>7.1f}  {','.join(r['reasons']) or 'none'}")

    # Highlight isolated-vs-concurrent delta if a 'concurrent' row is present.
    conc = next((r for l, r in rows if 'concurrent' in l.lower()), None)
    iso = [r for l, r in rows if 'concurrent' not in l.lower()]
    if conc and iso:
        iso_mean = np.mean([r['mean_clock'] for r in iso])
        delta = iso_mean - conc['mean_clock']
        print()
        print(f"Mean achieved SM clock: isolated avg {iso_mean:.0f} MHz vs "
              f"concurrent {conc['mean_clock']:.0f} MHz "
              f"(delta {delta:.0f} MHz, {100.0 * delta / iso_mean:.1f}% lower under concurrency)")
        print(f"Time perf-throttled under concurrency: {conc['pct_throttled']:.1f}%")


if __name__ == '__main__':
    main()
