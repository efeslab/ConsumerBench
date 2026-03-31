import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys

# Add parent directory to path so we can import the dcgm plotter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcgm_plotter_gpu_compute import parse_dcgm_output

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FONT SIZE — change this single value to resize all text
# ══════════════════════════════════════════════════════════════════════════════
GLOBAL_FONT_SIZE = 28

# ── SLO definitions ──────────────────────────────────────────────────────────
SLOs = {
    'chatbot-ttft': 1,
    'chatbot-tpot': 0.25,
    'imagegen': 28,
    'livecaption': 2,
}

# ── Data directories ─────────────────────────────────────────────────────────
BASE = '/home/yilegu/ConsumerBench/ConsumerBench-Results/New_Results'
DIRS = {
    'Greedy':  os.path.join(BASE, 'GreedyNew'),
    'Even MPS':    os.path.join(BASE, 'EvenMPSNew'),
    'TGS':    os.path.join(BASE, 'TGSBestNew'),
    'Tally':  os.path.join(BASE, 'TallyBestNew'),
    'Ideal MPS':     os.path.join(BASE, 'OptimalMPSNew'),
}

SAVE_DIR = '/home/yilegu/ConsumerBench/ConsumerBench/scripts/plots'
os.makedirs(SAVE_DIR, exist_ok=True)


# ── Helper: extract metrics from one experiment folder ───────────────────────
def extract_metrics(folder):
    """Return (latency_values, latency_stds, slo_values) for one experiment."""
    chat = pd.read_csv(os.path.join(folder, 'task_chat1_u0_perf.csv'))
    img  = pd.read_csv(os.path.join(folder, 'task_imagegen1_u0_perf.csv'))
    lv   = pd.read_csv(os.path.join(folder, 'task_lv1_u0_perf.csv'))

    # Chatbot SLO (both TTFT and TPOT must be within limits)
    chat_slo = 100 * (1 - ((chat['ttft'] > SLOs['chatbot-ttft']) |
                            (chat['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chat))

    # ImageGen SLO
    img_slo = 100 * (1 - (img['total time'] > SLOs['imagegen']).sum() / len(img))

    # LiveCaption SLO
    lv_slo = 100 * (1 - (lv['time'] > SLOs['livecaption']).sum() / len(lv))

    slo_vals = [chat_slo, img_slo, lv_slo]
    return slo_vals


# ── Main plotting function ───────────────────────────────────────────────────
def plot_combined():
    plt.rcParams.update({'font.size': GLOBAL_FONT_SIZE})

    config_names = list(DIRS.keys())
    n_configs = len(config_names)

    # Collect SLO metrics
    all_slo = []
    for name in config_names:
        sv = extract_metrics(DIRS[name])
        all_slo.append(sv)

    # ── Figure layout: 2 rows ────────────────────────────────────────────────
    # Row 1: SLO attainment bar plot (5 groups x 3 bars)
    # Row 2: 5 GPU utilization subplots
    fig = plt.figure(figsize=(28, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.8], hspace=0.50)

    # ── Row 1: SLO Attainment bars ───────────────────────────────────────────
    # 5 groups (one per config), 3 bars each (Chatbot, ImageGen, LiveCaption)
    ax_slo = fig.add_subplot(gs[0, 0])

    metric_labels = ['Chatbot', 'ImageGen', 'LiveCaption']
    n_metrics = len(metric_labels)

    # Colors / hatches for each metric (application)
    metric_colors = ['#778899', '#A0522D', '#C71585']
    hatches = ['///', '+++', 'xxx']

    bar_width = 0.17
    group_positions = np.arange(n_configs)

    for j in range(n_metrics):
        offset = (j - n_metrics / 2 + 0.5) * bar_width
        positions = group_positions + offset
        vals = [all_slo[i][j] for i in range(n_configs)]
        ax_slo.bar(positions, vals, width=bar_width,
                   facecolor='white', edgecolor=metric_colors[j],
                   hatch=hatches[j], linewidth=1.5, label=metric_labels[j])
        # Annotate percentages on top of each bar
        for pos, val in zip(positions, vals):
            ax_slo.text(pos, val + 1.5, f'{val:.0f}%', ha='center', va='bottom',
                        fontsize=GLOBAL_FONT_SIZE * 0.55, color=metric_colors[j])

    ax_slo.set_ylabel('SLO Attainment (%)')
    ax_slo.set_xticks(group_positions)
    ax_slo.set_xticklabels(config_names)
    ax_slo.set_ylim(0, 120)
    ax_slo.set_yticks([0, 20, 40, 60, 80, 100])
    ax_slo.hlines(y=100.0, xmin=-0.5, xmax=n_configs - 0.5,
                  color='green', linestyle='--', linewidth=2, label='SLO Target')

    handles, labels = ax_slo.get_legend_handles_labels()
    ax_slo.legend(handles, labels, loc='upper center', ncol=4,
                  frameon=False, bbox_to_anchor=(0.5, 1.15),
                  fontsize=GLOBAL_FONT_SIZE * 0.8)

    # ── Row 2: GPU Utilization time series ───────────────────────────────────
    gs_gpu = gridspec.GridSpecFromSubplotSpec(1, n_configs, subplot_spec=gs[1],
                                              wspace=0.25)

    sm_active_color = '#666A6D'
    sm_occupied_color = '#FF9800'

    for i, name in enumerate(config_names):
        ax_gpu = fig.add_subplot(gs_gpu[0, i])
        gpu_log = os.path.join(DIRS[name], 'gpu_utilization.log')

        timestamps, sm_active, sm_occupied, mem_bw = parse_dcgm_output(gpu_log)

        if timestamps:
            ax_gpu.fill_between(timestamps, sm_active, color=sm_active_color,
                                alpha=0.8, label='SMACT')
            ax_gpu.fill_between(timestamps, sm_occupied, color=sm_occupied_color,
                                alpha=0.8, label='SMOCC')

        ax_gpu.set_ylim(0, 100)
        ax_gpu.set_xlabel('Time (s)')
        if i == 0:
            ax_gpu.set_ylabel('GPU Util (%)')
        ax_gpu.set_title(name, fontsize=GLOBAL_FONT_SIZE * 0.75)
        ax_gpu.grid(True, linestyle='--', alpha=0.4)

    # Shared horizontal legend for GPU row, placed above the row
    gpu_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=sm_active_color, alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=sm_occupied_color, alpha=0.8),
    ]
    gpu_labels = ['SMACT (Reserved SMs)', 'SMOCC (Occupied SMs)']
    fig.legend(gpu_handles, gpu_labels, loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.46),
               fontsize=GLOBAL_FONT_SIZE * 0.8)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = os.path.join(SAVE_DIR, 'combined_bar_gpu_util.pdf')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    save_path_png = os.path.join(SAVE_DIR, 'combined_bar_gpu_util.png')
    fig.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path_png}")
    plt.close()


if __name__ == '__main__':
    plot_combined()
