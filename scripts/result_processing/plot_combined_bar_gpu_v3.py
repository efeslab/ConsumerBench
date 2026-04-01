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
GLOBAL_FONT_SIZE = 24

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
    'Greedy':    os.path.join(BASE, 'GreedyNew'),
    'Even MPS':  os.path.join(BASE, 'EvenMPSNew'),
    'TGS':       os.path.join(BASE, 'TGSBestNew'),
    'Tally':     os.path.join(BASE, 'TallyBestNew'),
    'Ideal MPS': os.path.join(BASE, 'OptimalMPSNew'),
}

SAVE_DIR = '/home/yilegu/ConsumerBench/ConsumerBench/scripts/plots'
os.makedirs(SAVE_DIR, exist_ok=True)


# ── Helper: extract metrics from one experiment folder ───────────────────────
def extract_metrics(folder):
    """Return (latency_values, latency_stds, slo_values) for one experiment."""
    chat = pd.read_csv(os.path.join(folder, 'task_chat1_u0_perf.csv'))
    img  = pd.read_csv(os.path.join(folder, 'task_imagegen1_u0_perf.csv'))
    lv   = pd.read_csv(os.path.join(folder, 'task_lv1_u0_perf.csv'))

    # Normalized latencies (mean / SLO)
    chatbot_ttft = chat['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_tpot = chat['tpot'].mean() / SLOs['chatbot-tpot']
    imagegen_lat = img['total time'].mean() / SLOs['imagegen']
    livecaption_lat = lv['time'].mean() / SLOs['livecaption']

    latency_vals = [chatbot_ttft, chatbot_tpot, imagegen_lat, livecaption_lat]

    # Normalized stds
    chatbot_ttft_std = chat['ttft'].std() / SLOs['chatbot-ttft']
    chatbot_tpot_std = chat['tpot'].std() / SLOs['chatbot-tpot']
    imagegen_std = img['total time'].std() / SLOs['imagegen']
    livecaption_std = lv['time'].std() / SLOs['livecaption']

    latency_stds = [chatbot_ttft_std, chatbot_tpot_std, imagegen_std, livecaption_std]

    # SLO attainment
    chat_slo = 100 * (1 - ((chat['ttft'] > SLOs['chatbot-ttft']) |
                            (chat['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chat))
    img_slo = 100 * (1 - (img['total time'] > SLOs['imagegen']).sum() / len(img))
    lv_slo = 100 * (1 - (lv['time'] > SLOs['livecaption']).sum() / len(lv))

    slo_vals = [chat_slo, img_slo, lv_slo]
    return latency_vals, latency_stds, slo_vals


# ── Main plotting function ───────────────────────────────────────────────────
def plot_combined():
    plt.rcParams.update({'font.size': GLOBAL_FONT_SIZE})

    config_names = list(DIRS.keys())
    n_configs = len(config_names)

    # Collect metrics
    all_latency = []
    all_latency_std = []
    all_slo = []
    for name in config_names:
        lv, ls, sv = extract_metrics(DIRS[name])
        all_latency.append(lv)
        all_latency_std.append(ls)
        all_slo.append(sv)

    # ── Figure layout: 3 rows ────────────────────────────────────────────────
    # Row 1: Normalized Latency (linear scale)
    # Row 2: SLO attainment
    # Row 3: GPU utilization subplots
    fig = plt.figure(figsize=(28, 18))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.45)

    # ── Colors / hatches for each metric (application) ───────────────────────
    latency_labels = ['Chatbot TTFT', 'Chatbot TPOT', 'ImageGen', 'LiveCaption']
    latency_colors = ['#778899', '#778899', '#A0522D', '#C71585']
    latency_hatches = ['///', '...', '+++', 'xxx']
    n_lat_metrics = len(latency_labels)

    slo_labels = ['Chatbot', 'ImageGen', 'LiveCaption']
    slo_colors = ['#778899', '#A0522D', '#C71585']
    slo_hatches = ['///', '+++', 'xxx']
    n_slo_metrics = len(slo_labels)

    bar_width = 0.15

    # ── Row 1: Normalized Latency (linear scale) ────────────────────────────
    ax_lat = fig.add_subplot(gs[0, 0])
    group_positions = np.arange(n_configs)

    for j in range(n_lat_metrics):
        offset = (j - n_lat_metrics / 2 + 0.5) * bar_width
        positions = group_positions + offset
        vals = [all_latency[i][j] for i in range(n_configs)]
        stds = [all_latency_std[i][j] for i in range(n_configs)]
        ax_lat.bar(positions, vals, yerr=stds, width=bar_width,
                   facecolor='white', edgecolor=latency_colors[j],
                   hatch=latency_hatches[j], linewidth=1.5,
                   label=latency_labels[j], capsize=3,
                   error_kw={'linewidth': 1.2})

    ax_lat.set_ylabel('Normalized Latency')
    ax_lat.set_xticks(group_positions)
    ax_lat.set_xticklabels(config_names)
    ax_lat.hlines(y=1.0, xmin=-0.5, xmax=n_configs - 0.5,
                  color='green', linestyle='--', linewidth=2, label='Latency Threshold')
    ax_lat.grid(axis='y', alpha=0.2)

    # Cap y-axis at a reasonable value to keep bars readable
    # (large error bars on outlier configs won't blow up the scale)
    all_means = [all_latency[i][j] for i in range(n_configs) for j in range(n_lat_metrics)]
    cap = max(max(all_means) * 1.4, 1.5)
    ax_lat.set_ylim(0, cap)

    handles_lat, labels_lat = ax_lat.get_legend_handles_labels()
    ax_lat.legend(handles_lat, labels_lat, loc='upper center', ncol=5,
                  frameon=False, bbox_to_anchor=(0.5, 1.18),
                  fontsize=GLOBAL_FONT_SIZE * 0.75)

    # ── Row 2: SLO Attainment bars ──────────────────────────────────────────
    ax_slo = fig.add_subplot(gs[1, 0])

    for j in range(n_slo_metrics):
        offset = (j - n_slo_metrics / 2 + 0.5) * bar_width
        positions = group_positions + offset
        vals = [all_slo[i][j] for i in range(n_configs)]
        ax_slo.bar(positions, vals, width=bar_width,
                   facecolor='white', edgecolor=slo_colors[j],
                   hatch=slo_hatches[j], linewidth=1.5, label=slo_labels[j])
        # Annotate percentages
        for pos, val in zip(positions, vals):
            ax_slo.text(pos, val + 1.5, f'{val:.0f}%', ha='center', va='bottom',
                        fontsize=GLOBAL_FONT_SIZE * 0.5, color=slo_colors[j])

    ax_slo.set_ylabel('SLO Attainment (%)')
    ax_slo.set_xticks(group_positions)
    ax_slo.set_xticklabels(config_names)
    ax_slo.set_ylim(0, 120)
    ax_slo.set_yticks([0, 20, 40, 60, 80, 100])
    ax_slo.hlines(y=100.0, xmin=-0.5, xmax=n_configs - 0.5,
                  color='green', linestyle='--', linewidth=2, label='SLO Target')

    handles_slo, labels_slo = ax_slo.get_legend_handles_labels()
    ax_slo.legend(handles_slo, labels_slo, loc='upper center', ncol=4,
                  frameon=False, bbox_to_anchor=(0.5, 1.18),
                  fontsize=GLOBAL_FONT_SIZE * 0.75)

    # ── Row 3: GPU Utilization time series ───────────────────────────────────
    gs_gpu = gridspec.GridSpecFromSubplotSpec(1, n_configs, subplot_spec=gs[2],
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
        ax_gpu.set_title(name, fontsize=GLOBAL_FONT_SIZE * 0.7)
        ax_gpu.grid(True, linestyle='--', alpha=0.4)

    # Shared horizontal legend for GPU row
    gpu_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=sm_active_color, alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=sm_occupied_color, alpha=0.8),
    ]
    gpu_labels = ['SMACT (Reserved SMs)', 'SMOCC (Occupied SMs)']
    fig.legend(gpu_handles, gpu_labels, loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.33),
               fontsize=GLOBAL_FONT_SIZE * 0.75)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = os.path.join(SAVE_DIR, 'combined_latency_slo_gpu_util.pdf')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    save_path_png = os.path.join(SAVE_DIR, 'combined_latency_slo_gpu_util.png')
    fig.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path_png}")
    plt.close()


if __name__ == '__main__':
    plot_combined()
