import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Global font size
plt.rcParams.update({'font.size': 24})

SLOs = {
    'chatbot-ttft': 1,
    'chatbot-tpot': 0.25,
}

def plot_performance_bar_plots(folder_path):
    # Read CSVs
    chatbot_offload_data = pd.read_csv(os.path.join(folder_path, 'task_chat1_u0_perf_kvcache_cpu.csv'))
    chatbot_data = pd.read_csv(os.path.join(folder_path, 'task_chat1_u0_perf.csv'))

    # Metrics
    chatbot_ttft = chatbot_data['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_tpot = chatbot_data['tpot'].mean() / SLOs['chatbot-tpot']
    chatbot_slo = 100 * (1 - ((chatbot_data['ttft'] > SLOs['chatbot-ttft']) |
                              (chatbot_data['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chatbot_data))
    chatbot_ttft_std = chatbot_data['ttft'].std() / SLOs['chatbot-ttft']
    chatbot_tpot_std = chatbot_data['tpot'].std() / SLOs['chatbot-tpot']

    chatbot_offload_ttft = chatbot_offload_data['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_offload_tpot = chatbot_offload_data['tpot'].mean() / SLOs['chatbot-tpot']
    chatbot_offload_slo = 100 * (1 - ((chatbot_offload_data['ttft'] > SLOs['chatbot-ttft']) |
                              (chatbot_offload_data['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chatbot_offload_data))
    chatbot_offload_ttft_std = chatbot_offload_data['ttft'].std() / SLOs['chatbot-ttft']
    chatbot_offload_tpot_std = chatbot_offload_data['tpot'].std() / SLOs['chatbot-tpot']


    # Bar data
    latency_values = [chatbot_ttft, chatbot_tpot, chatbot_offload_ttft, chatbot_offload_tpot]
    latency_stds = [chatbot_ttft_std, chatbot_tpot_std, chatbot_offload_ttft_std, chatbot_offload_tpot_std]
    latency_colors = ['#778899', '#778899', '#A0522D', '#A0522D']
    latency_hatch = '//'

    slo_values = [chatbot_slo, chatbot_offload_slo]
    slo_colors = ['#778899', '#A0522D']
    slo_hatch = 'x'

    # X positions
    latency_x = [0, 1, 2.5, 3.5]
    slo_x = [5.5, 6.5]
    divider_x = 4.5
    bar_width = 0.8
    alpha = 1
    edge_width = 2

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax2 = ax.twinx()

    # Latency bars
    ax.bar(latency_x, latency_values, yerr=latency_stds, width=bar_width,
           color=latency_colors, hatch=latency_hatch, alpha=alpha,
           edgecolor=latency_colors, linewidth=edge_width, facecolor='white', log=True)

    # SLO bars
    ax2.bar(slo_x, slo_values, width=bar_width, color=slo_colors, hatch=slo_hatch,
            alpha=alpha, edgecolor=slo_colors, linewidth=edge_width, facecolor='white')

    for i in [0, 1, 2, 3]:
        label = 'TTFT' if i == 0 or i == 2 else 'TPOT'
        height = latency_values[i]
        label_x = i if i < 2 else i + 0.5
        ax.text(label_x, 0.001, label,
                ha='center', va='bottom', fontsize=20, fontweight='normal')

    height = 1.25
    ax.text(0.5, height, 'Latency Threshold',
            ha='center', va='bottom', fontsize=20, color='green')


    # Vertical divider
    ax.axvline(x=divider_x, color='gray', linestyle='-', linewidth=2)

    # Horizontal thresholds
    ax.hlines(y=1.0, xmin=-1, xmax=divider_x, color='green', linestyle='--', linewidth=2)
    ax2.hlines(y=100.0, xmin=divider_x, xmax=8, color='green', linestyle='--', linewidth=2)

    # X-axis ticks and labels
    # x_ticks = [0, 1, 2.5, 3.5, 5.5, 6.5]
    # x_labels = ['Chatbot\n(TTFT)', 'Chatbot\n(TPOT)', 'Chatbot-Offload\n(TTFT)', 'Chatbot-Offload\n(TPOT)', 'Chatbot\n(SLO)', 'Chatbot-Offload\n(SLO)']

    # X-axis ticks just at group centers
    ax.set_xticks([0.5, 3])
    ax.set_xticklabels(['', ''])


    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels, rotation=30)
    # Axes formatting
    ax.set_ylabel('Normalized Latency', color='black')
    ax.set_yscale('log')
    ax.set_xlim(-1, 7.5)
    ax.set_ylim(1e-3, 1e3)
    ax.tick_params(axis='y', labelcolor='black')

    ax2.set_ylabel('SLO Attainment (%)', color='black')
    ax2.set_ylim(0, 110)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.tick_params(axis='y', labelcolor='black')

    # Sorted labels and handles
    legend_labels = [
        'KVCache-GPU',
        'KVCache-CPU',
        # 'Chatbot-SLO',
        # 'Chatbot-Offload-SLO'
    ]

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[0], linewidth=edge_width),
        # plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[1], hatch=latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[2], linewidth=edge_width),
        # plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[3], hatch=latency_hatch, linewidth=edge_width),
        # plt.Line2D([], [], color='green', linestyle='--', linewidth=2),
        # plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[0], hatch=slo_hatch, linewidth=edge_width),
        # plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[1], hatch=slo_hatch, linewidth=edge_width),
        # plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[2], hatch=slo_hatch, linewidth=edge_width),
    ]

    # # Display in two rows of 4 items max
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1), columnspacing=0.5)

    # Layout and save
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(folder_path, 'performance_kv_offload_sampling.pdf')
    plt.savefig(plot_path)
    # plt.show()
    plt.close()
    print(f"Plot saved to {plot_path}")

# Example usage:
plot_performance_bar_plots('scripts/plots/bar_plots_kv_offload_sampling')
