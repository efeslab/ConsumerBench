import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

import argparse

# Global font size
plt.rcParams.update({'font.size': 24})

SLOs = {
    'chatbot-ttft': 1,
    'chatbot-tpot': 0.25,
    'imagegen': 28,
    'livecaption': 2
}

def plot_performance(gpu_folder_path, cpu_folder_path, save_path="scripts/plots/gpu_vs_cpu_latency_and_slo_sampling"):
    # Read CSVs
    # gpu_data = pd.read_csv(os.path.join(gpu_folder_path, 'task_chat1_u0_perf.csv'))
    # cpu_data = pd.read_csv(os.path.join(cpu_folder_path, 'task_chat1_u0_perf.csv'))
    chat_bot_gpu_data = pd.read_csv(os.path.join(gpu_folder_path, 'task_chat1_u0_perf.csv'))
    chat_bot_cpu_data = pd.read_csv(os.path.join(cpu_folder_path, 'task_chat1_u0_perf.csv'))
    image_gen_gpu_data = pd.read_csv(os.path.join(gpu_folder_path, 'task_imagegen1_u0_perf.csv'))
    image_gen_cpu_data = pd.read_csv(os.path.join(cpu_folder_path, 'task_imagegen1_u0_perf.csv'))
    live_caption_gpu_data = pd.read_csv(os.path.join(gpu_folder_path, 'task_lv_u0_perf.csv'))
    live_caption_cpu_data = pd.read_csv(os.path.join(cpu_folder_path, 'task_lv_u0_perf.csv'))

    # print min max
    print(image_gen_gpu_data['total time'].min(), image_gen_gpu_data['total time'].max())
    print(live_caption_gpu_data['time'].min(), live_caption_gpu_data['time'].max())

    chatbot_ttft_gpu = chat_bot_gpu_data['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_tpot_gpu = chat_bot_gpu_data['tpot'].mean() / SLOs['chatbot-tpot']
    chatbot_slo_gpu = 100 * (1 - ((chat_bot_gpu_data['ttft'] > SLOs['chatbot-ttft']) |
                                  (chat_bot_gpu_data['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chat_bot_gpu_data))
    chatbot_ttft_cpu = chat_bot_cpu_data['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_tpot_cpu = chat_bot_cpu_data['tpot'].mean() / SLOs['chatbot-tpot']
    chatbot_slo_cpu = 100 * (1 - ((chat_bot_cpu_data['ttft'] > SLOs['chatbot-ttft']) |
                                  (chat_bot_cpu_data['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chat_bot_cpu_data))
    imagegen_latency_gpu = image_gen_gpu_data['total time'].mean() / SLOs['imagegen']
    imagegen_slo_gpu = 100 * (1 - (image_gen_gpu_data['total time'] > SLOs['imagegen']).sum() / len(image_gen_gpu_data))
    imagegen_latency_cpu = image_gen_cpu_data['total time'].mean() / SLOs['imagegen']
    imagegen_slo_cpu = 100 * (1 - (image_gen_cpu_data['total time'] > SLOs['imagegen']).sum() / len(image_gen_cpu_data))
    livecaption_latency_gpu = live_caption_gpu_data['time'].mean() / SLOs['livecaption']
    livecaption_slo_gpu = 100 * (1 - (live_caption_gpu_data['time'] > SLOs['livecaption']).sum() / len(live_caption_gpu_data))
    livecaption_latency_cpu = live_caption_cpu_data['time'].mean() / SLOs['livecaption']
    livecaption_slo_cpu = 100 * (1 - (live_caption_cpu_data['time'] > SLOs['livecaption']).sum() / len(live_caption_cpu_data))

    gpu_latency_values = [chatbot_ttft_gpu, chatbot_tpot_gpu, imagegen_latency_gpu, livecaption_latency_gpu]
    gpu_latency_stds = [chat_bot_gpu_data['ttft'].std() / SLOs['chatbot-ttft'],
                        chat_bot_gpu_data['tpot'].std() / SLOs['chatbot-tpot'],
                        image_gen_gpu_data['total time'].std() / SLOs['imagegen'],
                        live_caption_gpu_data['time'].std() / SLOs['livecaption']]
    latency_colors = ['#778899', '#778899', '#A0522D', '#C71585']
    gpu_latency_hatch = '//'

    cpu_latency_values = [chatbot_ttft_cpu, chatbot_tpot_cpu, imagegen_latency_cpu, livecaption_latency_cpu]
    cpu_latency_stds = [chat_bot_cpu_data['ttft'].std() / SLOs['chatbot-ttft'],
                        chat_bot_cpu_data['tpot'].std() / SLOs['chatbot-tpot'],
                        image_gen_cpu_data['total time'].std() / SLOs['imagegen'],
                        live_caption_cpu_data['time'].std() / SLOs['livecaption']]

    gpu_slo_values = [chatbot_slo_gpu, imagegen_slo_gpu, livecaption_slo_gpu]
    gpu_slo_colors = ['#778899', '#A0522D', '#C71585']
    cpu_slo_values = [chatbot_slo_cpu, imagegen_slo_cpu, livecaption_slo_cpu]
    cpu_slo_colors = ['#778899', '#A0522D', '#C71585']

    # X positions
    latency_x = [0, 1, 2, 3, 5, 6, 7, 8]
    alpha = 1
    bar_width = 0.8
    edge_width = 1
    fig, ax = plt.subplots(figsize=(12, 6))

    latency_values =  gpu_latency_values + cpu_latency_values
    latency_stds = gpu_latency_stds + cpu_latency_stds
    latency_colors *= 2

    ax.bar(latency_x, latency_values, yerr=latency_stds, width=bar_width,
           color=latency_colors, hatch=gpu_latency_hatch, alpha=alpha,
           edgecolor=latency_colors, linewidth=edge_width, facecolor='white', log=True)
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Latency (log scale)', color='black')
    # ax.set_ylim(1e-2, max(latency_values) * 1.8)
    ax.set_ylim(1e-3, 1e3 + 1e3)
    ax.set_xlim(-1, 9)

    # X-axis ticks just at group centers
    ax.set_xticks([1.5, 6.5])
    ax.set_xticklabels(['GPU', 'CPU'])

    for i in [0, 1, 5, 6]:
        label = 'TTFT' if i == 0 or i == 5 else 'TPOT'
        ax.text(i, 1e-3, label,
                ha='center', va='bottom')

    height = 1.5
    ax.text(0.8, height, 'Latency Threshold',
            ha='center', va='bottom', color='green')
    ax.hlines(y=1.0, xmin=-1, xmax=9, color='green', linestyle='--', linewidth=2)

    # Sorted labels and handles
    legend_labels = [
        'Chatbot',
        'ImageGen',
        'LiveCaptions',
    ]

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[0], hatch=gpu_latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[2], hatch=gpu_latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[3], hatch=gpu_latency_hatch, linewidth=edge_width),
    ]

    # Display in two rows of 4 items max
    ax.legend(legend_handles, legend_labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2))


    fig.tight_layout()
    save_path_lat = os.path.join(save_path, 'gpu_vs_cpu_latency.pdf')
    plt.savefig(save_path_lat)
    print(f"Latency saved to {save_path_lat}")

    # clear
    plt.clf()
    plt.close()

    # SLO bars
    fig, ax2 = plt.subplots(figsize=(12, 6))
    slo_x = [1, 2, 3, 5, 6, 7]
    slo_values = gpu_slo_values + cpu_slo_values
    slo_colors = gpu_slo_colors + cpu_slo_colors
    slo_hatch = 'x'

    # Annotate all SLO percentages
    for idx, val in enumerate(slo_values):
        ax2.text(slo_x[idx], val + 2, f'{int(val)}%', ha='center', va='bottom', color=slo_colors[idx])


    ax2.bar(slo_x, slo_values, width=bar_width, color=slo_colors, hatch=slo_hatch,
            alpha=alpha, edgecolor=slo_colors, linewidth=edge_width, facecolor='white')
    ax2.set_ylabel('SLO Attainment (%)', color='black')
    ax2.set_ylim(0, 115)

    # X-axis ticks just at group centers
    ax2.set_xticks([2, 6])
    ax2.set_xticklabels(['GPU', 'CPU'])
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_xlim(-1, 9)

    height = 102
    # ax2.text(6, height, 'SLO Threshold',
            # ha='center', va='bottom', fontsize=16, color='green')
    ax2.hlines(y=100.0, xmin=-1, xmax=9, color='green', linestyle='--', linewidth=2)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[0], hatch=slo_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[1], hatch=slo_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[2], hatch=slo_hatch, linewidth=edge_width),
    ]

    ax2.legend(legend_handles, legend_labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2))


    fig.tight_layout()
    save_path = os.path.join(save_path, 'gpu_vs_cpu_slo_sampling.pdf')
    plt.savefig(save_path)
    print(f"Latency saved to {save_path}")




def plot_performance_bar_plots(folder_path):
    # Read CSVs
    chatbot_data = pd.read_csv(os.path.join(folder_path, 'task_chat1_u0_perf.csv'))
    imagegen_data = pd.read_csv(os.path.join(folder_path, 'task_imagegen1_u0_perf.csv'))
    livecaption_data = pd.read_csv(os.path.join(folder_path, 'task_lv_u0_perf.csv'))

    # Metrics
    chatbot_ttft = chatbot_data['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_tpot = chatbot_data['tpot'].mean() / SLOs['chatbot-tpot']
    chatbot_slo = 100 * (1 - ((chatbot_data['ttft'] > SLOs['chatbot-ttft']) |
                              (chatbot_data['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chatbot_data))
    chatbot_ttft_std = chatbot_data['ttft'].std() / SLOs['chatbot-ttft']
    chatbot_tpot_std = chatbot_data['tpot'].std() / SLOs['chatbot-tpot']

    imagegen_latency = imagegen_data['total time'].mean() / SLOs['imagegen']
    imagegen_slo = 100 * (1 - (imagegen_data['total time'] > SLOs['imagegen']).sum() / len(imagegen_data))
    imagegen_latency_std = imagegen_data['total time'].std() / SLOs['imagegen']

    livecaption_latency = livecaption_data['time'].mean() / SLOs['livecaption']
    livecaption_slo = 100 * (1 - (livecaption_data['time'] > SLOs['livecaption']).sum() / len(livecaption_data))
    livecaption_latency_std = livecaption_data['time'].std() / SLOs['livecaption']

    # Bar data
    latency_values = [chatbot_ttft, chatbot_tpot, imagegen_latency, livecaption_latency]
    latency_stds = [chatbot_ttft_std, chatbot_tpot_std, imagegen_latency_std, livecaption_latency_std]
    latency_colors = ['#778899', '#778899', '#A0522D', '#C71585']
    latency_hatch = '+'

    slo_values = [chatbot_slo, imagegen_slo, livecaption_slo]
    slo_colors = ['#778899', '#A0522D', '#C71585']
    slo_hatch = 'x'

    # X positions
    latency_x = [0, 1, 2.5, 4]
    slo_x = [6, 7, 8]
    divider_x = 5.25
    bar_width = 0.8
    alpha = 1
    edge_width = 2

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    # Latency bars
    ax.bar(latency_x, latency_values, yerr=latency_stds, width=bar_width,
           color=latency_colors, hatch=latency_hatch, alpha=alpha,
           edgecolor=latency_colors, linewidth=edge_width, facecolor='white', log=True)

    # SLO bars
    ax2.bar(slo_x, slo_values, width=bar_width, color=slo_colors, hatch=slo_hatch,
            alpha=alpha, edgecolor=slo_colors, linewidth=edge_width, facecolor='white')

    # Vertical divider
    ax.axvline(x=divider_x, color='gray', linestyle='-', linewidth=2)

    # Horizontal thresholds
    ax.hlines(y=1.0, xmin=-1, xmax=divider_x, color='green', linestyle='--', linewidth=2)
    ax2.hlines(y=100.0, xmin=divider_x, xmax=9, color='green', linestyle='--', linewidth=2)

    # X-axis ticks and labels
    x_ticks = [0, 1, 2.5, 4, 6, 7, 8]
    x_labels = ['Chatbot-TTFT', 'Chatbot-TPOT', 'ImageGen', 'LiveCaption',
                'Chatbot', 'ImageGen', 'LiveCaption']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=15)

    # Axes formatting
    ax.set_ylabel('Normalized Latency (log scale)', color='black')
    ax.set_yscale('log')
    ax.set_xlim(-1, 9)
    ax.set_ylim(1e-3, 1e3)
    ax.tick_params(axis='y', labelcolor='black')

    ax2.set_ylabel('SLO Attainment (%)', color='black')
    ax2.set_ylim(0, 110)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.tick_params(axis='y', labelcolor='black')

    # Sorted labels and handles
    legend_labels = [
        'Chatbot - Latency',
        'ImageGen - Latency',
        'LiveCaption - Latency',
        'Latency / SLO Threshold',
        'Chatbot - SLO',
        'ImageGen - SLO',
        'LiveCaption - SLO'
    ]

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[0], hatch=latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[2], hatch=latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[3], hatch=latency_hatch, linewidth=edge_width),
        plt.Line2D([], [], color='green', linestyle='--', linewidth=2),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[0], hatch=slo_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[1], hatch=slo_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[2], hatch=slo_hatch, linewidth=edge_width),
    ]

    # Display in two rows of 4 items max
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1))

    # Layout and save
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    plot_path = os.path.join(folder_path, 'performance_split_barplot_final_labeled.pdf')
    plt.savefig(plot_path)
    # plt.show()
    print(f"Plot saved to {plot_path}")

# Example usage:
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu_folder_path', type=str, required=True)
    argparser.add_argument('--cpu_folder_path', type=str, required=True)
    argparser.add_argument('--save_path', type=str, required=True)
    args = argparser.parse_args()
    plot_performance(args.gpu_folder_path, args.cpu_folder_path, args.save_path)