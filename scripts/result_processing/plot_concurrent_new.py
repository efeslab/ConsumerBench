import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({'font.size': 24})

SLOs = {
    'chatbot-ttft': 1,
    'chatbot-tpot': 0.25,
    'imagegen': 28,
    'livecaption': 2
}

def extract_normalized_latencies(folder_path):
    chatbot_data = pd.read_csv(os.path.join(folder_path, 'task_chat1_u0_perf.csv'))
    imagegen_data = pd.read_csv(os.path.join(folder_path, 'task_imagegen1_u0_perf.csv'))
    livecaption_data = pd.read_csv(os.path.join(folder_path, 'task_lv_u0_perf.csv'))

    # print means
    # print(chatbot_data['ttft'].mean())
    # print(chatbot_data['tpot'].mean())
    # print(imagegen_data['total time'].mean())
    # print(livecaption_data['time'].mean())

    print(imagegen_data['total time'].min(), imagegen_data['total time'].max())
    print(livecaption_data['time'].min(), livecaption_data['time'].max())

    return {
        'Chatbot-TTFT': (chatbot_data['ttft'].mean() / SLOs['chatbot-ttft'],
                         chatbot_data['ttft'].std() / SLOs['chatbot-ttft']),
        'Chatbot-TPOT': (chatbot_data['tpot'].mean() / SLOs['chatbot-tpot'],
                         chatbot_data['tpot'].std() / SLOs['chatbot-tpot']),
        'ImageGen': (imagegen_data['total time'].mean() / SLOs['imagegen'],
                     imagegen_data['total time'].std() / SLOs['imagegen']),
        'LiveCaption': (livecaption_data['time'].mean() / SLOs['livecaption'],
                        livecaption_data['time'].std() / SLOs['livecaption'])
    }

def plot_latency_comparison(no_mps_path, with_mps_path, output_file):
    no_mps = extract_normalized_latencies(no_mps_path)
    with_mps = extract_normalized_latencies(with_mps_path)

    workloads = ['Chatbot-TTFT', 'Chatbot-TPOT', 'ImageGen', 'LiveCaption']
    colors = {
        'Chatbot-TTFT': '#778899',
        'Chatbot-TPOT': '#778899',
        'ImageGen': '#A0522D',
        'LiveCaption': '#C71585'
    }

    # Bar positions with gap between No-MPS and MPS
    x_no_mps = [0, 1, 2, 3]
    x_with_mps = [5, 6, 7, 8]  # gap at index 4
    x = x_no_mps + x_with_mps

    values = []
    errors = []
    bar_colors = []

    for w in workloads:
        values.append(no_mps[w][0])
        errors.append(no_mps[w][1])
        bar_colors.append(colors[w])
    for w in workloads:
        values.append(with_mps[w][0])
        errors.append(with_mps[w][1])
        bar_colors.append(colors[w])


    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, values, yerr=errors, width=0.8, color='white',
                  edgecolor=bar_colors, hatch='//', linewidth=2)

    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Latency')
    ax.set_ylim(1e-2, 1e2*3)

    # X-axis ticks just at group centers
    ax.set_xticks([1.5, 6.5])
    ax.set_xticklabels(['Greedy Allocation', 'GPU Partitioning'])

    # TTFT / TPOT labels inside bars
    for i in [0, 1, 5, 6]:
        label = 'TTFT' if i == 0 or i == 5 else 'TPOT'
        height = values[x.index(i)]
        ax.text(i, 0.01, label,
                ha='center', va='bottom', fontsize=20)

    height = values[x.index(3)]
    ax.text(3, height*1.6, 'Starvation',
            ha='center', va='bottom', fontsize=20, color='red')
    height = values[x.index(8)]
    ax.text(8, height*1.6, 'No\nStarvation',
            ha='center', va='bottom', fontsize=20, color='green')

    # Optional vertical divider
    # ax.axvline(x=4, color='gray', linestyle='--', linewidth=1)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#778899', hatch='//', linewidth=2, label='Chatbot'),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#A0522D', hatch='//', linewidth=2, label='ImageGen'),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#C71585', hatch='//', linewidth=2, label='LiveCaptions')
    ]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

    fig.tight_layout()
    plt.savefig(output_file)
    plt.show()
