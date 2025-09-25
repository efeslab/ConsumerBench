import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({'font.size': 18})

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

    return {
        'Chatbot-TTFT': chatbot_data['ttft'].mean() / SLOs['chatbot-ttft'],
        'Chatbot-TPOT': chatbot_data['tpot'].mean() / SLOs['chatbot-tpot'],
        'ImageGen': imagegen_data['total time'].mean() / SLOs['imagegen'],
        'LiveCaption': livecaption_data['time'].mean() / SLOs['livecaption']
    }, {
        'Chatbot-TTFT': chatbot_data['ttft'].std() / SLOs['chatbot-ttft'],
        'Chatbot-TPOT': chatbot_data['tpot'].std() / SLOs['chatbot-tpot'],
        'ImageGen': imagegen_data['total time'].std() / SLOs['imagegen'],
        'LiveCaption': livecaption_data['time'].std() / SLOs['livecaption']
    }

def plot_latency_comparison(no_mps_path, with_mps_path, output_file):
    no_mps_vals, no_mps_stds = extract_normalized_latencies(no_mps_path)
    with_mps_vals, with_mps_stds = extract_normalized_latencies(with_mps_path)

    workloads = ['Chatbot-TTFT', 'Chatbot-TPOT', 'ImageGen', 'LiveCaption']
    x_labels = []
    values = []
    errors = []
    colors = []

    base_colors = ['#778899', '#778899', '#A0522D', '#C71585']

    for i, workload in enumerate(workloads):
        x_labels.append(f'{workload}-No-MPS')
        values.append(no_mps_vals[workload])
        errors.append(no_mps_stds[workload])
        colors.append(base_colors[i])

        x_labels.append(f'{workload}-With-MPS')
        values.append(with_mps_vals[workload])
        errors.append(with_mps_stds[workload])
        colors.append(base_colors[i])

    x_pos = np.arange(len(x_labels))
    bar_width = 0.8

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, values, yerr=errors, width=bar_width, color='white',
           edgecolor=colors, hatch='//', linewidth=2)

    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.set_ylabel('Normalized Latency (log scale)')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)
    # ax.set_title('Latency Comparison With and Without MPS')

    fig.tight_layout()
    plt.savefig(output_file)
    plt.show()
