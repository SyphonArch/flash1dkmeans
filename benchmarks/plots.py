import json
import os
import matplotlib.pyplot as plt
from datasets import data

target_dir = 'i9-13900K'
#target_dir = 'm3'
sklearn_file = 'sklearn_bench.json'
k_cluster_file = 'k_cluster_bench.json'
two_cluster_file = 'two_cluster_bench.json'

os.makedirs(f'./fig_{target_dir}', exist_ok=True)

with open(f'{target_dir}/{sklearn_file}') as f:
    sklearn_data = json.load(f)

with open(f'{target_dir}/{k_cluster_file}') as f:
    k_cluster_data = json.load(f)

with open(f'{target_dir}/{two_cluster_file}') as f:
    two_cluster_data = json.load(f)

non_scaled_datasets = [dataset for dataset in sklearn_data if '^' not in dataset]
scaled_datasets = [dataset for dataset in sklearn_data if '^' in dataset]

k_range = list(map(str, [2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]))
k_range = ['two_cluster'] + k_range

for k in k_range:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if k == 'two_cluster':
        ax.set_title(f'Squared error comparison for the two cluster algorithm')
    else:
        ax.set_title(f'Squared error comparison for k={k}')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average squared error per element (1e-3)')

    kk = '2' if k == 'two_cluster' else k
    relevant_datasets = [dataset for dataset in non_scaled_datasets if kk in sklearn_data[dataset]]

    if int(kk) > 32:
        if 'iris' in relevant_datasets:
            relevant_datasets.remove('iris')

    sklearn_inertia = [sklearn_data[dataset][kk]['inertia'] for dataset in relevant_datasets]

    if k == 'two_cluster':
        flash1dkmeans_inertia = [two_cluster_data[dataset]['wrapper']['inertia'] for dataset in relevant_datasets]
    else:
        flash1dkmeans_inertia = [k_cluster_data[dataset][k]['wrapper']['inertia'] for dataset in relevant_datasets]

    sklearn_inertia = [inertia * 1e3 for inertia in sklearn_inertia]
    flash1dkmeans_inertia = [inertia * 1e3 for inertia in flash1dkmeans_inertia]

    sklearn_inertia_per_element = [inertia / len(data[dataset]) for inertia, dataset in
                                   zip(sklearn_inertia, relevant_datasets)]
    flash1dkmeans_inertia_per_element = [inertia / len(data[dataset]) for inertia, dataset in
                                         zip(flash1dkmeans_inertia, relevant_datasets)]

    bars1 = ax.bar([i - 0.2 for i in range(len(relevant_datasets))], sklearn_inertia_per_element, width=0.4,
                   label='sklearn', color='darkblue')
    bars2 = ax.bar([i + 0.2 for i in range(len(relevant_datasets))], flash1dkmeans_inertia_per_element, width=0.4,
                   label='flash1dkmeans', color='darkred')

    ax.bar_label(bars1, labels=[f'{val:.1f}' for val in sklearn_inertia_per_element], color='darkblue')
    ax.bar_label(bars2, labels=[f'{val:.1f}' for val in flash1dkmeans_inertia_per_element], color='darkred')

    ax.set_xticks(range(len(relevant_datasets)))
    ax.set_xticklabels(relevant_datasets)
    ax.legend()

    if k == 'two_cluster':
        plt.savefig(f'./fig_{target_dir}/inertia_comparison_two_cluster.png', dpi=600)
    else:
        plt.savefig(f'./fig_{target_dir}/inertia_comparison_k{k}.png', dpi=600)
    plt.close()

for k in k_range:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if k == 'two_cluster':
        ax.set_title(f'Runtime comparison for the two cluster algorithm')
    else:
        ax.set_title(f'Runtime comparison for k={k}')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Runtime (ms)')

    kk = '2' if k == 'two_cluster' else k
    relevant_datasets = [dataset for dataset in scaled_datasets if kk in sklearn_data[dataset]]

    sklearn_runtime = [sklearn_data[dataset][kk]['time (ms)'] for dataset in relevant_datasets]

    if k == 'two_cluster':
        flash1dkmeans_runtime = [two_cluster_data[dataset]['wrapper']['time (ms)'] for dataset in relevant_datasets]
        flash1dkmeans_numba_runtime = [two_cluster_data[dataset]['numba']['time (ms)'] for dataset in relevant_datasets]
    else:
        flash1dkmeans_runtime = [k_cluster_data[dataset][k]['wrapper']['time (ms)'] for dataset in relevant_datasets]
        flash1dkmeans_numba_runtime = [k_cluster_data[dataset][k]['numba']['time (ms)'] for dataset in
                                       relevant_datasets]

    relevant_datasets = [f"$2^{{{dataset_size.split('^')[1]}}}$" for dataset_size in relevant_datasets]

    ax.plot(relevant_datasets, sklearn_runtime, label='sklearn', color='darkblue', marker='o')
    ax.plot(relevant_datasets, flash1dkmeans_runtime, label='flash1dkmeans', color='darkgreen', marker='o')
    ax.plot(relevant_datasets, flash1dkmeans_numba_runtime, label='flash1dkmeans_numba', color='darkred', marker='o')

    all_values = sklearn_runtime + flash1dkmeans_runtime + flash1dkmeans_numba_runtime
    ymin, ymax = min(all_values) / 10, max(all_values) * 10
    ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')

    def label_format(value):
        # if value is less than 100, show one decimal place
        # else, show no decimal places
        return f'{value:.1f}' if value < 100 else f'{value:.0f}'


    for i, val in enumerate(sklearn_runtime):
        ax.annotate(label_format(val), (relevant_datasets[i], val), textcoords="offset points", xytext=(-3, 10),
                    ha='center', color='darkblue')

    for i, val in enumerate(flash1dkmeans_runtime):
        ax.annotate(label_format(val), (relevant_datasets[i], val), textcoords="offset points", xytext=(0, 8 if int(kk) >= 16 else -15),
                    ha='center', color='darkgreen')

    for i, val in enumerate(flash1dkmeans_numba_runtime):
        ax.annotate(label_format(val), (relevant_datasets[i], val), textcoords="offset points", xytext=(0, -12),
                    ha='center', color='darkred')

    ax.legend()
    #plt.xticks(rotation=45)
    plt.tight_layout()
    if k == 'two_cluster':
        plt.savefig(f'./fig_{target_dir}/runtime_comparison_two_cluster.png', dpi=600)
    else:
        plt.savefig(f'./fig_{target_dir}/runtime_comparison_k{k}.png', dpi=600)
    plt.close()