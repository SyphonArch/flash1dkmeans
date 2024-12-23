from matplotlib import pyplot as plt
import json
from datasets import data
import os

#target_dir = 'm3'
target_dir = 'i9-13900K'

sklearn_file = 'sklearn_bench.json'
k_cluster_file = 'k_cluster_bench.json'
two_cluster_file = 'two_cluster_bench.json'

os.makedirs(f'./fig_{target_dir}_pdf', exist_ok=True)


with open(f'{target_dir}/{sklearn_file}') as f:
    sklearn_data = json.load(f)

with open(f'{target_dir}/{k_cluster_file}') as f:
    k_cluster_data = json.load(f)

with open(f'{target_dir}/{two_cluster_file}') as f:
    two_cluster_data = json.load(f)

k_range = ['two_cluster', '8', '32', '128']
non_scaled_datasets = [dataset for dataset in sklearn_data if '^' not in dataset]
scaled_datasets = [dataset for dataset in sklearn_data if '^' in dataset]

def label_format(value):
    return f'{value:.1f}' if value < 100 else f'{value:.0f}'

# Inertia comparison in one 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(6.4 * 0.8 * 2, 4.8 * 0.8 * 2))
plt.subplots_adjust(hspace=0.4)  # Increase the value to add more space
for ax, k in zip(axes.flatten(), k_range):
    kk = '2' if k == 'two_cluster' else k
    relevant_datasets = [dataset for dataset in non_scaled_datasets if kk in sklearn_data[dataset]]
    if int(kk) > 32 and 'iris' in relevant_datasets:
        relevant_datasets.remove('iris')
    sklearn_inertia = [sklearn_data[dataset][kk]['inertia'] * 1e3 for dataset in relevant_datasets]
    flash1dkmeans_inertia = ([two_cluster_data[dataset]['wrapper']['inertia'] * 1e3 for dataset in relevant_datasets]
                             if k == 'two_cluster' else
                             [k_cluster_data[dataset][k]['wrapper']['inertia'] * 1e3 for dataset in relevant_datasets])
    sklearn_inertia_per_element = [inertia / len(data[dataset]) for inertia, dataset in zip(sklearn_inertia, relevant_datasets)]
    flash1dkmeans_inertia_per_element = [inertia / len(data[dataset]) for inertia, dataset in zip(flash1dkmeans_inertia, relevant_datasets)]
    if k == 'two_cluster':
        ax.set_title('two cluster', fontsize=14)
    else:
        ax.set_title(f'k={k}', fontsize=14)
    bars1 = ax.bar([i - 0.2 for i in range(len(relevant_datasets))], sklearn_inertia_per_element, width=0.4,
                   label='sklearn', color='darkblue')
    bars2 = ax.bar([i + 0.2 for i in range(len(relevant_datasets))], flash1dkmeans_inertia_per_element, width=0.4,
                   label='flash1dkmeans', color='darkred')
    ax.bar_label(bars1, labels=[label_format(val) for val in sklearn_inertia_per_element], color='darkblue', fontsize=9)
    ax.bar_label(bars2, labels=[label_format(val) for val in flash1dkmeans_inertia_per_element], color='darkred', fontsize=9)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)
    ax.set_xticks(range(len(relevant_datasets)))
    ax.set_xticklabels(relevant_datasets)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('squared error per element (1e-3)', fontsize=12)
handles, labels = [bars1, bars2], ['sklearn','flash1dkmeans']
fig.legend(handles, labels, loc='lower center', ncol=2)
plt.subplots_adjust(bottom=0.13)
plt.savefig(f'./fig_{target_dir}_pdf/inertia_comparison.pdf', bbox_inches='tight', pad_inches=0.01)
plt.close()

# Runtime comparison in one 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(6.4 * 0.8 * 2, 4.8 * 0.8 * 2))
plt.subplots_adjust(hspace=0.4)  # Increase the value to add more space
for ax, k in zip(axes.flatten(), k_range):
    kk = '2' if k == 'two_cluster' else k
    relevant_datasets = [dataset for dataset in scaled_datasets if kk in sklearn_data[dataset]]
    sklearn_runtime = [sklearn_data[dataset][kk]['time (ms)'] for dataset in relevant_datasets]
    if k == 'two_cluster':
        flash1dkmeans_runtime = [two_cluster_data[dataset]['wrapper']['time (ms)'] for dataset in relevant_datasets]
        flash1dkmeans_numba_runtime = [two_cluster_data[dataset]['numba']['time (ms)'] for dataset in relevant_datasets]
    else:
        flash1dkmeans_runtime = [k_cluster_data[dataset][k]['wrapper']['time (ms)'] for dataset in relevant_datasets]
        flash1dkmeans_numba_runtime = [k_cluster_data[dataset][k]['numba']['time (ms)'] for dataset in relevant_datasets]
    datasets_labels = [f"$2^{{{ds.split('^')[1]}}}$" for ds in relevant_datasets]
    if k == 'two_cluster':
        ax.set_title('two cluster', fontsize=14)
    else:
        ax.set_title(f'k={k}', fontsize=14)
    ax.plot(datasets_labels, sklearn_runtime, label='sklearn', color='darkblue', marker='o')
    ax.plot(datasets_labels, flash1dkmeans_runtime, label='flash1dkmeans', color='darkgreen', marker='o')
    ax.plot(datasets_labels, flash1dkmeans_numba_runtime, label='flash1dkmeans_numba', color='darkred', marker='o')
    all_values = sklearn_runtime + flash1dkmeans_runtime + flash1dkmeans_numba_runtime
    ymin, ymax = min(all_values) / 10, max(all_values) * 10
    ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    for i, val in enumerate(sklearn_runtime):
        ax.annotate(label_format(val), (datasets_labels[i], val), textcoords="offset points", xytext=(-3, 10),
                    ha='center', color='darkblue', fontsize=8)
    for i, val in enumerate(flash1dkmeans_runtime):
        ax.annotate(label_format(val), (datasets_labels[i], val), textcoords="offset points",
                    xytext=(0, 8 if int(kk) >= 16 else -15), ha='center', color='darkgreen', fontsize=8)
    for i, val in enumerate(flash1dkmeans_numba_runtime):
        ax.annotate(label_format(val), (datasets_labels[i], val), textcoords="offset points", xytext=(0, -12),
                    ha='center', color='darkred', fontsize=8)
    ax.set_xlabel('Dataset Size', fontsize=12)
    ax.set_ylabel('Runtime (ms)', fontsize=12)
handles = [plt.Line2D([0],[0], color='darkblue', marker='o'), plt.Line2D([0],[0], color='darkgreen', marker='o'), plt.Line2D([0],[0], color='darkred', marker='o')]
labels = ['sklearn','flash1dkmeans','flash1dkmeans_numba']
fig.legend(handles, labels, loc='lower center', ncol=3)
plt.subplots_adjust(bottom=0.13)
plt.savefig(f'./fig_{target_dir}_pdf/runtime_comparison.pdf', bbox_inches='tight', pad_inches=0.01)
plt.close()
