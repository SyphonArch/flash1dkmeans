import json
import matplotlib.pyplot as plt
from datasets import data

target_dir = 'i9-13900K'
sklearn_file = 'sklearn_bench.json'
k_cluster_file = 'k_cluster_bench.json'
two_cluster_file = 'two_cluster_bench.json'

with open(f'{target_dir}/{sklearn_file}') as f:
    # the sklearn data is structured as dataset_name -> k -> "time (ms)"/"inertia"
    sklearn_data = json.load(f)

with open(f'{target_dir}/{k_cluster_file}') as f:
    # the k_cluster data is structured as dataset_name -> k -> wrapper/numba -> "time (ms)"/"inertia"
    k_cluster_data = json.load(f)

with open(f'{target_dir}/{two_cluster_file}') as f:
    # the two_cluster data is structured as dataset_name -> wrapper/numba -> "time (ms)"/"inertia"
    two_cluster_data = json.load(f)

# Get and organize the dataset names
non_scaled_datasets = [dataset for dataset in sklearn_data if '^' not in dataset]
scaled_datasets = [dataset for dataset in sklearn_data if '^' in dataset]

# Get the k range
k_range = list(map(str, [2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]))

# first, let's compare the inertia values for all datasets which don't contain a '^' in their name,
# for sklearn and k_cluster. Plot a graph for each k value.
# In each graph, draw a bar for each dataset, with the sklearn inertia value and the k_cluster inertia value.
# The x-axis should be the dataset name, and the y-axis should be the inertia value.
# group the bars by dataset name, and color them differently for sklearn and k_cluster.


for k in k_range:
    fig, ax = plt.subplots()
    ax.set_title(f'Squared error comparison for k={k}')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average squared error per element (1e-3)')

    # find datasets that have k clusters
    relevant_datasets = [dataset for dataset in non_scaled_datasets if k in sklearn_data[dataset]]
    # relevant_datasets = ['california_housing', '32K-blobs']

    # Get the inertia values for each dataset
    sklearn_inertia = [sklearn_data[dataset][k]['inertia'] for dataset in relevant_datasets]

    if k == '2':
        flash1dkmeans_inertia = [two_cluster_data[dataset]['wrapper']['inertia'] for dataset in relevant_datasets]
    else:
        flash1dkmeans_inertia = [k_cluster_data[dataset][k]['wrapper']['inertia'] for dataset in relevant_datasets]

    # scale the inertia values by 1e-3
    sklearn_inertia = [inertia * 1e3 for inertia in sklearn_inertia]
    flash1dkmeans_inertia = [inertia * 1e3 for inertia in flash1dkmeans_inertia]

    # normalize by element count
    sklearn_inertia_per_element = [inertia / len(data[dataset]) for inertia, dataset in
                                   zip(sklearn_inertia, relevant_datasets)]
    flash1dkmeans_inertia_per_element = [inertia / len(data[dataset]) for inertia, dataset in
                                         zip(flash1dkmeans_inertia, relevant_datasets)]

    # Plot the inertia values
    bars1 = ax.bar([i - 0.2 for i in range(len(relevant_datasets))], sklearn_inertia_per_element, width=0.4,
                   label='sklearn')
    bars2 = ax.bar([i + 0.2 for i in range(len(relevant_datasets))], flash1dkmeans_inertia_per_element, width=0.4,
                   label='flash1dkmeans')

    # Add data labels
    ax.bar_label(bars1, labels=[f'{val:.1f}' for val in sklearn_inertia_per_element])
    ax.bar_label(bars2, labels=[f'{val:.1f}' for val in flash1dkmeans_inertia_per_element])

    ax.set_xticks(range(len(relevant_datasets)))
    ax.set_xticklabels(relevant_datasets)
    ax.legend()

    plt.savefig(f'./fig/inertia_comparison_k{k}.png')
    plt.close()

# Now we will do a runtime comparison for the same datasets, for sklearn and k_cluster.
# Use the scaled datasets this time, which are named like 2^20. Infer the number of elements from the name.
# Plot a line for sklearn, a line for k_cluster wrapper, and a line for k_cluster numba.
# For k=2, use the two_cluster data instead of k_cluster data.

for k in k_range:
    fig, ax = plt.subplots()
    ax.set_title(f'Runtime comparison for k={k}')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Runtime (ms)')

    # find datasets that have k clusters
    relevant_datasets = [dataset for dataset in scaled_datasets if k in sklearn_data[dataset]]
    # relevant_datasets = ['2^20-blobs']

    # Get the runtime values for each dataset
    sklearn_runtime = [sklearn_data[dataset][k]['time (ms)'] for dataset in relevant_datasets]

    if k == '2':
        flash1dkmeans_runtime = [two_cluster_data[dataset]['wrapper']['time (ms)'] for dataset in relevant_datasets]
        flash1dkmeans_numba_runtime = [two_cluster_data[dataset]['numba']['time (ms)'] for dataset in relevant_datasets]
    else:
        flash1dkmeans_runtime = [k_cluster_data[dataset][k]['wrapper']['time (ms)'] for dataset in relevant_datasets]
        flash1dkmeans_numba_runtime = [k_cluster_data[dataset][k]['numba']['time (ms)'] for dataset in
                                       relevant_datasets]

    # Plot the runtime values
    ax.plot(relevant_datasets, sklearn_runtime, label='sklearn')
    ax.plot(relevant_datasets, flash1dkmeans_runtime, label='flash1dkmeans')
    ax.plot(relevant_datasets, flash1dkmeans_numba_runtime, label='flash1dkmeans_numba')

    # Make the y-axis logarithmic
    ax.set_yscale('log')

    # Add data labels
    for i, val in enumerate(sklearn_runtime):
        ax.annotate(f'{val:.1f}', (relevant_datasets[i], val), textcoords="offset points", xytext=(0, 10), ha='center')

    for i, val in enumerate(flash1dkmeans_runtime):
        ax.annotate(f'{val:.1f}', (relevant_datasets[i], val), textcoords="offset points", xytext=(0, -15), ha='center')

    for i, val in enumerate(flash1dkmeans_numba_runtime):
        ax.annotate(f'{val:.1f}', (relevant_datasets[i], val), textcoords="offset points", xytext=(0, -10), ha='center')

    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./fig/runtime_comparison_k{k}.png')
    plt.close()
