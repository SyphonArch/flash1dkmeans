from datasets import data, scaled_data
from flash1dkmeans import kmeans_1d, numba_kmeans_1d_two_cluster
from utils import calculate_inertia
from time import time
import numpy as np
import json

np.random.seed(42)

bench = {}  # store the time taken as well as the inertia

all_data = {**data, **scaled_data}

for dataset_name, dataset in all_data.items():
    bench[dataset_name] = {}
    weights = np.random.random_sample(dataset.shape).astype(np.float64)

    print(f"Running flash1dkmeans on {dataset_name}")

    start = time()
    centroids, labels = kmeans_1d(dataset, 2, sample_weights=weights)
    stop = time()

    inertia = calculate_inertia(dataset, centroids, labels, weights)

    bench[dataset_name]['wrapper'] = {
        'time (ms)': (stop - start) * 1000,
        'inertia': inertia,
    }

    # test underlying numba implementation

    sorted_indices = np.argsort(dataset)

    sorted_data = dataset[sorted_indices].astype(np.float64)
    sorted_weights = weights[sorted_indices]

    weights_prefix_sum = np.cumsum(sorted_weights, dtype=np.float64)
    weighted_X_prefix_sum = np.cumsum(sorted_data * sorted_weights, dtype=np.float64)

    REPEAT = 1000
    start = time()
    for _ in range(REPEAT):
        centroids, cluster_borders = numba_kmeans_1d_two_cluster(
            sorted_data,
            weights_prefix_sum=weights_prefix_sum,
            weighted_X_prefix_sum=weighted_X_prefix_sum,
            start_idx=0,
            stop_idx=len(sorted_data),
        )
    stop = time()

    labels = np.empty_like(sorted_data, dtype=np.int32)
    division_point = cluster_borders[1]
    labels[:division_point] = 0
    labels[division_point:] = 1

    inertia = calculate_inertia(sorted_data, centroids, labels, sorted_weights)

    bench[dataset_name]['numba'] = {
        'time (ms)': (stop - start) / REPEAT * 1000,
        'inertia': inertia,
    }

    print(f"Done with {dataset_name}")

with open('two_cluster_bench_m3.json', 'w') as f:
    json.dump(bench, f, indent=2)
