from datasets import data, scaled_data, random_weights
from flash1dkmeans import kmeans_1d, numba_kmeans_1d_k_cluster
from utils import calculate_inertia
from time import time
import numpy as np
import json

np.random.seed(42)

k_to_test = [3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

bench = {}  # store the time taken as well as the inertia

all_data = {**data, **scaled_data}

for dataset_name, dataset in all_data.items():
    bench[dataset_name] = {}
    weights = random_weights[:len(dataset)]
    for k in k_to_test:
        if k > len(dataset):
            continue
        bench[dataset_name][k] = {}

        print(f"Running flash1dkmeans on {dataset_name} with k={k}...")

        start = time()
        centroids, labels = kmeans_1d(dataset, k, max_iter=300, random_state=42, sample_weights=weights)
        stop = time()

        inertia = calculate_inertia(dataset, centroids, labels, weights)

        bench[dataset_name][k]['wrapper'] = {
            'time (ms)': (stop - start) * 1000,
            'inertia': inertia,
        }

        # test underlying numba implementation

        sorted_data = np.sort(dataset)
        weights_prefix_sum = np.cumsum(weights)
        weighted_X_prefix_sum = np.cumsum(sorted_data * weights)
        weighted_X_squared_prefix_sum = np.cumsum(sorted_data ** 2 * weights)

        start = time()
        centroids, cluster_borders = numba_kmeans_1d_k_cluster(
            sorted_data, k,
            max_iter=300,
            weights_prefix_sum=weights_prefix_sum,
            weighted_X_prefix_sum=weighted_X_prefix_sum,
            weighted_X_squared_prefix_sum=weighted_X_squared_prefix_sum,
            start_idx=0,
            stop_idx=len(sorted_data),
            random_state=42,
        )
        stop = time()

        labels = np.empty_like(sorted_data, dtype=np.int32)
        for i in range(k):
            labels[cluster_borders[i]:cluster_borders[i + 1]] = i

        inertia = calculate_inertia(sorted_data, centroids, labels, weights)

        bench[dataset_name][k]['numba'] = {
            'time (ms)': (stop - start) * 1000,
            'inertia': inertia,
        }

        print(f"Done with {dataset_name} with k={k}.")

with open('k_cluster_bench.json', 'w') as f:
    json.dump(bench, f, indent=2)
