from datasets import data, scaled_data, random_weights
from flash1dkmeans import kmeans_1d_two_cluster, numba_kmeans_1d_two_cluster
from utils import calculate_inertia
from time import time
import numpy as np
import json

np.random.seed(42)

bench = {}  # store the time taken as well as the inertia

all_data = {**data, **scaled_data}

# warm-up on random data
print("Warming up on random data...")
kmeans_1d_two_cluster(np.random.rand(10000), sample_weights=np.random.rand(10000))

for dataset_name, dataset in all_data.items():
    bench[dataset_name] = {}
    weights = random_weights[:len(dataset)]

    print(f"Running flash1dkmeans on {dataset_name}")

    start = time()
    centroids, labels = kmeans_1d_two_cluster(dataset, sample_weights=weights)
    stop = time()

    inertia = calculate_inertia(dataset, centroids, labels, weights)

    bench[dataset_name]['wrapper'] = {
        'time (ms)': (stop - start) * 1000,
        'inertia': inertia,
    }

    # test underlying numba implementation
    sorted_indices = np.argsort(dataset)
    sorted_data = dataset[sorted_indices].astype(np.float64)
    sorted_weights = weights[sorted_indices].astype(np.float64)
    weights_prefix_sum = np.cumsum(sorted_weights)
    weighted_X_prefix_sum = np.cumsum(sorted_data * sorted_weights)

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

    labels2 = np.empty_like(sorted_data, dtype=np.int32)
    division_point = cluster_borders[1]
    labels2[:division_point] = 0
    labels2[division_point:] = 1

    inertia = calculate_inertia(sorted_data, centroids, labels2, sorted_weights)

    bench[dataset_name]['numba'] = {
        'time (ms)': (stop - start) / REPEAT * 1000,
        'inertia': inertia,
    }

    assert np.all(labels[sorted_indices] == labels2), "Labels do not match!"

    print(f"Done with {dataset_name}")

with open('two_cluster_bench.json', 'w') as f:
    json.dump(bench, f, indent=2)
