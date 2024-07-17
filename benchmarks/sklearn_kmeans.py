from datasets import data, scaled_data
from sklearn.cluster import KMeans
from utils import calculate_inertia
from time import time
import numpy as np
import json

np.random.seed(42)

k_to_test = [2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

bench = {}  # store the time taken as well as the inertia

all_data = {**data, **scaled_data}

for dataset_name, dataset in all_data.items():
    bench[dataset_name] = {}
    weights = np.random.random_sample(dataset.shape)
    for k in k_to_test:
        if k > len(dataset):
            continue

        print(f"Running kmeans on {dataset_name} with k={k}...")
        kmeans = KMeans(n_clusters=k, max_iter=300, random_state=42)

        start = time()
        results = kmeans.fit(dataset.reshape(-1, 1), sample_weight=weights)
        stop = time()

        centroids = results.cluster_centers_.flatten()
        labels = results.labels_
        inertia = calculate_inertia(dataset, centroids, labels, weights)

        bench[dataset_name][k] = {
            'time (ms)': (stop - start) * 1000,
            'inertia': inertia,
        }
        print(f"Done with {dataset_name} with k={k}.")

with open('sklearn_bench.json', 'w') as f:
    json.dump(bench, f, indent=2)
