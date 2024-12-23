import os
os.environ['OMP_NUM_THREADS'] = '1'  # for single-threaded execution
import numpy as np
import pickle
from sklearn.cluster import KMeans
from time import time
from flash1dkmeans import numba_kmeans_1d_two_cluster, numba_kmeans_1d_k_cluster

with open('Llama-3-8B_8_mlp_down_proj_214.pkl', 'rb') as f:
    data = pickle.load(f)

print("Simulating Any-Precision LLM Quantization")
print(f"Weight channel size: {len(data)}")
print()

np.random.seed(42)
weights = np.random.rand(len(data))

def recursive_clustering_array(data, weights, cluster_indices, n_clusters=2, depth=5):
    if depth == 0:
        return cluster_indices
    new_indices = []
    for start, end in cluster_indices:
        segment = data[start:end]
        seg_weights = weights[start:end]
        if len(segment) > 1:
            km = KMeans(n_clusters=n_clusters, max_iter=300, random_state=42)
            km.fit(segment.reshape(-1, 1), sample_weight=seg_weights)
            labels = km.labels_
            left_idx = np.where(labels == 0)[0] + start
            right_idx = np.where(labels == 1)[0] + start
            if len(left_idx) == 0:
                new_indices.append((start, start))
            else:
                new_indices.append((left_idx[0], left_idx[-1] + 1))
            if len(right_idx) == 0:
                new_indices.append((end, end))
            else:
                new_indices.append((right_idx[0], right_idx[-1] + 1))
        else:
            new_indices.append((start, end))
            new_indices.append((end, end))
    return recursive_clustering_array(data, weights, new_indices, n_clusters, depth - 1)

def recursive_numba_clustering(sorted_data, sorted_weights, cluster_borders, depth=5):
    if depth == 0:
        return cluster_borders
    new_cluster_borders = [cluster_borders[0]]
    for i in range(len(cluster_borders) - 1):
        start_idx = cluster_borders[i]
        stop_idx = cluster_borders[i + 1]
        if stop_idx - start_idx > 1:
            _, new_borders = numba_kmeans_1d_two_cluster(
                sorted_data,
                weights_prefix_sum=weights_prefix_sum,
                weighted_X_prefix_sum=weighted_X_prefix_sum,
                start_idx=start_idx,
                stop_idx=stop_idx,
            )
            new_cluster_borders.extend(new_borders[1:])
        else:
            new_cluster_borders.extend([stop_idx, stop_idx])
    return recursive_numba_clustering(sorted_data, sorted_weights, new_cluster_borders, depth - 1)

# Warmup
kmeans = KMeans(n_clusters=8, max_iter=300, random_state=42)
kmeans.fit(data.reshape(-1, 1), sample_weight=weights)

# Sklearn timing with sorting and upscaling measurement
start = time()
recursive_time_sklearn = 0
for _ in range(100):
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    km = KMeans(n_clusters=8, max_iter=300, random_state=42)
    km.fit(sorted_data.reshape(-1, 1), sample_weight=sorted_weights)
    labels = km.labels_
    cluster_indices = []
    for i in range(8):
        idx = np.where(labels == i)[0]
        if len(idx) > 0:
            cluster_indices.append((idx[0], idx[-1] + 1))
        else:
            cluster_indices.append((0, 0))
    
    upscale_start = time()
    final_clusters = recursive_clustering_array(sorted_data, sorted_weights, cluster_indices, 2, 5)
    recursive_time_sklearn += time() - upscale_start

    final_labels_sklearn_sorted = np.empty(len(data), dtype=int)
    final_labels_sklearn_sorted.fill(-1)
    for cid, (s, e) in enumerate(final_clusters):
        if s < e:
            final_labels_sklearn_sorted[s:e] = cid

    # Map labels back to original order
    final_labels_sklearn = final_labels_sklearn_sorted[np.argsort(sorted_indices)]
end = time()

sklearn_time = end - start
print(f"Sklearn 100 runs time: {sklearn_time:.3f} seconds")
print(f"Sklearn recursive clustering time: {recursive_time_sklearn:.3f} seconds")
print(f"Sklearn final centroid count: {len(final_clusters)}")
print()

# Warmup
sorted_indices = np.argsort(data)
sorted_data = data[sorted_indices].astype(np.float64)
sorted_weights = weights[sorted_indices].astype(np.float64)
weights_prefix_sum = np.cumsum(sorted_weights)
weighted_X_prefix_sum = np.cumsum(sorted_data * sorted_weights)
weighted_X_squared_prefix_sum = np.cumsum(sorted_data ** 2 * sorted_weights)

numba_kmeans_1d_k_cluster(
    sorted_data, 8,
    max_iter=300,
    weights_prefix_sum=weights_prefix_sum,
    weighted_X_prefix_sum=weighted_X_prefix_sum,
    weighted_X_squared_prefix_sum=weighted_X_squared_prefix_sum,
    start_idx=0,
    stop_idx=len(sorted_data),
    random_state=42,
)

numba_kmeans_1d_two_cluster(
    sorted_data,
    weights_prefix_sum=weights_prefix_sum,
    weighted_X_prefix_sum=weighted_X_prefix_sum,
    start_idx=0,
    stop_idx=len(sorted_data),
)

# Flash1dkmeans timing with upscaling measurement
start = time()
recursive_time_flash = 0
for _ in range(100):
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices].astype(np.float64)
    sorted_weights = weights[sorted_indices].astype(np.float64)
    weights_prefix_sum = np.cumsum(sorted_weights)
    weighted_X_prefix_sum = np.cumsum(sorted_data * sorted_weights)
    weighted_X_squared_prefix_sum = np.cumsum(sorted_data ** 2 * sorted_weights)

    centroids, cluster_borders = numba_kmeans_1d_k_cluster(
        sorted_data, 8,
        max_iter=300,
        weights_prefix_sum=weights_prefix_sum,
        weighted_X_prefix_sum=weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum=weighted_X_squared_prefix_sum,
        start_idx=0,
        stop_idx=len(sorted_data),
        random_state=42,
    )

    upscale_start = time()

    final_cluster_borders = recursive_numba_clustering(sorted_data, sorted_weights, cluster_borders)
    recursive_time_flash += time() - upscale_start

    labels2 = np.empty_like(sorted_data, dtype=np.int32)
    for i in range(len(final_cluster_borders) - 1):
        labels2[final_cluster_borders[i]:final_cluster_borders[i + 1]] = i
    unsorted_indices = np.argsort(sorted_indices)
    final_labels_flash = labels2[unsorted_indices]
end = time()

flash_time = end - start
print(f"Flash1dkmeans 100 runs time: {flash_time:.3f} seconds")
print(f"Flash1dkmeans recursive clustering time: {recursive_time_flash:.3f} seconds")
print(f"Flash1dkmeans final centroid count: {len(final_cluster_borders) - 1}")
print()

# speedup
print(f"Total speedup: {sklearn_time / flash_time:.3f}x")
print(f"Upscaling speedup: {recursive_time_sklearn / recursive_time_flash:.3f}x")
