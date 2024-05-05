from flash1dkmeans import numba_kmeans_1d_k_cluster
import numpy as np

n, k = 1024, 4

# Generate random data
data = np.random.random_sample(n)
data = np.sort(data)

print(data)

# Generate random weights
weights = np.random.random_sample(data.shape)

# Calculate prefix sums
weights_prefix_sum = np.cumsum(weights)
weighted_X_prefix_sum = np.cumsum(data * weights)
weighted_X_squared_prefix_sum = np.cumsum(data ** 2 * weights)

middle_idx = n // 2

# Providing prefix sums reduces redundant calculations
# This is useful when the algorithm is run multiple times on different segments of the data
for start_idx, stop_idx in [(0, middle_idx), (middle_idx, n)]:
    centroids, cluster_borders = numba_kmeans_1d_k_cluster(
        data, k,  # Note how the sample weights are not provided when the prefix sums are provided
        max_iter=100,  # maximum number of iterations
        weights_prefix_sum=weights_prefix_sum,  # prefix sum of the sample weights, leave empty for unwieghted data
        weighted_X_prefix_sum=weighted_X_prefix_sum,  # prefix sum of the weighted data
        weighted_X_squared_prefix_sum=weighted_X_squared_prefix_sum,  # prefix sum of the squared weighted data
        start_idx=start_idx,  # start index of the data
        stop_idx=stop_idx,  # stop index of the data
    )

    print(centroids, cluster_borders)
