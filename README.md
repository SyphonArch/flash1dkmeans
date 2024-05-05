# flash1dkmeans
An optimized K-means implementation for the one-dimensional case

Exploits the fact that one-dimensional data can be sorted.

Numba acceleration is used, so callers can further use parallelization with Numba's `@njit(parallel=True)`
if multiple instances of the algorithm are run in parallel.

## Features

### Two clusters

Guaranteed to find the optimal solution for two clusters.

### K clusters

Uses the K-means++ initialization algorithm to find the initial centroids.
Then uses the Lloyd's algorithm to find the final centroids, except with optimizations for the one-dimensional case.

## Time Complexity

- **2 clusters**: O(log(n))
- (+ (O(n) for prefix sum calculation if not provided))
- **k clusters**: O(k * 2 + log(k) * log(n) + max_iter * log(n) * k)  
  (+ (O(n) for prefix sum calculation if not provided))

This is a significant improvement over the standard K-means algorithm, which has a time complexity of O(n * k * max_iter),
even excluding the time complexity of the K-means++ initialization.

## How fast is it?

TODO: Comparision with sklearn's KMeans


## Installation
```bash
pip install flash1dkmeans
```

## Usage

### Basic usage
```python
from flash1dkmeans import kmeans_1d

data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
k = 2

centroids, labels = kmeans_1d(data, k)
```

### More Options
```python
from flash1dkmeans import kmeans_1d
import numpy as np

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
weights = np.random.random_sample(data.shape)
k = 3

centroids, labels = kmeans_1d(
    data, k,
    sample_weights=weights,  # sample weights
    n_local_trials=(2 + int(np.log(k))),  # number of local tries
    max_iter=100,  # maximum number of iterations
)
```

### Even More Options
```python
from flash1dkmeans import kmeans_1d
import numpy as np

n, k = 1024, 4

# Generate random data
data = np.random.random_sample(n)
data = np.sort(data)

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
  centroids, cluster_borders = kmeans_1d(
      data, k,  # Note how the sample weights are not provided when the prefix sums are provided
      n_local_trials=(2 + int(np.log(k))),  # number of local tries
      max_iter=100,  # maximum number of iterations
      return_cluster_borders=True,  # return cluster borders instead of labels
      weights_prefix_sum=weights_prefix_sum,  # prefix sum of the sample weights, leave empty for unwieghted data
      weighted_X_prefix_sum=weighted_X_prefix_sum,  # prefix sum of the weighted data
      weighted_X_squared_prefix_sum=weighted_X_squared_prefix_sum,  # prefix sum of the squared weighted data
      start_idx=start_idx,  # start index of the data
      stop_idx=stop_idx,  # stop index of the data
  )
```

## Notes

This repository has been created to be used in [Any-Precision-LLM](https://github.com/SNU-ARC/any-precision-llm) project,
where multiple 1D K-means instances are run in parallel for LLM quantization.

However the algorithm is general and can be used for any 1D K-means problem.

