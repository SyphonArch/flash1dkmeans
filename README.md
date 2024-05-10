# flash1dkmeans
An optimized K-means implementation for the one-dimensional case

Exploits the fact that one-dimensional data can be sorted.

For the lower level functions prefixed with `numba_`, Numba acceleration is used,
so callers can utilize these functions within their own Numba-accelerated functions.

## Important Notice

This library utilizes [Numba](https://numba.pydata.org/), a JIT compiler, for acceleration.
As there is a compile time overhead, the first invocation may be slower than usual.

Numba caches the compiled functions, so execution times should stabilize after the first invocation.

## Features

### Two clusters

Guaranteed to find the optimal solution for two clusters by employing a clever binary search.

### K clusters

Uses the K-means++ initialization algorithm to find the initial centroids.
Then uses the Lloyd's algorithm to find the final centroids, except with optimizations for the one-dimensional case.

## Time Complexity

For number of elements $n$, number of clusteres $k$, number of Lloyd's algorithm iterations $i$, and assuming one-dimensional data (which is the only case covered by this implementation):

- **2 clusters**: $O(\log{n})$  
  ($+ O(n)$ for prefix sum calculation if not provided, $+ O(n \cdot \log {n})$ for sorting if not sorted)
- **$k$ clusters**: $O(k ^ 2 \cdot \log {k} \cdot \log {n}) + O(i \cdot \log {n} \cdot k)$  
  (The first term is for K-means++ initialization, and the latter for Lloyd's algorithm)  
  ($+ O(n)$ for prefix sum calculation if not provided, $+ O(n \cdot \log {n})$ for sorting if not sorted)

This is a significant improvement over common K-means implementations.
For example, general implementations for $d$-dimensional data using a combination of greedy K-means++ initialization and Lloyd's algorithm for convergence,
when given one-dimensional data, spends $O(k ^ 2 \cdot \log {k} \cdot n)$ time in initialization and $O(i \cdot n \cdot k)$ time in iterations.

**Note that you must use the underlying `numba_` functions directly in order to directly supply prefix sums and skip sorting.**

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
    max_iter=100,  # maximum number of iterations
)
```

### Even More Options
The underlying Numba-accelerated function `numba_kmeans_1d_k_clusters` can be used directly for more control.

This is useful when the algorithm is run multiple times on different segments of the data,
or to use within another Numba-accelerated function.

The list of available functions are as follows:
- `numba_kmeans_1d_two_clusters`
- `numba_kmeans_1d_two_clusters_unweighted`
- `numba_kmeans_1d_k_cluster`
- `numba_kmeans_1d_k_cluster_unweighted`

All of these functions assume the data is sorted beforehand.

```python
from flash1dkmeans import numba_kmeans_1d_k_cluster
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
  centroids, cluster_borders = numba_kmeans_1d_k_cluster(  # Note that data MUST be sorted beforehand
    data, k,  # Note how the sample weights are not provided when the prefix sums are provided
    max_iter=100,  # maximum number of iterations
    weights_prefix_sum=weights_prefix_sum,  # prefix sum of the sample weights, leave empty for unwieghted data
    weighted_X_prefix_sum=weighted_X_prefix_sum,  # prefix sum of the weighted data
    weighted_X_squared_prefix_sum=weighted_X_squared_prefix_sum,  # prefix sum of the squared weighted data
    start_idx=start_idx,  # start index of the data
    stop_idx=stop_idx,  # stop index of the data
  )
```

Refer to the docstrings for more information.

## Notes

This repository has been created to be used in [Any-Precision-LLM](https://github.com/SNU-ARC/any-precision-llm) project,
where multiple 1D K-means instances are run in parallel for LLM quantization.

However, the algorithm is general and can be used for any 1D K-means problem.

I have proved the validity for the two cluster case - detailed version of the proof might be posted in the future.

