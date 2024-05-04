# flash1dkmeans
An optimized K-means implementation for the one-dimensional case

## Installation
```bash
pip install flash1dkmeans
```

## Usage
```python
from flash1dkmeans import kmeans_1d

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
k = 2
max_iter = 100
tol = 1e-4

centroids, labels = kmeans_1d(data, k, max_iter, tol)
```