import sklearn.datasets as datasets
import numpy as np

np.random.seed(42)

print("Generating datasets...")

random_weights = np.random.random_sample(2 ** 24)

data = {
    'california_housing': datasets.fetch_california_housing()['data'][:, 0],
    'iris': datasets.load_iris()['data'][:, 0],
    '32K-blobs': datasets.make_blobs(n_samples=32000, centers=8, n_features=1, random_state=42)[0].flatten(),
    '32K-rand': np.random.random_sample(32000)
}

scaled_data = {}

for power in range(10, 24):
    size = 2 ** power
    scaled_data[f'2^{power}'] = np.random.random_sample(size)

print("Datasets generated.")
