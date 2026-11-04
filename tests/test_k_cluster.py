import unittest

import numpy as np

from flash1dkmeans import numba_kmeans_1d_k_cluster, numba_kmeans_1d_k_cluster_unweighted
from flash1dkmeans.k_cluster import _kmeans_plusplus, _kmeans_plusplus_unweighted
from flash1dkmeans.utils import set_np_seed_njit


class LloydIterationTests(unittest.TestCase):
    def test_lloyd_runs_after_initialization(self):
        # The cluster borders compared in the first Lloyd's iteration used to be uninitialized memory. When that
        # memory still held the borders of the initial centroids, for example after a call with the same data and
        # random_state, Lloyd's algorithm stopped before updating any centroid. After at least one iteration,
        # every centroid is the mean of its cluster.
        rng = np.random.default_rng(0)
        for n in (1000, 100000):
            X = np.sort(rng.random(n))
            weights = rng.random(n)
            weighted_sums = (np.cumsum(weights), np.cumsum(X * weights), np.cumsum(X ** 2 * weights))
            unweighted_sums = (np.cumsum(X), np.cumsum(X ** 2))
            for k in (2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512):
                for weighted in (True, False):
                    with self.subTest(n=n, k=k, weighted=weighted):
                        set_np_seed_njit(0)
                        if weighted:
                            _kmeans_plusplus(X, k, *weighted_sums, 0, n)
                            centroids, borders = numba_kmeans_1d_k_cluster(X, k, 300, *weighted_sums, 0, n, 0)
                        else:
                            _kmeans_plusplus_unweighted(X, k, *unweighted_sums, 0, n)
                            centroids, borders = numba_kmeans_1d_k_cluster_unweighted(
                                X, k, 300, *unweighted_sums, 0, n, 0)
                        for c, left, right in zip(centroids, borders[:-1], borders[1:]):
                            if right > left:
                                w = weights[left:right] if weighted else None
                                self.assertAlmostEqual(c, np.average(X[left:right], weights=w), delta=1e-9)


if __name__ == "__main__":
    unittest.main()
