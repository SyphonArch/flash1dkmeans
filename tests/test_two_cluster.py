import unittest

import numpy as np

from flash1dkmeans import kmeans_1d_two_cluster
from flash1dkmeans.two_cluster import (
    numba_kmeans_1d_two_cluster,
    numba_kmeans_1d_two_cluster_unweighted,
)


class TwoClusterCentroidTests(unittest.TestCase):
    def test_public_api_final_split(self):
        data = np.array([0, 0, 1, 1, 100], dtype=np.float64)
        for weights in (None, np.ones(5, dtype=np.float64)):
            with self.subTest(weighted=weights is not None):
                centroids, borders = kmeans_1d_two_cluster(
                    data, is_sorted=True, sample_weights=weights,
                    return_cluster_borders=True,
                )
                np.testing.assert_array_equal(borders, [0, 4, 5])
                np.testing.assert_allclose(centroids, [0.5, 100])

    def test_centroids_match_returned_partition(self):
        # Exercise both search directions and convergence inside the loop.
        for values, split in (
            ([0, 0, 1, 1, 100], 4),
            ([-100, -1, -1, 0, 0], 1),
            ([0, 0, 100, 100, 100], 2),
        ):
            for offset in (0, 1):
                data = np.array(
                    values if offset == 0 else [-200] + values + [200],
                    dtype=np.float64,
                )
                start, stop = offset, offset + len(values)
                for weight_kind in ("unweighted", "unit", "nonuniform"):
                    with self.subTest(values=values, offset=offset, weights=weight_kind):
                        weights = np.ones(len(data), dtype=np.float64)
                        if weight_kind == "nonuniform":
                            weights[start:stop] = [1, 2, 3, 4, 5]
                        if weight_kind == "unweighted":
                            centroids, borders = numba_kmeans_1d_two_cluster_unweighted(
                                data, np.cumsum(data), start, stop,
                            )
                        else:
                            centroids, borders = numba_kmeans_1d_two_cluster(
                                data, np.cumsum(weights), np.cumsum(data * weights),
                                start, stop,
                            )
                        np.testing.assert_array_equal(borders, [start, start + split, stop])
                        expected = [
                            np.average(data[left:right], weights=weights[left:right])
                            for left, right in zip(borders[:-1], borders[1:])
                        ]
                        np.testing.assert_allclose(centroids, expected)


if __name__ == "__main__":
    unittest.main()
