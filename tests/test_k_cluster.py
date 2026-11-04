import unittest

import numba
import numpy as np

from flash1dkmeans import kmeans_1d, numba_kmeans_1d_k_cluster, numba_kmeans_1d_k_cluster_unweighted
from flash1dkmeans.k_cluster import (
    _centroids_to_cluster_borders,
    _insert_centroid,
    _insertion_borders,
    _kmeans_plusplus,
    _kmeans_plusplus_unweighted,
    _rand_choice_centroids,
    _rand_choice_centroids_unweighted,
)
from flash1dkmeans.utils import set_np_seed_njit

import reference_kmeans_plusplus as reference


@numba.njit
def random_sample_njit(size):
    """Draws from Numba's random state, which is separate from NumPy's."""
    return np.random.random_sample(size)


def make_case(rng, sizes=(2, 3, 5, 10, 50, 300, 2000), exact=False):
    """Draws a random input for the equivalence tests: sorted data, optional weights, k and a range.
    With exact=True the data and weights are small integers, so that every prefix sum and inertia is exact."""
    n = int(rng.choice(sizes))
    kind = rng.choice(["normal", "uniform", "heavy_tailed", "discrete", "constant"])
    if exact:
        X = rng.integers(-20, 21, size=n).astype(np.float64)
    elif kind == "normal":
        X = rng.normal(size=n)
    elif kind == "uniform":
        X = rng.random(n)
    elif kind == "heavy_tailed":
        X = rng.standard_t(2, size=n)
    elif kind == "discrete":
        X = rng.integers(0, 8, size=n).astype(np.float64)
    else:
        X = np.full(n, 0.3)
    X = np.sort(X)
    if not exact and rng.random() < 0.2:
        X = X.astype(np.float32)

    weight_kind = rng.choice(["unweighted", "uniform", "sparse", "integer"])
    if weight_kind == "unweighted":
        weights = None
    elif weight_kind == "integer" or exact:
        weights = rng.integers(0, 4, size=n).astype(np.float64)
    elif weight_kind == "uniform":
        weights = rng.random(n)
    else:
        weights = rng.random(n) * (rng.random(n) < 0.3)

    start, stop = 0, n
    if n > 3 and rng.random() < 0.3:
        start = int(rng.integers(0, n // 2))
        stop = int(rng.integers(start + 2, n + 1))
    size = stop - start
    k = int(rng.choice([2, 3, min(size, 8), min(size, 64), size - 1, size]))
    k = max(k, 2)
    return X, weights, k, start, stop


def prefix_sums(X, weights):
    """Returns the prefix sums used by the numba functions, unweighted if weights is None."""
    X64 = X.astype(np.float64)
    if weights is None:
        return np.cumsum(X64), np.cumsum(X64 ** 2)
    return np.cumsum(weights), np.cumsum(X64 * weights), np.cumsum(X64 ** 2 * weights)


def clustering_wcss(X, weights, centroids, start, stop):
    """WCSS of the samples in [start, stop) around their closest centroid (ties to the lower centroid)."""
    sorted_centroids = np.sort(centroids)
    borders = _centroids_to_cluster_borders(X, sorted_centroids, start, stop)
    labels = np.repeat(np.arange(len(sorted_centroids)), np.diff(borders))
    w = np.ones(stop - start) if weights is None else weights[start:stop]
    return float(np.sum(w * (X[start:stop].astype(np.float64) - sorted_centroids[labels]) ** 2))


def compare_with_reference(X, weights, k, start, stop, seed):
    """Runs the incremental and the reference k-cluster algorithm with the same seed.

    Returns a dict with whether the initial centroids are identical, the WCSS of the initial centroids and of the
    final clustering for both, and for the first step where the initial centroids differ, whether the new choice is
    one of the reference's candidates at that step.
    """
    sums = prefix_sums(X, weights)
    if weights is None:
        new_init, reference_init = _kmeans_plusplus_unweighted, reference._kmeans_plusplus_unweighted
        new_full, reference_full = numba_kmeans_1d_k_cluster_unweighted, reference.numba_kmeans_1d_k_cluster_unweighted
    else:
        new_init, reference_init = _kmeans_plusplus, reference._kmeans_plusplus
        new_full, reference_full = numba_kmeans_1d_k_cluster, reference.numba_kmeans_1d_k_cluster

    set_np_seed_njit(seed)
    new_centroids = new_init(X, k, *sums, start, stop)
    set_np_seed_njit(seed)
    reference_centroids, candidates, inertias = reference_init(X, k, *sums, start, stop)
    new_final, _ = new_full(X, k, 300, *sums, start, stop, seed)
    reference_final, _ = reference_full(X, k, 300, *sums, start, stop, seed)

    result = {
        "identical": bool(np.array_equal(new_centroids, reference_centroids)),
        "new_init_wcss": clustering_wcss(X, weights, new_centroids, start, stop),
        "reference_init_wcss": clustering_wcss(X, weights, reference_centroids, start, stop),
        "new_final_wcss": clustering_wcss(X, weights, new_final, start, stop),
        "reference_final_wcss": clustering_wcss(X, weights, reference_final, start, stop),
    }
    if not result["identical"]:
        step = int(np.flatnonzero(new_centroids != reference_centroids)[0])
        # Each cluster's inertia is a difference of prefix sums, whose magnitude is at most max(x^2) * total weight.
        # Its rounding error is a few machine epsilons of that, and the reference sums k such terms.
        # The reference squares float32 centroids in float32, so its epsilon is that of the data type.
        total_weight = len(X) if weights is None else np.sum(weights)
        rounding_bound = 16 * (k + 1) * np.finfo(X.dtype).eps * np.max(X.astype(np.float64) ** 2) * total_weight
        is_candidate = bool(np.any(candidates[step] == new_centroids[step]))
        new_inertia = inertias[step][np.argmax(candidates[step] == new_centroids[step])]
        reference_inertia = inertias[step][np.argmax(candidates[step] == reference_centroids[step])]
        if np.max(inertias[step]) <= rounding_bound:
            # Every remaining sample already has zero distance (or zero weight), up to rounding
            result["cause"] = "zero remaining inertia"
        elif is_candidate and new_inertia - reference_inertia <= rounding_bound:
            result["cause"] = "candidates within rounding error"
        else:
            result["cause"] = "other"
        result["step"] = step
    return result


class KMeansPlusPlusEquivalenceTests(unittest.TestCase):
    def test_matches_reference_on_exact_data(self):
        # With small integer data and weights every inertia is exact, so the incremental initialization must
        # draw the same candidates and choose the same centroids as the reference.
        rng = np.random.default_rng(0)
        for seed in range(500):
            X, weights, k, start, stop = make_case(rng, sizes=(2, 3, 5, 10, 50, 300), exact=True)
            with self.subTest(seed=seed, n=len(X), k=k, weighted=weights is not None):
                self.assertTrue(compare_with_reference(X, weights, k, start, stop, seed)["identical"])

    def test_matches_reference_on_float_data(self):
        # On floating-point data the two may choose differently only where rounding decides: when every remaining
        # sample has zero inertia, so that no sample is preferred, or when two candidates are within rounding error.
        rng = np.random.default_rng(0)
        for seed in range(1000):
            X, weights, k, start, stop = make_case(rng, sizes=(2, 3, 5, 10, 50, 300))
            with self.subTest(seed=seed, n=len(X), k=k, weighted=weights is not None):
                result = compare_with_reference(X, weights, k, start, stop, seed)
                self.assertNotEqual(result.get("cause"), "other")

    def test_sampling_distribution(self):
        # The candidates must follow the D^2 distribution: each sample with probability proportional to its weight
        # times its squared distance to the closest centroid. Compared with a chi-square test over many draws.
        from scipy.stats import chisquare
        rng = np.random.default_rng(4)
        for weighted in (True, False):
            n = 40
            X = np.sort(rng.normal(size=n))
            weights = rng.random(n) if weighted else np.ones(n)
            sums = prefix_sums(X, weights if weighted else None)
            sorted_centroids = np.sort(rng.choice(X, size=4, replace=False))
            cluster_borders = _centroids_to_cluster_borders(X, sorted_centroids, 0, n)
            labels = np.repeat(np.arange(4), np.diff(cluster_borders))
            distances = weights * (X - sorted_centroids[labels]) ** 2
            cumulative_inertia = np.concatenate(([0.0], np.cumsum(
                [distances[left:right].sum() for left, right in zip(cluster_borders[:-1], cluster_borders[1:])])))

            set_np_seed_njit(0)
            choose = _rand_choice_centroids if weighted else _rand_choice_centroids_unweighted
            samples = choose(X, sorted_centroids, cluster_borders, cumulative_inertia, 4, *sums, 400000, 0, n)
            counts = np.bincount(np.searchsorted(X, samples), minlength=n)
            with self.subTest(weighted=weighted):
                # Samples at a centroid have zero probability
                self.assertEqual(counts[distances == 0].sum(), 0)
                observed, expected = counts[distances > 0], distances[distances > 0] / distances.sum() * len(samples)
                self.assertGreater(chisquare(observed, expected).pvalue, 1e-3)


class IncrementalStateTests(unittest.TestCase):
    def test_insertions_match_recomputed_borders_and_inertia(self):
        # Inserting centroids one by one must give the same borders as computing them from scratch,
        # and the cumulative inertia must match the inertia of each cluster computed directly.
        rng = np.random.default_rng(1)
        for trial in range(200):
            n = int(rng.integers(2, 200))
            X = np.sort(rng.integers(0, 20, size=n).astype(np.float64))
            weights = rng.integers(0, 3, size=n).astype(np.float64)
            weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum = prefix_sums(X, weights)
            start = int(rng.integers(0, n - 1))
            stop = int(rng.integers(start + 1, n + 1))
            k = int(rng.integers(1, stop - start + 1))
            new_centroids = rng.choice(X[start:stop], size=k)

            sorted_centroids = np.empty(k)
            cluster_borders = np.empty(k + 1, dtype=np.int32)
            cluster_inertia = np.empty(k)
            cumulative_inertia = np.zeros(k + 1)
            sorted_centroids[0] = new_centroids[0]
            cluster_borders[:2] = start, stop
            cluster_inertia[0] = np.sum(weights[start:stop] * (X[start:stop] - new_centroids[0]) ** 2)
            cumulative_inertia[1] = cluster_inertia[0]

            for n_centroids in range(1, k):
                centroid = new_centroids[n_centroids]
                pos, left_border, right_border = _insertion_borders(
                    X, sorted_centroids, n_centroids, centroid, start, stop)

                def inertia(c, left, right):
                    return np.sum(weights[left:right] * (X[left:right] - c) ** 2)

                left_inertia = inertia(sorted_centroids[pos - 1], cluster_borders[pos - 1], left_border) \
                    if pos > 0 else 0.0
                right_inertia = inertia(sorted_centroids[pos], right_border, cluster_borders[pos + 1]) \
                    if pos < n_centroids else 0.0
                _insert_centroid(sorted_centroids, cluster_borders, cluster_inertia, cumulative_inertia,
                                 n_centroids, pos, centroid, left_border, right_border,
                                 left_inertia, inertia(centroid, left_border, right_border), right_inertia)

                with self.subTest(trial=trial, n_centroids=n_centroids + 1):
                    expected_centroids = np.sort(new_centroids[:n_centroids + 1])
                    expected_borders = _centroids_to_cluster_borders(X, expected_centroids, start, stop)
                    np.testing.assert_array_equal(sorted_centroids[:n_centroids + 1], expected_centroids)
                    np.testing.assert_array_equal(cluster_borders[:n_centroids + 2], expected_borders)
                    expected_inertia = [inertia(c, left, right) for c, left, right in
                                        zip(expected_centroids, expected_borders[:-1], expected_borders[1:])]
                    np.testing.assert_allclose(np.diff(cumulative_inertia[:n_centroids + 2]), expected_inertia,
                                               rtol=0, atol=1e-9)

    def test_sampling_follows_squared_distance(self):
        # With small integer data, weights and centroids all prefix sums are exact, so the sampled index must
        # be exactly the first index where the cumulative weighted squared distance reaches the selector.
        rng = np.random.default_rng(2)
        for trial in range(100):
            n = int(rng.integers(2, 300))
            X = np.sort(rng.integers(-10, 10, size=n).astype(np.float64))
            weights = rng.integers(0, 3, size=n).astype(np.float64)
            weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum = prefix_sums(X, weights)
            k = int(rng.integers(1, min(n, 10) + 1))
            sorted_centroids = np.sort(rng.choice(X, size=k, replace=False) if k <= len(np.unique(X))
                                       else rng.choice(X, size=k))
            cluster_borders = _centroids_to_cluster_borders(X, sorted_centroids, 0, n)
            labels = np.repeat(np.arange(k), np.diff(cluster_borders))
            distances = weights * (X - sorted_centroids[labels]) ** 2
            cumulative_inertia = np.concatenate(([0.0], np.cumsum(
                [distances[left:right].sum() for left, right in zip(cluster_borders[:-1], cluster_borders[1:])])))

            sample_size = 5
            set_np_seed_njit(trial)
            samples = _rand_choice_centroids(
                X, sorted_centroids, cluster_borders, cumulative_inertia, k,
                weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                sample_size, 0, n)
            set_np_seed_njit(trial)
            selectors = random_sample_njit(sample_size) * np.sum(distances)
            expected_idx = np.minimum(np.searchsorted(np.cumsum(distances), selectors), n - 1)
            with self.subTest(trial=trial):
                np.testing.assert_array_equal(samples, X[expected_idx])


class KClusterEdgeCaseTests(unittest.TestCase):
    def assert_valid_clustering(self, X, weights, centroids, borders, k, start, stop):
        """Checks that the borders partition [start, stop) in order and the centroids are the cluster means."""
        self.assertEqual(len(centroids), k)
        self.assertEqual(borders[0], start)
        self.assertEqual(borders[-1], stop)
        self.assertTrue(np.all(np.diff(borders) >= 0))
        self.assertTrue(np.all(np.diff(centroids) >= 0))
        for c, left, right in zip(centroids, borders[:-1], borders[1:]):
            if right > left:
                w = np.ones(right - left) if weights is None else weights[left:right]
                if w.sum() > 0:
                    self.assertAlmostEqual(c, np.average(X[left:right], weights=w))

    def test_edge_cases(self):
        cases = {
            "two_samples": (np.array([0.0, 1.0]), None, 2),
            "k_equals_n": (np.arange(7, dtype=np.float64), None, 7),
            "k_equals_n_weighted": (np.arange(7, dtype=np.float64), np.arange(1, 8, dtype=np.float64), 7),
            "duplicates_k_equals_n": (np.array([1.0, 1.0, 1.0, 2.0, 2.0]), None, 5),
            "constant": (np.full(20, -0.5), None, 4),
            "constant_weighted": (np.full(20, -0.5), np.ones(20), 4),
            "zero_weights": (np.arange(10, dtype=np.float64), np.array([0, 0, 1, 0, 0, 0, 2, 0, 0, 0.0]), 3),
            "float32": (np.linspace(-1, 1, 100).astype(np.float32), None, 5),
        }
        for name, (X, weights, k) in cases.items():
            for random_state in range(5):
                with self.subTest(name=name, random_state=random_state):
                    centroids, borders = kmeans_1d(X, k, is_sorted=True, sample_weights=weights,
                                                   random_state=random_state, return_cluster_borders=True)
                    self.assert_valid_clustering(X, weights, centroids, borders, k, 0, len(X))

    def test_sub_range(self):
        rng = np.random.default_rng(3)
        X = np.sort(rng.normal(size=1000))
        weights = rng.random(1000)
        weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum = prefix_sums(X, weights)
        for start, stop in ((0, 500), (500, 1000), (123, 130)):
            with self.subTest(start=start, stop=stop):
                centroids, borders = numba_kmeans_1d_k_cluster(
                    X, 4, 300, weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                    start, stop, 0)
                self.assert_valid_clustering(X, weights, centroids, borders, 4, start, stop)

    def test_k_equals_n_has_zero_inertia(self):
        X = np.array([3.0, -1.0, 2.5, 7.0, 0.0])
        for weights in (None, np.array([1.0, 2.0, 3.0, 4.0, 5.0])):
            for random_state in range(10):
                with self.subTest(weighted=weights is not None, random_state=random_state):
                    centroids, labels = kmeans_1d(X, len(X), sample_weights=weights, random_state=random_state)
                    np.testing.assert_allclose(centroids[labels], X, rtol=1e-12)


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
