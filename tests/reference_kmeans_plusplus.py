"""reference_kmeans_plusplus.py
Reference copy of the k-cluster algorithm before the incremental kmeans++ initialization (commit 49ab272).

Used only by the equivalence tests. The functions are unchanged except that _kmeans_plusplus and
_kmeans_plusplus_unweighted also return the candidates of every step and the inertia computed for each candidate,
which numba_kmeans_1d_k_cluster and numba_kmeans_1d_k_cluster_unweighted discard.
"""

import numba
import numpy as np
from flash1dkmeans.config import ARRAY_INDEX_DTYPE
from flash1dkmeans.utils import query_prefix_sum, set_np_seed_njit
from flash1dkmeans.k_cluster import _rand_choice_prefix_sum, _centroids_to_cluster_borders


@numba.njit(cache=True)
def numba_kmeans_1d_k_cluster(
        sorted_X,
        n_clusters,
        max_iter,
        weights_prefix_sum, weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum,
        start_idx,
        stop_idx,
        random_state=None,
):
    """An optimized kmeans for 1D data with n clusters.
    Exploits the fact that the data is 1D to optimize the calculations.
    Time complexity: O(k ^ 2 * log(k) * log(n) + i * log(n) * k)

    Args:
        sorted_X: np.ndarray
            The input data. Should be sorted in ascending order.
        n_clusters: int
            The number of clusters to generate
        max_iter: int
            The maximum number of iterations to run
        weights_prefix_sum: np.ndarray
            The prefix sum of the weights. Should be None if the data is unweighted.
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of the weighted X
        weighted_X_squared_prefix_sum: np.ndarray
            The prefix sum of the weighted X squared
        start_idx: int
            The start index of the range to consider
        stop_idx: int
            The stop index of the range to consider
        random_state: int or None
            The random seed to use.

    Returns:
        centroids: np.ndarray
            The centroids of the clusters
        cluster_borders: np.ndarray
            The borders of the clusters
    """
    # set random_state
    set_np_seed_njit(random_state)

    cluster_borders = np.empty(n_clusters + 1, dtype=ARRAY_INDEX_DTYPE)
    cluster_borders[0] = start_idx
    cluster_borders[-1] = stop_idx
    cluster_borders[1:-1] = -1  # not valid borders, so that the first iteration always runs

    centroids, _, _ = _kmeans_plusplus(
        sorted_X, n_clusters,
        weights_prefix_sum, weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum,
        start_idx, stop_idx,
    )
    sorted_centroids = np.sort(centroids)

    for _ in range(max_iter):
        new_cluster_borders = _centroids_to_cluster_borders(sorted_X, sorted_centroids, start_idx, stop_idx)

        if np.array_equal(cluster_borders, new_cluster_borders):
            break

        cluster_borders[:] = new_cluster_borders
        for i in range(n_clusters):
            cluster_start = cluster_borders[i]
            cluster_end = cluster_borders[i + 1]

            if cluster_end < cluster_start:
                raise ValueError("Cluster end is less than cluster start")

            if cluster_start == cluster_end:
                continue

            cluster_weighted_X_sum = query_prefix_sum(weighted_X_prefix_sum, cluster_start, cluster_end)
            cluster_weight_sum = query_prefix_sum(weights_prefix_sum, cluster_start, cluster_end)

            if cluster_weight_sum == 0:
                # if the sum of the weights is zero, we set the centroid to the mean of the cluster
                sorted_centroids[i] = sorted_X[cluster_start:cluster_end].mean()
            else:
                sorted_centroids[i] = cluster_weighted_X_sum / cluster_weight_sum

    return sorted_centroids, cluster_borders


@numba.njit(cache=True)
def numba_kmeans_1d_k_cluster_unweighted(
        sorted_X,
        n_clusters,
        max_iter,
        X_prefix_sum,
        X_squared_prefix_sum,
        start_idx,
        stop_idx,
        random_state=None,
):
    """Unweighted version of flash_1d_kmeans_k_cluster"""
    # set random_state
    set_np_seed_njit(random_state)

    cluster_borders = np.empty(n_clusters + 1, dtype=ARRAY_INDEX_DTYPE)
    cluster_borders[0] = start_idx
    cluster_borders[-1] = stop_idx
    cluster_borders[1:-1] = -1  # not valid borders, so that the first iteration always runs

    centroids, _, _ = _kmeans_plusplus_unweighted(
        sorted_X, n_clusters,
        X_prefix_sum,
        X_squared_prefix_sum,
        start_idx, stop_idx,
    )
    sorted_centroids = np.sort(centroids)

    for _ in range(max_iter):
        new_cluster_borders = _centroids_to_cluster_borders(sorted_X, sorted_centroids, start_idx, stop_idx)

        if np.array_equal(cluster_borders, new_cluster_borders):
            break

        cluster_borders[:] = new_cluster_borders
        for i in range(n_clusters):
            cluster_start = cluster_borders[i]
            cluster_end = cluster_borders[i + 1]

            if cluster_end < cluster_start:
                raise ValueError("Cluster end is less than cluster start")

            if cluster_start == cluster_end:
                continue

            cluster_weighted_X_sum = query_prefix_sum(X_prefix_sum, cluster_start, cluster_end)
            cluster_weight_sum = cluster_end - cluster_start

            if cluster_weight_sum == 0:
                # if the sum of the weights is zero, we set the centroid to the mean of the cluster
                sorted_centroids[i] = sorted_X[cluster_start:cluster_end].mean()
            else:
                sorted_centroids[i] = cluster_weighted_X_sum / cluster_weight_sum

    return sorted_centroids, cluster_borders


@numba.njit(cache=True)
def _calculate_inertia(sorted_centroids, centroid_ranges,
                       weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                       stop_idx):
    """Calculates the inertia of the clusters given the centroids.
    The inertia is the sum of the squared distances of each sample to the closest centroid.
    The calculations are done efficiently using prefix sums.

    Time complexity: O(k)

    Args:
        sorted_centroids: np.ndarray
            The centroids of the clusters
        centroid_ranges: np.ndarray
            The borders of the clusters
        weights_prefix_sum: np.ndarray
            The prefix sum of the weights. Should be None if the data is unweighted.
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of the weighted X
        weighted_X_squared_prefix_sum: np.ndarray
            The prefix sum of the weighted X squared
        stop_idx: int
            The stop index of the range to consider
    """
    # inertia = sigma_i(w_i * abs(x_i - c)^2) = sigma_i(w_i * (x_i^2 - 2 * x_i * c + c^2))
    #         = sigma_i(w_i * x_i^2) - 2 * c * sigma_i(w_i * x_i) + c^2 * sigma_i(w_i)
    #         = sigma_i(weighted_X_squared) - 2 * c * sigma_i(weighted_X) + c^2 * sigma_i(weight)
    #  Note that the centroid c is the CLOSEST centroid to x_i, so the above calculation must be done for each cluster

    inertia = 0
    for i in range(len(sorted_centroids)):
        start = centroid_ranges[i]
        end = centroid_ranges[i + 1]

        if start >= stop_idx:
            break
        if end >= stop_idx:
            end = stop_idx

        if start == end:
            continue

        cluster_weighted_X_squared_sum = query_prefix_sum(weighted_X_squared_prefix_sum, start, end)
        cluster_weighted_X_sum = query_prefix_sum(weighted_X_prefix_sum, start, end)
        cluster_weight_sum = query_prefix_sum(weights_prefix_sum, start, end)

        inertia += (cluster_weighted_X_squared_sum - 2 * sorted_centroids[i] * cluster_weighted_X_sum +
                    sorted_centroids[i] ** 2 * cluster_weight_sum)

    return inertia


@numba.njit(cache=True)
def _calculate_inertia_unweighted(sorted_centroids, centroid_ranges,
                                  X_prefix_sum, X_squared_prefix_sum,
                                  stop_idx):
    """Unweighted version of _calculate_inertia"""
    # inertia = sigma_i(w_i * abs(x_i - c)^2) = sigma_i(w_i * (x_i^2 - 2 * x_i * c + c^2))
    #         = sigma_i(w_i * x_i^2) - 2 * c * sigma_i(w_i * x_i) + c^2 * sigma_i(w_i)
    #         = sigma_i(weighted_X_squared) - 2 * c * sigma_i(weighted_X) + c^2 * sigma_i(weight)
    #  Note that the centroid c is the CLOSEST centroid to x_i, so the above calculation must be done for each cluster

    inertia = 0
    for i in range(len(sorted_centroids)):
        start = centroid_ranges[i]
        end = centroid_ranges[i + 1]

        if start >= stop_idx:
            break
        if end >= stop_idx:
            end = stop_idx

        if start == end:
            continue

        cluster_weighted_X_squared_sum = query_prefix_sum(X_squared_prefix_sum, start, end)
        cluster_weighted_X_sum = query_prefix_sum(X_prefix_sum, start, end)
        cluster_weight_sum = end - start

        inertia += (cluster_weighted_X_squared_sum - 2 * sorted_centroids[i] * cluster_weighted_X_sum +
                    sorted_centroids[i] ** 2 * cluster_weight_sum)

    return inertia


@numba.njit(cache=True)
def _rand_choice_centroids(X, centroids,
                           weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                           sample_size, start_idx, stop_idx):
    """Randomly choose sample_size elements from X, weighted by the distance to the closest centroid.
    The weighted logic is implemented efficiently by utilizing the _calculate_inertia function.

    Time complexity: O(l * k * log n)

    Args:
        X: np.ndarray
            The input data. Should be sorted in ascending order.
        centroids: np.ndarray
            The centroids of the clusters
        is_weighted: bool
            Whether the data is weighted. If True, the weighted versions of the arrays should be provided.
        weights_prefix_sum: np.ndarray
            The prefix sum of the weights. Should be None if the data is unweighted.
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of the weighted X
        weighted_X_squared_prefix_sum: np.ndarray
            The prefix sum of the weighted X squared
        sample_size: int
            The number of samples to choose
        start_idx: int
            The start index of the range to consider
        stop_idx: int
            The stop index of the range to consider

    Returns:
        np.ndarray: The chosen samples
    """
    sorted_centroids = np.sort(centroids)  # O(k log k)
    cluster_borders = _centroids_to_cluster_borders(X, sorted_centroids, start_idx, stop_idx)  # O(k log n)
    total_inertia = _calculate_inertia(sorted_centroids, cluster_borders,  # O(k)
                                       weights_prefix_sum, weighted_X_prefix_sum,
                                       weighted_X_squared_prefix_sum, stop_idx)
    selectors = np.random.random_sample(sample_size) * total_inertia
    results = np.empty(sample_size, dtype=centroids.dtype)

    for i in range(sample_size):  # O(l k log n)
        selector = selectors[i]
        floor = start_idx + 1
        ceiling = stop_idx
        while floor < ceiling:
            stop_idx_cand = (floor + ceiling) // 2
            inertia = _calculate_inertia(sorted_centroids, cluster_borders,  # O(k)
                                         weights_prefix_sum, weighted_X_prefix_sum,
                                         weighted_X_squared_prefix_sum, stop_idx_cand)
            if inertia < selector:
                floor = stop_idx_cand + 1
            else:
                ceiling = stop_idx_cand
        results[i] = X[floor - 1]

    return results


@numba.njit(cache=True)
def _rand_choice_centroids_unweighted(X, centroids,
                                      X_prefix_sum,
                                      X_squared_prefix_sum,
                                      sample_size, start_idx, stop_idx):
    """Unweighted version of _rand_choice_centroids"""
    sorted_centroids = np.sort(centroids)
    cluster_borders = _centroids_to_cluster_borders(X, sorted_centroids, start_idx, stop_idx)
    total_inertia = _calculate_inertia_unweighted(sorted_centroids, cluster_borders,
                                                  X_prefix_sum, X_squared_prefix_sum, stop_idx)
    selectors = np.random.random_sample(sample_size) * total_inertia
    results = np.empty(sample_size, dtype=centroids.dtype)

    for i in range(sample_size):
        selector = selectors[i]
        floor = start_idx + 1
        ceiling = stop_idx
        while floor < ceiling:
            stop_idx_cand = (floor + ceiling) // 2
            inertia = _calculate_inertia_unweighted(sorted_centroids, cluster_borders,
                                                    X_prefix_sum, X_squared_prefix_sum, stop_idx_cand)
            if inertia < selector:
                floor = stop_idx_cand + 1
            else:
                ceiling = stop_idx_cand
        results[i] = X[floor - 1]

    return results


@numba.njit(cache=True)
def _kmeans_plusplus(X, n_clusters,
                     weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                     start_idx, stop_idx):
    """An optimized version of the kmeans++ initialization algorithm for 1D data.
    The algorithm is optimized for 1D data and utilizes prefix sums for efficient calculations.

    Time complexity: = O(k ^ 2 * log k * log n)

    Args:
        X: np.ndarray
            The input data
        n_clusters: int
            The number of clusters to choose
        weights_prefix_sum: np.ndarray
            The prefix sum of the weights. Should be None if the data is unweighted.
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of the weighted X
        weighted_X_squared_prefix_sum: np.ndarray
            The prefix sum of the weighted X squared

    Returns:
        np.ndarray: The chosen centroids
        np.ndarray: The candidates of every step, shape (n_clusters, n_local_trials)
        np.ndarray: The inertia of every candidate, shape (n_clusters, n_local_trials)
    """
    centroids = np.empty(n_clusters, dtype=X.dtype)
    n_local_trials = 2 + int(np.log(n_clusters))
    all_candidates = np.zeros((n_clusters, n_local_trials), dtype=X.dtype)  # recorded for the tests
    all_inertias = np.zeros((n_clusters, n_local_trials))  # recorded for the tests

    # First centroid is chosen randomly according to sample_weight
    centroids[0] = _rand_choice_prefix_sum(X, weights_prefix_sum, start_idx, stop_idx)  # O(log n)

    for c_id in range(1, n_clusters):  # O(k^2 l log n)
        # Choose the next centroid randomly according to the weighted distances
        # Sample n_local_trials candidates and choose the best one

        centroid_candidates = _rand_choice_centroids(  # O(l k log n)
            X, centroids[:c_id],
            weights_prefix_sum, weighted_X_prefix_sum,
            weighted_X_squared_prefix_sum, n_local_trials,
            start_idx, stop_idx
        )

        best_inertia = np.inf
        best_centroid = None
        for i in range(len(centroid_candidates)): # O(l k log n)
            # O(k log k + k log n + k) = O(k log n), as k <= n
            centroids[c_id] = centroid_candidates[i]
            sorted_centroids = np.sort(centroids[:c_id + 1]) # O(k log k), I think we could avoid centroid sorting and use some linear algorithm, but the gain would be minimal, especially considering that k <= n, and most times k << n
            centroid_ranges = _centroids_to_cluster_borders(X, sorted_centroids, start_idx, stop_idx)  # O(k log n)
            inertia = _calculate_inertia(sorted_centroids, centroid_ranges,  # O(k)
                                         weights_prefix_sum, weighted_X_prefix_sum,
                                         weighted_X_squared_prefix_sum, stop_idx)
            all_candidates[c_id, i] = centroid_candidates[i]  # recorded for the tests
            all_inertias[c_id, i] = inertia  # recorded for the tests
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroid = centroid_candidates[i]
        centroids[c_id] = best_centroid

    return centroids, all_candidates, all_inertias


@numba.njit(cache=True)
def _kmeans_plusplus_unweighted(X, n_clusters,
                                X_prefix_sum, X_squared_prefix_sum,
                                start_idx, stop_idx):
    """Unweighted version of _kmeans_plusplus"""
    centroids = np.empty(n_clusters, dtype=X.dtype)
    n_local_trials = 2 + int(np.log(n_clusters))
    all_candidates = np.zeros((n_clusters, n_local_trials), dtype=X.dtype)  # recorded for the tests
    all_inertias = np.zeros((n_clusters, n_local_trials))  # recorded for the tests

    # First centroid is chosen randomly according to sample_weight
    centroids[0] = X[np.random.randint(start_idx, stop_idx)]

    for c_id in range(1, n_clusters):
        # Choose the next centroid randomly according to the weighted distances
        # Sample n_local_trials candidates and choose the best one
        centroid_candidates = _rand_choice_centroids_unweighted(
            X, centroids[:c_id], X_prefix_sum,
            X_squared_prefix_sum, n_local_trials,
            start_idx, stop_idx
        )

        best_inertia = np.inf
        best_centroid = None
        for i in range(len(centroid_candidates)):
            centroids[c_id] = centroid_candidates[i]
            sorted_centroids = np.sort(centroids[:c_id + 1])
            centroid_ranges = _centroids_to_cluster_borders(X, sorted_centroids, start_idx, stop_idx)
            inertia = _calculate_inertia_unweighted(sorted_centroids, centroid_ranges,
                                                    X_prefix_sum, X_squared_prefix_sum, stop_idx)
            all_candidates[c_id, i] = centroid_candidates[i]  # recorded for the tests
            all_inertias[c_id, i] = inertia  # recorded for the tests
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroid = centroid_candidates[i]
        centroids[c_id] = best_centroid

    return centroids, all_candidates, all_inertias
