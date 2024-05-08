"""n_cluster.py
Optimized kmeans for 1D data with n clusters.

This module contains an optimized kmeans for 1D data with n clusters.

The following is the main function:
- flash_1d_kmeans_n_cluster

Inputs must be sorted in ascending order, no default values are provided, and no error checking is done.
This is because this module is intended to be used internally.
"""

import numba
from .utils import query_prefix_sum
import numpy as np
from .config import ARRAY_INDEX_DTYPE


@numba.njit(cache=True)
def numba_kmeans_1d_k_cluster(
        sorted_X,
        n_clusters,
        max_iter,
        weights_prefix_sum, weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum,
        start_idx,
        stop_idx,
):
    """An optimized kmeans for 1D data with n clusters.
    Exploits the fact that the data is 1D to optimize the calculations.
    Time complexity: O(n_clusters * log(n_clusters) * log(len(X)) * n_clusters + max_iter * log(len(X)) * n_clusters)
                      = O(k ^ 2 * log(k) * log(n) + max_iter * log(n) * k)

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

    Returns:
        centroids: np.ndarray
            The centroids of the clusters
        cluster_borders: np.ndarray
            The borders of the clusters
    """
    assert n_clusters > 2, "n_clusters must be greater than 2, for 2 clusters use the faster two cluster implementation"

    cluster_borders = np.empty(n_clusters + 1, dtype=ARRAY_INDEX_DTYPE)
    cluster_borders[0] = start_idx
    cluster_borders[-1] = stop_idx

    centroids = _kmeans_plusplus(
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
):
    """Unweighted version of flash_1d_kmeans_k_cluster"""
    assert n_clusters > 2, "n_clusters must be greater than 2, for 2 clusters use the faster two cluster implementation"

    cluster_borders = np.empty(n_clusters + 1, dtype=ARRAY_INDEX_DTYPE)
    cluster_borders[0] = start_idx
    cluster_borders[-1] = stop_idx

    centroids = _kmeans_plusplus_unweighted(
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
def _rand_choice_prefix_sum(arr, prob_prefix_sum, start_idx, stop_idx):
    """Randomly choose an element from arr according to the probability distribution given by prob_prefix_sum
    Time complexity: O(log(n))

    Args:
        arr: np.ndarray
            The array to choose from
        prob_prefix_sum: np.ndarray
            The prefix sum of the probability distribution
        start_idx: int
            The start index of the range to consider
        stop_idx: int
            The stop index of the range to consider

    Returns:
        The chosen element
    """
    total_prob = query_prefix_sum(prob_prefix_sum, start_idx, stop_idx)
    selector = np.random.random_sample() * total_prob

    # Because we are using start_idx as the base, but the prefix sum is calculated from 0,
    # we need to adjust the selector if start_idx is not 0.
    adjusted_selector = selector + prob_prefix_sum[start_idx - 1] if start_idx > 0 else selector

    # Search for the index of the selector in the prefix sum, and add start_idx to get the index in the original array
    idx = np.searchsorted(prob_prefix_sum[start_idx:stop_idx], adjusted_selector) + start_idx

    return arr[idx]


@numba.njit(cache=True)
def _centroids_to_cluster_borders(X, sorted_centroids, start_idx, stop_idx):
    """Converts the centroids to cluster borders.
    The cluster borders are where the clusters are divided.
    The centroids must be sorted.

    Time complexity: O(log(len(X)) * len(centroids))

    Args:
        X: np.ndarray
            The input data. Should be sorted in ascending order.
        sorted_centroids: np.ndarray
            The sorted centroids
        start_idx: int
            The start index of the range to consider
        stop_idx: int
            The stop index of the range to consider

    Returns:
        np.ndarray: The cluster borders
    """
    midpoints = (sorted_centroids[:-1] + sorted_centroids[1:]) / 2
    cluster_borders = np.empty(len(sorted_centroids) + 1, dtype=ARRAY_INDEX_DTYPE)
    cluster_borders[0] = start_idx
    cluster_borders[-1] = stop_idx
    cluster_borders[1:-1] = np.searchsorted(X[start_idx:stop_idx], midpoints) + start_idx
    return cluster_borders


@numba.njit(cache=True)
def _calculate_inertia(sorted_centroids, centroid_ranges,
                       weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                       stop_idx):
    """Calculates the inertia of the clusters given the centroids.
    The inertia is the sum of the squared distances of each sample to the closest centroid.
    The calculations are done efficiently using prefix sums.

    Time complexity: O(len(centroids))

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

    Time complexity: O(sample_size * log(len(X)) * len(centroids))

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
    sorted_centroids = np.sort(centroids)
    cluster_borders = _centroids_to_cluster_borders(X, sorted_centroids, start_idx, stop_idx)
    total_inertia = _calculate_inertia(sorted_centroids, cluster_borders,
                                       weights_prefix_sum, weighted_X_prefix_sum,
                                       weighted_X_squared_prefix_sum, stop_idx)
    selectors = np.random.random_sample(sample_size) * total_inertia
    results = np.empty(sample_size, dtype=centroids.dtype)

    for i in range(sample_size):
        selector = selectors[i]
        floor = start_idx + 1
        ceiling = stop_idx
        while floor < ceiling:
            stop_idx_cand = (floor + ceiling) // 2
            inertia = _calculate_inertia(sorted_centroids, cluster_borders,
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

    Time complexity: O(n_clusters * (log(n_clusters) * log(len(X)) * n_clusters + log(len(X)) * n_clusters + n_clusters))
                     = O(k ^ 2 * log(k) * log(n))

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
    """
    centroids = np.empty(n_clusters, dtype=X.dtype)
    n_local_trials = 2 + int(np.log(n_clusters))

    # First centroid is chosen randomly according to sample_weight
    centroids[0] = _rand_choice_prefix_sum(X, weights_prefix_sum, start_idx, stop_idx)

    for c_id in range(1, n_clusters):
        # Choose the next centroid randomly according to the weighted distances
        # Sample n_local_trials candidates and choose the best one
        centroid_candidates = _rand_choice_centroids(
            X, centroids[:c_id],
            weights_prefix_sum, weighted_X_prefix_sum,
            weighted_X_squared_prefix_sum, n_local_trials,
            start_idx, stop_idx
        )

        best_inertia = np.inf
        best_centroid = None
        for i in range(len(centroid_candidates)):
            centroids[c_id] = centroid_candidates[i]
            sorted_centroids = np.sort(centroids[:c_id + 1])
            centroid_ranges = _centroids_to_cluster_borders(X, sorted_centroids, start_idx, stop_idx)
            inertia = _calculate_inertia(sorted_centroids, centroid_ranges,
                                         weights_prefix_sum, weighted_X_prefix_sum,
                                         weighted_X_squared_prefix_sum, stop_idx)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroid = centroid_candidates[i]
        centroids[c_id] = best_centroid

    return centroids


@numba.njit(cache=True)
def _kmeans_plusplus_unweighted(X, n_clusters,
                                X_prefix_sum, X_squared_prefix_sum,
                                start_idx, stop_idx):
    """Unweighted version of _kmeans_plusplus"""
    centroids = np.empty(n_clusters, dtype=X.dtype)
    n_local_trials = 2 + int(np.log(n_clusters))

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
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroid = centroid_candidates[i]
        centroids[c_id] = best_centroid

    return centroids
