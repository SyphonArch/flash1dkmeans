"""n_cluster.py
Optimized kmeans for 1D data with n clusters.

This module contains an optimized kmeans for 1D data with n clusters.

There are two functions intended to be used outside of this module:
- flash_1d_kmeans_n_cluster
- flash_1d_kmeans_n_cluster_unweighted

The first function is a weighted version of the kmeans algorithm for 1D data with n clusters.
The second function is an unweighted version of the kmeans algorithm for 1D data with n clusters.

Inputs must be sorted in ascending order, and no default values are provided -
this is because this module is intended to be used internally.
"""

import numba
from .utils import query_prefix_sum
import numpy as np


@numba.njit(cache=True)
def flash_1d_kmeans_n_cluster(
        X, weights_prefix_sum, weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum,
        n_clusters,
        max_iter,
):
    """WARNING: All inputs must be sorted in ascending order of X"""
    cluster_borders = np.empty(n_clusters + 1, dtype=np.int32)
    cluster_borders[0] = 0
    cluster_borders[-1] = len(X)

    new_cluster_borders = np.empty(n_clusters + 1, dtype=np.int32)
    new_cluster_borders[0] = 0
    new_cluster_borders[-1] = len(X)

    centroids = _kmeans_plusplus(
        X, n_clusters,
        weights_prefix_sum, weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum
    )
    centroids.sort()

    for _ in range(max_iter):
        cluster_midpoints = (centroids[:-1] + centroids[1:]) / 2
        for i in range(n_clusters - 1):
            new_cluster_borders[i + 1] = np.searchsorted(X, cluster_midpoints[i])

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
                centroids[i] = X[cluster_start:cluster_end].mean()
            else:
                centroids[i] = cluster_weighted_X_sum / cluster_weight_sum

    return centroids, cluster_borders


@numba.njit(cache=True)
def _rand_choice_prefix_sum(arr, prob_prefix_sum):
    """Randomly choose an element from arr according to the probability distribution given by prob_prefix_sum
    Time complexity: O(log(n))

    Args:
        arr: np.ndarray
            The array to choose from
        prob_prefix_sum: np.ndarray
            The prefix sum of the probability distribution

    Returns:
        The chosen element
    """
    total_prob = query_prefix_sum(prob_prefix_sum, 0, len(prob_prefix_sum))
    selector = np.random.random_sample() * total_prob
    return arr[np.searchsorted(prob_prefix_sum, selector)]


@numba.njit(cache=True)
def centroids_to_cluster_borders(X, sorted_centroids):
    """Converts the centroids to cluster borders.
    The cluster borders are where the clusters are divided.
    The centroids must be sorted.

    Time complexity: O(log(len(X)) * len(centroids))

    Args:
        X: np.ndarray
            The input data. Should be sorted in ascending order.
        sorted_centroids: np.ndarray
            The sorted centroids

    Returns:
        np.ndarray: The cluster borders
    """
    midpoints = (sorted_centroids[:-1] + sorted_centroids[1:]) / 2
    cluster_borders = np.empty(len(sorted_centroids) + 1, dtype=np.int32)
    cluster_borders[0] = 0
    cluster_borders[-1] = len(X)
    cluster_borders[1:-1] = np.searchsorted(X, midpoints)
    return cluster_borders


@numba.njit(cache=True)
def _calculate_inertia(X, sorted_centroids, centroid_ranges,
                       weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                       stop_idx=None):
    """Calculates the inertia of the clusters given the centroids.
    The inertia is the sum of the squared distances of each sample to the closest centroid.
    The calculations are done efficiently using prefix sums.

    Time complexity: O(len(centroids) * log(len(X)))

    Args:
        X: np.ndarray
            The input data. Should be sorted in ascending order.
        sorted_centroids: np.ndarray
            The centroids of the clusters
        weights_prefix_sum: np.ndarray
            The prefix sum of the weights
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of the weighted X
        weighted_X_squared_prefix_sum: np.ndarray
            The prefix sum of the weighted X squared
        stop_idx: int
            The index to stop calculating the inertia. If None, calculate the inertia for the entire X
    """
    if stop_idx is None:
        stop_idx = len(X)

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
def _rand_choice_centroids(X, centroids, weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                           sample_size):
    """Randomly choose sample_size elements from X, weighted by the distance to the closest centroid.
    The weighted logic is implemented efficiently by utilizing the _calculate_inertia function.

    Time complexity: O(sample_size * log(len(X)) * len(centroids) * log(len(X)))

    Args:
        X: np.ndarray
            The input data. Should be sorted in ascending order.
        centroids: np.ndarray
            The centroids of the clusters
        weights_prefix_sum: np.ndarray
            The prefix sum of the weights
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of the weighted X
        weighted_X_squared_prefix_sum: np.ndarray
            The prefix sum of the weighted X squared
        sample_size: int
            The number of samples to choose

    Returns:
        np.ndarray: The chosen samples
    """
    sorted_centroids = np.sort(centroids)
    cluster_borders = centroids_to_cluster_borders(X, sorted_centroids)
    total_inertia = _calculate_inertia(X, sorted_centroids, cluster_borders,
                                       weights_prefix_sum, weighted_X_prefix_sum,
                                       weighted_X_squared_prefix_sum)
    selectors = np.random.random_sample(sample_size) * total_inertia
    results = np.empty(sample_size, dtype=centroids.dtype)

    for i in range(sample_size):
        selector = selectors[i]
        left = 1
        right = len(X)
        while left < right:
            mid = (left + right) // 2
            inertia = _calculate_inertia(X, sorted_centroids, cluster_borders,
                                         weights_prefix_sum, weighted_X_prefix_sum,
                                         weighted_X_squared_prefix_sum,
                                         stop_idx=mid)
            if inertia < selector:
                left = mid + 1
            else:
                right = mid
        results[i] = X[left - 1]

    return results


@numba.njit(cache=True)
def _kmeans_plusplus(X, n_clusters, weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum):
    """An optimized version of the kmeans++ initialization algorithm for 1D data.
    The algorithm is optimized for 1D data and utilizes prefix sums for efficient calculations.

    Time complexity: O(n_clusters * log(n_clusters) * log(len(X)) * len(n_clusters) * log(len(X)))
                     = O(k ^ 2 * log(k) * log(n) ^ 2)

    Args:
        X: np.ndarray
            The input data
        n_clusters: int
            The number of clusters to choose
        weights_prefix_sum: np.ndarray
            The prefix sum of the weights
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of the weighted X
        weighted_X_squared_prefix_sum: np.ndarray
            The prefix sum of the weighted X squared

    Returns:
        np.ndarray: The chosen centroids
    """
    n_local_trials = 2 + int(np.log(n_clusters))

    centroids = np.empty(n_clusters, dtype=X.dtype)

    # First centroid is chosen randomly according to sample_weight
    centroids[0] = _rand_choice_prefix_sum(X, weights_prefix_sum, 0, len(X))

    for c_id in range(1, n_clusters):
        # Choose the next centroid randomly according to the weighted distances
        # Sample n_local_trials candidates and choose the best one
        centroid_candidates = _rand_choice_centroids(X, centroids[:c_id], weights_prefix_sum, weighted_X_prefix_sum,
                                                     weighted_X_squared_prefix_sum, n_local_trials)

        best_inertia = np.inf
        best_centroid = None
        for i in range(len(centroid_candidates)):
            centroids[c_id] = centroid_candidates[i]
            sorted_centroids = np.sort(centroids[:c_id + 1])
            centroid_ranges = centroids_to_cluster_borders(X, sorted_centroids)
            inertia = _calculate_inertia(X, sorted_centroids, centroid_ranges,
                                         weights_prefix_sum, weighted_X_prefix_sum,
                                         weighted_X_squared_prefix_sum)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroid = centroid_candidates[i]
        centroids[c_id] = best_centroid

    return centroids
