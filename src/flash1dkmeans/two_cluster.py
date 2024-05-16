"""two_cluster.py
Optimized kmeans for 1D data with 2 clusters.

This module contains an optimized kmeans for 1D data with 2 clusters.

The following is the main function:
- flash_1d_kmeans_two_cluster

Inputs must be sorted in ascending order, no default values are provided, and no error checking is done.
This is because this module is intended to be used internally.
"""

import numpy as np
import numba
from .utils import query_prefix_sum
from .config import PREFIX_SUM_DTYPE, ARRAY_INDEX_DTYPE


@numba.njit(cache=True)
def numba_kmeans_1d_two_cluster(
        sorted_X,
        weights_prefix_sum,
        weighted_X_prefix_sum,
        start_idx,
        stop_idx,
):
    """An optimized kmeans for 1D data with 2 clusters, weighted version.
    Utilizes a binary search to find the optimal division point.
    Time complexity: O(log(n))

    Args:
        sorted_X: np.ndarray
            The input data. Should be sorted in ascending order.
        weights_prefix_sum: np.ndarray
            The prefix sum of the sample weights. Should be None if the data is unweighted.
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of (the weighted) X.
        start_idx: int
            The start index of the range to consider.
        stop_idx: int
            The stop index of the range to consider.

    Returns:
        centroids: np.ndarray
            The centroids of the two clusters, shape (2,)
        cluster_borders: np.ndarray
            The borders of the two clusters, shape (3,)

    WARNING: X should be sorted in ascending order before calling this function.
    """
    size = stop_idx - start_idx
    centroids = np.empty(2, dtype=sorted_X.dtype)
    cluster_borders = np.empty(3, dtype=ARRAY_INDEX_DTYPE)
    cluster_borders[0] = start_idx
    cluster_borders[2] = stop_idx
    # Remember to set cluster_borders[1] as the division point

    if size == 1:
        centroids[0], centroids[1] = sorted_X[start_idx], sorted_X[start_idx]
        cluster_borders[1] = start_idx + 1
        return centroids, cluster_borders

    if size == 2:
        centroids[0], centroids[1] = sorted_X[start_idx], sorted_X[start_idx + 1]
        cluster_borders[1] = start_idx + 1
        return centroids, cluster_borders

    # Now we know that there are at least 3 elements

    # If the sum of the sample weight in the range is 0, we assume that the data is unweighted
    if query_prefix_sum(weights_prefix_sum, start_idx, stop_idx) == 0:
        # We need to recalculate the prefix sum, as previously it would have been all zeros
        X_casted = sorted_X.astype(PREFIX_SUM_DTYPE)
        X_prefix_sum = np.cumsum(X_casted)
        return numba_kmeans_1d_two_cluster_unweighted(sorted_X, X_prefix_sum, start_idx, stop_idx)
    else:
        # Check if there is only one nonzero sample weight
        total_weight = query_prefix_sum(weights_prefix_sum, start_idx, stop_idx)
        sample_weight_prefix_sum_within_range = weights_prefix_sum[start_idx:stop_idx]
        final_increase_idx = np.searchsorted(sample_weight_prefix_sum_within_range,
                                             sample_weight_prefix_sum_within_range[-1])
        final_increase_amount = query_prefix_sum(weights_prefix_sum,
                                                 start_idx + final_increase_idx,
                                                 start_idx + final_increase_idx + 1)
        if total_weight == final_increase_amount:
            # If there is only one nonzero sample weight, we need to return the corresponding weight as the centroid
            # and set all elements to the left cluster
            nonzero_weight_index = start_idx + final_increase_idx
            centroids[0], centroids[1] = sorted_X[nonzero_weight_index], sorted_X[nonzero_weight_index]
            cluster_borders[1] = stop_idx
            return centroids, cluster_borders

    # Now we know that there are at least 3 elements and at least 2 nonzero weights

    # KMeans with 2 clusters on 1D data is equivalent to finding a division point.
    # The division point can be found by doing a binary search on the prefix sum.

    # We will do a search for the division point,
    # where we search for the optimum number of elements in the first cluster
    # We don't want empty clusters, so we set the floor and ceiling to start_idx + 1 and stop_idx - 1
    floor = start_idx + 1
    ceiling = stop_idx - 1
    left_centroid = None
    right_centroid = None
    division_point = None

    while floor < ceiling:
        division_point = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_weight_sum = query_prefix_sum(weights_prefix_sum, start_idx, division_point)
        if left_weight_sum == 0:
            floor = division_point + 1
            continue
        right_weight_sum = query_prefix_sum(weights_prefix_sum, division_point, stop_idx)
        # If the right cluster has no weight, we need to move the ceiling down
        if right_weight_sum == 0:
            ceiling = division_point - 1
            continue

        left_centroid = query_prefix_sum(weighted_X_prefix_sum, start_idx, division_point) / left_weight_sum
        right_centroid = query_prefix_sum(weighted_X_prefix_sum, division_point, stop_idx) / right_weight_sum

        new_division_point_value = (left_centroid + right_centroid) / 2
        if sorted_X[division_point - 1] <= new_division_point_value:
            if new_division_point_value <= sorted_X[division_point]:
                # The new division point matches the previous one, so we can stop
                break
            else:
                floor = division_point + 1
        else:
            ceiling = division_point - 1

    # initialize variables in case the loop above does not run through
    if left_centroid is None:
        division_point = (floor + ceiling) // 2
        left_centroid = (query_prefix_sum(weighted_X_prefix_sum, start_idx, division_point) /
                         query_prefix_sum(weights_prefix_sum, start_idx, division_point))
    if right_centroid is None:
        division_point = (floor + ceiling) // 2
        right_centroid = (query_prefix_sum(weighted_X_prefix_sum, division_point, stop_idx) /
                          query_prefix_sum(weights_prefix_sum, division_point, stop_idx))

    # avoid using lists to allow numba.njit
    centroids[0] = left_centroid
    centroids[1] = right_centroid

    cluster_borders[1] = division_point
    return centroids, cluster_borders


@numba.njit(cache=True)
def numba_kmeans_1d_two_cluster_unweighted(
        sorted_X,
        X_prefix_sum,
        start_idx,
        stop_idx,
):
    """Unweighted version of flash_1d_kmeans_two_cluster."""
    size = stop_idx - start_idx
    centroids = np.empty(2, dtype=sorted_X.dtype)
    cluster_borders = np.empty(3, dtype=ARRAY_INDEX_DTYPE)
    cluster_borders[0] = start_idx
    cluster_borders[2] = stop_idx
    # Remember to set cluster_borders[1] as the division point

    if size == 1:
        centroids[0], centroids[1] = sorted_X[start_idx], sorted_X[start_idx]
        cluster_borders[1] = start_idx + 1
        return centroids, cluster_borders

    if size == 2:
        centroids[0], centroids[1] = sorted_X[start_idx], sorted_X[start_idx + 1]
        cluster_borders[1] = start_idx + 1
        return centroids, cluster_borders

    # Now we know that there are at least 3 elements and at least 2 nonzero weights

    # KMeans with 2 clusters on 1D data is equivalent to finding a division point.
    # The division point can be found by doing a binary search on the prefix sum.

    # We will do a search for the division point,
    # where we search for the optimum number of elements in the first cluster
    # We don't want empty clusters, so we set the floor and ceiling to start_idx + 1 and stop_idx - 1
    floor = start_idx + 1
    ceiling = stop_idx - 1
    left_centroid = None
    right_centroid = None
    division_point = None

    while floor < ceiling:
        division_point = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_weight_sum = division_point - start_idx
        if left_weight_sum == 0:
            floor = division_point + 1
            continue
        right_weight_sum = stop_idx - division_point
        # If the right cluster has no weight, we need to move the ceiling down
        if right_weight_sum == 0:
            ceiling = division_point - 1
            continue

        left_centroid = query_prefix_sum(X_prefix_sum, start_idx, division_point) / left_weight_sum
        right_centroid = query_prefix_sum(X_prefix_sum, division_point, stop_idx) / right_weight_sum

        new_division_point_value = (left_centroid + right_centroid) / 2
        if sorted_X[division_point - 1] <= new_division_point_value:
            if new_division_point_value <= sorted_X[division_point]:
                # The new division point matches the previous one, so we can stop
                break
            else:
                floor = division_point + 1
        else:
            ceiling = division_point - 1

    # initialize variables in case the loop above does not run through
    if left_centroid is None:
        division_point = (floor + ceiling) // 2
        left_centroid = query_prefix_sum(X_prefix_sum, start_idx, division_point) / (division_point - start_idx)
    if right_centroid is None:
        division_point = (floor + ceiling) // 2
        right_centroid = query_prefix_sum(X_prefix_sum, division_point, stop_idx) / (stop_idx - division_point)

    # avoid using lists to allow numba.njit
    centroids[0] = left_centroid
    centroids[1] = right_centroid

    cluster_borders[1] = division_point
    return centroids, cluster_borders
