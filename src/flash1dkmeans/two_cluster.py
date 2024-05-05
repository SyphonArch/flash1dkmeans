import numpy as np
import numba
from .utils import query_prefix_sum


@numba.njit(cache=True)
def flash_1d_kmeans_two_cluster(
        X,
        weighted_X_prefix_sum,
        sample_weight_prefix_sum,
):
    """An optimized kmeans for 1D data with 2 clusters.
    Utilizes a binary search to find the optimal division point.
    Time complexity: O(log(n))

    Args:
        X: np.ndarray
            The input data. Should be sorted in ascending order.
        weighted_X_prefix_sum: np.ndarray
            The prefix sum of the weighted X.
        sample_weight_prefix_sum: np.ndarray
            The prefix sum of the sample weights.

    Returns:
        centroids: np.ndarray
            The centroids of the two clusters, shape (2,)
        cluster_borders: np.ndarray
            The borders of the two clusters, shape (3,)

    WARNING: X should be sorted in ascending order before calling this function.
    """
    if len(X) == 0:
        raise ValueError("X should not be empty")

    centroids = np.empty(2, dtype=X.dtype)
    cluster_borders = np.empty(3, dtype=np.int32)
    cluster_borders[0] = 0
    cluster_borders[2] = len(X)
    # Remember to set cluster_borders[1] as the division point

    if len(X) == 1:
        centroids[0], centroids[1] = X[0], X[0]
        cluster_borders[1] = 1
        return centroids, cluster_borders

    if len(X) == 2:
        centroids[0], centroids[1] = X[0], X[1]
        cluster_borders[1] = 1
        return centroids, cluster_borders

    # Now we know that there are at least 3 elements

    # If the sum of the sample weight in the range is 0, we call an unweighted version of the function
    if query_prefix_sum(sample_weight_prefix_sum, 0, len(X)) == 0:
        X_prefix_sum = np.cumsum(X, dtype=np.float64)
        return flash_1d_kmeans_two_cluster_unweighted(X, X_prefix_sum)

    # Check if there is only one nonzero sample weight
    total_weight = query_prefix_sum(sample_weight_prefix_sum, 0, len(X))
    sample_weight_prefix_sum_within_range = sample_weight_prefix_sum[0:len(X)]
    final_increase_idx = np.searchsorted(sample_weight_prefix_sum_within_range,
                                         sample_weight_prefix_sum_within_range[-1])
    final_increase_amount = query_prefix_sum(sample_weight_prefix_sum,
                                             final_increase_idx,
                                             final_increase_idx + 1)
    if total_weight == final_increase_amount:
        # If there is only one nonzero sample weight, we need to return the corresponding weight as the centroid
        # and set all elements to the left cluster
        nonzero_weight_index = final_increase_idx
        centroids[0], centroids[1] = X[nonzero_weight_index], X[nonzero_weight_index]
        cluster_borders[1] = len(X)
        return centroids, cluster_borders

    # Now we know that there are at least 3 elements and at least 2 nonzero weights

    # KMeans with 2 clusters on 1D data is equivalent to finding a division point.
    # The division point can be found by doing a binary search on the prefix sum.

    # We will do a search for the division point,
    # where we search for the optimum number of elements in the first cluster
    # We don't want empty clusters, so we set the floor and ceiling to 1 and len(X) - 1
    floor = 1
    ceiling = len(X) - 1
    left_centroid = None
    right_centroid = None
    division_point = None

    while floor + 1 < ceiling:
        division_point = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_weight_sum = query_prefix_sum(sample_weight_prefix_sum, 0, division_point)
        if left_weight_sum == 0:
            floor = division_point
            continue
        right_weight_sum = query_prefix_sum(sample_weight_prefix_sum, division_point, len(X))
        # If the right cluster has no weight, we need to move the ceiling down
        if right_weight_sum == 0:
            ceiling = division_point
            continue

        left_centroid = query_prefix_sum(weighted_X_prefix_sum, 0, division_point) / left_weight_sum
        right_centroid = query_prefix_sum(weighted_X_prefix_sum, division_point, len(X)) / right_weight_sum

        new_division_point_value = (left_centroid + right_centroid) / 2
        if X[division_point - 1] <= new_division_point_value:
            if new_division_point_value <= X[division_point]:
                # The new division point matches the previous one, so we can stop
                break
            else:
                floor = division_point
        else:
            ceiling = division_point

    # initialize variables in case the loop above does not run through
    if left_centroid is None:
        division_point = (floor + ceiling) // 2
        left_centroid = query_prefix_sum(weighted_X_prefix_sum, 0, division_point) / \
                        query_prefix_sum(sample_weight_prefix_sum, 0, division_point)
    if right_centroid is None:
        division_point = (floor + ceiling) // 2
        right_centroid = query_prefix_sum(weighted_X_prefix_sum, division_point, len(X)) / \
                         query_prefix_sum(sample_weight_prefix_sum, division_point, len(X))

    # avoid using lists to allow numba.njit
    centroids[0] = left_centroid
    centroids[1] = right_centroid

    cluster_borders[1] = division_point
    return centroids, cluster_borders


@numba.njit(cache=True)
def flash_1d_kmeans_two_cluster_unweighted(X, X_prefix_sum):
    """Unweighted version of flash_1d_kmeans_two_cluster.

    WARNING: X should be sorted in ascending order before calling this function.
    """
    if len(X) == 0:
        raise ValueError("X should not be empty")

    centroids = np.empty(2, dtype=X.dtype)
    cluster_borders = np.empty(3, dtype=np.int32)
    cluster_borders[0] = 0
    cluster_borders[2] = len(X)
    # Remember to set cluster_borders[1] as the division point

    if len(X) == 1:
        centroids[0], centroids[1] = X[0], X[0]
        cluster_borders[1] = 1
        return centroids, cluster_borders

    if len(X) == 2:
        centroids[0], centroids[1] = X[0], X[1]
        cluster_borders[1] = 1
        return centroids, cluster_borders

    # Now we know that there are at least 3 elements
    floor = 1
    ceiling = len(X) - 1
    left_centroid = None
    right_centroid = None

    while floor + 1 < ceiling:
        division_point = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_cluster_size = division_point
        if left_cluster_size == 0:
            floor = division_point
            continue
        right_cluster_size = len(X) - division_point
        # If the right cluster has no weight, we need to move the ceiling down
        if right_cluster_size == 0:
            ceiling = division_point
            continue

        left_centroid = query_prefix_sum(X_prefix_sum, 0, division_point) / left_cluster_size
        right_centroid = query_prefix_sum(X_prefix_sum, division_point, len(X)) / right_cluster_size

        new_division_point_value = (left_centroid + right_centroid) / 2
        if X[division_point - 1] <= new_division_point_value:
            if new_division_point_value <= X[division_point]:
                # The new division point matches the previous one, so we can stop
                break
            else:
                floor = division_point
        else:
            ceiling = division_point

    # initialize variables in case the loop above does not run through
    if left_centroid is None:
        division_point = (floor + ceiling) // 2
        left_centroid = query_prefix_sum(X_prefix_sum, 0, division_point) / division_point
    if right_centroid is None:
        division_point = (floor + ceiling) // 2
        right_centroid = query_prefix_sum(X_prefix_sum, division_point, len(X)) / (len(X) - division_point)

    # avoid using lists to allow numba.njit
    centroids[0] = left_centroid
    centroids[1] = right_centroid

    cluster_borders[1] = division_point
    return centroids, cluster_borders
