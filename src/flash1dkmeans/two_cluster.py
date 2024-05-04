import numpy as np
import numba
from .utils import query_prefix_sum


@numba.njit(cache=True)
def flash_1d_kmeans_two_cluster(
        X,
        weighted_X_prefix_sum,
        sample_weight_prefix_sum,
        start_idx,
        stop_idx
):
    """An optimized kmeans for 1D data with 2 clusters.
    Only operates on the range [start_idx, end_idx) of the input arrays.

    This function uses np.float32 instead of np.float16 for the centroids so that numba can compile it.
    Please cast the result back to np.float16 before saving it.

    WARNING: X should be sorted in ascending order.
    """
    size = stop_idx - start_idx
    if size < 0:
        raise ValueError("The end index should be greater than or equal to the start index")

    if size == 0:
        raise ValueError("This function should not be called with an empty range")

    centroids = np.empty(2, dtype=np.float32)

    if size == 1:
        centroids[0], centroids[1] = X[start_idx], X[start_idx]
        return centroids, start_idx + 1

    if size == 2:
        centroids[0], centroids[1] = X[start_idx], X[start_idx + 1]
        return centroids, start_idx + 1

    # Now we know that there are at least 3 elements

    # If the sum of the sample weight in the range is 0, we call an unweighted version of the function
    if query_prefix_sum(sample_weight_prefix_sum, start_idx, stop_idx) == 0:
        return flash_1d_kmeans_two_cluster_unweighted(X, start_idx, stop_idx)

    # Check if there is only one nonzero sample weight
    total_weight = query_prefix_sum(sample_weight_prefix_sum, start_idx, stop_idx)
    sample_weight_prefix_sum_within_range = sample_weight_prefix_sum[start_idx:stop_idx]
    final_increase_idx = np.searchsorted(sample_weight_prefix_sum_within_range,
                                         sample_weight_prefix_sum_within_range[-1])
    final_increase_amount = query_prefix_sum(sample_weight_prefix_sum,
                                             start_idx + final_increase_idx,
                                             start_idx + final_increase_idx + 1)
    if total_weight == final_increase_amount:
        # If there is only one nonzero sample weight, we need to return the corresponding weight as the centroid
        # and set all elements to the left cluster
        nonzero_weight_index = start_idx + final_increase_idx
        centroids[0], centroids[1] = X[nonzero_weight_index], X[nonzero_weight_index]
        return centroids, stop_idx

    # Now we know that there are at least 3 elements and at least 2 nonzero weights

    # KMeans with 2 clusters on 1D data is equivalent to finding a division point.
    # The division point can be found by doing a binary search on the prefix sum.

    # We will do a search for the division point,
    # where we search for the optimum number of elements in the first cluster
    # We don't want empty clusters, so we set the floor and ceiling to 1 and len(X) - 1
    floor = start_idx
    ceiling = stop_idx
    left_centroid = None
    right_centroid = None
    division_point = None

    while floor + 1 < ceiling:
        division_point = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_weight_sum = query_prefix_sum(sample_weight_prefix_sum, start_idx, division_point)
        if left_weight_sum == 0:
            floor = division_point
            continue
        right_weight_sum = query_prefix_sum(sample_weight_prefix_sum, division_point, stop_idx)
        # If the right cluster has no weight, we need to move the ceiling down
        if right_weight_sum == 0:
            ceiling = division_point
            continue

        left_centroid = query_prefix_sum(weighted_X_prefix_sum, start_idx, division_point) / left_weight_sum
        right_centroid = query_prefix_sum(weighted_X_prefix_sum, division_point, stop_idx) / right_weight_sum

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
        left_centroid = query_prefix_sum(weighted_X_prefix_sum, start_idx, division_point) / \
                        query_prefix_sum(sample_weight_prefix_sum, start_idx, division_point)
    if right_centroid is None:
        division_point = (floor + ceiling) // 2
        right_centroid = query_prefix_sum(weighted_X_prefix_sum, division_point, stop_idx) / \
                         query_prefix_sum(sample_weight_prefix_sum, division_point, stop_idx)

    # avoid using lists to allow numba.njit
    centroids[0] = left_centroid
    centroids[1] = right_centroid

    return centroids, division_point


@numba.njit(cache=True)
def flash_1d_kmeans_two_cluster_unweighted(X, start_idx, stop_idx):
    """Unweighted version of _faster_1d_two_cluster_kmeans.

    WARNING: X should have more than 3 elements and should be sorted in ascending order.
    """
    centroids = np.empty(2, dtype=np.float32)

    floor = start_idx
    ceiling = stop_idx
    left_centroid = None
    right_centroid = None
    division_point = None

    X_prefix_sum = np.cumsum(X.astype(np.float64))

    while floor + 1 < ceiling:
        division_point = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_cluster_size = division_point - start_idx
        if left_cluster_size == 0:
            floor = division_point
            continue
        right_cluster_size = stop_idx - division_point
        # If the right cluster has no weight, we need to move the ceiling down
        if right_cluster_size == 0:
            ceiling = division_point
            continue

        left_centroid = query_prefix_sum(X_prefix_sum, start_idx, division_point) / left_cluster_size
        right_centroid = query_prefix_sum(X_prefix_sum, division_point, stop_idx) / right_cluster_size

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
        left_centroid = query_prefix_sum(X_prefix_sum, start_idx, division_point) / (division_point - start_idx)
    if right_centroid is None:
        division_point = (floor + ceiling) // 2
        right_centroid = query_prefix_sum(X_prefix_sum, division_point, stop_idx) / (stop_idx - division_point)

    # avoid using lists to allow numba.njit
    centroids[0] = left_centroid
    centroids[1] = right_centroid

    return centroids, division_point
