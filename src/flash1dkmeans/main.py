from .k_cluster import numba_kmeans_1d_k_cluster, numba_kmeans_1d_k_cluster_unweighted
from .two_cluster import numba_kmeans_1d_two_cluster, numba_kmeans_1d_two_cluster_unweighted
from .config import LABEL_DTYPE, PREFIX_SUM_DTYPE
import numpy as np
import logging


def kmeans_1d(
        X,
        n_clusters,
        max_iter=None,
        is_sorted=False,
        sample_weights=None,
        n_local_trials=None,
        return_cluster_borders=False,
):
    """An optimized kmeans for 1D data.
    Utilizes a binary search to find the optimal division points for 2 clusters,
    and an optimized kmeans++ initialization for more than 2 clusters.

    Exploits the fact that 1D data can be sorted.

    Time complexity:
        2 clusters:
            O(log(n)) (+ (O(n) for prefix sum calculation if not provided))
        n clusters:
            O(n_clusters * 2 + log(n_clusters) * log(n) + max_iter * log(n) * n_clusters)
            (+ (O(n) for prefix sum calculation if not provided))

    Args:
        X: np.ndarray or list
            The input data. Should be sorted in ascending order if is_sorted is False.
        n_clusters: int
            The number of clusters.
        max_iter: int
            The maximum number of iterations. Only relevant for n_clusters > 2.
            Default is None, which becomes 300 for n_clusters > 2.
        is_sorted: bool
            Whether the data is already sorted.
        sample_weights: np.ndarray or list or None
            The sample weights. If None, all samples are weighted equally.
        n_local_trials: int
            The number of local trials for kmeans++ initialization. Only relevant for n_clusters > 2.
        return_cluster_borders: bool
            Whether to return the cluster border indices instead of the labels.

    Returns:
        centroids: np.ndarray
            The centroids of the clusters.
        labels: np.ndarray
            The labels of the samples. Only returned if return_cluster_borders is False.
        cluster_borders: np.ndarray
            The borders of the clusters. Only returned if return_cluster_borders is True.
    """
    # -------------- Input validation --------------
    # Check if data is weighted
    if sample_weights is not None:
        is_weighted = True
    else:
        is_weighted = False

    # Check if all lengths are the same
    if sample_weights is not None:
        assert len(X) == len(sample_weights), "X and sample_weights must have the same length"

    # Check that return_cluster_borders is called with is_sorted=True, warn the user otherwise
    if return_cluster_borders:  # User requested cluster borders
        if not is_sorted:  # However, the user did not provide sorted data
            logging.warning(  # Warn the user that the data will be sorted
                "The returned cluster borders will be indexed in the sorted order of X, not the original order.")

    # max_iter is redundant if n_clusters == 2
    if n_clusters == 2:
        assert max_iter is None, "max_iter should not be provided for 2 clusters"
    else:
        if max_iter is None:
            max_iter = 300

    # Check some values
    assert n_clusters >= 2, "n_clusters must be at least 2"
    assert max_iter is None or max_iter >= 0, "max_iter must be non-negative"
    assert n_local_trials is None or n_local_trials > 0, "n_local_trials must be positive"

    # -------------- Convert all arrays to numpy arrays --------------
    X = np.asarray(X)

    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    # -------------- Sorting --------------

    # Sort the data if it is not already sorted
    # Note that we checked that start_idx and stop_idx are not given if the data is not sorted
    if not is_sorted:
        sorted_indices = np.argsort(X)
        sorted_X = X[sorted_indices]
        sorted_sample_weights = None if sample_weights is None else sample_weights[sorted_indices]
    else:
        sorted_indices = None
        sorted_X = X
        sorted_sample_weights = sample_weights

    # -------------- Calculate prefix sums & Execute main algorithm --------------

    sorted_X_casted = sorted_X.astype(PREFIX_SUM_DTYPE)

    if is_weighted:
        sample_weights_casted = sorted_sample_weights.astype(PREFIX_SUM_DTYPE)
        sample_weight_prefix_sum = np.cumsum(sample_weights_casted)
        weighted_X_prefix_sum = np.cumsum(sorted_X_casted * sample_weights_casted)
        if n_clusters == 2:
            centroids, cluster_borders = numba_kmeans_1d_two_cluster(
                sorted_X=sorted_X,
                weights_prefix_sum=sample_weight_prefix_sum,
                weighted_X_prefix_sum=weighted_X_prefix_sum,
                start_idx=0,
                stop_idx=len(sorted_X),
            )
        else:
            weighted_X_squared_prefix_sum = np.cumsum(sorted_X_casted ** 2 * sample_weights_casted)
            centroids, cluster_borders = numba_kmeans_1d_k_cluster(
                sorted_X=sorted_X,
                n_clusters=n_clusters,
                max_iter=max_iter,
                weights_prefix_sum=sample_weight_prefix_sum,
                weighted_X_prefix_sum=weighted_X_prefix_sum,
                weighted_X_squared_prefix_sum=weighted_X_squared_prefix_sum,
                start_idx=0,
                stop_idx=len(sorted_X),
            )
    else:
        X_prefix_sum = np.cumsum(sorted_X_casted)
        if n_clusters == 2:
            centroids, cluster_borders = numba_kmeans_1d_two_cluster_unweighted(
                sorted_X=sorted_X,
                X_prefix_sum=X_prefix_sum,
                start_idx=0,
                stop_idx=len(sorted_X),
            )
        else:
            X_squared_prefix_sum = np.cumsum(sorted_X_casted ** 2)
            centroids, cluster_borders = numba_kmeans_1d_k_cluster_unweighted(
                sorted_X=sorted_X,
                n_clusters=n_clusters,
                max_iter=max_iter,
                X_prefix_sum=X_prefix_sum,
                X_squared_prefix_sum=X_squared_prefix_sum,
                start_idx=0,
                stop_idx=len(sorted_X),
            )

    # -------------- Post-processing --------------

    if return_cluster_borders:
        # We checked that the data is sorted if cluster borders are requested, so no post-processing is needed
        return centroids, cluster_borders
    else:
        # Convert cluster borders to labels
        labels = np.zeros(len(X), dtype=LABEL_DTYPE)
        for i in range(n_clusters):
            labels[cluster_borders[i]:cluster_borders[i + 1]] = i

        # Unsort the labels if the data was not sorted
        if not is_sorted:
            assert sorted_indices is not None, "sorted_indices should be available here"
            labels = labels[np.argsort(sorted_indices)]

        return centroids, labels
