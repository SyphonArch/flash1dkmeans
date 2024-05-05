from .k_cluster import flash_1d_kmeans_k_cluster
from .two_cluster import flash_1d_kmeans_two_cluster
from .config import LABEL_DTYPE, PREFIX_SUM_DTYPE
import numpy as np
import logging


def kmeans_1d(
        X,
        n_clusters,
        max_iter=300,
        is_sorted=False,
        sample_weights=None,
        n_local_trials=None,
        return_cluster_borders=False,
        weights_prefix_sum=None,
        weighted_X_prefix_sum=None,
        weighted_X_squared_prefix_sum=None,
        start_idx=None,
        stop_idx=None,
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
        is_sorted: bool
            Whether the data is already sorted.
        sample_weights: np.ndarray or list or None
            The sample weights. If None, all samples are weighted equally.
        n_local_trials: int
            The number of local trials for kmeans++ initialization. Only relevant for n_clusters > 2.
        return_cluster_borders: bool
            Whether to return the cluster border indices instead of the labels.

        In the case of repeatedly calling kmeans on different ranges of the same data, the following arguments
        can be provided to avoid redundant calculations. Make sure that the data is sorted if these are provided.

        weights_prefix_sum: np.ndarray or list or None
            The prefix sum of the sample weights. Should be None if the data is unweighted.
        weighted_X_prefix_sum: np.ndarray or list or None
            The prefix sum of (the weighted) X.
        weighted_X_squared_prefix_sum: np.ndarray or list or None
            The prefix sum of (weighted) X squared.
        start_idx: int
            The start index of the range to consider.
        stop_idx: int
            The stop index of the range to consider.

    Returns:
        centroids: np.ndarray
            The centroids of the clusters.
        labels: np.ndarray
            The labels of the samples. Only returned if return_cluster_borders is False.
        cluster_borders: np.ndarray
            The borders of the clusters. Only returned if return_cluster_borders is True.
    """
    # -------------- Input validation --------------
    if weighted_X_prefix_sum is not None:
        assert weighted_X_squared_prefix_sum is not None, \
            "If weighted_X_prefix_sum is provided, weighted_X_squared_prefix_sum should also be provided."
        # Note that weights_prefix_sum may be None if the data is unweighted

    # Check if user provided both sample_weights and weights_prefix_sum
    if sample_weights is not None and weights_prefix_sum is not None:
        raise ValueError("Both sample_weights and weights_prefix_sum cannot be provided. Please provide only one.")

    # Check if data is weighted
    if sample_weights is not None or weights_prefix_sum is not None:
        is_weighted = True
    else:
        is_weighted = False

    # Check that data is sorted if weighted_X_prefix_sum is provided
    if weighted_X_prefix_sum is not None and not is_sorted:
        raise ValueError("The data must be sorted if weighted_X_prefix_sum is provided.")

    # Check if all lengths are the same
    if sample_weights is not None:
        assert len(X) == len(sample_weights), "X and sample_weights must have the same length"
    if weights_prefix_sum is not None:
        assert len(X) == len(weights_prefix_sum), "X and weights_prefix_sum must have the same length"
    if weighted_X_prefix_sum is not None:
        assert len(X) == len(weighted_X_prefix_sum), "X and weighted_X_prefix_sum must have the same length"
    if weighted_X_squared_prefix_sum is not None:
        assert len(X) == len(weighted_X_squared_prefix_sum), \
            "X and weighted_X_squared_prefix_sum must have the same length"

    # Check that return_cluster_borders is called with is_sorted=True, warn the user otherwise
    if return_cluster_borders:  # User requested cluster borders
        if not is_sorted:  # However, the user did not provide sorted data
            logging.warning(  # Warn the user that the data will be sorted
                "The returned cluster borders will be indexed in the sorted order of X, not the original order.")

    # Check if start_idx and stop_idx are given with sorted data
    if not is_sorted:
        assert start_idx is None and stop_idx is None, "start_idx and stop_idx are only valid if the data is sorted."

    # Check some values
    assert n_clusters >= 2, "n_clusters must be at least 2"
    assert max_iter >= 0, "max_iter must be non-negative"
    assert n_local_trials is None or n_local_trials > 0, "n_local_trials must be positive"

    # -------------- Convert all arrays to numpy arrays --------------
    X = np.asarray(X)

    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    if weights_prefix_sum is not None:
        weights_prefix_sum = np.asarray(weights_prefix_sum)

    if weighted_X_prefix_sum is not None:
        weighted_X_prefix_sum = np.asarray(weighted_X_prefix_sum)

    if weighted_X_squared_prefix_sum is not None:
        weighted_X_squared_prefix_sum = np.asarray(weighted_X_squared_prefix_sum)

    # -------------- Setting up defaults --------------

    if start_idx is None:
        start_idx = 0
    if stop_idx is None:
        stop_idx = len(X)

    assert 0 <= start_idx < stop_idx <= len(X), "Invalid start_idx and stop_idx"

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

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

    # -------------- Main algorithm --------------

    centroids, cluster_borders = _sorted_kmeans_1d(
        sorted_X,
        n_clusters,
        max_iter,
        is_weighted,
        sorted_sample_weights,
        n_local_trials,
        weights_prefix_sum,
        weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum,
        start_idx,
        stop_idx,
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


def _sorted_kmeans_1d(
        sorted_X,  # caller should ensure that X is sorted
        n_clusters,  # caller should ensure n_clusters >= 2
        max_iter,  # caller should ensure max_iter >= 0
        is_weighted,
        sample_weights,
        n_local_trials,
        weights_prefix_sum,
        weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum,
        start_idx,  # caller should ensure 0 <= start_idx < stop_idx <= len(X)
        stop_idx,  # caller should ensure 0 <= start_idx < stop_idx <= len(X)
):
    """The main algorithm for flash_1d_kmeans.
    This function assumes that the input data is sorted, and all input is assumed to be valid.

    Time complexity:
        2 clusters:
            O(log(n)) (+ (O(n) for prefix sum calculation if not provided))
        n clusters:
            O(n_clusters * 2 + log(n_clusters) * log(n) + max_iter * log(n) * n_clusters)
            (+ (O(n) for prefix sum calculation if not provided))

    Args:
        sorted_X: np.ndarray
            The input data. Should be sorted in ascending order.
        n_clusters: int
            The number of clusters.
        max_iter: int
            The maximum number of iterations.
        is_weighted: bool
            Whether the data is weighted.
        sample_weights: np.ndarray or list or None
            The sample weights.
        n_local_trials: int
            The number of local trials for kmeans++ initialization.
        weights_prefix_sum: np.ndarray or list or None
            The prefix sum of the sample weights.
        weighted_X_prefix_sum: np.ndarray or list or None
            The prefix sum of (the weighted) X.
        weighted_X_squared_prefix_sum: np.ndarray or list or None
            The prefix sum of (the squared weighted) X.
        start_idx: int
            The start index of the range to consider.
        stop_idx: int
            The stop index of the range to consider.

    Returns:
        centroids: np.ndarray
            The centroids of the clusters.
        cluster_borders: np.ndarray
            The borders of the clusters.
    """
    if n_clusters == 2:
        # If weighted_X_prefix_sum is not provided, neither is weights_prefix_sum (checked by the caller)
        # They will be generated from sample_weights if needed
        if weighted_X_prefix_sum is None:
            sorted_X_casted = sorted_X.astype(PREFIX_SUM_DTYPE)
            if is_weighted:
                weights_casted = sample_weights.astype(PREFIX_SUM_DTYPE)
                weights_prefix_sum = np.cumsum(weights_casted)

                weighted_X_prefix_sum = np.cumsum(sorted_X_casted * weights_casted)
            else:
                weights_prefix_sum = None  # weights_prefix_sum is not needed
                weighted_X_prefix_sum = np.cumsum(sorted_X_casted)  # no weighting needed
        else:
            pass  # weighted_X_prefix_sum and weights_prefix_sum are already provided
        centroids, cluster_borders = flash_1d_kmeans_two_cluster(
            sorted_X,
            is_weighted,
            weighted_X_prefix_sum,
            weights_prefix_sum,
            start_idx,
            stop_idx,
        )
    else:
        # If weighted_X_prefix_sum is not provided, neither is weights_prefix_sum and weighted_X_squared_prefix_sum
        # They will be generated from sample_weights if needed
        if weighted_X_prefix_sum is None:
            sorted_X_casted = sorted_X.astype(PREFIX_SUM_DTYPE)
            if is_weighted:
                weights_casted = sample_weights.astype(PREFIX_SUM_DTYPE)
                weights_prefix_sum = np.cumsum(weights_casted)

                weighted_X_casted = sorted_X_casted * weights_casted
                weighted_X_prefix_sum = np.cumsum(weighted_X_casted)
                weighted_X_squared_prefix_sum = np.cumsum(weighted_X_casted * sorted_X_casted)
            else:
                weights_prefix_sum = None  # weights_prefix_sum is not needed
                weighted_X_prefix_sum = np.cumsum(sorted_X_casted)
                weighted_X_squared_prefix_sum = np.cumsum(sorted_X_casted ** 2)

        centroids, cluster_borders = flash_1d_kmeans_k_cluster(
            sorted_X,
            n_clusters,
            max_iter,
            is_weighted,
            weights_prefix_sum,
            weighted_X_prefix_sum,
            weighted_X_squared_prefix_sum,
            n_local_trials,
            start_idx,
            stop_idx,
        )

    return centroids, cluster_borders
