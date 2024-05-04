from .n_cluster import flash_1d_kmeans_n_cluster
from .two_cluster import flash_1d_kmeans_two_cluster, flash_1d_kmeans_two_cluster_unweighted
import numpy as np
import logging


def flash_1d_kmeans(
        X,
        n_clusters,
        start_idx=0,
        stop_idx=None,
        max_iter=300,
        is_sorted=False,
        return_cluster_borders=False,
        sample_weights=None,
        weights_prefix_sum=None,
        weighted_X_prefix_sum=None,
        weighted_X_squared_prefix_sum=None,
):
    if return_cluster_borders and not is_sorted:
        logging.warning(
            "The returned cluster borders will be indexed in the sorted order of X, not the original order.")

    if sample_weights is not None and weights_prefix_sum is not None:
        raise ValueError("Both sample_weights and weights_prefix_sum cannot be provided. Please provide only one.")

    if weights_prefix_sum is not None:
        assert weighted_X_prefix_sum is not None and weighted_X_squared_prefix_sum is not None, \
            ("If weights_prefix_sum is provided, "
             "weighted_X_prefix_sum and weighted_X_squared_prefix_sum must also be provided")
        assert len(weights_prefix_sum) == len(X), "weights_prefix_sum must have the same length as X"
        assert len(weighted_X_prefix_sum) == len(X), "weighted_X_prefix_sum must have the same length as X"
        assert len(weighted_X_squared_prefix_sum) == len(X), \
            "weighted_X_squared_prefix_sum must have the same length as X"

    if not is_sorted:
        sorted_indices = np.argsort(X)
        sorted_X = X[sorted_indices]
        if weights_prefix_sum is not None:
            weights_prefix_sum = weights_prefix_sum[sorted_indices]
        if weighted_X_prefix_sum is not None:
            weighted_X_prefix_sum = weighted_X_prefix_sum[sorted_indices]
        if weighted_X_squared_prefix_sum is not None:
            weighted_X_squared_prefix_sum = weighted_X_squared_prefix_sum[sorted_indices]
    else:
        sorted_X = X

    if sample_weights is None and weights_prefix_sum is None:
        # Unweighted case
        assert weighted_X_prefix_sum is None and weighted_X_squared_prefix_sum is None, \
            ("If weights_prefix_sum is not provided, weighted_X_prefix_sum and "
             "weighted_X_squared_prefix_sum must not be provided")
        if n_clusters == 2:
            centroids, division_point = flash_1d_kmeans_two_cluster_unweighted(sorted_X, start_idx, stop_idx)
            if return_cluster_borders:
                cluster_borders = np.zeros(3, dtype=np.int32)
                cluster_borders[0] = start_idx
                cluster_borders[1] = division_point
                cluster_borders[2] = stop_idx
                return centroids, cluster_borders

    else:
        # Weighted case
        if sample_weights is not None:
            assert len(sample_weights) == len(X), "sample_weights must have the same length as X"
            weights_prefix_sum = np.cumsum(sample_weights, dtype=np.float64)
            weighted_X_prefix_sum = np.cumsum(sample_weights * X, dtype=np.float64)
            weighted_X_squared_prefix_sum = np.cumsum(sample_weights * X ** 2, dtype=np.float64)



def _sorted_flash_1d_kmeans(
        X,
        n_clusters,
        start_idx,
        stop_idx,
        max_iter,
        sample_weights=None,
        weights_prefix_sum=None,
        weighted_X_prefix_sum=None,
        weighted_X_squared_prefix_sum=None,
):
    pass