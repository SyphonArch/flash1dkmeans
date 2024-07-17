def calculate_inertia(X, centroids, labels, weights):
    """
    Calculate the inertia, i.e. the sum of the squared distances of samples to their closest cluster center.

    Args:
        X: array of floats
            The data to calculate the inertia for.
        centroids: array of floats
            The cluster centers.
        labels: array of ints
            The cluster labels for each sample.
        weights: array of floats
            The sample weights.

    Returns:
        float: The inertia of the data.
    """
    wcss = (X - centroids[labels]) ** 2  # within-cluster sum of squares
    weighted_wcss = wcss * weights
    return weighted_wcss.sum()
