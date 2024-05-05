import numba


@numba.njit(cache=True)
def query_prefix_sum(arr_prefix_sum, start, stop):
    """Returns the sum of elements in the range [start, stop) of arr.

    Args:
        arr_prefix_sum: The prefix sum of the array arr.
        start: The start index of the range.
        stop: The stop index of the range.

    Returns:
        The sum of elements in the range [start, stop) of arr.
    """
    return arr_prefix_sum[stop - 1] - arr_prefix_sum[start - 1] if start > 0 else arr_prefix_sum[stop - 1]


@numba.njit(cache=True)
def query_weights_prefix_sum(is_weighted, sample_weights, start, stop):
    """Returns the sum of sample_weights in the range [start, stop).
    If is_weighted is False, simply returns stop - start.

    Args:
        is_weighted: bool
            Whether the data is weighted. Unweighted data is equivalent to weights of all 1.
        sample_weights: np.ndarray
            The sample weights.
        start: int
            The start index of the range.
        stop: int
            The stop index of the range.
    """
    if is_weighted:
        return query_prefix_sum(sample_weights, start, stop)
    else:
        return stop - start
