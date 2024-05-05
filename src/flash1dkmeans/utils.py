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
