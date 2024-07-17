import numba
import numpy as np


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
def set_np_seed_njit(random_state: int = None):
    """Set the seed for numpy random number generator. Must be used in a numba.jit function.

    Args:
        random_state: The seed to set. If None, no seed is set.

    Returns:
        None
    """
    if random_state is not None:
        # Only integer arguments allowed for np.random.seed in Numba, unlike in NumPy.
        np.random.seed(random_state)