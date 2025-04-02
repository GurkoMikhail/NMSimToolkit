import numba as nb
import numpy as np

@nb.njit(parallel=True)
def numba_fast_index(source, indices, target):
    """Copy elements from source[indices] to target"""
    for i in nb.prange(len(indices)):
        target[i] = source[indices[i]]
    return target


@nb.njit(parallel=True)
def numba_fast_insert(target, indices, values):
    for i in nb.prange(len(indices)):
        target[indices[i]] = values[i]
    return target


@nb.njit(parallel=True)
def numba_fast_assign(target, indices, source):
    """Fast assignment for arrays"""
    for i in nb.prange(len(indices)):
        idx = indices[i]
        target[idx] = source[idx]
    return target

@nb.njit(parallel=True)
def numba_fast_index_float(source, indices):
    """Fast indexing for float arrays"""
    result = np.empty(len(indices), dtype=np.float64)
    for i in nb.prange(len(indices)):
        result[i] = source[indices[i]]
    return result
