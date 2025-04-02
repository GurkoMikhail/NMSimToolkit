import numba
import numpy as np

@numba.njit(parallel=True)
def numba_check_inside(position, half_size):
    """
    Numba-accelerated implementation of checking which positions are inside.
    
    Parameters:
    -----------
    position : numpy.ndarray
        Array of shape (n, 3) containing the positions
    half_size : numpy.ndarray
        Array of shape (3,) containing the half-sizes of the box
        
    Returns:
    --------
    inside : numpy.ndarray
        Boolean array indicating which positions are inside
    """
    n = position.shape[0]
    inside = np.zeros(n, dtype=numba.boolean)

    for i in numba.prange(n):
        pos = position[i]

        is_inside = True
        for j in range(3):
            if abs(pos[j]) > half_size[j]:
                is_inside = False
                break

        inside[i] = is_inside

    return inside
