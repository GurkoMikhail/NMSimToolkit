import numba
import numpy as np

@numba.njit(parallel=True)
def numba_ray_casting(position, direction, half_size, distance_epsilon):
    """
    Numba-accelerated implementation of ray casting.
    
    Parameters:
    -----------
    position : numpy.ndarray
        Array of shape (n, 3) containing the positions
    direction : numpy.ndarray
        Array of shape (n, 3) containing the directions
    half_size : numpy.ndarray
        Array of shape (3,) containing the half-sizes of the box
    distance_epsilon : float
        Small epsilon value to avoid numerical issues
        
    Returns:
    --------
    distance : numpy.ndarray
        Array of shape (n,) containing the distances
    inside : numpy.ndarray
        Array of shape (n,) containing boolean values indicating if the position is inside
    """
    n = position.shape[0]
    distance = np.full(n, np.inf)
    inside = np.zeros(n, dtype=numba.boolean)

    for i in numba.prange(n):
        pos = position[i]
        dir = direction[i]

        is_inside = True
        for j in range(3):
            if abs(pos[j]) > half_size[j]:
                is_inside = False
                break

        inside[i] = is_inside

        t_min = -np.inf
        t_max = np.inf
        
        for j in range(3):
            if abs(dir[j]) < 1e-10:
                if pos[j] > half_size[j] or pos[j] < -half_size[j]:
                    t_max = -np.inf
                    break
            else:
                inv_dir = 1.0 / dir[j]
                t1 = (-half_size[j] - pos[j]) * inv_dir
                t2 = (half_size[j] - pos[j]) * inv_dir

                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    t_max = -np.inf
                    break

        if is_inside:
            distance[i] = t_max + distance_epsilon
        else:
            if t_max > 0 and t_min < t_max:
                distance[i] = max(0.0, t_min) + distance_epsilon

    return distance, inside