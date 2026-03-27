import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.other.typing_definitions import Index, Float
from core.particles.kinematic_state import KinematicState
from core.geometry.navigation_state import NavigationState

@njit(cache=True, inline='always')
def _move_particle(state: KinematicState, p_idx: Index, distance: Float) -> None:
    """
    Moves a single particle by a given distance IN-PLACE.
    Updates position and distance_traveled.
    """
    state.distance_traveled[p_idx] += distance

    state.position.x[p_idx] += state.direction.x[p_idx] * distance
    state.position.y[p_idx] += state.direction.y[p_idx] * distance
    state.position.z[p_idx] += state.direction.z[p_idx] * distance

@njit(cache=True, inline='always')
def _rotate_particle(state: KinematicState, p_idx: Index, theta: Float, phi: Float) -> None:
    """
    Rotates a single particle's direction vector by theta and phi IN-PLACE.
    """
    dir_x = state.direction.x[p_idx]
    dir_y = state.direction.y[p_idx]
    dir_z = state.direction.z[p_idx]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    delta1 = sin_theta * np.cos(phi)
    delta2 = sin_theta * np.sin(phi)

    delta = 1.0 - 2.0 * (dir_z < 0.0)

    b = dir_x * delta1 + dir_y * delta2
    tmp = cos_theta - b / (1.0 + np.abs(dir_z))

    state.direction.x[p_idx] = dir_x * tmp + delta1
    state.direction.y[p_idx] = dir_y * tmp + delta2
    state.direction.z[p_idx] = dir_z * cos_theta - delta * b


@njit(cache=True)
def move_kernel(
    state: KinematicState,
    target_indices: NDArray[Index],
    distances: NDArray[Float]
) -> None:
    """
    Applies _move_particle sequentially across the given indices.
    """
    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        _move_particle(state, p_idx, distances[j])


@njit(cache=True)
def rotate_kernel(
    state: KinematicState,
    target_indices: NDArray[Index],
    thetas: NDArray[Float],
    phis: NDArray[Float]
) -> None:
    """
    Applies _rotate_particle sequentially across the given indices.
    """
    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        _rotate_particle(state, p_idx, thetas[j], phis[j])


@njit(cache=True)
def update_navigation_state_move_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index],
    distances: NDArray[Float]
) -> None:
    """
    Decrements boundary distance after a particle has moved.
    """
    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        nav_state.boundary_distance[p_idx] -= distances[j]

@njit(cache=True)
def update_navigation_state_rotate_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index]
) -> None:
    """
    Invalidates boundary distance after a particle has changed direction.
    """
    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        nav_state.boundary_distance[p_idx] = 0.0

@njit(cache=True)
def update_navigation_state_inject_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index]
) -> None:
    """
    Resets navigation state for newly injected particles.
    """
    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        nav_state.current_volume[p_idx] = -1
        nav_state.next_volume[p_idx] = -1
        nav_state.boundary_distance[p_idx] = 0.0
