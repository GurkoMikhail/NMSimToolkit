"""
Модуль Numba-ядер кинематических операций над частицами (перемещение, поворот направления)
и обновления состояния геометрической навигации.
"""

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.geometry.navigation_state import NavigationState
from core.other.typing_definitions import Float, Index
from core.particles.kinematic_state import KinematicState


@njit(inline='always')
def _move_particle(state: KinematicState, p_idx: Index, distance: Float) -> None:
    """
    Встраиваемая функция перемещения одной частицы на расстояние distance вдоль ее вектора направления.
    """
    state.distance_traveled[p_idx] += distance
    state.position.x[p_idx] += state.direction.x[p_idx] * distance
    state.position.y[p_idx] += state.direction.y[p_idx] * distance
    state.position.z[p_idx] += state.direction.z[p_idx] * distance


@njit(inline='always')
def _rotate_particle(state: KinematicState, p_idx: Index, theta: Float, phi: Float) -> None:
    """
    Встраиваемая функция поворота вектора направления одной частицы на сферические углы (theta, phi).
    """
    dir_x = state.direction.x[p_idx]
    dir_y = state.direction.y[p_idx]
    dir_z = state.direction.z[p_idx]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    delta1 = sin_theta * np.cos(phi)
    delta2 = sin_theta * np.sin(phi)

    delta = 1.0
    if dir_z < 0.0:
        delta = -1.0

    proj_b = dir_x * delta1 + dir_y * delta2
    abs_z = np.abs(dir_z)
    tmp = cos_theta - proj_b / (1.0 + abs_z)

    state.direction.x[p_idx] = dir_x * tmp + delta1
    state.direction.y[p_idx] = dir_y * tmp + delta2
    state.direction.z[p_idx] = dir_z * cos_theta - delta * proj_b


@njit(cache=True)
def move_kernel(state: KinematicState, target_indices: NDArray[np.int64], distances: NDArray[Float]) -> None:
    """
    Пакетное ядро перемещения выбранных активных частиц на заданные расстояния на месте.
    """
    for j in range(len(target_indices)):
        i = target_indices[j]
        _move_particle(state, i, distances[j])


@njit(inline='always')
def _invalidate_navigation_state(nav_state: NavigationState, p_idx: Index) -> None:
    """
    Сбрасывает состояние геометрической навигации одной частицы для принудительного поиска объема.
    """
    nav_state.current_volume[p_idx] = -1
    nav_state.boundary_distance[p_idx] = 0.0


@njit(cache=True)
def update_navigation_state_inject_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index],
) -> None:
    """Инвалидирует состояние навигации для вновь инжектированных частиц."""
    for i in range(target_indices.shape[0]):
        p_idx = target_indices[i]
        _invalidate_navigation_state(nav_state, p_idx)


@njit(cache=True)
def update_navigation_state_rotate_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index],
) -> None:
    """Сбрасывает расстояние до границы для частиц, изменивших направление движения."""
    for i in range(target_indices.shape[0]):
        p_idx = target_indices[i]
        nav_state.boundary_distance[p_idx] = 0.0


@njit(cache=True)
def update_navigation_state_move_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index],
    distances: NDArray[Float],
) -> None:
    """Обновляет остаточное расстояние до границы после перемещения и инвалидирует при пересечении."""
    for i in range(target_indices.shape[0]):
        p_idx = target_indices[i]
        nav_state.boundary_distance[p_idx] -= distances[i]

        if nav_state.boundary_distance[p_idx] <= 0.0:
            _invalidate_navigation_state(nav_state, p_idx)


@njit(cache=True)
def rotate_kernel(
    state: KinematicState,
    target_indices: NDArray[np.int64],
    thetas: NDArray[Float],
    phis: NDArray[Float],
) -> None:
    """
    Пакетное ядро поворота векторов направления выбранных активных частиц на углы (thetas, phis).
    """
    for j in range(len(target_indices)):
        i = target_indices[j]
        _rotate_particle(state, i, thetas[j], phis[j])


__all__ = [
    "_move_particle",
    "_rotate_particle",
    "move_kernel",
    "_invalidate_navigation_state",
    "update_navigation_state_inject_kernel",
    "update_navigation_state_rotate_kernel",
    "update_navigation_state_move_kernel",
    "rotate_kernel",
]

