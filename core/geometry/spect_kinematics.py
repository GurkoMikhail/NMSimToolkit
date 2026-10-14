"""
Модуль кинематики ракурсов гантри и детекторов ОФЭКТ (SPECT).
"""
from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


@runtime_checkable
class ISpectProtocolConfig(Protocol):
    """Протокол параметров кинематики сканирования гантри ОФЭКТ."""

    views: int
    gamma_cameras: int
    start_angle: float
    end_angle: float
    head_angles: Optional[List[float]]
    endpoint: bool


@runtime_checkable
class IStepKinematicsConfig(Protocol):
    """Протокол параметров пошаговой кинематики сканирования."""

    steps: int
    start_angle: float
    end_angle: float
    endpoint: bool


def compute_spect_poses(
    views_or_protocol: Any,
    gamma_cameras: int = 1,
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    head_angle_offsets: Optional[List[float]] = None,
    endpoint: bool = False,
    *,
    start_angle_deg: Optional[float] = None,
    end_angle_deg: Optional[float] = None,
) -> List[List[float]]:
    """
    Вычисляет список углов для всех детекторных головок на каждом шаге гантри ОФЭКТ.
    Возвращает список позиций гантри, где каждая позиция — список углов [head_0, head_1, ..., head_{N-1}].
    Поддерживает передачу как объекта протокола IStepKinematicsConfig / ISpectProtocolConfig, так и скалярных параметров.
    """
    if start_angle_deg is not None:
        start_angle = start_angle_deg
    if end_angle_deg is not None:
        end_angle = end_angle_deg

    if isinstance(views_or_protocol, IStepKinematicsConfig):
        positions_count = max(1, int(views_or_protocol.steps))
        cameras_count = max(1, int(gamma_cameras))
        start_angle = float(np.degrees(views_or_protocol.start_angle))
        end_angle = float(np.degrees(views_or_protocol.end_angle))
        endpoint = bool(views_or_protocol.endpoint)
        total_views = positions_count * cameras_count
    elif isinstance(views_or_protocol, ISpectProtocolConfig):
        total_views = int(views_or_protocol.views)
        cameras_count = max(1, int(views_or_protocol.gamma_cameras))
        start_angle = float(np.degrees(views_or_protocol.start_angle))
        end_angle = float(np.degrees(views_or_protocol.end_angle))
        head_angle_offsets = (
            [float(np.degrees(angle_val)) for angle_val in views_or_protocol.head_angles]
            if views_or_protocol.head_angles
            else None
        )
        endpoint = bool(views_or_protocol.endpoint)
        positions_count = max(1, total_views // cameras_count)
    else:
        total_views = int(views_or_protocol)
        cameras_count = max(1, int(gamma_cameras))
        positions_count = max(1, total_views // cameras_count)

    base_angles = np.linspace(start_angle, end_angle, positions_count, endpoint=endpoint)
    if head_angle_offsets is not None and len(head_angle_offsets) == cameras_count:
        offsets = head_angle_offsets
    else:
        step_offset = 360.0 / cameras_count if cameras_count > 0 else 0.0
        offsets = [step_offset * camera_index for camera_index in range(cameras_count)]

    calculated_poses: List[List[float]] = []
    for base_angle in base_angles:
        single_pose = [(float(base_angle) + float(offset_val)) % 360.0 for offset_val in offsets]
        calculated_poses.append(single_pose)
    return calculated_poses


def compute_orbit_matrix(
    radius: float,
    angle: Optional[float] = None,
    z: float = 0.0,
    half_thickness: float = 0.0,
    roll: float = 0.0,
    axial_z: Optional[float] = None,
    *,
    angle_deg: Optional[float] = None,
    roll_deg: Optional[float] = None,
) -> NDArray[Float]:
    """
    Вычисляет кинематическую матрицу трансформации 4x4 для круговой орбиты ОФЭКТ,
    ориентирующую гамма-камеру к центру орбиты (0, 0, z).
    Лицевая поверхность гамма-камеры находится на заданном расстоянии radius от центра орбиты.
    При half_thickness > 0 геометрический центр камеры смещается на radius + half_thickness,
    что обеспечивает точный отсчет радиуса орбиты по лицевой поверхности гамма-камеры.
    Лицевая нормаль детектора (+Z) направлена к центру орбиты.
    Ось стола (+Y детектора) направлена вдоль глобальной оси Z.
    Поперечная ось (+X детектора) направлена тангенциально к орбите.
    Параметр roll задает вращение детектора в его собственной плоскости вокруг нормали Z (в градусах).
    """
    if angle is None:
        angle = 0.0 if angle_deg is None else float(angle_deg)
    if roll == 0.0 and roll_deg is not None:
        roll = float(roll_deg)
    if axial_z is not None:
        z = float(axial_z)

    orbit_angle = float(np.radians(angle))
    center_radius = float(radius) + float(half_thickness)
    pos_x = center_radius * float(np.cos(orbit_angle))
    pos_y = center_radius * float(np.sin(orbit_angle))
    orbit_matrix: NDArray[Float] = np.eye(4, dtype=Float)
    orbit_matrix[0, 3] = pos_x
    orbit_matrix[1, 3] = pos_y
    orbit_matrix[2, 3] = z

    # Лицевая нормаль детектора (локальная ось +Z, столбец 2) смотрит в центр орбиты (0, 0, z)
    orbit_matrix[0, 2] = -np.cos(orbit_angle)
    orbit_matrix[1, 2] = -np.sin(orbit_angle)
    orbit_matrix[2, 2] = 0.0

    # Осевая ось детектора (локальная ось +Y, столбец 1) направлена вдоль глобальной оси +Z (ось стола)
    axis_y = np.array([0.0, 0.0, 1.0], dtype=Float)

    # Трансверсионная ось детектора (локальная ось +X, столбец 0) образует правую тройку: X = Y x Z
    axis_x = np.array([np.sin(orbit_angle), -np.cos(orbit_angle), 0.0], dtype=Float)

    if roll != 0.0:
        roll_angle = float(np.radians(roll))
        cos_roll = float(np.cos(roll_angle))
        sin_roll = float(np.sin(roll_angle))
        rotated_x = cos_roll * axis_x + sin_roll * axis_y
        rotated_y = -sin_roll * axis_x + cos_roll * axis_y
        orbit_matrix[0:3, 0] = rotated_x
        orbit_matrix[0:3, 1] = rotated_y
    else:
        orbit_matrix[0:3, 0] = axis_x
        orbit_matrix[0:3, 1] = axis_y

    return orbit_matrix


__all__ = [
    "ISpectProtocolConfig",
    "IStepKinematicsConfig",
    "compute_spect_poses",
    "compute_orbit_matrix",
]

