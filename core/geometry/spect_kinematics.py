"""
Модуль кинематики ракурсов гантри и детекторов ОФЭКТ (SPECT).
"""

from typing import Any, List, Optional, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ISpectProtocolConfig(Protocol):
    """Протокол параметров кинематики сканирования гантри ОФЭКТ."""
    views: int
    gamma_cameras: int
    start_angle: float
    end_angle: float
    head_angles: Optional[List[float]]
    endpoint: bool


def compute_spect_poses(
    views_or_protocol: Any,
    gamma_cameras: int = 1,
    start_angle_deg: float = 0.0,
    end_angle_deg: float = 360.0,
    head_angle_offsets: Optional[List[float]] = None,
    endpoint: bool = False,
) -> List[List[float]]:
    """
    Вычисляет список углов для всех детекторных головок на каждом шаге гантри ОФЭКТ.
    Возвращает список позиций гантри, где каждая позиция — список углов [head_0, head_1, ..., head_{N-1}].
    Поддерживает передачу как объекта протокола ISpectProtocolConfig, так и скалярных параметров.
    """
    if isinstance(views_or_protocol, ISpectProtocolConfig):
        total_views = int(views_or_protocol.views)
        cameras_count = max(1, int(views_or_protocol.gamma_cameras))
        start_angle_deg = float(np.degrees(views_or_protocol.start_angle))
        end_angle_deg = float(np.degrees(views_or_protocol.end_angle))
        head_angle_offsets = (
            [float(np.degrees(angle_val)) for angle_val in views_or_protocol.head_angles]
            if views_or_protocol.head_angles
            else None
        )
        endpoint = bool(views_or_protocol.endpoint)
    else:
        total_views = int(views_or_protocol)
        cameras_count = max(1, int(gamma_cameras))

    positions_count = max(1, total_views // cameras_count)
    base_angles = np.linspace(start_angle_deg, end_angle_deg, positions_count, endpoint=endpoint)
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
    angle_deg: float,
    z: float = 0.0,
    half_thickness: float = 0.0,
    roll_deg: float = 0.0,
    axial_z: Optional[float] = None,
) -> np.ndarray:
    """
    Вычисляет кинематическую матрицу трансформации 4x4 для круговой орбиты ОФЭКТ,
    ориентирующую гамма-камеру к центру орбиты (0, 0, z).
    Лицевая поверхность гамма-камеры находится на заданном расстоянии radius от центра орбиты.
    При half_thickness > 0 геометрический центр камеры смещается на radius + half_thickness,
    что обеспечивает точный отсчет радиуса орбиты по лицевой поверхности гамма-камеры.
    Лицевая нормаль детектора (+Z) направлена к центру орбиты.
    Ось стола (+Y детектора) направлена вдоль глобальной оси Z.
    Поперечная ось (+X детектора) направлена тангенциально к орбите.
    Параметр roll_deg задает вращение детектора в его собственной плоскости вокруг нормали Z.
    """
    if axial_z is not None:
        z = float(axial_z)

    rad = np.radians(angle_deg)
    center_radius = float(radius) + float(half_thickness)
    pos_x = center_radius * np.cos(rad)
    pos_y = center_radius * np.sin(rad)
    mat = np.eye(4, dtype=np.float64)
    mat[0, 3] = pos_x
    mat[1, 3] = pos_y
    mat[2, 3] = z

    # Лицевая нормаль детектора (локальная ось +Z, столбец 2) смотрит в центр орбиты (0, 0, z)
    mat[0, 2] = -np.cos(rad)
    mat[1, 2] = -np.sin(rad)
    mat[2, 2] = 0.0

    # Осевая ось детектора (локальная ось +Y, столбец 1) направлена вдоль глобальной оси +Z (ось стола)
    axis_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Трансверсионная ось детектора (локальная ось +X, столбец 0) образует правую тройку: X = Y x Z
    axis_x = np.array([np.sin(rad), -np.cos(rad), 0.0], dtype=np.float64)

    if roll_deg != 0.0:
        roll_rad = np.radians(roll_deg)
        cos_roll = np.cos(roll_rad)
        sin_roll = np.sin(roll_rad)
        # Вращение вокруг нормали Z
        rotated_x = cos_roll * axis_x + sin_roll * axis_y
        rotated_y = -sin_roll * axis_x + cos_roll * axis_y
        mat[0:3, 0] = rotated_x
        mat[0:3, 1] = rotated_y
    else:
        mat[0:3, 0] = axis_x
        mat[0:3, 1] = axis_y

    return mat


__all__ = ["compute_spect_poses", "compute_orbit_matrix"]
