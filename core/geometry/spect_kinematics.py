"""
Модуль кинематики ракурсов гантри и детекторов ОФЭКТ (SPECT).
"""

from typing import Any, List, Optional
import numpy as np

from core.config.models import SpectProtocolConfig


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
    Поддерживает передачу как объекта SpectProtocolConfig, так и скалярных параметров.
    """
    if isinstance(views_or_protocol, SpectProtocolConfig):
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


__all__ = ["compute_spect_poses"]
