import logging
from typing import Any, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal
import pyvista as pv

from gui.viewmodels.node_viewmodel import GammaCameraViewModel

_logger = logging.getLogger(__name__)


class SPECTManipulator(QObject):
    """
    Специализированный процедурный 3D-манипулятор для ОФЭКТ системы.
    Вместо произвольного 6-DOF Gizmo накладывает строгие кинематические ограничения:
    - Радиус орбиты R (расстояние от оси вращения до лицевой поверхности гамма-камеры)
    - Угол поворота гантри Theta (угол проекции)
    - Осевой сдвиг Z (высота вдоль продольной оси стола)
    """

    orbit_changed = Signal(float, float, float)  # radius, angle_deg, z_pos

    def __init__(
        self,
        viewport: Any,
        initial_radius: float = 250.0,
        initial_angle: float = 0.0,
        initial_z: float = 0.0,
        auto_render: bool = False
    ) -> None:
        super().__init__()
        self.viewport = viewport
        self.radius = float(initial_radius)
        self.angle_deg = float(initial_angle)
        self.z_pos = float(initial_z)
        self.half_thickness: float = 0.0

        self.orbit_actor_name = "spect_orbit_trajectory"
        self.detector_indicator_name = "spect_detector_indicator"

        self._min_radius = 50.0   # Минимальный безопасный радиус до фантома
        self._max_radius = 600.0  # Максимальный вылет штатива
        if auto_render:
            self.update_visuals(render=True)

    def set_orbit_parameters(
        self,
        radius: float,
        angle_deg: float,
        z: Optional[float] = None,
        render: bool = False,
        emit_signal: bool = True
    ) -> None:
        """
        Установка параметров орбиты с проверкой граничных ограничений.
        """
        self.radius = float(np.clip(radius, self._min_radius, self._max_radius))
        self.angle_deg = float(angle_deg % 360.0)
        if z is not None:
            self.z_pos = float(z)

        self.update_visuals(render=render)
        if emit_signal:
            self.orbit_changed.emit(self.radius, self.angle_deg, self.z_pos)

    def rotate_by(self, delta_angle_deg: float) -> None:
        """
        Инкрементальный поворот гантри ОФЭКТ на заданный угол.
        """
        self.set_orbit_parameters(self.radius, self.angle_deg + delta_angle_deg, self.z_pos)

    def set_radius(self, new_radius: float) -> None:
        """
        Изменение радиуса орбиты до лицевой поверхности гамма-камеры.
        """
        self.set_orbit_parameters(new_radius, self.angle_deg, self.z_pos)

    def get_cartesian_position(self, half_thickness: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Вычисляет декартовы координаты (X, Y, Z) центра гамма-камеры с учетом
        радиуса орбиты до лицевой поверхности.
        """
        rad = np.radians(self.angle_deg)
        h = self.half_thickness if half_thickness is None else float(half_thickness)
        center_r = self.radius + h
        x = center_r * np.cos(rad)
        y = center_r * np.sin(rad)
        return (float(x), float(y), float(self.z_pos))

    def get_orientation_matrix(self, half_thickness: Optional[float] = None) -> np.ndarray:
        """
        Возвращает матрицу трансформации 4x4, ориентирующую гамма-камеру к центру вращения
        с учетом радиуса орбиты до лицевой поверхности гамма-камеры.
        """
        h = self.half_thickness if half_thickness is None else float(half_thickness)
        return GammaCameraViewModel.compute_orbit_matrix(self.radius, self.angle_deg, self.z_pos, half_thickness=h)

    def update_visuals(self, render: bool = False) -> None:
        """
        Отрисовка круговой направляющей орбиты и маркера детектора в 3D.
        """
        if self.viewport is None or self.viewport.plotter is None:
            return

        try:
            # 1. Траектория орбиты (окружность)
            angles = np.linspace(0, 2 * np.pi, 120)
            xs = self.radius * np.cos(angles)
            ys = self.radius * np.sin(angles)
            zs = np.full_like(xs, self.z_pos)
            orbit_points = np.column_stack((xs, ys, zs))

            orbit_poly = pv.lines_from_points(orbit_points, close=True)
            self.viewport.add_mesh_actor(
                self.orbit_actor_name,
                orbit_poly,
                color='#3498db',
                opacity=0.6,
                wireframe=True
            )

            # 2. Индикатор положения головки детектора
            pos = self.get_cartesian_position()
            indicator = pv.Sphere(radius=12.0, center=pos)
            self.viewport.add_mesh_actor(
                self.detector_indicator_name,
                indicator,
                color='#e74c3c',
                opacity=0.9
            )
            if render:
                self.viewport.render()

        except Exception as e:
            _logger.debug(f"Ошибка визуализации ОФЭКТ-манипулятора: {e}")

    def remove_visuals(self) -> None:
        """
        Очистка визуальных направляющих манипулятора.
        """
        if self.viewport is not None:
            self.viewport.remove_actor(self.orbit_actor_name)
            self.viewport.remove_actor(self.detector_indicator_name)
            self.viewport.render()
