import logging
from typing import Any, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal
import pyvista as pv

_logger = logging.getLogger(__name__)


class PETManipulator(QObject):
    """
    Процедурный манипулятор геометрии кольцевого ПЭТ-сканера.
    Управляет параметрами цилиндрической детекторной системы:
    - Диаметр кольца сканера (D)
    - Аксиальная длина поля зрения (Axial Length, L)
    - Радиальная толщина детекторов
    - Количество детекторных секторов / блоков
    """

    geometry_changed = Signal(float, float, int)  # diameter, axial_length, num_sectors

    def __init__(
        self,
        viewport: Any,
        diameter: float = 600.0,
        axial_length: float = 200.0,
        num_sectors: int = 24,
        auto_render: bool = False
    ) -> None:
        super().__init__()
        self.viewport = viewport
        self.diameter = float(diameter)
        self.axial_length = float(axial_length)
        self.num_sectors = int(num_sectors)

        self.actor_name = "pet_ring_geometry"
        if auto_render:
            self.update_visuals()

    def set_parameters(
        self,
        diameter: Optional[float] = None,
        axial_length: Optional[float] = None,
        num_sectors: Optional[int] = None
    ) -> None:
        """
        Обновление параметров геометрии ПЭТ сканера.
        """
        if diameter is not None:
            self.diameter = max(100.0, float(diameter))
        if axial_length is not None:
            self.axial_length = max(20.0, float(axial_length))
        if num_sectors is not None:
            self.num_sectors = max(4, int(num_sectors))

        self.update_visuals()
        self.geometry_changed.emit(self.diameter, self.axial_length, self.num_sectors)

    def update_visuals(self) -> None:
        """
        Отрисовка детекторного кольца блоков сцинтилляторов в 3D.
        """
        if self.viewport is None or self.viewport.plotter is None:
            return

        try:
            radius = self.diameter / 2.0
            half_len = self.axial_length / 2.0

            # Создаем цилиндрический каркас сканера
            cylinder = pv.Cylinder(
                center=(0.0, 0.0, 0.0),
                direction=(0.0, 0.0, 1.0),
                radius=radius,
                height=self.axial_length,
                resolution=self.num_sectors,
                capping=False
            )

            self.viewport.add_mesh_actor(
                self.actor_name,
                cylinder,
                color='#1abc9c',
                opacity=0.35,
                wireframe=True
            )
            self.viewport.render()

        except Exception as e:
            _logger.debug(f"Ошибка визуализации ПЭТ манипулятора: {e}")

    def remove_visuals(self) -> None:
        """
        Удаление визуализации кольца ПЭТ.
        """
        if self.viewport is not None:
            self.viewport.remove_actor(self.actor_name)
            self.viewport.render()
