"""
Модуль специализированных источников частиц для верификационных экспериментов.
Включает конусный точечный источник ConePointSource с равномерным сэмплированием направления
в заданном телесном угле и опциональной рандомизацией фазы в пределах элементарной ячейки решетки.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple
import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Activity, Energy, Float, Time
from core.source.sources import PointSource


class ConePointSource(PointSource):
    """
    Точечный (или усредненный по элементарной ячейке периодичности) источник излучения,
    испускающий частицы равномерно внутри конуса с половинным углом раствора max_polar_angle
    вокруг положительного направления оси +Z.

    При max_polar_angle = pi (180 градусов) полностью эквивалентен изотропному 4*pi PointSource.
    Хранит точный весовой множитель телесного угла solid_angle_fraction = (1 - cos(max_polar_angle)) / 2
    для корректного пересчета абсолютной чувствительности к 4*pi.
    """

    _max_polar_angle: Float

    def __init__(
        self,
        activity: Activity,
        energy: Energy | float | Sequence[Any] | NDArray[Any],
        max_polar_angle: Float | float = Float(np.pi),
        cell_jitter_span: Optional[Tuple[Float | float, Float | float]] = None,
        half_life: Time = Float(6.0 * units.hour),
    ) -> None:
        angle_val = Float(max_polar_angle)
        if angle_val <= 0.0 or angle_val > Float(np.pi):
            raise ValueError(
                f"Половинный угол раствора конуса должен лежать в (0, pi], получено {max_polar_angle}"
            )
        if cell_jitter_span is not None:
            if len(cell_jitter_span) != 2 or cell_jitter_span[0] < 0.0 or cell_jitter_span[1] < 0.0:
                raise ValueError(
                    f"Диапазон рандомизации ячейки cell_jitter_span должен содержать 2 неотрицательных числа, "
                    f"получено {cell_jitter_span}"
                )

        normalized_energy: Energy | Sequence[Any] | NDArray[Any] = (
            Float(energy) if isinstance(energy, (int, float, np.floating)) else energy
        )
        super().__init__(
            activity=activity,
            energy=normalized_energy,
            size=Float(0.0),
            half_life=half_life,
        )
        self._max_polar_angle = angle_val
        self._cos_theta_min = Float(np.cos(angle_val))
        self._cell_jitter_span = (
            (Float(cell_jitter_span[0]), Float(cell_jitter_span[1]))
            if cell_jitter_span is not None
            else None
        )

    @property
    def max_polar_angle(self) -> Float:
        """Половинный угол раствора конуса излучения относительно оси +Z (рад)."""
        return self._max_polar_angle

    @property
    def solid_angle_fraction(self) -> Float:
        """
        Отношение телесного угла конуса к полному телесному углу 4*pi:
        Omega / (4*pi) = (1 - cos(theta_max)) / 2.
        """
        return Float(0.5 * (1.0 - self._cos_theta_min))


__all__ = [
    "ConePointSource",
]

