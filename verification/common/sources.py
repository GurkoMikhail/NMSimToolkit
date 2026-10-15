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

from core.other.typing_definitions import Activity, Energy, Float, Length, Time, Vector3D
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
    _cos_theta_min: Float
    _cell_jitter_span: Optional[Tuple[Float, Float]]

    def __init__(
        self,
        activity: Activity,
        energy: Energy | float | Sequence[Any] | NDArray[Any],
        max_polar_angle: Float | float = Float(np.pi),
        cell_jitter_span: Optional[Tuple[Float | float, Float | float]] = None,
        half_life: Time = Float(6.0 * units.hour),
        rng: Optional[np.random.Generator] = None,
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
            rng=rng,
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

    def generate_position(self, count: int) -> Vector3D:
        """
        Генерирует глобальные координаты рождения частиц.
        Если задан cell_jitter_span = (pitch_x, pitch_y), равномерно распределяет начальные точки
        в пределах одной элементарной ячейки [-pitch_x/2, +pitch_x/2] x [-pitch_y/2, +pitch_y/2]
        для получения shift-invariant PSF Ангера (с сохранением точной точки рождения в birth_x, birth_y).
        """
        local_positions = np.zeros((count, 3), dtype=Float)
        if self._cell_jitter_span is not None:
            span_x, span_y = self._cell_jitter_span
            if span_x > 0.0:
                local_positions[:, 0] = self.rng.uniform(-0.5 * span_x, 0.5 * span_x, count)
            if span_y > 0.0:
                local_positions[:, 1] = self.rng.uniform(-0.5 * span_y, 0.5 * span_y, count)
        global_positions = self.convert_to_global_position(local_positions)
        return np.asarray(global_positions, dtype=Float)

    def generate_direction(self, count: int) -> Vector3D:
        """
        Генерирует единичные векторы направления равномерно по телесному углу
        внутри конуса с осью +Z и половинным углом раствора max_polar_angle.
        """
        rand_u1 = self.rng.random(count)
        rand_u2 = self.rng.random(count)

        cos_theta = 1.0 - rand_u1 * (1.0 - self._cos_theta_min)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
        phi_angle = 2.0 * np.pi * rand_u2

        dir_x = sin_theta * np.cos(phi_angle)
        dir_y = sin_theta * np.sin(phi_angle)
        dir_z = cos_theta

        direction = np.column_stack((dir_x, dir_y, dir_z))
        return np.asarray(direction, dtype=Float)


__all__ = [
    "ConePointSource",
]
