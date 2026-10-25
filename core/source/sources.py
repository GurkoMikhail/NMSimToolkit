"""
Модуль источников ионизирующего излучения (воксельных и точечных) для архитектуры SoA.
"""

from typing import Any, Optional, Sequence, Tuple

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import (
    Activity,
    Energy,
    Float,
    Length,
    Time,
    Vector3D,
)
from core.scene.nodes import CompositeNode
from core.source.alias_tables import build_alias_tables

_ENERGY_DTYPE = np.dtype([("energy", Float), ("probability", Float)])


class Source(CompositeNode):
    """
    Класс источника частиц для Data-Oriented Design (SoA).

    Базовые единицы (hepunits):
    - активность: Бк (1 / ns)
    - размер вокселя: мм (units.mm)
    - энергия: МэВ (units.MeV)
    - период полураспада: нс (units.ns)
    """

    _distribution: NDArray[Float]
    initial_activity: Float
    _voxel_size: Length
    size: Vector3D
    radiation_type: str
    _energy: NDArray[Any]
    half_life: Time
    voxel_indices: NDArray[np.int32]
    voxel_probabilities: NDArray[Float]
    voxel_prob: NDArray[Float]
    voxel_alias: NDArray[np.int32]
    line_prob: NDArray[Float]
    line_alias: NDArray[np.int32]
    _cos_theta_min: Float
    _cell_jitter_span: Optional[Tuple[Float, Float]]

    def __init__(
        self,
        distribution: Sequence[Any] | NDArray[Any],
        activity: Optional[Activity] = None,
        voxel_size: Length = Float(4.0 * units.mm),
        radiation_type: str = 'Gamma',
        energy: Energy | Sequence[Any] | NDArray[Any] = Float(140.5 * units.keV),
        half_life: Time = Float(6.0 * units.hour),
    ) -> None:
        super().__init__()
        voxel_size_val = Float(voxel_size)
        if not np.isfinite(voxel_size_val) or voxel_size_val < 0.0:
            raise ValueError(
                f"Размер вокселя источника не может быть отрицательным (< 0), получено {voxel_size_val}"
            )
        half_life_val = Float(half_life)
        if np.isnan(half_life_val) or half_life_val <= 0.0:
            raise ValueError(
                f"Период полураспада half_life должен быть строго положительным, получено {half_life}."
            )

        self._voxel_size = voxel_size_val
        self.radiation_type = radiation_type
        self.half_life: Float = half_life_val
        self._cos_theta_min = Float(-1.0)
        self._cell_jitter_span = None

        dist_array = np.asarray(distribution, dtype=Float)
        self.distribution = dist_array

        init_act = Float(np.sum(dist_array)) if activity is None else Float(activity)
        if not np.isfinite(init_act) or init_act <= 0.0:
            raise ValueError(
                f"Начальная активность источника должна быть строго положительной (> 0), получено {init_act}"
            )
        self.initial_activity = init_act
        self.energy = energy

    @property
    def cos_theta_min(self) -> Float:
        """Минимальный косинус полярного угла испускания (-1.0 для изотропного 4*pi источника)."""
        return self._cos_theta_min

    @property
    def cell_jitter_span(self) -> Optional[Tuple[Float, Float]]:
        """Опциональный диапазон рандомизации положения в плоскости XY (мм)."""
        return self._cell_jitter_span

    @property
    def energy(self) -> NDArray[Any]:
        """
        Спектр энергии частиц источника в виде структурированного массива
        с полями 'energy' (МэВ) и 'probability' (нормированная вероятность, sum(p) == 1.0).
        """
        return self._energy

    @energy.setter
    def energy(self, value: Energy | Sequence[Any] | NDArray[Any]) -> None:
        if (
            isinstance(value, np.ndarray)
            and value.dtype.names is not None
            and "energy" in value.dtype.names
            and "probability" in value.dtype.names
        ):
            energies_1d = np.asarray(value["energy"], dtype=Float).ravel()
            probs_1d = np.asarray(value["probability"], dtype=Float).ravel()
        else:
            raw_array = np.asarray(value, dtype=Float)
            if raw_array.ndim <= 1:
                energies_1d = raw_array.ravel()
                probs_1d = np.ones(energies_1d.shape[0], dtype=Float)
            elif raw_array.ndim == 2 and raw_array.shape[1] in (1, 2):
                energies_1d = raw_array[:, 0]
                probs_1d = (
                    raw_array[:, 1]
                    if raw_array.shape[1] == 2
                    else np.ones(raw_array.shape[0], dtype=Float)
                )
            else:
                raise ValueError(f"Массив энергий должен иметь форму (N,) или (N, 2), получено {raw_array.shape}")

        if energies_1d.size == 0:
            raise ValueError("Спектр энергий источника не может быть пустым.")
        if np.any(~np.isfinite(energies_1d)) or np.any(energies_1d <= 0.0):
            raise ValueError("Энергия излучения должна быть строго положительной (> 0)")
        if np.any(~np.isfinite(probs_1d)) or np.any(probs_1d < 0.0):
            raise ValueError("Вероятности излучения в спектре не могут быть отрицательными (< 0)")
        prob_sum = Float(np.sum(probs_1d))
        if prob_sum <= 0.0:
            raise ValueError(
                f"Сумма вероятностей спектра должна быть строго положительной (> 0), получено {prob_sum}"
            )

        structured_table: NDArray[Any] = np.empty(energies_1d.shape[0], dtype=_ENERGY_DTYPE)
        structured_table["energy"] = energies_1d
        structured_table["probability"] = probs_1d / prob_sum
        structured_table.flags.writeable = False
        self._energy = structured_table
        self.line_prob, self.line_alias = build_alias_tables(self._energy["probability"])

    @property
    def distribution(self) -> NDArray[Float]:
        """Нормированное пространственное распределение вероятностей испускания частиц."""
        return self._distribution

    @distribution.setter
    def distribution(self, value: Any) -> None:
        dist_array = np.asarray(value, dtype=Float)
        if dist_array.ndim != 3 or dist_array.size == 0:
            raise ValueError(
                f"Распределение активности должно быть непустым 3D массивом, получено shape={dist_array.shape}"
            )
        if np.any(~np.isfinite(dist_array)) or np.any(dist_array < 0.0):
            raise ValueError(
                "Распределение активности не может содержать отрицательные или неконечные (NaN/Inf) значения."
            )
        total_activity = Float(np.sum(dist_array))
        if total_activity <= 0.0:
            raise ValueError(
                "Распределение активности не может быть нулевым или отрицательным — источник не содержит активного вещества."
            )
        self._distribution = dist_array / total_activity
        self._distribution.flags.writeable = False
        self.size = np.asarray(self._distribution.shape, dtype=Float) * self._voxel_size
        self._rebuild_voxel_tables()

    @property
    def voxel_size(self) -> Length:
        """Шаг воксельной сетки распределения активности (мм)."""
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, value: Length) -> None:
        voxel_size_val = Float(value)
        if not np.isfinite(voxel_size_val) or voxel_size_val < 0.0:
            raise ValueError(
                f"Размер вокселя источника не может быть отрицательным (< 0), получено {voxel_size_val}"
            )
        self._voxel_size = voxel_size_val
        self.size = np.asarray(self._distribution.shape, dtype=Float) * self._voxel_size

    def _rebuild_voxel_tables(self) -> None:
        """
        Формирует компактные плоские индексы активных вокселей (int32), нормированные вероятности
        и таблицы метода алиасов Воза (prob, alias) без выделения 3D-сетки координат центров.
        """
        flat_dist = self._distribution.ravel()
        if flat_dist.size > np.iinfo(np.int32).max:
            raise ValueError(
                f"Число вокселей распределения ({flat_dist.size}) превышает предел int32."
            )
        active_indices = np.flatnonzero(flat_dist > 0.0).astype(np.int32)
        if active_indices.size == 0:
            raise ValueError("Распределение активности не содержит положительных вокселей.")

        prob_values = np.asarray(flat_dist[active_indices], dtype=Float)
        prob_sum = float(np.sum(prob_values))
        if prob_sum <= 0.0:
            raise ValueError("Сумма вероятностей активных вокселей должна быть строго положительной (> 0).")

        norm_probs = np.ascontiguousarray(prob_values / prob_sum, dtype=Float)
        prob_table, alias_table = build_alias_tables(norm_probs)

        self.voxel_indices = np.ascontiguousarray(active_indices, dtype=np.int32)
        self.voxel_probabilities = norm_probs
        self.voxel_prob = prob_table
        self.voxel_alias = alias_table

    @property
    def decay_constant(self) -> Float:
        """Постоянная радиоактивного распада lambda = ln(2) / T_1/2."""
        if np.isinf(self.half_life):
            return Float(0.0)
        return Float(np.log(2.0) / self.half_life)


class PointSource(Source):
    """
    Идеальный точечный источник излучения в формате SoA.
    По умолчанию имеет нулевой размер вокселя (size=0.0, K=1) и испускает частицы строго
    из начала своей локальной системы координат (0, 0, 0) без потребления ГСЧ на смещение.
    """

    def __init__(
        self,
        activity: Activity,
        energy: Energy | Sequence[Any] | NDArray[Any],
        size: Length = Float(0.0),
        half_life: Time = Float(6.0 * units.hour),
    ) -> None:
        distribution = np.ones((1, 1, 1), dtype=Float)
        super().__init__(
            distribution=distribution,
            activity=activity,
            voxel_size=size,
            energy=energy,
            half_life=half_life,
        )


__all__ = [
    "Source",
    "PointSource",
]
