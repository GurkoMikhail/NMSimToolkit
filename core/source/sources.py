"""
Модуль композиционной модели первичных источников ионизирующего излучения для архитектуры SoA.

Декомпозирует источник излучения Source(CompositeNode) на ортогональные компоненты:
1. SpatialDistribution (PointDistribution, VoxelDistribution) — пространственная модель в локальной СК;
2. RadiationChannel(species, spectrum, angular, branching_ratio) — физический канал испускания
   с поддержкой дискретных (DiscreteSpectrum) и непрерывных кусочно-линейных (ContinuousSpectrum)
   энергетических спектров, а также изотропного (IsotropicAngular) и конусного (ConeAngular)
   угловых распределений.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import (
    Activity,
    Angle,
    Energy,
    Float,
    Index,
    Length,
    ParticleSpecies,
    Time,
    Vector3D,
)
from core.scene.nodes import CompositeNode
from core.source.alias_tables import build_alias_tables

if TYPE_CHECKING:
    from core.particles.particles import ParticleBank
    from core.physics.interaction_buffers import RNGContext


class SpatialDistribution(abc.ABC):
    """Абстрактный базовый класс пространственного распределения источника в локальной СК."""

    @property
    @abc.abstractmethod
    def size(self) -> Vector3D:
        """Габаритные размеры области источника в локальной СК (мм)."""


@dataclass(frozen=True)
class PointDistribution(SpatialDistribution):
    """
    Пространственная модель точечного (или равномерно размытого по ячейке XY) источника.

    Не выделяет воксельных массивов (0 байт). При cell_jitter == (0.0, 0.0) частицы
    рождаются строго в начале локальной СК (0, 0, 0) без вызова ГСЧ на выбор позиции.
    """

    cell_jitter: Tuple[Float, Float] = (Float(0.0), Float(0.0))

    def __post_init__(self) -> None:
        if len(self.cell_jitter) != 2:
            raise ValueError(
                f"Параметр cell_jitter должен быть кортежем из 2 элементов (dx, dy), получено {self.cell_jitter}."
            )
        jitter_x = Float(self.cell_jitter[0])
        jitter_y = Float(self.cell_jitter[1])
        if not np.isfinite(jitter_x) or not np.isfinite(jitter_y) or jitter_x < 0.0 or jitter_y < 0.0:
            raise ValueError(
                f"Значения cell_jitter должны быть конечными и неотрицательными (>= 0.0), получено {self.cell_jitter}."
            )
        object.__setattr__(self, "cell_jitter", (jitter_x, jitter_y))

    @property
    def size(self) -> Vector3D:
        res = np.array([self.cell_jitter[0], self.cell_jitter[1], 0.0], dtype=Float)
        res.flags.writeable = False
        return res


@dataclass(frozen=True)
class VoxelDistribution(SpatialDistribution):
    """
    Пространственная модель трёхмерного воксельного распределения активности.

    Хранит нормированное 3D-распределение, шаг сетки voxel_size, сжатый список
    активных вокселей voxel_indices и таблицы метода алиасов Воза (voxel_prob, voxel_alias).
    """

    distribution: NDArray[Float]
    voxel_size: Length = Float(4.0 * units.mm)
    raw_total_activity: Float = field(init=False, repr=False)
    voxel_indices: NDArray[np.int32] = field(init=False, repr=False)
    voxel_probabilities: NDArray[Float] = field(init=False, repr=False)
    voxel_prob: NDArray[Float] = field(init=False, repr=False)
    voxel_alias: NDArray[np.int32] = field(init=False, repr=False)
    _size: Vector3D = field(init=False, repr=False)

    def __post_init__(self) -> None:
        voxel_size_val = Float(self.voxel_size)
        if not np.isfinite(voxel_size_val) or voxel_size_val <= 0.0:
            raise ValueError(
                f"Размер вокселя VoxelDistribution должен быть строго положительным (> 0), получено {voxel_size_val}."
            )

        dist_array = np.asarray(self.distribution, dtype=Float)
        if dist_array.ndim != 3 or dist_array.size == 0:
            raise ValueError(
                f"Распределение активности должно быть непустым 3D массивом, получено shape={dist_array.shape}."
            )
        if np.any(~np.isfinite(dist_array)) or np.any(dist_array < 0.0):
            raise ValueError(
                "Распределение активности не может содержать отрицательные или неконечные (NaN/Inf) значения."
            )
        total_act = Float(np.sum(dist_array))
        if total_act <= 0.0:
            raise ValueError(
                "Распределение активности не может быть нулевым или отрицательным — источник не содержит активного вещества."
            )

        norm_dist = np.ascontiguousarray(dist_array / total_act, dtype=Float)
        norm_dist.flags.writeable = False

        flat_dist = norm_dist.ravel()
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
        norm_probs.flags.writeable = False
        prob_table, alias_table = build_alias_tables(norm_probs)
        prob_table.flags.writeable = False
        alias_table.flags.writeable = False
        active_indices_contig = np.ascontiguousarray(active_indices, dtype=np.int32)
        active_indices_contig.flags.writeable = False

        size_vec = np.asarray(norm_dist.shape, dtype=Float) * voxel_size_val
        size_vec.flags.writeable = False

        object.__setattr__(self, "distribution", norm_dist)
        object.__setattr__(self, "voxel_size", voxel_size_val)
        object.__setattr__(self, "raw_total_activity", total_act)
        object.__setattr__(self, "voxel_indices", active_indices_contig)
        object.__setattr__(self, "voxel_probabilities", norm_probs)
        object.__setattr__(self, "voxel_prob", prob_table)
        object.__setattr__(self, "voxel_alias", alias_table)
        object.__setattr__(self, "_size", size_vec)

    @property
    def size(self) -> Vector3D:
        return self._size

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Форма 3D-сетки воксельного распределения (nx, ny, nz)."""
        shape = self.distribution.shape
        return int(shape[0]), int(shape[1]), int(shape[2])


@dataclass(frozen=True)
class SpectralLine:
    """Отдельная моноэнергетическая линия дискретного спектра с относительным весом/вероятностью."""

    energy: Energy
    probability: Float = Float(1.0)

    def __post_init__(self) -> None:
        energy_val = Float(self.energy)
        prob_val = Float(self.probability)
        if not np.isfinite(energy_val) or energy_val <= 0.0:
            raise ValueError(f"Энергия спектральной линии должна быть строго положительной (> 0), получено {energy_val}.")
        if not np.isfinite(prob_val) or prob_val < 0.0:
            raise ValueError(f"Вес спектральной линии не может быть отрицательным (< 0), получено {prob_val}.")
        object.__setattr__(self, "energy", energy_val)
        object.__setattr__(self, "probability", prob_val)


class EnergySpectrum(abc.ABC):
    """Абстрактный базовый класс энергетического спектра канала излучения."""

    @property
    @abc.abstractmethod
    def energy_bounds(self) -> Tuple[Float, Float]:
        """Минимальная и максимальная энергии спектра (МэВ)."""


@dataclass(frozen=True)
class DiscreteSpectrum(EnergySpectrum):
    """
    Линейчатый (дискретный) энергетический спектр.

    Нормирует вероятности линий к сумме 1.0 и предвычисляет таблицы метода алиасов Воза
    (line_energy, line_prob, line_alias) для O(1) выборки в Numba JIT-ядрах.
    """

    lines: Tuple[SpectralLine, ...]
    line_energy: NDArray[Float] = field(init=False, repr=False)
    line_probabilities: NDArray[Float] = field(init=False, repr=False)
    line_prob: NDArray[Float] = field(init=False, repr=False)
    line_alias: NDArray[np.int32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("Спектр энергий DiscreteSpectrum не может быть пустым.")

        parsed_lines: list[SpectralLine] = []
        for item in self.lines:
            if isinstance(item, SpectralLine):
                parsed_lines.append(item)
            elif isinstance(item, (tuple, list, np.ndarray)) and len(item) == 2:
                parsed_lines.append(SpectralLine(energy=Float(item[0]), probability=Float(item[1])))
            elif isinstance(item, (int, float, np.floating)):
                parsed_lines.append(SpectralLine(energy=Float(item), probability=Float(1.0)))
            else:
                raise ValueError(f"Некорректный элемент спектральной линии: {item}.")

        energies_1d = np.array([line.energy for line in parsed_lines], dtype=Float)
        probs_1d = np.array([line.probability for line in parsed_lines], dtype=Float)
        prob_sum = Float(np.sum(probs_1d))
        if prob_sum <= 0.0:
            raise ValueError(
                f"Сумма вероятностей спектра должна быть строго положительной (> 0), получено {prob_sum}."
            )

        norm_probs = np.ascontiguousarray(probs_1d / prob_sum, dtype=Float)
        norm_probs.flags.writeable = False
        energies_contig = np.ascontiguousarray(energies_1d, dtype=Float)
        energies_contig.flags.writeable = False

        prob_table, alias_table = build_alias_tables(norm_probs)
        prob_table.flags.writeable = False
        alias_table.flags.writeable = False

        object.__setattr__(self, "lines", tuple(parsed_lines))
        object.__setattr__(self, "line_energy", energies_contig)
        object.__setattr__(self, "line_probabilities", norm_probs)
        object.__setattr__(self, "line_prob", prob_table)
        object.__setattr__(self, "line_alias", alias_table)

    @classmethod
    def from_energies(
        cls,
        value: Energy | Sequence[Any] | NDArray[Any] | "DiscreteSpectrum",
    ) -> "DiscreteSpectrum":
        """Создаёт DiscreteSpectrum из скаляра энергии, списка пар [E, p] или массива NumPy."""
        if isinstance(value, DiscreteSpectrum):
            return value
        if isinstance(value, (int, float, np.floating)):
            return cls(lines=(SpectralLine(energy=Float(value), probability=Float(1.0)),))
        if (
            isinstance(value, np.ndarray)
            and value.dtype.names is not None
            and "energy" in value.dtype.names
            and "probability" in value.dtype.names
        ):
            energies_1d = np.asarray(value["energy"], dtype=Float).ravel()
            probs_1d = np.asarray(value["probability"], dtype=Float).ravel()
            if energies_1d.size == 0:
                raise ValueError("Спектр энергий источника не может быть пустым.")
            return cls(
                lines=tuple(
                    SpectralLine(energy=Float(e_val), probability=Float(p_val))
                    for e_val, p_val in zip(energies_1d, probs_1d)
                )
            )
        if isinstance(value, Sequence) and len(value) > 0 and isinstance(value[0], SpectralLine):
            return cls(lines=tuple(value))

        raw_array = np.asarray(value, dtype=Float)
        if raw_array.ndim <= 1:
            energies_1d = raw_array.ravel()
            num_items = max(1, int(energies_1d.shape[0]))
            probs_1d = np.full(energies_1d.shape[0], 1.0 / num_items, dtype=Float)
        elif raw_array.ndim == 2 and raw_array.shape[1] in (1, 2):
            energies_1d = raw_array[:, 0]
            num_items = max(1, int(raw_array.shape[0]))
            probs_1d = (
                raw_array[:, 1]
                if raw_array.shape[1] == 2
                else np.full(raw_array.shape[0], 1.0 / num_items, dtype=Float)
            )
        else:
            raise ValueError(f"Массив энергий должен иметь форму (N,) или (N, 2), получено {raw_array.shape}.")

        if energies_1d.size == 0:
            raise ValueError("Спектр энергий источника не может быть пустым.")
        return cls(
            lines=tuple(
                SpectralLine(energy=Float(e_val), probability=Float(p_val))
                for e_val, p_val in zip(energies_1d, probs_1d)
            )
        )

    @property
    def energy_bounds(self) -> Tuple[Float, Float]:
        return Float(np.min(self.line_energy)), Float(np.max(self.line_energy))


@dataclass(frozen=True)
class ContinuousSpectrum(EnergySpectrum):
    """
    Непрерывный кусочно-линейный энергетический спектр p(E) на сетке узлов energy_grid.

    На этапе инициализации вычисляет площади трапеций A_k = 0.5 * (p_k + p_{k+1}) * (E_{k+1} - E_k),
    строит таблицу Воза интервалов (bin_prob, bin_alias) и предвычисляет отношения высот
    bin_pdf_ratio = p_k / (p_k + p_{k+1}) для O(1) аналитического обращения трапеции в @njit-ядре.
    """

    energy_grid: NDArray[Float]
    pdf_values: NDArray[Float]
    bin_energy_low: NDArray[Float] = field(init=False, repr=False)
    bin_energy_high: NDArray[Float] = field(init=False, repr=False)
    bin_probabilities: NDArray[Float] = field(init=False, repr=False)
    bin_pdf_ratio: NDArray[Float] = field(init=False, repr=False)
    bin_prob: NDArray[Float] = field(init=False, repr=False)
    bin_alias: NDArray[np.int32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grid_arr = np.ascontiguousarray(np.asarray(self.energy_grid, dtype=Float))
        pdf_arr = np.ascontiguousarray(np.asarray(self.pdf_values, dtype=Float))

        if grid_arr.ndim != 1 or pdf_arr.ndim != 1:
            raise ValueError("Массивы energy_grid и pdf_values в ContinuousSpectrum должны быть одномерными.")
        if grid_arr.size < 2 or grid_arr.size != pdf_arr.size:
            raise ValueError(
                f"Сетка ContinuousSpectrum должна содержать >= 2 узлов и совпадать по длине с pdf_values, "
                f"получено len(energy_grid)={grid_arr.size}, len(pdf_values)={pdf_arr.size}."
            )
        if np.any(~np.isfinite(grid_arr)) or np.any(grid_arr <= 0.0):
            raise ValueError("Узлы energy_grid в ContinuousSpectrum должны быть конечными и строго положительными (> 0).")
        if np.any(np.diff(grid_arr) <= 0.0):
            raise ValueError("Узлы energy_grid в ContinuousSpectrum должны строго возрастать.")
        if np.any(~np.isfinite(pdf_arr)) or np.any(pdf_arr < 0.0):
            raise ValueError("Значения pdf_values в ContinuousSpectrum должны быть конечными и неотрицательными (>= 0).")

        e_low = np.ascontiguousarray(grid_arr[:-1], dtype=Float)
        e_high = np.ascontiguousarray(grid_arr[1:], dtype=Float)
        p_left = pdf_arr[:-1]
        p_right = pdf_arr[1:]
        p_sum_pair = p_left + p_right
        areas = 0.5 * p_sum_pair * (e_high - e_low)
        total_area = float(np.sum(areas))
        if total_area <= 0.0:
            raise ValueError("Интегральная площадь ContinuousSpectrum должна быть строго положительной (> 0).")

        norm_areas = np.ascontiguousarray(areas / total_area, dtype=Float)
        pdf_ratio = np.where(p_sum_pair > 0.0, p_left / np.where(p_sum_pair > 0.0, p_sum_pair, 1.0), 0.5)
        pdf_ratio_contig = np.ascontiguousarray(pdf_ratio, dtype=Float)

        prob_table, alias_table = build_alias_tables(norm_areas)

        grid_arr.flags.writeable = False
        pdf_arr.flags.writeable = False
        e_low.flags.writeable = False
        e_high.flags.writeable = False
        norm_areas.flags.writeable = False
        pdf_ratio_contig.flags.writeable = False
        prob_table.flags.writeable = False
        alias_table.flags.writeable = False

        object.__setattr__(self, "energy_grid", grid_arr)
        object.__setattr__(self, "pdf_values", pdf_arr)
        object.__setattr__(self, "bin_energy_low", e_low)
        object.__setattr__(self, "bin_energy_high", e_high)
        object.__setattr__(self, "bin_probabilities", norm_areas)
        object.__setattr__(self, "bin_pdf_ratio", pdf_ratio_contig)
        object.__setattr__(self, "bin_prob", prob_table)
        object.__setattr__(self, "bin_alias", alias_table)

    @property
    def energy_bounds(self) -> Tuple[Float, Float]:
        return Float(self.energy_grid[0]), Float(self.energy_grid[-1])


class AngularDistribution(abc.ABC):
    """Абстрактный базовый класс углового распределения относительно опорного орта +Z."""

    @property
    @abc.abstractmethod
    def cos_theta_min(self) -> Float:
        """Минимальный косинус полярного угла вылета (-1.0 для изотропного 4*pi распределения)."""

    @property
    @abc.abstractmethod
    def solid_angle_fraction(self) -> Float:
        """Доля телесного угла вылета относительно полной сферы 4*pi."""


@dataclass(frozen=True)
class IsotropicAngular(AngularDistribution):
    """Изотропное угловое распределение в полном телесном угле 4*pi."""

    @property
    def cos_theta_min(self) -> Float:
        return Float(-1.0)

    @property
    def solid_angle_fraction(self) -> Float:
        return Float(1.0)


@dataclass(frozen=True)
class ConeAngular(AngularDistribution):
    """
    Равномерное угловое распределение внутри конуса с половинным углом раствора max_polar_angle
    вокруг локальной оси +Z.
    """

    max_polar_angle: Angle

    def __post_init__(self) -> None:
        angle_val = Float(self.max_polar_angle)
        if not np.isfinite(angle_val) or angle_val <= 0.0 or angle_val > Float(np.pi):
            raise ValueError(
                f"Половинный угол раствора конуса max_polar_angle должен лежать в (0, pi], получено {self.max_polar_angle}."
            )
        object.__setattr__(self, "max_polar_angle", angle_val)

    @property
    def cos_theta_min(self) -> Float:
        if self.max_polar_angle >= Float(np.pi):
            return Float(-1.0)
        return Float(math.cos(float(self.max_polar_angle)))

    @property
    def solid_angle_fraction(self) -> Float:
        return Float(0.5 * (1.0 - float(self.cos_theta_min)))


@dataclass(frozen=True)
class RadiationChannel:
    """
    Физический канал испускания частиц источником излучения.

    Определяется четвёркой: тип частицы (species), энергетический спектр (spectrum),
    угловое распределение (angular) и выход на распад (branching_ratio).
    """

    species: ParticleSpecies = ParticleSpecies.PHOTON
    spectrum: EnergySpectrum = field(
        default_factory=lambda: DiscreteSpectrum.from_energies(Float(140.5 * units.keV))
    )
    angular: AngularDistribution = field(default_factory=IsotropicAngular)
    branching_ratio: Float = Float(1.0)

    def __post_init__(self) -> None:
        species_enum = ParticleSpecies(int(self.species))
        br_val = Float(self.branching_ratio)
        if not np.isfinite(br_val) or br_val <= 0.0:
            raise ValueError(
                f"Коэффициент ветвления канала branching_ratio должен быть конечным и > 0, получено {self.branching_ratio}."
            )
        object.__setattr__(self, "species", species_enum)
        object.__setattr__(self, "branching_ratio", br_val)


class Source(CompositeNode):
    """
    Композиционный узел первичного источника частиц в дереве 3D-сцены.

    Объединяет пространственную модель в локальной СК (spatial: SpatialDistribution),
    набор каналов излучения (channels: Tuple[RadiationChannel, ...]) и кинетику распада
    (initial_activity, half_life).

    Базовые единицы (hepunits):
    - активность: Бк (1 / ns)
    - линейные размеры: мм (units.mm)
    - энергия: МэВ (units.MeV)
    - период полураспада: нс (units.ns)
    """

    _spatial: SpatialDistribution
    _channels: Tuple[RadiationChannel, ...]
    _initial_activity: Float
    _half_life: Time

    def __init__(
        self,
        spatial: SpatialDistribution,
        channels: Sequence[RadiationChannel],
        activity: Optional[Activity] = None,
        half_life: Time = Float(6.0 * units.hour),
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.spatial = spatial
        self.channels = tuple(channels)
        self.half_life = half_life

        if activity is None:
            if isinstance(spatial, VoxelDistribution):
                init_act = spatial.raw_total_activity
            else:
                raise ValueError(
                    "Для невоксельного источника (PointDistribution) необходимо явно указать начальную активность activity > 0."
                )
        else:
            init_act = Float(activity)
        self.initial_activity = init_act

    @classmethod
    def from_point(
        cls,
        activity: Activity,
        energy: Energy | Sequence[Any] | NDArray[Any] | EnergySpectrum = Float(140.5 * units.keV),
        half_life: Time = Float(6.0 * units.hour),
        *,
        species: ParticleSpecies = ParticleSpecies.PHOTON,
        angular: Optional[AngularDistribution] = None,
        max_polar_angle: Optional[Angle] = None,
        cell_jitter: Tuple[Float, Float] = (Float(0.0), Float(0.0)),
        branching_ratio: Float = Float(1.0),
        name: Optional[str] = None,
    ) -> "Source":
        """
        Фабричный конструктор одноканального точечного (или ячейкового) источника.
        """
        spectrum_obj = energy if isinstance(energy, EnergySpectrum) else DiscreteSpectrum.from_energies(energy)
        if angular is not None and max_polar_angle is not None:
            raise ValueError("Нельзя одновременно задавать параметры angular и max_polar_angle в Source.from_point.")
        if angular is not None:
            angular_obj = angular
        elif max_polar_angle is not None:
            angular_obj = ConeAngular(max_polar_angle=Float(max_polar_angle))
        else:
            angular_obj = IsotropicAngular()

        spatial_obj = PointDistribution(cell_jitter=(Float(cell_jitter[0]), Float(cell_jitter[1])))
        channel = RadiationChannel(
            species=species,
            spectrum=spectrum_obj,
            angular=angular_obj,
            branching_ratio=Float(branching_ratio),
        )
        return cls(
            spatial=spatial_obj,
            channels=(channel,),
            activity=activity,
            half_life=half_life,
            name=name,
        )

    @classmethod
    def from_voxels(
        cls,
        distribution: Sequence[Any] | NDArray[Any],
        activity: Optional[Activity] = None,
        voxel_size: Length = Float(4.0 * units.mm),
        energy: Energy | Sequence[Any] | NDArray[Any] | EnergySpectrum = Float(140.5 * units.keV),
        half_life: Time = Float(6.0 * units.hour),
        *,
        species: ParticleSpecies = ParticleSpecies.PHOTON,
        angular: Optional[AngularDistribution] = None,
        branching_ratio: Float = Float(1.0),
        name: Optional[str] = None,
    ) -> "Source":
        """
        Фабричный конструктор одноканального воксельного источника.
        """
        spectrum_obj = energy if isinstance(energy, EnergySpectrum) else DiscreteSpectrum.from_energies(energy)
        angular_obj = angular if angular is not None else IsotropicAngular()
        spatial_obj = VoxelDistribution(distribution=np.asarray(distribution, dtype=Float), voxel_size=Float(voxel_size))
        channel = RadiationChannel(
            species=species,
            spectrum=spectrum_obj,
            angular=angular_obj,
            branching_ratio=Float(branching_ratio),
        )
        return cls(
            spatial=spatial_obj,
            channels=(channel,),
            activity=activity,
            half_life=half_life,
            name=name,
        )

    @property
    def spatial(self) -> SpatialDistribution:
        """Пространственная модель распределения источника в локальной СК."""
        return self._spatial

    @spatial.setter
    def spatial(self, value: SpatialDistribution) -> None:
        self._spatial = value

    @property
    def channels(self) -> Tuple[RadiationChannel, ...]:
        """Кортеж физических каналов испускания частиц источником."""
        return self._channels

    @channels.setter
    def channels(self, value: Sequence[RadiationChannel]) -> None:
        ch_tuple = tuple(value)
        if not ch_tuple:
            raise ValueError("Источник Source должен содержать хотя бы один канал излучения (RadiationChannel).")
        self._channels = ch_tuple

    @property
    def initial_activity(self) -> Float:
        """Начальная активность источника A_0 в распадах за наносекунду (Бк в единицах hepunits)."""
        return self._initial_activity

    @initial_activity.setter
    def initial_activity(self, value: Activity) -> None:
        init_act = Float(value)
        if not np.isfinite(init_act) or init_act <= 0.0:
            raise ValueError(
                f"Начальная активность источника должна быть строго положительной (> 0), получено {init_act}."
            )
        self._initial_activity = init_act

    @property
    def half_life(self) -> Time:
        """Период полураспада радионуклида T_1/2 (нс)."""
        return self._half_life

    @half_life.setter
    def half_life(self, value: Time) -> None:
        half_life_val = Float(value)
        if np.isnan(half_life_val) or half_life_val <= 0.0:
            raise ValueError(
                f"Период полураспада half_life должен быть строго положительным, получено {value}."
            )
        self._half_life = half_life_val

    @property
    def decay_constant(self) -> Float:
        """Постоянная радиоактивного распада lambda = ln(2) / T_1/2."""
        if np.isinf(self._half_life):
            return Float(0.0)
        return Float(np.log(2.0) / self._half_life)

    @property
    def size(self) -> Vector3D:
        """Габаритные размеры пространственного распределения источника в локальной СК (мм)."""
        return self._spatial.size

    @property
    def solid_angle_fraction(self) -> Float:
        """
        Отношение телесного угла вылета к полному телесному углу 4*pi
        (взвешенное по коэффициентам ветвления каналов).
        """
        total_br = sum(float(ch.branching_ratio) for ch in self._channels)
        if total_br <= 0.0:
            return Float(1.0)
        weighted = sum(float(ch.branching_ratio) * float(ch.angular.solid_angle_fraction) for ch in self._channels)
        return Float(weighted / total_br)


class ParticleSource(abc.ABC):
    """
    Абстрактный базовый класс рантайм-источника эмиссии частиц в ParticleBank.

    Определяет единый полиморфный контракт взаимодействия SimulationManager с первичным
    пуассоновским потоком (PrimarySource) и наведенным вторичным источником (SecondarySource):
    1. Проверка наличия ожидающих инжекции частиц (has_pending);
    2. Пакетное Zero-Allocation пополнение свободных слотов банка с квотированием (replenish_bank);
    3. Экспорт и восстановление состояния для чекпоинтов Stateful Resume (export_state / restore_state).
    """

    @property
    @abc.abstractmethod
    def has_pending(self) -> bool:
        """True, если источник готов эмитировать частицы в текущем окне симуляции."""

    @abc.abstractmethod
    def replenish_bank(
        self,
        bank: ParticleBank,
        rng_ctx: RNGContext,
        start_slot_cursor: Index = Index(0),
        max_slots: Index = Index(-1),
    ) -> Tuple[Index, Index]:
        """
        Инжектирует частицы в свободные слоты ParticleBank со start_slot_cursor,
        занимая не более max_slots слотов (при max_slots >= 0).
        Возвращает кортеж (injected_count, next_slot_cursor).
        """

    @abc.abstractmethod
    def export_state(self) -> Optional[dict[str, Any]]:
        """Экспортирует состояние источника для сохранения чекпоинта Stateful Resume."""

    @abc.abstractmethod
    def restore_state(self, state: Optional[Mapping[str, Any]]) -> None:
        """Восстанавливает состояние источника из словаря чекпоинта Stateful Resume."""


__all__ = [
    "ParticleSpecies",
    "SpatialDistribution",
    "PointDistribution",
    "VoxelDistribution",
    "SpectralLine",
    "EnergySpectrum",
    "DiscreteSpectrum",
    "ContinuousSpectrum",
    "AngularDistribution",
    "IsotropicAngular",
    "ConeAngular",
    "RadiationChannel",
    "Source",
    "ParticleSource",
]
