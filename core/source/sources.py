"""
Модуль источников ионизирующего излучения (воксельных и точечных) для архитектуры SoA.
"""

from typing import Any, List, Optional, Sequence

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import (
    Activity,
    Energy,
    Float,
    Index,
    Length,
    Species,
    Time,
    Vector3D,
)
from core.other.vectors import Vector3DSoA
from core.particles.particles import ParticleBank
from core.scene.nodes import CompositeNode


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
    rng: np.random.Generator
    emission_table: List[NDArray[Float]]

    def __init__(
        self,
        distribution: Sequence[Any] | NDArray[Any],
        activity: Optional[Activity] = None,
        voxel_size: Length = Float(4.0 * units.mm),
        radiation_type: str = 'Gamma',
        energy: Energy | Sequence[Any] | NDArray[Any] = Float(140.5 * units.keV),
        half_life: Time = Float(6.0 * units.hour),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()
        dist_array = np.asarray(distribution, dtype=Float)
        if dist_array.ndim != 3 and dist_array.size > 0:
            raise ValueError(
                f"Распределение активности должно быть 3D массивом, получено shape={dist_array.shape}"
            )

        total_activity = Float(np.sum(dist_array))
        init_act = total_activity if activity is None else Float(activity)
        if init_act <= 0.0 and dist_array.size > 0:
            raise ValueError(
                f"Начальная активность источника должна быть строго положительной (> 0), получено {init_act}"
            )
        self.initial_activity = init_act
        self._voxel_size = Float(voxel_size)
        self.radiation_type = radiation_type
        self.size = np.zeros(3, dtype=Float)
        self.emission_table = []
        self._distribution = np.empty((0, 0, 0), dtype=Float)
        self.distribution = dist_array

        self._energy = np.empty(0, dtype=[("energy", Float), ("probability", Float)])
        self.energy = energy

        self.half_life = Float(half_life)
        self.rng = np.random.default_rng() if rng is None else rng

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
            energy_arr: NDArray[Any] = value.copy()
            prob_sum = float(np.sum(energy_arr["probability"]))
            if prob_sum > 0.0:
                energy_arr["probability"] = energy_arr["probability"] / prob_sum
            self._energy = energy_arr
            return

        raw_array = np.asarray(value, dtype=Float)
        if raw_array.size == 0:
            raise ValueError("Спектр энергий источника не может быть пустым.")

        if raw_array.ndim == 0:
            structured: NDArray[Any] = np.zeros(1, dtype=[("energy", Float), ("probability", Float)])
            scalar_energy = float(raw_array)
            if scalar_energy <= 0.0:
                raise ValueError("Энергия излучения должна быть строго положительной (> 0)")
            structured["energy"] = Float(scalar_energy)
            structured["probability"] = 1.0
            self._energy = structured
            return

        if raw_array.ndim == 1:
            raw_array = raw_array.reshape(-1, 1)

        if raw_array.shape[1] == 1:
            energies = raw_array[:, 0]
            if np.any(energies <= 0.0):
                raise ValueError("Энергия излучения должна быть строго положительной (> 0)")
            probabilities = np.ones(raw_array.shape[0], dtype=Float) / raw_array.shape[0]
        elif raw_array.shape[1] == 2:
            energies = raw_array[:, 0]
            if np.any(energies <= 0.0):
                raise ValueError("Энергия излучения должна быть строго положительной (> 0)")
            prob_col = raw_array[:, 1]
            if np.any(prob_col < 0.0):
                raise ValueError("Вероятности излучения в спектре не могут быть отрицательными (< 0)")
            prob_sum_val = Float(np.sum(prob_col))
            if prob_sum_val <= 0.0:
                raise ValueError(
                    f"Сумма вероятностей спектра должна быть строго положительной (> 0), получено {prob_sum_val}"
                )
            probabilities = prob_col / prob_sum_val
        else:
            raise ValueError(f"Массив энергий должен иметь форму (N,) или (N, 2), получено {raw_array.shape}")

        structured_table: NDArray[Any] = np.zeros(
            raw_array.shape[0], dtype=[("energy", Float), ("probability", Float)]
        )
        structured_table["energy"] = energies
        structured_table["probability"] = probabilities
        self._energy = structured_table

    @property
    def distribution(self) -> NDArray[Float]:
        """Нормированное пространственное распределение вероятностей испускания частиц."""
        return self._distribution

    @distribution.setter
    def distribution(self, value: Any) -> None:
        dist_array = np.asarray(value, dtype=Float)
        if dist_array.size == 0:
            self._distribution = dist_array
            self.size = np.zeros(3, dtype=Float)
            self._generate_emission_table()
            return
        total_activity = Float(np.sum(dist_array))
        if total_activity <= 0.0:
            raise ValueError(
                "Распределение активности не может быть нулевым или отрицательным — источник не содержит активного вещества."
            )
        self._distribution = dist_array / total_activity
        self.size = np.asarray(self._distribution.shape, dtype=Float) * self._voxel_size
        self._generate_emission_table()

    @property
    def voxel_size(self) -> Length:
        """Шаг воксельной сетки распределения активности (мм)."""
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, value: Length) -> None:
        self._voxel_size = Float(value)
        if self._distribution.size > 0:
            self.size = np.asarray(self._distribution.shape, dtype=Float) * self._voxel_size
            self._generate_emission_table()

    def _generate_emission_table(self) -> None:
        """
        Генерация таблицы координат и нормированных вероятностей испускания для непустых вокселей.
        Гарантирует, что сумма вероятностей строго равна 1.0 (без машинной погрешности округления).
        """
        if self._distribution.size == 0:
            self.emission_table = [np.zeros((0, 3), dtype=Float), np.zeros(0, dtype=Float)]
            return

        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(0.0, self.size[0], self._distribution.shape[0], endpoint=False),
            np.linspace(0.0, self.size[1], self._distribution.shape[1], endpoint=False),
            np.linspace(0.0, self.size[2], self._distribution.shape[2], endpoint=False),
            indexing='ij',
        )
        position = np.stack((grid_x, grid_y, grid_z), axis=3).reshape(-1, 3) - self.size / 2.0
        probability = self._distribution.ravel()
        indices = probability.nonzero()[0]
        prob_values = probability[indices]
        prob_sum = float(np.sum(prob_values))
        if prob_sum > 0.0:
            prob_values = prob_values / prob_sum
        self.emission_table = [
            np.asarray(position[indices], dtype=Float),
            np.asarray(prob_values, dtype=Float),
        ]

    @property
    def decay_constant(self) -> Float:
        """Постоянная радиоактивного распада lambda = ln(2) / T_1/2."""
        if np.isinf(self.half_life):
            return Float(0.0)
        return Float(np.log(2.0) / self.half_life)

    def _get_effective_dt(self, delta_time: Float) -> Float:
        lambd = self.decay_constant
        if lambd == 0.0 or lambd * delta_time < 1e-6:
            return delta_time
        return Float((1.0 - np.exp(-lambd * delta_time)) / lambd)

    def get_activity(self, current_time: Float) -> Float:
        """Мгновенная активность источника в момент времени current_time."""
        if self.decay_constant == 0.0:
            return Float(self.initial_activity)
        return Float(self.initial_activity * (2.0 ** (-current_time / self.half_life)))

    def get_expected_particles(self, time_start: Float, time_end: Float) -> Float:
        """Точный интеграл числа распадов на интервале [time_start, time_end]."""
        delta_time = Float(time_end - time_start)
        if delta_time <= 0.0:
            return Float(0.0)
        return Float(self.get_activity(time_start) * self._get_effective_dt(delta_time))

    def set_state(self, rng_state: Optional[Any] = None) -> None:
        """Устанавливает внутреннее состояние битового генератора источника."""
        if rng_state is None:
            return
        state_dict = dict(self.rng.bit_generator.state)
        state_dict['state'] = rng_state
        self.rng.bit_generator.state = state_dict

    def generate_energy(self, count: int) -> NDArray[Float]:
        """Генерирует энергии испускаемых частиц согласно спектру источника."""
        energies = self.rng.choice(self.energy["energy"], count, p=self.energy["probability"])
        return np.asarray(energies, dtype=Float)

    def generate_position(self, count: int) -> Vector3D:
        """Генерирует глобальные координаты рождения частиц."""
        voxel_origins = self.emission_table[0]
        probability = self.emission_table[1]
        sampled_positions = np.array(self.rng.choice(voxel_origins, count, p=probability), dtype=Float, copy=True)
        if self._voxel_size > 0.0:
            sampled_positions += self.rng.uniform(0.0, self._voxel_size, sampled_positions.shape)
        global_positions = self.convert_to_global_position(sampled_positions)
        return np.asarray(global_positions, dtype=Float)

    def generate_emission_time(self, count: int, time_start: Float, time_end: Float) -> NDArray[Float]:
        """Генерирует моменты времени испускания частиц на интервале [time_start, time_end]."""
        delta_time = Float(time_end - time_start)
        uniform_samples = self.rng.uniform(0.0, 1.0, count)
        lambd = self.decay_constant

        if lambd == 0.0 or lambd * delta_time < 1e-6:
            emission_time = time_start + uniform_samples * delta_time
        else:
            effective_dt = self._get_effective_dt(delta_time)
            emission_time = time_start - (1.0 / lambd) * np.log(1.0 - uniform_samples * lambd * effective_dt)
        return np.asarray(emission_time, dtype=Float)

    def generate_direction(self, count: int) -> Vector3D:
        """Генерирует изотропные единичные векторы направления разлета частиц."""
        rand_u1 = self.rng.random(count)
        rand_u2 = self.rng.random(count)
        cos_alpha = 1.0 - 2.0 * rand_u1
        sin_alpha = np.sqrt(np.maximum(0.0, 1.0 - cos_alpha ** 2))
        cos_beta = sin_alpha * np.cos(2.0 * np.pi * rand_u2)
        cos_gamma = sin_alpha * np.sin(2.0 * np.pi * rand_u2)
        direction = np.column_stack((cos_alpha, cos_beta, cos_gamma))
        return np.asarray(direction, dtype=Float)

    def inject(self, bank: ParticleBank, batch_size: int, time_start: Float, time_end: Float) -> NDArray[Index]:
        """
        Генерирует частицы и инжектирует их напрямую в банк через механизм Direct Injection (SoA).
        """
        num_particles = min(batch_size, bank.capacity - int(np.count_nonzero(bank.state.is_active)))
        if num_particles <= 0:
            return np.array([], dtype=Index)

        energy = self.generate_energy(num_particles)
        direction_arr = self.generate_direction(num_particles)
        position_arr = self.generate_position(num_particles)
        emission_time = self.generate_emission_time(num_particles, time_start, time_end)

        position = Vector3DSoA(
            x=position_arr[:, 0].astype(Length),
            y=position_arr[:, 1].astype(Length),
            z=position_arr[:, 2].astype(Length),
        )
        direction = Vector3DSoA(
            x=direction_arr[:, 0].astype(Float),
            y=direction_arr[:, 1].astype(Float),
            z=direction_arr[:, 2].astype(Float),
        )

        species = np.zeros(num_particles, dtype=Species)
        distance_traveled = np.zeros(num_particles, dtype=Length)

        target_indices = bank.inject_particles(
            species=species,
            position=position,
            direction=direction,
            energy=energy,
            emission_time=emission_time,
            distance_traveled=distance_traveled,
        )
        return target_indices


class PointSource(Source):
    """
    Идеальный точечный источник излучения в формате SoA.
    По умолчанию имеет нулевой размер вокселя (size=0.0) и испускает частицы строго из начала
    своей локальной системы координат (0, 0, 0).
    """

    def __init__(
        self,
        activity: Activity,
        energy: Energy | Sequence[Any] | NDArray[Any],
        size: Length = Float(0.0),
        half_life: Time = Float(6.0 * units.hour),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        distribution = np.ones((1, 1, 1), dtype=Float)
        super().__init__(
            distribution=distribution,
            activity=activity,
            voxel_size=size,
            energy=energy,
            half_life=half_life,
            rng=rng,
        )


__all__ = [
    "Source",
    "PointSource",
]


