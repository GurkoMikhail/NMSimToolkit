"""
Модуль классов физических процессов взаимодействия излучения с веществом (фотоэффект,
комптоновское и когерентное рассеяние, рождение пар) для SoA-архитектуры.
"""

from abc import ABC
from typing import Any, Optional

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.materials.attenuation_functions import AttenuationFunction
from core.other.typing_definitions import Float, Index, ProcessID
from core.particles.particles import ParticleBank
from core.particles.particles_kernels import update_navigation_state_rotate_kernel
from core.physics.interaction_buffers import InteractionBuffer, RNGContext
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.processes_kernels import (
    make_coherent_kernel,
    make_compton_kernel,
    make_photoelectric_kernel,
)
import settings.database_setting as settings


class Process(ABC):
    """Абстрактный базовый класс физического процесса взаимодействия частиц с веществом."""

    process_id: Optional[ProcessID] = None
    invalidates_navigation: bool = False
    rng: np.random.Generator
    _energy_range: NDArray[Float]
    attenuation_function: AttenuationFunction
    attenuation_database: Any
    _kernel: Any = None

    def __init__(
        self,
        attenuation_database: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Инициализирует физический процесс и строит функцию ослабления на сетке энергий."""
        self.attenuation_database = (
            settings.attenuation_database if attenuation_database is None else attenuation_database
        )
        self.rng = np.random.default_rng() if rng is None else rng
        self._energy_range = np.array([1.0 * units.keV, 20.0 * units.MeV], dtype=Float)
        self._construct_attenuation_function()

    def _construct_attenuation_function(self) -> None:
        self.attenuation_function = AttenuationFunction(self, self.attenuation_database)

    @property
    def name(self) -> str:
        """Строковое имя физического процесса."""
        return self.__class__.__name__

    @property
    def energy_range(self) -> NDArray[Float]:
        """Рабочий энергетический диапазон таблиц сечений процесса."""
        return self._energy_range

    @energy_range.setter
    def energy_range(self, value: NDArray[Float]) -> None:
        self._energy_range = np.asarray(value, dtype=Float)
        self._construct_attenuation_function()

    def apply(
        self,
        bank: ParticleBank,
        target_indices: NDArray[Index],
        interaction_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        material_ids: NDArray[Index],
        rng_ctx: RNGContext,
    ) -> None:
        """Применяет Numba-ядро процесса к выбранным частицам и обновляет состояние навигации."""
        self._kernel(
            bank.state,
            bank.initial_state.ID,
            target_indices,
            bank.navigation_state.current_volume,
            material_ids,
            interaction_buffer,
            physics_buffer,
            rng_ctx,
        )
        if self.invalidates_navigation:
            update_navigation_state_rotate_kernel(bank.navigation_state, target_indices)


class PhotoelectricEffect(Process):
    """Фотоэлектрическое поглощение фотонов."""

    process_id = ProcessID(0)

    def __init__(
        self,
        attenuation_database: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(attenuation_database, rng)
        self._kernel = make_photoelectric_kernel(self.process_id)


class CoherentScattering(Process):
    """Когерентное (рэлеевское) рассеяние фотонов."""

    process_id = ProcessID(2)
    invalidates_navigation = True

    def __init__(
        self,
        attenuation_database: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(attenuation_database, rng)
        self._kernel = make_coherent_kernel(self.process_id)

    def generate_phi(self, size: int) -> NDArray[Float]:
        """Генерирует азимутальный угол рассеяния phi равномерно на интервале [-pi, pi]."""
        phi = np.pi * (self.rng.random(size) * 2.0 - 1.0)
        return np.asarray(phi, dtype=Float)


class ComptonScattering(CoherentScattering):
    """Некогерентное (комптоновское) рассеяние фотонов на электронах оболочки."""

    process_id = ProcessID(1)
    invalidates_navigation = True

    def __init__(
        self,
        attenuation_database: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(attenuation_database, rng)
        self._kernel = make_compton_kernel(self.process_id)

    def calculate_energy_deposit(
        self, theta: NDArray[Float], particle_energy: NDArray[Float]
    ) -> NDArray[Float]:
        """Вычисляет потерю энергии фотона при комптоновском рассеянии на угол theta."""
        reduced_energy = particle_energy / (0.510998910 * units.MeV)
        k1_cos = reduced_energy * (1.0 - np.cos(theta))
        energy_deposit = particle_energy * k1_cos / (1.0 + k1_cos)
        return np.asarray(energy_deposit, dtype=Float)


class PairProduction(Process):
    """Образование электрон-позитронных пар в поле ядра."""

    process_id = ProcessID(3)


def get_process_mapping() -> dict[int, str]:
    """
    Возвращает словарь соответствия числового кода процесса его строковому имени,
    динамически инспектируя иерархию зарегистрированных подклассов Process.
    """
    mapping_result: dict[int, str] = {}

    def _collect_subclasses(base_class: type[Process]) -> None:
        for subclass_item in base_class.__subclasses__():
            if subclass_item.process_id is not None:
                mapping_result[int(subclass_item.process_id)] = subclass_item.__name__
            _collect_subclasses(subclass_item)

    _collect_subclasses(Process)
    return dict(sorted(mapping_result.items()))


def get_species_mapping() -> dict[int, str]:
    """
    Возвращает словарь соответствия числового кода типа частицы ее строковому имени.
    """
    return {
        0: "Photon",
        1: "Electron",
        2: "Positron",
    }


__all__ = [
    "Process",
    "PhotoelectricEffect",
    "CoherentScattering",
    "ComptonScattering",
    "PairProduction",
    "get_process_mapping",
    "get_species_mapping",
]

