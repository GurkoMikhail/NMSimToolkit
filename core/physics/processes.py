from core.physics.physics_buffer import PhysicsBuffer

from core.physics.processes_kernels import make_photoelectric_kernel, make_compton_kernel, make_coherent_kernel
from core.particles.particles_kernels import update_navigation_state_rotate_kernel
from abc import ABC
from typing import Any, Optional, Union

import numpy as np
import hepunits as units
from numpy.typing import NDArray

import settings.database_setting as settings
from core.materials.attenuation_functions import AttenuationFunction
from core.other.typing_definitions import Float, ProcessID
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import InteractionBuffer, RNGContext
from core.other.typing_definitions import Index

class Process(ABC):
    """ Класс процесса """
    process_id: ProcessID
    invalidates_navigation: bool = False
    rng: np.random.Generator
    _energy_range: NDArray[Float]
    attenuation_function: AttenuationFunction
    attenuation_database: Optional[Any]

    def __init__(self, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None:
        """ Конструктор процесса """
        self.attenuation_database = settings.attenuation_database if attenuation_database is None else attenuation_database
        self.rng = np.random.default_rng() if rng is None else rng
        self._energy_range = np.array([1*units.keV, 1*units.MeV])
        self._construct_attenuation_function()

    def _construct_attenuation_function(self):
        self.attenuation_function = AttenuationFunction(self, self.attenuation_database)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def energy_range(self) -> NDArray[Float]:
        return self._energy_range

    @energy_range.setter
    def energy_range(self, value: NDArray[Float]) -> None:
        self._energy_range = value
        self._construct_attenuation_function()

    def apply(self, bank: ParticleBank, target_indices: NDArray[Index], interaction_buffer: InteractionBuffer, physics_buffer: PhysicsBuffer, material_ids: NDArray[Index], rng_ctx: RNGContext) -> None:
        self._kernel(bank.state, bank.initial_state.ID, target_indices, bank.navigation_state.current_volume, material_ids, interaction_buffer, physics_buffer, rng_ctx)
        if self.invalidates_navigation:
            update_navigation_state_rotate_kernel(bank.navigation_state, target_indices)

class PhotoelectricEffect(Process):
    """ Класс фотоэффекта """
    process_id = ProcessID(0)

    def __init__(self, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(attenuation_database, rng)
        self._kernel = make_photoelectric_kernel(self.process_id)

class CoherentScattering(Process):
    """ Класс когерентного рассеяния """
    process_id = ProcessID(2)
    invalidates_navigation = True
    
    def __init__(self, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None:
        Process.__init__(self, attenuation_database, rng)                
        self._kernel = make_coherent_kernel(self.process_id)

    def generate_phi(self, size: int) -> NDArray[Float]:
        """ Сгенерировать угол рассеяния - phi """
        phi = np.pi * (self.rng.random(size) * 2 - 1)
        return phi

class ComptonScattering(CoherentScattering):
    """ Класс эффекта Комптона """
    process_id = ProcessID(1)
    invalidates_navigation = True

    def __init__(self, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None:
        Process.__init__(self, attenuation_database, rng)
        self._kernel = make_compton_kernel(self.process_id)

    def culculate_energy_deposit(self, theta: NDArray[Float], particle_energy: NDArray[Float]) -> NDArray[Float]:
        """ Вычислить изменения энергий """
        k = particle_energy / (0.510998910 * units.MeV)
        k1_cos = k * (1 - np.cos(theta))
        energy_deposit = particle_energy * k1_cos / (1 + k1_cos)
        return energy_deposit

class PairProduction(Process):
    """ Класс эффекта образования электрон-позитронных пар """
