import core.physics.g4compton as g4compton
import core.physics.g4coherent as g4coherent
from core.materials.attenuation_functions import AttenuationFunction
import settings.database_setting as settings
from abc import ABC
import numpy as np
from numpy import pi, cos
from hepunits import*


from typing import Optional, Any, Union, Tuple, cast
from numpy.typing import NDArray
from core.particles.particles import Particle, ParticleArray
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Float
from core.data.interaction_data import InteractionArray


class Process(ABC):
    """ Класс процесса """
    rng: np.random.Generator
    _energy_range: NDArray[np.float64] # type: ignore
    attenuation_function: AttenuationFunction
    attenuation_database: Any

    def __init__(self, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None:
        """ Конструктор процесса """
        self.attenuation_database = settings.attenuation_database if attenuation_database is None else attenuation_database
        self.rng = np.random.default_rng() if rng is None else rng
        self._energy_range = np.array([1*keV, 1*MeV])
        self._construct_attenuation_function()

    def _construct_attenuation_function(self):
        self.attenuation_function = AttenuationFunction(self, self.attenuation_database)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def energy_range(self) -> np.ndarray:
        return self._energy_range

    @energy_range.setter
    def energy_range(self, value: np.ndarray) -> None:
        self._energy_range = value
        self._construct_attenuation_function()

    def get_LAC(self, particle: ParticleArray, material: Union[Material, MaterialArray]) -> Any: # type: ignore
        energy = particle.energy
        LAC = self.attenuation_function(material, energy)
        return LAC

    def generate_free_path(self, particle: ParticleArray, material: Union[Material, MaterialArray]) -> Any: # type: ignore
        LAC = self.get_LAC(particle, material)
        freePath = self.rng.exponential(1/LAC)
        return freePath

    def __call__(self, particle: ParticleArray, material: Union[Material, MaterialArray]) -> InteractionArray: # type: ignore
        """ Применить процесс """
        size = particle.size
        # Определение точности
        precision = particle.energy.dtype.type
        interaction_data = cast(InteractionArray, InteractionArray(size, precision=precision))
        interaction_data.position = particle.position
        interaction_data.direction = particle.direction
        interaction_data.process_name = self.name
        interaction_data.particle_ID = particle.ID
        interaction_data.energy_deposit = 0.
        interaction_data.scattering_angles = 0.
        interaction_data.emission_time = particle.emission_time
        interaction_data.emission_energy = particle.emission_energy
        interaction_data.emission_position = particle.emission_position
        interaction_data.emission_direction = particle.emission_direction
        interaction_data.distance_traveled = particle.distance_traveled
        return interaction_data


class PhotoelectricEffect(Process):
    """ Класс фотоэффекта """

    def __call__(self, particle: ParticleArray, material: Union[Material, MaterialArray]) -> InteractionArray: # type: ignore
        """ Применить фотоэффект """
        interaction_data = cast(InteractionArray, super().__call__(particle, material)) # type: ignore
        energy_deposit = particle.energy
        particle.change_energy(energy_deposit)
        interaction_data.energy_deposit = energy_deposit
        return interaction_data
        

class CoherentScattering(Process):
    """ Класс когерентного рассеяния """
    
    def __init__(self, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None:
        Process.__init__(self, attenuation_database, rng)                
        self.theta_generator = g4coherent.initialize(self.rng)

    def generate_theta(self, particle: ParticleArray, material: Union[Material, MaterialArray]) -> Any: # type: ignore
        """ Сгенерировать угол рассеяния - theta """
        energy = particle.energy
        Z = np.array(material.Zeff, dtype=int)
        theta = self.theta_generator(energy, Z)
        return theta

    def generate_phi(self, size):
        """ Сгенерировать угол рассеяния - phi """
        phi = pi*(self.rng.random(size)*2 - 1)
        return phi

    def __call__(self, particle: ParticleArray, material: Union[Material, MaterialArray]) -> InteractionArray: # type: ignore
        """ Применить когерентное рассеяние """
        size = particle.size
        theta = self.generate_theta(particle, material)
        phi = self.generate_phi(size)
        particle.rotate(theta, phi)
        interaction_data = cast(InteractionArray, super().__call__(particle, material)) # type: ignore
        interaction_data.scattering_angles = np.column_stack((theta, phi))
        return interaction_data


class ComptonScattering(CoherentScattering):
    """ Класс эффекта Комптона """

    def __init__(self, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None:
        Process.__init__(self, attenuation_database, rng)
        self.theta_generator = g4compton.initialize(self.rng)

    def culculate_energy_deposit(self, theta, particle_energy):
        """ Вычислить изменения энергий """
        k = particle_energy/0.510998910*MeV
        k1_cos = k*(1 - cos(theta))
        energy_deposit = particle_energy*k1_cos/(1 + k1_cos)
        return energy_deposit

    def __call__(self, particle: ParticleArray, material: Union[Material, MaterialArray]) -> InteractionArray: # type: ignore
        """ Применить эффект Комптона """
        interaction_data = cast(InteractionArray, super().__call__(particle, material)) # type: ignore
        theta = interaction_data.scattering_angles[:, 0]
        energy_deposit = self.culculate_energy_deposit(theta, particle.energy)
        particle.change_energy(energy_deposit)
        interaction_data.energy_deposit = energy_deposit
        return interaction_data


class PairProduction(Process):
    """ Класс эффекта образования электрон-позитронных пар """
