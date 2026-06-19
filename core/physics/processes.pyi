import numpy as np
from abc import ABC
from typing import Optional, Any
from numpy.typing import NDArray
from core.materials.attenuation_functions import AttenuationFunction
from core.other.typing_definitions import Float

class Process(ABC):
    rng: np.random.Generator
    _energy_range: NDArray[Float]
    attenuation_function: AttenuationFunction
    def __init__(self, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def energy_range(self) -> NDArray[Float]: ...
    @energy_range.setter
    def energy_range(self, value: NDArray[Float]) -> None: ...

class PhotoelectricEffect(Process): ...

class CoherentScattering(Process):
    theta_generator: Any
    def generate_phi(self, size: int) -> NDArray[Float]: ...

class ComptonScattering(CoherentScattering):
    def culculate_energy_deposit(self, theta: NDArray[Float], particle_energy: NDArray[Float]) -> NDArray[Float]: ...

class PairProduction(Process): ...

process_data_dtype: np.dtype
