import numpy as np
from typing import Union, overload, Any, Tuple, Optional
from numpy.typing import NDArray
from core.other.typing_definitions import Float, Vector3D, Energy, Time, Length, ID

class ParticleCore:
    species: Union[np.uint64, NDArray[np.uint64]]
    position: Union[Vector3D, NDArray[Float]]
    direction: Union[Vector3D, NDArray[Float]]
    energy: Union[Float, NDArray[Float]]
    emission_time: Union[Float, NDArray[Float]]
    emission_energy: Union[Float, NDArray[Float]]
    emission_position: Union[Vector3D, NDArray[Float]]
    emission_direction: Union[Vector3D, NDArray[Float]]
    distance_traveled: Union[Float, NDArray[Float]]
    ID: Union[ID, NDArray[ID]]

    def move(self, distance: Union[Float, NDArray[Float]]) -> None: ...
    def rotate(self, theta: Union[Float, NDArray[Float]], phi: Union[Float, NDArray[Float]]) -> None: ...

class Particle(np.void, ParticleCore):
    species: np.uint64
    position: Vector3D
    direction: Vector3D
    energy: Float
    emission_time: Float
    emission_energy: Float
    emission_position: Vector3D
    emission_direction: Vector3D
    distance_traveled: Float
    ID: ID

class ParticleArray(np.ndarray, ParticleCore):
    count: int

    species: NDArray[np.uint64]
    position: NDArray[Float]
    direction: NDArray[Float]
    energy: NDArray[Float]
    emission_time: NDArray[Float]
    emission_energy: NDArray[Float]
    emission_position: NDArray[Float]
    emission_direction: NDArray[Float]
    distance_traveled: NDArray[Float]
    ID: NDArray[ID]

    def __new__(cls, shape: Union[int, Tuple[int, ...]]) -> 'ParticleArray': ...

    @classmethod
    def create(
        cls,
        species: NDArray[np.uint64],
        position: Vector3D,
        direction: Vector3D,
        energy: NDArray[Float],
        emission_time: Optional[NDArray[Float]] = None,
        emission_position: Optional[Vector3D] = None,
        emission_direction: Optional[Vector3D] = None,
        distance_traveled: Optional[NDArray[Float]] = None
    ) -> 'ParticleArray': ...

def get_particle_dtype() -> np.dtype: ...

dtype_of_particle: np.dtype
