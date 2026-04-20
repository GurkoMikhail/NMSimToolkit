import numpy as np
from typing import Union, overload, Any, Tuple, Optional
from numpy.typing import NDArray
from core.other.typing_definitions import Float, Vector3D, Energy, Time, Length, ID, Species

class ParticleCore:
    species: Union[Species, NDArray[Species]]
    position: Vector3D
    direction: Vector3D
    energy: Union[Energy, NDArray[Energy]]
    emission_time: Union[Time, NDArray[Time]]
    emission_energy: Union[Energy, NDArray[Energy]]
    emission_position: Vector3D
    emission_direction: Vector3D
    distance_traveled: Union[Length, NDArray[Length]]
    ID: Union[ID, NDArray[ID]]

    def move(self, distance: Union[Length, NDArray[Length]]) -> None: ...
    def rotate(self, theta: Union[Float, NDArray[Float]], phi: Union[Float, NDArray[Float]]) -> None: ...

    @classmethod
    def get_dtype(cls) -> np.dtype: ...

class Particle(np.void, ParticleCore):
    species: Species
    position: Vector3D
    direction: Vector3D
    energy: Energy
    emission_time: Time
    emission_energy: Energy
    emission_position: Vector3D
    emission_direction: Vector3D
    distance_traveled: Length
    ID: ID

class ParticleArray(np.ndarray, ParticleCore):
    count: int

    species: NDArray[Species]
    position: Vector3D
    direction: Vector3D
    energy: NDArray[Energy]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]
    emission_position: Vector3D
    emission_direction: Vector3D
    distance_traveled: NDArray[Length]
    ID: NDArray[ID]

    def __new__(cls, shape: Union[int, Tuple[int, ...]]) -> 'ParticleArray': ...

    @classmethod
    def create(
        cls,
        species: NDArray[Species],
        position: Vector3D,
        direction: Vector3D,
        energy: NDArray[Energy],
        emission_time: Optional[NDArray[Time]] = None,
        emission_position: Optional[Vector3D] = None,
        emission_direction: Optional[Vector3D] = None,
        distance_traveled: Optional[NDArray[Length]] = None
    ) -> 'ParticleArray': ...

