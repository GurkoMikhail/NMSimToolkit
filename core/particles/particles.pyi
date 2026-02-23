import numpy as np
from typing import Union, overload, Any, Tuple, Optional
from numpy.typing import NDArray
from core.other.typing_definitions import Float, Vector3D, Energy, Time, Length, ID

class ParticleProperties:
    @property
    def type(self) -> NDArray[np.uint64]: ...
    @property
    def position(self) -> Vector3D: ...
    @property
    def direction(self) -> Vector3D: ...
    @property
    def energy(self) -> NDArray[Float]: ...
    @property
    def emission_time(self) -> NDArray[Float]: ...
    @property
    def emission_energy(self) -> NDArray[Float]: ...
    @property
    def emission_position(self) -> Vector3D: ...
    @property
    def emission_direction(self) -> Vector3D: ...
    @property
    def distance_traveled(self) -> NDArray[Float]: ...
    @property
    def ID(self) -> NDArray[ID]: ...

class Particle(np.void, ParticleProperties): # type: ignore
    @property
    def type(self) -> np.uint64: ... # type: ignore
    @property
    def position(self) -> Vector3D: ... # type: ignore
    @property
    def direction(self) -> Vector3D: ... # type: ignore
    @property
    def energy(self) -> float: ... # type: ignore
    @property
    def emission_time(self) -> float: ... # type: ignore
    @property
    def emission_energy(self) -> float: ... # type: ignore
    @property
    def emission_position(self) -> Vector3D: ... # type: ignore
    @property
    def emission_direction(self) -> Vector3D: ... # type: ignore
    @property
    def distance_traveled(self) -> float: ... # type: ignore
    @property
    def ID(self) -> ID: ... # type: ignore

class ParticleArray(np.ndarray, ParticleProperties):
    count: int
    def __new__(
        cls,
        type: NDArray[np.uint64],
        position: Vector3D,
        direction: Vector3D,
        energy: NDArray[Float],
        emission_time: Optional[NDArray[Float]] = None,
        emission_position: Optional[Vector3D] = None,
        emission_direction: Optional[Vector3D] = None,
        distance_traveled: Optional[NDArray[Float]] = None
    ) -> 'ParticleArray': ...

    def move(self, distance: NDArray[Float]) -> None: ...
    def change_energy(self, delta_energy: NDArray[Float]) -> None: ...
    def rotate(self, theta: NDArray[Float], phi: NDArray[Float]) -> None: ...

def get_particle_dtype(precision: Any = ...) -> np.dtype: ...

dtype_of_particle: np.dtype
