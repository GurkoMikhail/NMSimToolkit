import numpy as np
from typing import TypeVar, Generic, Union, overload, Any, Tuple, Optional
from numpy.typing import NDArray
from core.other.typing_definitions import Precision, Vector3D, Scalar, Energy, Time, Length, ID

class ParticleProperties:
    @property
    def type(self) -> NDArray[np.uint64]: ...
    @property
    def position(self) -> NDArray[Any]: ...
    @property
    def direction(self) -> NDArray[Any]: ...
    @property
    def energy(self) -> NDArray[Any]: ...
    @property
    def emission_time(self) -> NDArray[Any]: ...
    @property
    def emission_energy(self) -> NDArray[Any]: ...
    @property
    def emission_position(self) -> NDArray[Any]: ...
    @property
    def emission_direction(self) -> NDArray[Any]: ...
    @property
    def distance_traveled(self) -> NDArray[Any]: ...
    @property
    def ID(self) -> NDArray[np.uint64]: ...

class Particle(np.void, ParticleProperties):  # type: ignore
    @property
    def type(self) -> np.uint64: ... # type: ignore
    @property
    def position(self) -> NDArray[Any]: ...
    @property
    def direction(self) -> NDArray[Any]: ...
    @property
    def energy(self) -> float: ... # type: ignore
    @property
    def emission_time(self) -> float: ... # type: ignore
    @property
    def emission_energy(self) -> float: ... # type: ignore
    @property
    def emission_position(self) -> NDArray[Any]: ...
    @property
    def emission_direction(self) -> NDArray[Any]: ...
    @property
    def distance_traveled(self) -> float: ... # type: ignore
    @property
    def ID(self) -> np.uint64: ... # type: ignore

class ParticleArray(np.ndarray, ParticleProperties, Generic[Precision]):
    count: int
    def __new__(
        cls,
        type: NDArray[np.uint64],
        position: NDArray[Any],
        direction: NDArray[Any],
        energy: NDArray[Precision],
        emission_time: Optional[NDArray[Precision]] = None,
        emission_position: Optional[NDArray[Any]] = None,
        emission_direction: Optional[NDArray[Any]] = None,
        distance_traveled: Optional[NDArray[Precision]] = None
    ) -> 'ParticleArray[Precision]': ...

    def move(self, distance: NDArray[Precision]) -> None: ...
    def change_energy(self, delta_energy: NDArray[Precision]) -> None: ...
    def rotate(self, theta: NDArray[Precision], phi: NDArray[Precision]) -> None: ...

def get_particle_dtype(precision: Any = ...) -> np.dtype: ...

dtype_of_particle: np.dtype
