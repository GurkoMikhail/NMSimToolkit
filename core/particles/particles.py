from typing import Any, Optional, Union, cast, Tuple

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Vector3D, Species


class ParticleCore:
    """ Базовый класс свойств частицы, обеспечивающий доступ к полям структурированного массива и методы для работы с ним """

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

    def __getattr__(self, name: str) -> Any:
        try:
            return cast(Any, self)[name]
        except ValueError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        dtype = getattr(self, 'dtype', None)
        if dtype is not None and name in dtype.names:
            cast(Any, self)[name] = value
        else:
            super().__setattr__(name, value)

    def move(self, distance: Union[Length, NDArray[Length]]) -> None:
        """ Переместить частицы (работает для скаляра и массива) """
        dist_arr = np.asarray(distance)
        self.distance_traveled += dist_arr

        # Для скаляра: direction [3] * dist_arr [] -> [3]
        # Для массива: direction [N, 3] * dist_arr [N, 1] -> [N, 3]
        # [..., np.newaxis] добавит ось только если dist_arr не скаляр:
        # но если dist_arr - скаляр (shape=()), то dist_arr[..., np.newaxis] будет иметь shape=(1,).
        # direction [3] * (1,) -> [3].
        # Это работает корректно в обоих случаях!
        self.position += self.direction * dist_arr[..., np.newaxis]

    def rotate(self, theta: Union[Float, NDArray[Float]], phi: Union[Float, NDArray[Float]]) -> None:
        """ Повернуть направления частиц (dimension-agnostic) """
        direction = cast(NDArray[Float], self.direction)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        delta1 = sin_theta * np.cos(phi)
        delta2 = sin_theta * np.sin(phi)

        # Используем `...` для индексации
        delta = np.ones_like(cos_theta) - 2 * (direction[..., 2] < 0)
        b = direction[..., 0] * delta1 + direction[..., 1] * delta2
        tmp = cos_theta - b / (1 + np.abs(direction[..., 2]))

        direction[..., 0] = direction[..., 0] * tmp + delta1
        direction[..., 1] = direction[..., 1] * tmp + delta2
        direction[..., 2] = direction[..., 2] * cos_theta - delta * b

        self.direction = direction


    @classmethod
    def get_dtype(cls) -> np.dtype:
        """ Генерирует dtype для частиц """
        return np.dtype([
            ('species', Species),
            ('position', (Length, 3)),
            ('direction', (Length, 3)),
            ('energy', Energy),
            ('emission_time', Time),
            ('emission_energy', Energy),
            ('emission_position', (Length, 3)),
            ('emission_direction', (Length, 3)),
            ('distance_traveled', Length),
            ('ID', ID)
        ])


class Particle(np.void, ParticleCore):
    """ Класс одиночной частицы (элемент структурированного массива) """
    pass

class ParticleArray(np.ndarray, ParticleCore):
    """ 
    Класс массива частиц
    """

    count: int = 0

    def __new__(cls, shape: Union[int, Tuple[int, ...]]) -> 'ParticleArray':
        current_dtype = ParticleCore.get_dtype()
        obj = super().__new__(cls, shape=shape, dtype=(Particle, current_dtype))
        return obj

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
    ) -> 'ParticleArray':

        obj = cls(shape=energy.size)

        obj['species'] = species
        obj['position'] = position
        obj['direction'] = direction
        obj['energy'] = energy
        obj['emission_time'] = 0 if emission_time is None else emission_time
        obj['emission_energy'] = energy
        obj['emission_position'] = position if emission_position is None else emission_position
        obj['emission_direction'] = direction if emission_direction is None else emission_direction
        obj['distance_traveled'] = 0 if distance_traveled is None else distance_traveled
        obj['ID'] = cls.__get_ID(obj.size)
        return obj

    @classmethod
    def __get_ID(cls, n: int) -> NDArray[ID]:
        ID_vals = np.arange(cls.count, cls.count + n, dtype=ID)
        cls.count += n
        return ID_vals
