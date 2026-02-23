from abc import ABC
import numpy as np
from numpy import cos, sin, abs
from typing import Optional, Union, Any, cast
from numpy.typing import NDArray
from core.other.typing_definitions import Precision, Scalar, Vector3D, Length, Energy, Time

# Define internal precision for default operations from typing_definitions
from core.other.typing_definitions import Precision as DEFAULT_PRECISION

class ParticleProperties(ABC):
    """ Базовый класс свойств частицы, обеспечивающий доступ к полям структурированного массива """

    def __getitem__(self, key: str) -> Any:
        # Это будет реализовано в подклассах np.void или np.ndarray
        raise NotImplementedError

    @property
    def type(self) -> NDArray[np.uint64]:
        """ Тип частиц """
        return self['type'].copy()

    @property
    def position(self) -> Vector3D:
        """ Положение частиц """
        return self['position'].copy()

    @property
    def direction(self) -> Vector3D:
        """ Направление частиц """
        return self['direction'].copy()

    @property
    def energy(self) -> NDArray[Precision]:
        """ Энергия частиц """
        return self['energy'].copy()

    @property
    def emission_time(self) -> NDArray[Precision]:
        """ Время эмиссии частиц """
        return self['emission_time'].copy()

    @property
    def emission_energy(self) -> NDArray[Precision]:
        """ Энергия частицы при эмиссии """
        return self['emission_energy'].copy()

    @property
    def emission_position(self) -> Vector3D:
        """ Положение эмиссии частиц """
        return self['emission_position'].copy()

    @property
    def emission_direction(self) -> Vector3D:
        """ Направление эмиссии частиц """
        return self['emission_direction'].copy()

    @property
    def distance_traveled(self) -> NDArray[Precision]:
        """ Пройденное расстояние частицами """
        return self['distance_traveled'].copy()
    
    @property
    def ID(self) -> NDArray[np.uint64]:
        """ Идентификатор частицы """
        return self['ID'].copy()


class Particle(np.void, ParticleProperties): # type: ignore
    """ Класс одиночной частицы (элемент структурированного массива) """
    pass


def get_particle_dtype(precision: Any = DEFAULT_PRECISION) -> np.dtype:
    """ Генерирует dtype для частиц с заданной точностью """
    p_char = 'd' if precision == np.float64 else 'f'
    return np.dtype([
        ('type', 'u8'),
        ('position', f'3{p_char}'),
        ('direction', f'3{p_char}'),
        ('energy', p_char),
        ('emission_time', p_char),
        ('emission_energy', p_char),
        ('emission_position', f'3{p_char}'),
        ('emission_direction', f'3{p_char}'),
        ('distance_traveled', p_char),
        ('ID', 'u8')
    ])

dtype_of_particle = get_particle_dtype(DEFAULT_PRECISION)

class ParticleArray(np.ndarray, ParticleProperties):
    """ 
    Класс массива частиц
    """

    count: int = 0

    def __new__(
        cls,
        type: NDArray[np.uint64],
        position: Vector3D,
        direction: Vector3D,
        energy: NDArray[Precision],
        emission_time: Optional[NDArray[Precision]] = None,
        emission_position: Optional[Vector3D] = None,
        emission_direction: Optional[Vector3D] = None,
        distance_traveled: Optional[NDArray[Precision]] = None
    ) -> 'ParticleArray':

        # Определение точности на основе конфигурации проекта
        current_dtype = get_particle_dtype(DEFAULT_PRECISION)

        obj = super().__new__(cls, shape=energy.size, dtype=(Particle, current_dtype))
        obj = cast('ParticleArray', obj)

        obj['type'] = type
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
    def __get_ID(cls, n: int) -> NDArray[np.uint64]:
        ID_vals = np.arange(cls.count, cls.count + n, dtype='uint64')
        cls.count += n
        return ID_vals

    def move(self, distance: NDArray[Precision]) -> None:
        """ Переместить частицы """
        self['distance_traveled'] += distance
        self['position'] += self.direction * distance.reshape((-1, 1))

    def change_energy(self, delta_energy: NDArray[Precision]) -> None:
        """ Изменить энергию частиц """
        self['energy'] -= delta_energy

    def rotate(self, theta: NDArray[Precision], phi: NDArray[Precision]) -> None:
        """
        Повернуть направления частиц
        """
        direction = self['direction']
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        delta1 = sin_theta * cos(phi)
        delta2 = sin_theta * sin(phi)
        delta = np.ones_like(cos_theta) - 2 * (direction[:, 2] < 0)
        b = direction[:, 0] * delta1 + direction[:, 1] * delta2
        tmp = cos_theta - b / (1 + abs(direction[:, 2]))
        direction[:, 0] = direction[:, 0] * tmp + delta1
        direction[:, 1] = direction[:, 1] * tmp + delta2
        direction[:, 2] = direction[:, 2] * cos_theta - delta * b
