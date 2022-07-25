from abc import ABC
import numpy as np
from numpy import cos, sin, abs


class ParticleProperties(ABC):
    """ Класс свойств частицы """


    @property
    def type(self):
        """ Тип частиц """
        return self['type']

    @property
    def position(self):
        """ Положение частиц """
        return self['position']

    @property
    def direction(self):
        """ Направление частиц """
        return self['direction']

    @property
    def energy(self):
        """ Энергия частиц """
        return self['energy']

    @property
    def emission_time(self):
        """ Время эмиссии частиц """
        return self['emission_time']

    @property
    def emission_position(self):
        """ Положение эмиссии частиц """
        return self['emission_position']

    @property
    def distance_traveled(self):
        """ Пройденное расстояние частицами """
        return self['distance_traveled']
    
    @property
    def ID(self):
        """ Идентификатор частицы """
        return self['ID']


class Particle(np.void, ParticleProperties):
    """ Класс частицы """


class ParticleArray(np.ndarray, ParticleProperties):
    """ 
    Класс массива частиц

    [type] = uint
    [position] = mm\n
    [direction] = mm\n
    [energy] = MeV\n
    [emission_time] = ns\n
    [emission_position] = mm\n
    [distance_traveled] = mm\n
    """

    count = 0

    def __new__(subtype, type, position, direction, energy, emission_time=None, emission_position=None, distance_traveled=None):
        obj = super().__new__(subtype, shape=energy.size, dtype=(Particle, dtype_of_particle))
        obj['type'] = type
        obj['position'] = position
        obj['direction'] = direction
        obj['energy'] = energy
        obj['emission_time'] = 0 if emission_time is None else emission_time
        obj['emission_position'] = position if emission_position is None else emission_position
        obj['distance_traveled'] = 0 if distance_traveled is None else distance_traveled
        obj['ID'] = subtype.__get_ID(obj.size)
        return obj

    @classmethod
    def __get_ID(cls, n):
        ID = np.arange(cls.count, cls.count + n, dtype='uint')
        cls.count += n
        return ID

    def move(self, distance):
        """ Переместить частицы """
        self['distance_traveled'] += distance
        self['position'] += self.direction*distance.reshape((-1, 1))

    def change_energy(self, delta_energy):
        self['energy'] -= delta_energy

    def rotate(self, theta, phi):
        """
        Повернуть направления частиц

        [theta] = radian\n
        [phi] = radian
        """
        direction = self['direction']
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        delta1 = sin_theta*cos(phi)
        delta2 = sin_theta*sin(phi)
        delta = np.ones_like(cos_theta) - 2*(direction[:, 2] < 0)
        b = direction[:, 0]*delta1 + direction[:, 1]*delta2
        tmp = cos_theta - b/(1 + abs(direction[:, 2]))
        direction[:, 0] = direction[:, 0]*tmp + delta1
        direction[:, 1] = direction[:, 1]*tmp + delta2
        direction[:, 2] = direction[:, 2]*cos_theta - delta*b


dtype_of_particle = np.dtype([
    ('type', 'u8'),
    ('position', '3d'),
    ('direction', '3d'),
    ('energy', 'd'),
    ('emission_time', 'd'),
    ('emission_position', '3d'),
    ('distance_traveled', 'd'),
    ('ID', 'u8')
])

