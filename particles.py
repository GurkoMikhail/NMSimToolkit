import numpy as np
from numpy import cos, sin, abs


class ParticleProperties:
    """ Класс свойств частицы """

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
    def emissionTime(self):
        """ Время эмиссии частиц """
        return self['emissionTime']

    @property
    def emissionPosition(self):
        """ Положение эмиссии частиц """
        return self['emissionPosition']

    @property
    def distanceTraveled(self):
        """ Пройденное расстояние частицами """
        return self['distanceTraveled']
    
    @property
    def ID(self):
        """ Идентификатор частицы """
        return self['ID']


class Particle(np.void, ParticleProperties):
    """ Класс частицы """


class Particles(np.ndarray, ParticleProperties):
    """ 
    Класс массива частиц

    [position] = mm\n
    [direction] = mm\n
    [energy] = MeV\n
    [emissionTime] = ns\n
    [emissionPosition] = mm\n
    [distanceTraveled] = mm\n
    """

    processes = []
    count = 0

    def __new__(subtype, position, direction, energy, emissionTime=None, emissionPosition=None, distanceTraveled=None):
        obj = super().__new__(subtype, shape=energy.size, dtype=(Particle, dtypeOfParticle))
        obj['position'] = position
        obj['direction'] = direction
        obj['energy'] = energy
        obj['emissionTime'] = 0 if emissionTime is None else emissionTime
        obj['emissionPosition'] = position if emissionPosition is None else emissionPosition
        obj['distanceTraveled'] = 0 if distanceTraveled is None else distanceTraveled
        obj['ID'] = subtype.__getID(obj.size)
        return obj

    @classmethod
    def __getID(cls, n):
        ID = np.arange(cls.count, cls.count + n, dtype='uint')
        cls.count += n
        return ID

    def move(self, distance):
        """ Переместить частицы """
        self['distanceTraveled'] += distance
        self['position'] += self.direction*distance.reshape((-1, 1))

    def changeEnergy(self, deltaEnergy):
        self['energy'] -= deltaEnergy

    def rotate(self, theta, phi):
        """
        Повернуть направления частиц

        [theta] = radian\n
        [phi] = radian
        """
        direction = self['direction']
        cosTheta = cos(theta)
        sinTheta = sin(theta)
        delta1 = sinTheta*cos(phi)
        delta2 = sinTheta*sin(phi)
        delta = np.ones_like(cosTheta) - 2*(direction[:, 2] < 0)
        b = direction[:, 0]*delta1 + direction[:, 1]*delta2
        tmp = cosTheta - b/(1 + abs(direction[:, 2]))
        direction[:, 0] = direction[:, 0]*tmp + delta1
        direction[:, 1] = direction[:, 1]*tmp + delta2
        direction[:, 2] = direction[:, 2]*cosTheta - delta*b


class Photons(Particles):
    """ 
    Класс массива частиц

    [position] = mm\n
    [direction] = mm\n
    [energy] = MeV\n
    [emissionTime] = ns\n
    [emissionPosition] = mm\n
    [distanceTraveled] = mm
    """

    processes = ['PhotoelectricEffect', 'ComptonScattering', 'CoherentScattering']


dtypeOfParticle = np.dtype([
    ('position', '3d'),
    ('direction', '3d'),
    ('energy', 'd'),
    ('emissionTime', 'd'),
    ('emissionPosition', '3d'),
    ('distanceTraveled', 'd'),
    ('ID', 'u8')
])

