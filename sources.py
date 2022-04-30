import numpy as np
from numpy import load, cos, sin, log, sqrt
from particles import Photons
from hepunits import*


class Source:
    """
    Класс источника частиц

    [position = (x, y, z)] = cm

    [activity] = Bq

    [distribution] = float[:,:,:]

    [voxelSize] = cm

    [energy] = eV

    [halfLife] = sec
    """

    def __init__(self, position, activity, distribution, voxelSize=0.4, radiationType='Gamma', energy=140.*10**3, halfLife=6*60*60, rotationAngles=None, rotationCenter=None):
        self.position = np.asarray(position)
        self.initialActivity = np.asarray(activity)
        self.distribution = np.asarray(distribution)
        self.distribution /= np.sum(self.distribution)
        self.voxelSize = voxelSize
        self.size = np.asarray(self.distribution.shape)*self.voxelSize
        self.radiationType = radiationType
        self.energy = energy
        self.halfLife = halfLife
        self.timer = 0.
        self._rotated = False
        self._generateEmissionTable()
        self.rotate(rotationAngles, rotationCenter)
        self.rng = np.random.default_rng()

    def rotate(self, rotationAngles, rotationCenter=None):
        if rotationAngles is not None:
            self._rotated = True
        else:
            rotationAngles = [0., 0., 0.]
        self.rotationAngles = np.asarray(rotationAngles)
        if rotationCenter is None:
            rotationCenter = np.asarray(self.size/2)
        self.rotationCenter = rotationCenter
        alpha, beta, gamma = -self.rotationAngles
        R = np.asarray([
            [cos(alpha)*cos(beta),  cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma),    cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma) ],
            [sin(alpha)*cos(beta),  sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma),    sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma) ],
            [-sin(beta),            cos(beta)*sin(gamma),                                       cos(beta)*cos(gamma)                                    ]
        ])
        self.R = R.T
        self._generateEmissionTable()

    def _generateEmissionTable(self):
        xs, ys, zs = np.meshgrid(
            np.linspace(0, self.size[0], self.distribution.shape[0], endpoint=False),
            np.linspace(0, self.size[1], self.distribution.shape[1], endpoint=False),
            np.linspace(0, self.size[2], self.distribution.shape[2], endpoint=False),
            indexing = 'ij'
        )
        position = np.stack((xs, ys, zs), axis=3).reshape(-1, 3)
        probability = self.distribution.ravel()
        indices = probability.nonzero()[0]
        self.emission_table = [position[indices], probability[indices]]

    @property
    def activity(self):
        return self.initialActivity*2**(-self.timer/self.halfLife)
    
    @property
    def nucleiNumber(self):
        return self.activity*self.halfLife/log(2)

    def set_state(self, timer, rng_state=None):
        if timer is not None:
            self.timer = timer
        if rng_state is None:
            return
        self.rng.bit_generator.state['state'] = rng_state

    def generatePosition(self, n):
        position = self.emission_table[0]
        probability = self.emission_table[1]
        position = self.rng.choice(position, n, p=probability)
        position += self.rng.uniform(0., self.voxelSize, position.shape)
        if self._rotated:
            position -= self.rotationCenter
            np.matmul(position, self.R, out=position)
            position += self.rotationCenter
        position += self.position
        return position

    def generateEmissionTime(self, n):
        dt = log((self.nucleiNumber + n)/self.nucleiNumber)*self.halfLife/log(2)
        a = 2**(-self.timer/self.halfLife)
        b = 2**(-(self.timer + dt)/self.halfLife)
        alpha = self.rng.uniform(b, a, n)
        emissionTime = -log(alpha)*self.halfLife/log(2)
        return emissionTime, dt

    def generateDirection(self, n):
        a1 = self.rng.random(n)
        a2 = self.rng.random(n)
        cosAlpha = 1 - 2*a1
        sq = sqrt(1 - cosAlpha**2)
        cosBeta = sq*cos(2*pi*a2)
        cosGamma = sq*sin(2*pi*a2)
        direction = np.column_stack((cosAlpha, cosBeta, cosGamma))
        return direction

    def generateParticles(self, n):
        energy = np.full(n, self.energy)
        direction = self.generateDirection(n)
        position = self.generatePosition(n)
        emissionTime, dt = self.generateEmissionTime(n)
        self.timer += dt
        particles = Photons(position, direction, energy, emissionTime)
        return particles


class PointSource(Source):
    """
    Источник 99mТс-MIBI

    [position = (x, y, z)] = cm

    [activity] = Bq
    
    [energy] = eV
    """

    def __init__(self, position, activity, energy):
        distribution = [[[1.]]]
        voxelSize = 1.*mm
        radiationType = 'Gamma'
        halfLife = 6.*hour
        rotationAngles = None
        rotationCenter = None
        super().__init__(position, activity, distribution, voxelSize, radiationType, energy, halfLife, rotationAngles, rotationCenter)

    def generatePosition(self, n):
        return np.full((n, 3), self.position)

class Тс99m_MIBI(Source):
    """
    Источник 99mТс-MIBI

    [position = (x, y, z)] = cm

    [activity] = Bq

    [distribution] = float[:,:,:]

    [voxelSize] = cm
    """

    def __init__(self, position, activity, distribution, voxelSize, rotationAngles=None, rotationCenter=None):
        radiationType = 'Gamma'
        energy = 140.5*keV
        halfLife = 6.*hour
        super().__init__(position, activity, distribution, voxelSize, radiationType, energy, halfLife, rotationAngles, rotationCenter)


class SourcePhantom(Тс99m_MIBI):
    """
    Источник 99mТс-MIBI

    [position = (x, y, z)] = cm

    [activity] = Bq

    [phantom_name] = string

    [voxelSize] = cm
    """

    def __init__(self, position, activity, phantom_name, voxelSize, rotationAngles=None, rotationCenter=None):
        distribution = load(f'Phantoms/{phantom_name}.npy')
        super().__init__(position, activity, distribution, voxelSize, rotationAngles=rotationAngles, rotationCenter=rotationCenter)


class efg3(SourcePhantom):
    """
    Источник efg3

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, position, activity, rotationAngles=None, rotationCenter=None):
        phantom_name = 'efg3'
        voxelSize = 4.*mm
        super().__init__(position, activity, phantom_name, voxelSize, rotationAngles, rotationCenter)
        

class efg3cut(SourcePhantom):
    """
    Источник efg3cut

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, position, activity, rotationAngles=None, rotationCenter=None):
        phantom_name = 'efg3cut'
        voxelSize = 4.*mm
        super().__init__(position, activity, phantom_name, voxelSize, rotationAngles, rotationCenter)


class efg3cutDefect(SourcePhantom):
    """
    Источник efg3cutDefect

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, position, activity, rotationAngles=None, rotationCenter=None):
        phantom_name = 'efg3cutDefect'
        voxelSize = 4.*mm
        super().__init__(position, activity, phantom_name, voxelSize, rotationAngles, rotationCenter)

