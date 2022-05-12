import numpy as np
from numpy import load, cos, sin, log, sqrt, matmul
import utils
from particles import Photons
from hepunits import*


class Source:
    """
    Класс источника частиц

    [activity] = Bq

    [distribution] = float[:,:,:]

    [voxelSize] = cm

    [energy] = eV

    [halfLife] = sec
    """

    def __init__(self, distribution, activity=None, voxelSize=4*mm, radiationType='Gamma', energy=140.5*keV, halfLife=6*hour):
        self.initialActivity = np.sum(self.distribution) if activity is None else np.asarray(activity)
        self.distribution = np.asarray(distribution)
        self.distribution /= np.sum(self.distribution)
        self.voxelSize = voxelSize
        self.size = np.asarray(self.distribution.shape)*self.voxelSize
        self.radiationType = radiationType
        self.energy = energy
        self.halfLife = halfLife
        self.timer = 0.
        self._generateEmissionTable()
        self.transformationMatrix = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        self.rng = np.random.default_rng()

    def translate(self, x=0., y=0., z=0., inLocal=False):
        """ Переместить объём """
        translation = np.asarray([x, y, z])
        translationMatrix = utils.computeTranslationMatrix(translation)
        if inLocal:
            self.transformationMatrix = self.transformationMatrix@translationMatrix
        else:
            self.transformationMatrix = translationMatrix@self.transformationMatrix

    def rotate(self, alpha=0., beta=0., gamma=0., rotationCenter=[0., 0., 0.], inLocal=False):
        """ Повернуть объём вокруг координатных осей """
        rotationAngles = np.asarray([alpha, beta, gamma])
        rotationCenter = np.asarray(rotationCenter)
        rotationMatrix = utils.computeTranslationMatrix(rotationCenter)
        rotationMatrix = rotationMatrix@utils.computeRotationMatrix(rotationAngles)
        rotationMatrix = rotationMatrix@utils.computeTranslationMatrix(-rotationCenter)
        if inLocal:
            self.transformationMatrix = self.transformationMatrix@rotationMatrix
        else:
            self.transformationMatrix = rotationMatrix@self.transformationMatrix

    def convertToGlobalPosition(self, position):
        globalPosition = np.ones((position.shape[0], 4), dtype=float)
        globalPosition[:, :3] = position
        matmul(globalPosition, self.transformationMatrix.T, out=globalPosition)
        return globalPosition[:, :3]

    def _generateEmissionTable(self):
        xs, ys, zs = np.meshgrid(
            np.linspace(0, self.size[0], self.distribution.shape[0], endpoint=False),
            np.linspace(0, self.size[1], self.distribution.shape[1], endpoint=False),
            np.linspace(0, self.size[2], self.distribution.shape[2], endpoint=False),
            indexing = 'ij'
        )
        position = np.stack((xs, ys, zs), axis=3).reshape(-1, 3) - self.size/2
        probability = self.distribution.ravel()
        indices = probability.nonzero()[0]
        self.emissionTable = [position[indices], probability[indices]]

    @property
    def activity(self):
        return self.initialActivity*2**(-self.timer/self.halfLife)
    
    @property
    def nucleiNumber(self):
        return self.activity*self.halfLife/log(2)

    def setState(self, timer, rngState=None):
        if timer is not None:
            self.timer = timer
        if rngState is None:
            return
        self.rng.bit_generator.state['state'] = rngState

    def generatePosition(self, n):
        position = self.emissionTable[0]
        probability = self.emissionTable[1]
        position = self.rng.choice(position, n, p=probability)
        position += self.rng.uniform(0., self.voxelSize, position.shape)
        position = self.convertToGlobalPosition(position)
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

    def __init__(self, distribution, activity=None, voxelSize=4*mm):
        radiationType = 'Gamma'
        energy = 140.5*keV
        halfLife = 6.*hour
        super().__init__(distribution, activity, voxelSize, radiationType, energy, halfLife)


class SourcePhantom(Тс99m_MIBI):
    """
    Источник 99mТс-MIBI

    [position = (x, y, z)] = cm

    [activity] = Bq

    [phantomName] = string

    [voxelSize] = cm
    """

    def __init__(self, phantomName, activity=None, voxelSize=4*mm):
        distribution = load(f'Phantoms/{phantomName}.npy')
        super().__init__(distribution, activity, voxelSize)


class efg3(SourcePhantom):
    """
    Источник efg3

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, activity):
        phantomName = 'efg3'
        voxelSize = 4.*mm
        super().__init__(phantomName, activity, voxelSize)
        

class efg3cut(SourcePhantom):
    """
    Источник efg3cut

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, activity):
        phantomName = 'efg3cut'
        voxelSize = 4.*mm
        super().__init__(phantomName, activity, voxelSize)


class efg3cutDefect(SourcePhantom):
    """
    Источник efg3cutDefect

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, position, activity, rotationAngles=None, rotationCenter=None):
        phantomName = 'efg3cutDefect'
        voxelSize = 4.*mm
        super().__init__(position, activity, phantomName, voxelSize, rotationAngles, rotationCenter)

