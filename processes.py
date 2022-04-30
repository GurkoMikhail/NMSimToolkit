import g4compton
import g4coherent
import numpy as np
from numpy import pi, cos
from attenuationFunctions import LACfunction
from materials import MaterialsDataBase
from hepunits import*



class Process:
    """ Класс процесса """

    def __init__(self, materialsDataBase=MaterialsDataBase(), rng=None):
        """ Конструктор процесса """
        self.rng = np.random.default_rng() if rng is None else rng
        self._energyRange = np.array([1*keV, 1*MeV])
        self.materialsDataBase = materialsDataBase
        self._constructLACfunctions()

    def _constructLACfunctions(self):
        materialsList = self.materialsDataBase.materialsList
        self.LACfunctions = {material.name: LACfunction.construct(self, material, self.materialsDataBase) for material in materialsList}

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def energyRange(self):
        return self._energyRange

    @energyRange.setter
    def energyRange(self, value):
        self._energyRange = value
        self._constructLACfunctions()

    def getLAC(self, particles, materials):
        LAC = np.ndarray(materials.size, dtype=float)
        energy = particles.energy
        IDs = np.array([material.ID for material in materials])
        for ID in np.unique(IDs):
            match = IDs == ID
            LAC[match] = self.LACfunctions[materials[match][0].name](energy[match])
        # for material, indices in uniqueWithIndices(materials):
        #     LAC[indices] = self.LACfunctions[material.name](energy[indices])
        # LAC = [self.LACfunctions[material.name](particle.energy) for material, particle in zip(materials, particles)]
        # LAC = [self.LACfunctions[material.name](particle.energy) if self.name in particle.processes else 0. for material, particle in zip(materials, particles)]
        return LAC

    def generateFreePath(self, particles, materials):
        LAC = self.getLAC(particles, materials)
        freePath = self.rng.exponential(1/LAC)
        return freePath

    def __call__(self, particles, materials):
        """ Применить процесс """
        size = particles.size
        interactionData = np.recarray(size, dtype=processDataDType)
        interactionData.position = particles.position
        interactionData.direction = particles.direction
        interactionData.processName = self.name
        interactionData.particleID = particles.ID
        interactionData.energyTransfer = 0
        interactionData.scatteringAngles = 0
        interactionData.emissionTime = particles.emissionTime
        interactionData.emissionPosition = particles.emissionPosition
        interactionData.distanceTraveled = particles.distanceTraveled
        return interactionData


class PhotoelectricEffect(Process):
    """ Класс фотоэффекта """

    def __call__(self, particles, materials):
        """ Применить фотоэффект """
        interactionData = super().__call__(particles, materials)
        energyTransfer = particles.energy
        particles.changeEnergy(energyTransfer)
        interactionData.energyTransfer = energyTransfer
        return interactionData
        

class CoherentScattering(Process):
    """ Класс когерентного рассеяния """
    
    def __init__(self, materialsDataBase=MaterialsDataBase(), rng=None):
        super().__init__(materialsDataBase, rng)                
        self.thetaGenerator = g4coherent.initialize(self.rng)

    def generateTheta(self, particles, materials):
        """ Сгенерировать угол рассеяния - theta """
        energy = particles.energy
        Z = np.array([round(material.Zeff) for material in materials], dtype=int)
        theta = self.thetaGenerator(energy, Z)
        return theta

    def generatePhi(self, size):
        """ Сгенерировать угол рассеяния - phi """
        phi = pi*(self.rng.random(size)*2 - 1)
        return phi

    def __call__(self, particles, materials):
        """ Применить эффект Комптона """
        size = particles.size
        theta = self.generateTheta(particles, materials)
        phi = self.generatePhi(size)
        particles.rotate(theta, phi)
        interactionData = super().__call__(particles, materials)
        interactionData.scatteringAngles = np.column_stack((theta, phi))
        return interactionData


class ComptonScattering(CoherentScattering):
    """ Класс эффекта Комптона """

    def __init__(self, materialsDataBase=MaterialsDataBase(), rng=None):
        super().__init__(materialsDataBase, rng)
        self.thetaGenerator = g4compton.initialize(self.rng)

    def culculateEnergyTransfer(self, theta, particlesEnergy):
        """ Вычислить изменения энергий """
        k = particlesEnergy/0.510998910*MeV
        k1_cos = k*(1 - cos(theta))
        energyTransfer = particlesEnergy*k1_cos/(1 + k1_cos)
        return energyTransfer

    def __call__(self, particles, materials):
        """ Применить эффект Комптона """
        interactionData = super().__call__(particles, materials)
        theta = interactionData.scatteringAngles[:, 0]
        energyTransfer = self.culculateEnergyTransfer(theta, particles.energy)
        particles.changeEnergy(energyTransfer)
        interactionData.energyTransfer = energyTransfer
        return interactionData


class PairProduction(Process):
    """ Класс эффекта образования электрон-позитронных пар """


processDataDType = np.dtype([
    ('position', '3d'),
    ('direction', '3d'),
    ('processName', 'U30'),
    ('particleType', 'U30'),
    ('particleID', 'u8'),
    ('energyTransfer', 'd'),
    ('scatteringAngles', '2d'),
    ('emissionTime', 'd'),
    ('emissionPosition', '3d'),
    ('distanceTraveled', 'd'),
])


processesList = {
    'PhotoelectricEffect': PhotoelectricEffect,
    'ComptonScattering': ComptonScattering,
    'CoherentScattering': CoherentScattering,
#     'PairProduction': PairProduction
}

