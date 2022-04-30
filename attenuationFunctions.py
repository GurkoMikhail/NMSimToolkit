import numpy as np
from scipy.interpolate import interp1d
from h5py import File
from materials import MaterialsDataBase
from hepunits import*


class MACfunction(interp1d):
    """ Функция массового коэффициента ослабления """

    def __init__(self, energy, MAC, kind='linear'):
        super().__init__(energy, MAC, kind)

    @classmethod
    def construct(cls, process, material, materialsDataBase=MaterialsDataBase(), kind='linear'):
        AC = caculateAC(process, material, materialsDataBase)
        energy = AC['energy']
        MAC = AC['MAC']
        return cls(energy, MAC, kind)


class LACfunction(interp1d):
    """ Функция линейного коэффициента ослабления """

    def __init__(self, energy, LAC, kind='linear'):
        super().__init__(energy, LAC, kind)

    @classmethod
    def construct(cls, process, material, materialsDataBase=MaterialsDataBase(), kind='linear'):
        AC = caculateAC(process, material, materialsDataBase)
        energy = AC['energy']
        LAC = AC['LAC']
        return cls(energy, LAC, kind)


def loadMACofElement(process, name, materialsDataBase=MaterialsDataBase()):
    fileOfAC = File(materialsDataBase.pathToTableOfAC, 'r')
    groupOfMAC = fileOfAC[f'{materialsDataBase.baseName}/Mass attenuation coefficients/Elemental media']
    processName = processesNames[process.name]
    for elementName, elementGroup in groupOfMAC.items():
        if elementName == name:
            energy = np.array(elementGroup['Energy'])
            arrayOfMAC = np.array(elementGroup[processName])*(cm2/g)
            fileOfAC.close()
            return {'energy': energy, 'AC': arrayOfMAC}
    fileOfAC.close()
    raise NameError(f'Материал {name} не найден')


def caculateAC(process, material, materialsDataBase=MaterialsDataBase()):
    energySet = np.zeros(2, dtype=float)
    MAC = np.zeros(2, dtype=float)
    for elementName, weight in material.elements.items():
        MACofElement = loadMACofElement(process, elementName, materialsDataBase)
        addEnergySet = MACofElement['energy']
        min = np.searchsorted(addEnergySet, process.energyRange[0], side='left')
        max = np.searchsorted(addEnergySet, process.energyRange[1], side='right')
        addEnergySet = addEnergySet[min:max]
        addMAC = MACofElement['AC'][min:max]*weight
        newEnergySet = np.sort(np.concatenate([
            energySet,
            addEnergySet[~np.in1d(addEnergySet, energySet)]
        ]))
        _, indices, counts = np.unique(newEnergySet, return_index=True, return_counts=True)
        indicesForDisplace = indices[(counts > 1).nonzero()[0]]
        newEnergySet[indicesForDisplace] -= 1*eV
        newMAC = np.interp(newEnergySet, energySet, MAC)
        newMAC += np.interp(newEnergySet, addEnergySet, addMAC)
        energySet = newEnergySet
        MAC = newMAC
        energySet = energySet[2:]
        MAC = MAC[2:]
        LAC = MAC*material.density
        AC = {
            'energy': energySet,
            'MAC': MAC,
            'LAC': LAC
        }
    return AC

    
processesNames = {
    'PhotoelectricEffect':  'Photoelectric absorption',
    'ComptonScattering':    'Incoherent scattering',
    'CoherentScattering':   'Coherent scattering'
}

