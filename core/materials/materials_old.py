from dataclasses import dataclass, astuple, asdict
from functools import cache
from itertools import count
from atomicProperties import atomicNumber, elementSymbol
import numpy as np
from h5py import File
from hepunits import*


class MaterialsDataBase:
    """ Класс базы данных материалов """

    counter = count(1)

    def __init__(
        self,
        pathToMaterialsTable = 'tables/materials.hdf',
        pathToTableOfAC = 'tables/attenuationCoefficients.hdf',
        baseName = 'NIST'
    ):
        self.pathToMaterialsTable = pathToMaterialsTable
        self.pathToTableOfAC = pathToTableOfAC
        self.baseName = baseName
        self._loadMaterials()

    def _loadMaterials(self):
        self.materialsDict = {}
        materialsFile = File(self.pathToMaterialsTable, 'r')
        baseGroup = materialsFile[self.baseName]
        for groupName, materialTypeGroup in baseGroup.items():
            for materialName, materialGroup in materialTypeGroup.items():
                type = groupName
                density = float(np.copy(materialGroup['Density']))*(g/cm3)
                if groupName == 'Elemental media':
                    Z = float(np.copy(materialGroup['Z']))
                    composition = {elementSymbol[Z]: 1.}
                else:
                    composition = {}
                    for element, weight in materialGroup['Composition'].items():
                        Z = atomicNumber[element]
                        composition.update({elementSymbol[Z]: float(np.copy(weight))})
                composition = MaterialComposition(**composition)
                try:
                    ZtoAratio = float(np.copy(materialGroup['Z\\A']))
                except:
                    # print(f'Для {materialName} отсутствует Z\\A')
                    ZtoAratio = 0.5
                ID = next(self.counter)
                material = Material(materialName, type, density, composition, ZtoAratio, ID)
                self.materialsDict.update({materialName: material})
                self.materialsDict.update({ID: material})

    @property
    def materialsNamesList(self):
        """ Список имён материалов """
        return self.materialsDict.keys()

    @property
    def materialsList(self):
        """ Список материалов """
        return self.materialsDict.values()

    def __getitem__(self, material):
        return self.materialsDict[material]


@dataclass(eq=True, frozen=True)
class MaterialComposition:
    """ Класс состава материала """
    H: float = 0
    He: float = 0
    Li: float = 0
    Be: float = 0
    B: float = 0
    C: float = 0
    N: float = 0
    O: float = 0
    F: float = 0
    Ne: float = 0
    Na: float = 0
    Mg: float = 0
    Al: float = 0
    Si: float = 0
    P: float = 0
    S: float = 0
    Cl: float = 0
    Ar: float = 0
    K: float = 0
    Ca: float = 0
    Sc: float = 0
    Ti: float = 0
    V: float = 0
    Cr: float = 0
    Mn: float = 0
    Fe: float = 0
    Co: float = 0
    Ni: float = 0
    Cu: float = 0
    Zn: float = 0
    Ga: float = 0
    Ge: float = 0
    As: float = 0
    Se: float = 0
    Br: float = 0
    Kr: float = 0
    Rb: float = 0
    Sr: float = 0
    Y: float = 0
    Zr: float = 0
    Nb: float = 0
    Mo: float = 0
    Tc: float = 0
    Ru: float = 0
    Rh: float = 0
    Pd: float = 0
    Ag: float = 0
    Cd: float = 0
    In: float = 0
    Sn: float = 0
    Sb: float = 0
    Te: float = 0
    I: float = 0
    Xe: float = 0
    Cs: float = 0
    Ba: float = 0
    La: float = 0
    Ce: float = 0
    Pr: float = 0
    Nd: float = 0
    Pm: float = 0
    Sm: float = 0
    Eu: float = 0
    Gd: float = 0
    Tb: float = 0
    Dy: float = 0
    Ho: float = 0
    Er: float = 0
    Tm: float = 0
    Yb: float = 0
    Lu: float = 0
    Hf: float = 0
    Ta: float = 0
    W: float = 0
    Re: float = 0
    Os: float = 0
    Ir: float = 0
    Pt: float = 0
    Au: float = 0
    Hg: float = 0
    Tl: float = 0
    Pb: float = 0
    Bi: float = 0
    Po: float = 0
    At: float = 0
    Rn: float = 0
    Fr: float = 0
    Ra: float = 0
    Ac: float = 0
    Th: float = 0
    Pa: float = 0
    U: float = 0

    @property
    @cache
    def Zeff(self):
        return np.average(range(1, 93), weights=self.compositionArray)

    @property
    @cache
    def elementsPresent(self):
        return {name: weight for name, weight in asdict(self).items() if weight > 0}

    @property
    @cache
    def compositionArray(self):
        return np.array(astuple(self))


@dataclass(eq=True, frozen=True)
class Material:
    """ Класс материала """
    name: str
    type: str
    density: float
    composition: MaterialComposition
    ZtoAratio: float
    ID: int

    def __lt__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density < other.Zeff*other.density
        return False

    def __le__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density <= other.Zeff*other.density
        return False

    def __gt__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density > other.Zeff*other.density
        return True

    def __ge__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density >= other.Zeff*other.density
        return True

    @property
    def Zeff(self):
        return self.composition.Zeff

    @property
    def elements(self):
        return self.composition.elementsPresent

