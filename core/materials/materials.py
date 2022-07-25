from abc import ABC
import numpy as np


class MaterialProperties(ABC):
    """ Класс свойств материала """

    @property
    def name(self):
        """ Название материала """
        return self['name'].copy()

    @property
    def type(self):
        """ Тип материала """
        return self['type'].copy()

    @property
    def density(self):
        """ Плотность материала """
        return self['density'].copy()

    @property
    def composition(self):
        """ Состав материала" """
        return self['composition'].copy()

    @property
    def ZtoA_ratio(self):
        """ Оношение Z к A """
        return self['ZtoA_ratio'].copy()

    @property
    def Zeff(self):
        """ Эффективной заряд материала """
        shape = self.shape
        Z = np.linspace(np.zeros(shape), np.full(shape, 93), 93, endpoint=False, dtype=int).T
        Zeff = np.average(Z, axis=-1, weights=self['composition'])
        return Zeff

    @property
    def ID(self):
        """ Идентификатор материала """
        return self['ID'].copy()


class Material(np.void, MaterialProperties):
    """ Класс материала """


class MaterialArray(np.ndarray, MaterialProperties):
    """ 
    Класс массива материалов

    [name] = mm\n
    [type] = mm\n
    [density] = MeV\n
    [composition] = ns\n
    [ZtoA_ratio] = mm\n
    """
    
    def __new__(cls, shape):
        obj = super().__new__(cls, shape, dtype=(Material, dtype_of_material))
        obj['name'] = 'Vacuum'
        obj['composition'][..., 0] = 1
        return obj

    @classmethod
    def create(cls, name, type, density, composition, ZtoA_ratio, ID):
        shape = density.size
        obj = cls(shape)
        obj['name'] = name
        obj['type'] = type
        obj['density'] = density
        obj['composition'] = composition
        obj['ZtoA_ratio'] = ZtoA_ratio
        obj['ID'] = ID
        return obj


dtype_of_material = np.dtype([
    ('name', '<U50'),
    ('type', '<U30'),
    ('density', 'f'),
    ('composition', '93f'),
    ('ZtoA_ratio', 'f'),
    ('ID', 'u8')
])

