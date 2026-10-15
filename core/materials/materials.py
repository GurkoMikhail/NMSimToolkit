"""
Модуль описания физических материалов и массивов материалов для воксельных и гетерогенных объемов.
"""

from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, List, Tuple

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.materials.atomic_properties import atomic_number
from core.other.nonunique_array import NonuniqueArray
from core.other.typing_definitions import Float, Index

Composition = namedtuple('Composition', ['H'])


@dataclass(eq=True, frozen=True)
class Material:
    """Неизменяемое представление физического материала с химическим составом и плотностью."""

    name: str = 'Vacuum'
    type: str = ''
    density: Float = Float(0.4 * (10 ** (-29)) * units.g / units.cm3)
    composition: Any = Composition(H=1.0)
    ZtoA_ratio: Float = Float(0.0)
    ID: int = 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Material):
            return NotImplemented
        return bool(self.Zeff * self.density < other.Zeff * other.density)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Material):
            return NotImplemented
        return bool(self.Zeff * self.density <= other.Zeff * other.density)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Material):
            return NotImplemented
        return bool(self.Zeff * self.density > other.Zeff * other.density)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Material):
            return NotImplemented
        return bool(self.Zeff * self.density >= other.Zeff * other.density)

    @property
    @cache
    def Zeff(self) -> Float:
        """Вычисляет эффективный атомный номер материала по массовым долям элементов."""
        z_eff = 0.0
        for element, weight in self.composition_dict.items():
            z_eff += atomic_number[element] * float(weight)
        return Float(z_eff)

    @property
    @cache
    def composition_dict(self) -> Dict[str, Float]:
        """Возвращает химический состав материала в виде словаря {элемент: массовая_доля}."""
        raw_dict: Dict[str, Float] = self.composition._asdict()
        return raw_dict

    @property
    @cache
    def composition_array(self) -> NDArray[Float]:
        """Возвращает массив массовых долей элементов по индексу атомного номера Z (0..92)."""
        comp_arr = np.zeros(shape=93, dtype=Float)
        for element, weight in self.composition_dict.items():
            atomic_z = atomic_number[element]
            comp_arr[atomic_z] = weight
        return comp_arr


class MaterialArray(NonuniqueArray):
    """
    Индексированный массив материалов для воксельных фантомов с обратным отображением индексов.
    """

    def __new__(cls, shape: Tuple[int, ...]) -> 'MaterialArray':
        obj = super().__new__(cls, shape)
        obj.element_list = [Material()]
        return obj

    @property
    def material_list(self) -> List[Material]:
        """Список уникальных материалов в массиве."""
        return self.element_list

    @property
    def Zeff(self) -> NDArray[Float]:
        """Массив эффективных атомных номеров по форме сетки."""
        z_eff_arr = np.zeros(self.shape, dtype=Float)
        for material, indices in self.inverse_indices.items():
            z_eff_arr[indices] = material.Zeff
        return z_eff_arr

    @property
    def density(self) -> NDArray[Float]:
        """Массив плотностей материалов по форме сетки."""
        density_arr = np.zeros(self.shape, dtype=Float)
        for material, indices in self.inverse_indices.items():
            density_arr[indices] = material.density
        return density_arr

    @property
    def ID(self) -> NDArray[Index]:
        """Массив числовых идентификаторов материалов по форме сетки."""
        material_id_arr = np.zeros(self.shape, dtype=Index)
        for material, indices in self.inverse_indices.items():
            material_id_arr[indices] = material.ID
        return material_id_arr


__all__ = [
    "Composition",
    "Material",
    "MaterialArray",
]

