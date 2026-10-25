"""
Модуль структур буфера физических данных сцены (PhysicsBuffer и ElementCSR).
"""

from typing import NamedTuple

from numpy.typing import NDArray

from core.materials.material_bank import MaterialBank
from core.other.typing_definitions import CFuncAddress, Charge, Float, Index


class ElementCSR(NamedTuple):
    """
    CSR-представление элементного состава материалов: смещения element_offsets,
    атомные номера element_Z и таблица нормированных парциальных долей элементов
    element_partial_fractions формы (num_grid_rows, num_processes, max_elements)
    на строках финальной энергетической сетки MaterialBank.
    """

    element_offsets: NDArray[Index]
    element_Z: NDArray[Charge]
    element_partial_fractions: NDArray[Float]


class PhysicsBuffer(NamedTuple):
    """
    Буфер скомпилированных физических данных сцены, содержащий:
    - динамический банк материалов (MaterialBank),
    - карту мажорантных материалов объемов для трекинга ВудКока,
    - массив адресов скомпилированных C-функций гетерогенных материалов (@cfunc),
    - CSR-таблицу элементного состава материалов (ElementCSR).
    """

    material_bank: MaterialBank
    majorant_material_map: NDArray[Index]
    woodcock_function_pointers: NDArray[CFuncAddress]
    element_csr: ElementCSR


__all__ = [
    "ElementCSR",
    "PhysicsBuffer",
]

