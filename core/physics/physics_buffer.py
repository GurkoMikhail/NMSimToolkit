"""
Модуль структур буфера физических данных сцены (PhysicsBuffer, ElementCSR и AtomicRelaxationBank).
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.materials.material_bank import MaterialBank
from core.other.typing_definitions import CFuncAddress, Charge, Float, Index

MAX_RELAXATION_Z: int = 100


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


class AtomicRelaxationBank(NamedTuple):
    """
    Предкомпилированный банк атомной релаксации и скачков фотоэффекта (Z = 0..100)
    с таблицами Воза для O(1) выборки характеристических линий и Оже-электронов в Numba.
    Подоболочки каждого элемента упорядочены по строгому возрастанию энергии связи binding_energy
    и индексируются через shell_offsets.
    """

    shell_offsets: NDArray[Index]
    subshell_designator: NDArray[np.int32]
    binding_energy: NDArray[Float]
    jump_cumulative_weight: NDArray[Float]
    fluorescence_yield: NDArray[Float]
    decay_constant: NDArray[Float]
    line_offsets: NDArray[Index]
    line_energy: NDArray[Float]
    line_prob: NDArray[Float]
    line_alias: NDArray[np.int32]
    line_vacancy_designator: NDArray[np.int32]
    auger_offsets: NDArray[Index]
    auger_energy: NDArray[Float]
    auger_prob: NDArray[Float]
    auger_alias: NDArray[np.int32]
    auger_vacancy_primary: NDArray[np.int32]
    auger_vacancy_secondary: NDArray[np.int32]


class PhysicsBuffer(NamedTuple):
    """
    Буфер скомпилированных физических данных сцены, содержащий:
    - динамический банк материалов (MaterialBank),
    - карту мажорантных материалов объемов для трекинга ВудКока,
    - массив адресов скомпилированных C-функций гетерогенных материалов (@cfunc),
    - CSR-таблицу элементного состава материалов (ElementCSR),
    - предкомпилированный банк атомной релаксации с таблицами Воза (AtomicRelaxationBank).
    """

    material_bank: MaterialBank
    majorant_material_map: NDArray[Index]
    woodcock_function_pointers: NDArray[CFuncAddress]
    element_csr: ElementCSR
    relaxation_bank: AtomicRelaxationBank


__all__ = [
    "AtomicRelaxationBank",
    "ElementCSR",
    "MAX_RELAXATION_Z",
    "PhysicsBuffer",
]
