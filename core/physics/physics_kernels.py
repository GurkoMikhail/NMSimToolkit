"""
Модуль Numba-ядер интерполяции макроскопических сечений взаимодействия (LAC) по таблицам MaterialBank.
"""

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.materials.material_bank import MaterialBank
from core.other.typing_definitions import Float, Index


@njit(cache=True, inline='always')
def _get_macroscopic_cross_sections(
    energy: Float,
    material_id: Index,
    material_bank: MaterialBank,
    out_lacs_slice: NDArray[Float],
) -> None:
    """
    Вычисляет макроскопические сечения (LAC) всех активных процессов для указанного материала
    и энергии частицы методом быстрой линейной интерполяции на месте (без аллокаций памяти).
    """
    pointer = material_bank.mat_pointers[material_id]
    start_idx = pointer['start_idx']
    length = pointer['length']
    num_procs = min(len(out_lacs_slice), material_bank.physics_lac_table.shape[1])

    if length == 0:
        for i in range(len(out_lacs_slice)):
            out_lacs_slice[i] = 0.0
        return

    mat_energy_grid = material_bank.physics_energy_grid[start_idx : start_idx + length]
    idx = np.searchsorted(mat_energy_grid, energy)

    if idx == 0:
        for i in range(num_procs):
            out_lacs_slice[i] = material_bank.physics_lac_table[start_idx, i]
    elif idx >= length:
        for i in range(num_procs):
            out_lacs_slice[i] = material_bank.physics_lac_table[start_idx + length - 1, i]
    else:
        e0 = mat_energy_grid[idx - 1]
        e1 = mat_energy_grid[idx]

        delta_e = e1 - e0
        fraction = (energy - e0) / delta_e if delta_e > 0.0 else 0.0

        for i in range(num_procs):
            v0 = material_bank.physics_lac_table[start_idx + idx - 1, i]
            v1 = material_bank.physics_lac_table[start_idx + idx, i]
            out_lacs_slice[i] = v0 + fraction * (v1 - v0)


__all__ = ["_get_macroscopic_cross_sections"]

