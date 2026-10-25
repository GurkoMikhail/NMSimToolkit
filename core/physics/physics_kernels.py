"""
Модуль Numba-ядер интерполяции макроскопических сечений взаимодействия (LAC) по таблицам MaterialBank
с использованием двухуровневого логарифмического индекса за O(1).
"""

import math

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.materials.material_bank import MaterialBank
from core.other.typing_definitions import Float, Index


@njit(cache=True)
def _compute_log_1d_array(values: NDArray[Float]) -> NDArray[Float]:
    """
    Вычисляет натуральные логарифмы одномерного массива скалярным math.log
    для полной побитовой согласованности с JIT-ядрами интерполяции.
    """
    count = values.shape[0]
    out = np.empty(count, dtype=Float)
    for idx in range(count):
        val = values[idx]
        out[idx] = Float(math.log(val)) if val > 0.0 else Float(-np.inf)
    return out


@njit(cache=True)
def _compute_log_2d_array(values: NDArray[Float]) -> NDArray[Float]:
    """
    Вычисляет натуральные логарифмы двумерной таблицы скалярным math.log
    для полной побитовой согласованности с JIT-ядрами интерполяции.
    """
    rows, cols = values.shape
    out = np.empty((rows, cols), dtype=Float)
    for row_idx in range(rows):
        for col_idx in range(cols):
            val = values[row_idx, col_idx]
            out[row_idx, col_idx] = Float(math.log(val)) if val > 0.0 else Float(-np.inf)
    return out


@njit(cache=True, inline='always')
def _compute_log_energy_and_bin(
    energy: Float,
    material_bank: MaterialBank,
) -> tuple[Float, Index]:
    """
    Вычисляет натуральный логарифм энергии частицы и индекс корзины двухуровневого
    логарифмического индекса в диапазоне [0, lookup_num_bins - 1].
    Единый блок выхода (single return) позволяет Numba полностью удалить NRT_incref/NRT_decref.
    """
    log_e_min = material_bank.lookup_log_e_min
    inv_dlog_e = material_bank.lookup_inv_dlog_e
    max_bin = Index(material_bank.lookup_num_bins - 1)
    log_e = Float(-np.inf)
    bin_idx = Index(0)
    if energy > 0.0:
        log_e = Float(math.log(energy))
        if max_bin > 0:
            scaled = (log_e - log_e_min) * inv_dlog_e
            if scaled >= max_bin:
                bin_idx = max_bin
            elif scaled > 0.0:
                bin_idx = Index(int(scaled))
    return log_e, bin_idx


@njit(cache=True, inline='always')
def _locate_row_with_linear(
    energy: Float,
    log_e: Float,
    log_bin: Index,
    material_id: Index,
    material_bank: MaterialBank,
) -> tuple[Index, Float, Float, Float]:
    """
    Внутреннее ядро поиска нижней строки интервала row0, логарифмической доли log_fraction,
    линейной доли lin_fraction и нижней границы интервала energy_low за одно обращение к physics_energy_grid.
    """
    mat_pointers = material_bank.mat_pointers
    grid = material_bank.physics_energy_grid
    log_grid = material_bank.physics_log_energy_grid
    bin_starts = material_bank.lookup_bin_start

    pointer = mat_pointers[material_id]
    start_idx = Index(pointer['start_idx'])
    length = Index(pointer['length'])
    lookup_offset = Index(pointer['lookup_offset'])

    has_nodes = Index(length > 0)
    has_intervals = Index(length > 1)
    safe_bin_idx = Index((lookup_offset + log_bin) * has_intervals)
    idx = Index(bin_starts[safe_bin_idx] * has_intervals)

    while idx + 1 < length - 1 and energy >= grid[start_idx + idx + 1]:
        idx = Index(idx + 1)

    safe_start = Index(start_idx * has_nodes)
    safe_last = Index((start_idx + length - 1) * has_nodes)
    row0 = Index(safe_start + idx * has_intervals)
    row1 = Index(row0 + has_intervals)

    e_first = grid[safe_start]
    e_last = grid[safe_last]
    energy_low = grid[row0]
    energy_high = grid[row1]
    log_low = log_grid[row0]
    log_high = log_grid[row1]
    log_fraction = Float(0.0)
    lin_fraction = Float(0.0)

    if length <= 0:
        row0 = Index(0)
        lin_fraction = Float(-1.0)
    elif length == 1 or energy < e_first:
        row0 = safe_start
    elif energy >= e_last:
        row0 = safe_last
    elif energy > energy_low and energy_high > energy_low:
        lin_fraction = Float((energy - energy_low) / (energy_high - energy_low))
        if energy_low > 0.0:
            log_fraction = Float((log_e - log_low) / (log_high - log_low))
        else:
            log_fraction = lin_fraction

    return row0, log_fraction, lin_fraction, energy_low


@njit(cache=True, inline='always')
def _locate_row(
    energy: Float,
    log_e: Float,
    log_bin: Index,
    material_id: Index,
    material_bank: MaterialBank,
) -> tuple[Index, Float]:
    """
    Находит индекс нижней строки интервала row в MaterialBank и логарифмическую долю
    интерполяции log_fraction в [0, 1] за O(1) по корзине log_bin и линейному досмотру вперёд
    с право-непрерывным переходом на парах/тройках узлов краев поглощения.
    Расположение цикла while на верхнем уровне CFG и безусловное чтение массивов до ветвлений
    позволяют проходу RefCountPrunePass в Numba полностью удалить внутренние NRT_incref/NRT_decref.
    """
    row0, log_fraction, _, _ = _locate_row_with_linear(
        energy, log_e, log_bin, material_id, material_bank
    )
    return row0, log_fraction


@njit(cache=True, inline='always')
def _locate_energy_interval(
    energy: Float,
    material_id: Index,
    material_bank: MaterialBank,
) -> tuple[Index, Float]:
    """
    Находит индекс нижней строки интервала row в MaterialBank и логарифмическую долю
    интерполяции fraction в [0, 1] за O(1) для заданной энергии и материала с право-непрерывным
    переходом на парах узлов краев поглощения.
    """
    log_e, log_bin = _compute_log_energy_and_bin(energy, material_bank)
    row0, log_fraction, _, _ = _locate_row_with_linear(
        energy, log_e, log_bin, material_id, material_bank
    )
    return row0, log_fraction


@njit(cache=True, inline='always')
def _get_macroscopic_cross_sections(
    energy: Float,
    log_e: Float,
    log_bin: Index,
    material_id: Index,
    material_bank: MaterialBank,
    out_lacs_slice: NDArray[Float],
) -> None:
    """
    Вычисляет макроскопические сечения (LAC) всех активных процессов для указанного материала
    и энергии частицы методом кусочной log-log интерполяции на месте (без аллокаций памяти)
    за O(1) через _locate_row с учетом односторонних пределов на парах узлов краев поглощения.
    Извлечение таблиц до вызова _locate_row_with_linear и развернутая интерполяция без второго цикла
    гарантируют полное отсутствие атомарных операций NRT_incref/NRT_decref в горячем цикле.
    """
    lacs = material_bank.physics_lac_table
    log_lacs = material_bank.physics_log_lac_table

    num_cols = Index(lacs.shape[1])
    out_len = Index(out_lacs_slice.shape[0])
    col1 = Index(1 * Index(num_cols > 1))
    col2 = Index(2 * Index(num_cols > 2))
    idx1 = Index(1 * Index(out_len > 1))
    idx2 = Index(2 * Index(out_len > 2))

    row0, log_fraction, lin_fraction, energy_low = _locate_row_with_linear(
        energy, log_e, log_bin, material_id, material_bank
    )
    row1 = Index(row0 + Index(lin_fraction > 0.0))

    val0_low = lacs[row0, 0]
    val0_high = lacs[row1, 0]
    log_val0_low = log_lacs[row0, 0]
    log_val0_high = log_lacs[row1, 0]

    val1_low = lacs[row0, col1]
    val1_high = lacs[row1, col1]
    log_val1_low = log_lacs[row0, col1]
    log_val1_high = log_lacs[row1, col1]

    val2_low = lacs[row0, col2]
    val2_high = lacs[row1, col2]
    log_val2_low = log_lacs[row0, col2]
    log_val2_high = log_lacs[row1, col2]

    res0 = val0_low
    res1 = val1_low
    res2 = val2_low

    if lin_fraction < 0.0:
        res0 = Float(0.0)
        res1 = Float(0.0)
        res2 = Float(0.0)
    elif lin_fraction > 0.0:
        if val0_low > 0.0 and val0_high > 0.0 and energy_low > 0.0:
            res0 = Float(math.exp(log_val0_low + log_fraction * (log_val0_high - log_val0_low)))
        else:
            res0 = Float(val0_low + lin_fraction * (val0_high - val0_low))

        if num_cols > 1:
            if val1_low > 0.0 and val1_high > 0.0 and energy_low > 0.0:
                res1 = Float(math.exp(log_val1_low + log_fraction * (log_val1_high - log_val1_low)))
            else:
                res1 = Float(val1_low + lin_fraction * (val1_high - val1_low))

        if num_cols > 2:
            if val2_low > 0.0 and val2_high > 0.0 and energy_low > 0.0:
                res2 = Float(math.exp(log_val2_low + log_fraction * (log_val2_high - log_val2_low)))
            else:
                res2 = Float(val2_low + lin_fraction * (val2_high - val2_low))

    has_proc0 = Float((num_cols > 0) & (out_len > 0))
    has_proc1 = Float((num_cols > 1) & (out_len > 1))
    has_proc2 = Float((num_cols > 2) & (out_len > 2))

    out_lacs_slice[idx2] = Float(res2 * has_proc2)
    out_lacs_slice[idx1] = Float(res1 * has_proc1)
    out_lacs_slice[0] = Float(res0 * has_proc0)


__all__ = [
    "_compute_log_1d_array",
    "_compute_log_2d_array",
    "_compute_log_energy_and_bin",
    "_locate_row",
    "_locate_energy_interval",
    "_get_macroscopic_cross_sections",
]


