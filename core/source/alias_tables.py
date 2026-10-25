"""
Модуль построения таблиц метода алиасов Воза (Vose's Alias Method) за время O(K).
"""

from typing import Any, Sequence, Tuple

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


@njit(cache=True)
def _build_alias_tables_kernel(
    scaled_probs: NDArray[Float],
) -> Tuple[NDArray[Float], NDArray[np.int32]]:
    """
    Внутреннее Numba-ядро построения таблиц prob и alias метода Воза за O(K)
    с устойчивой обработкой остатков очередей.
    """
    num_items = scaled_probs.shape[0]
    prob = np.empty(num_items, dtype=Float)
    alias = np.zeros(num_items, dtype=np.int32)

    if num_items == 1:
        prob[0] = Float(1.0)
        alias[0] = np.int32(0)
        return prob, alias

    small_queue = np.empty(num_items, dtype=np.int32)
    large_queue = np.empty(num_items, dtype=np.int32)
    small_top = 0
    large_top = 0

    for item_idx in range(num_items):
        if scaled_probs[item_idx] < 1.0:
            small_queue[small_top] = np.int32(item_idx)
            small_top += 1
        else:
            large_queue[large_top] = np.int32(item_idx)
            large_top += 1

    while small_top > 0 and large_top > 0:
        small_top -= 1
        less_idx = small_queue[small_top]
        large_top -= 1
        more_idx = large_queue[large_top]

        prob_val = scaled_probs[less_idx]
        if prob_val < 0.0:
            prob_val = Float(0.0)
        elif prob_val > 1.0:
            prob_val = Float(1.0)
        prob[less_idx] = prob_val
        alias[less_idx] = more_idx

        scaled_probs[more_idx] = (scaled_probs[more_idx] + scaled_probs[less_idx]) - 1.0
        if scaled_probs[more_idx] < 1.0:
            small_queue[small_top] = more_idx
            small_top += 1
        else:
            large_queue[large_top] = more_idx
            large_top += 1

    while large_top > 0:
        large_top -= 1
        rem_idx = large_queue[large_top]
        prob[rem_idx] = Float(1.0)
        alias[rem_idx] = rem_idx

    while small_top > 0:
        small_top -= 1
        rem_idx = small_queue[small_top]
        prob[rem_idx] = Float(1.0)
        alias[rem_idx] = rem_idx

    return prob, alias


def build_alias_tables(
    probabilities: Sequence[float] | NDArray[Any],
) -> Tuple[NDArray[Float], NDArray[np.int32]]:
    """
    Строит таблицы вероятностей и алиасов по алгоритму Воза (Vose's Alias Method) за O(K).

    Проверяет предусловия: одномерность, непустоту, конечность, неотрицательность и
    строго положительную сумму вероятностей. Финальные элементы обеих очередей
    получают prob = 1.0 для численной устойчивости.
    """
    probs_arr = np.asarray(probabilities, dtype=Float)
    if probs_arr.ndim != 1 or probs_arr.size == 0:
        raise ValueError(
            f"Массив вероятностей для таблицы Воза должен быть одномерным и непустым, "
            f"получено shape={probs_arr.shape}."
        )
    if probs_arr.size > np.iinfo(np.int32).max:
        raise ValueError(
            f"Размер распределения ({probs_arr.size}) превышает допустимый диапазон int32."
        )
    if np.any(~np.isfinite(probs_arr)):
        raise ValueError("Массив вероятностей содержит неконечные значения (NaN или Inf).")
    if np.any(probs_arr < 0.0):
        raise ValueError("Вероятности элементов не могут быть отрицательными (< 0).")

    prob_sum = float(np.sum(probs_arr))
    if prob_sum <= 0.0 or abs(prob_sum - 1.0) > 1e-5:
        raise ValueError(
            f"Сумма вероятностей должна быть равна 1.0 (допуск 1e-5), получено {prob_sum}."
        )

    num_items = int(probs_arr.shape[0])
    scaled_probs = np.ascontiguousarray(probs_arr * (float(num_items) / prob_sum), dtype=Float)
    return _build_alias_tables_kernel(scaled_probs)


__all__ = [
    "build_alias_tables",
]
