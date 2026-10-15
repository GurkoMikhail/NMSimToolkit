"""
Вспомогательные геометрические и математические утилиты расчетного ядра.
"""
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


def compute_translation_matrix(translation: Sequence[Float] | NDArray[Float]) -> NDArray[Float]:
    """
    Вычисляет матрицу аффинного переноса 4x4 по заданному 3-вектору смещения.
    """
    trans_arr = np.asarray(translation, dtype=Float)
    if trans_arr.shape != (3,):
        raise ValueError(f"Вектор переноса должен иметь форму (3,), получено {trans_arr.shape}")
    translation_x, translation_y, translation_z = trans_arr[0], trans_arr[1], trans_arr[2]
    translation_matrix = np.array([
        [1., 0., 0., translation_x],
        [0., 1., 0., translation_y],
        [0., 0., 1., translation_z],
        [0., 0., 0., 1.]
    ], dtype=Float)
    return translation_matrix


def compute_rotation_matrix(angles: Sequence[Float] | NDArray[Float]) -> NDArray[Float]:
    """
    Вычисляет матрицу 4x4 трехмерного поворота по углам Эйлера (alpha, beta, gamma).
    """
    rot_arr = np.asarray(angles, dtype=Float)
    if rot_arr.shape != (3,):
        raise ValueError(f"Вектор углов вращения должен иметь форму (3,), получено {rot_arr.shape}")
    alpha, beta, gamma = rot_arr[0], rot_arr[1], rot_arr[2]
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

    rotation_matrix = np.array([
        [cos_alpha * cos_beta, cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma, cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma, 0.],
        [sin_alpha * cos_beta, sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma, sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma, 0.],
        [-sin_beta,           cos_beta * sin_gamma,                                      cos_beta * cos_gamma,                                      0.],
        [0.,                  0.,                                                        0.,                                                        1.]
    ], dtype=Float)
    return rotation_matrix


def unique_with_indices(array: Sequence[Any]) -> List[Tuple[Any, NDArray[np.int64]]]:
    """
    Возвращает список пар (уникальный_элемент, индексы_вхождений) для переданной последовательности.
    """
    unique_items = set(array)
    return [
        (item, np.array([item_idx for item_idx, element in enumerate(array) if element is item], dtype=np.int64))
        for item in unique_items
    ]


__all__ = [
    'compute_translation_matrix',
    'compute_rotation_matrix',
    'unique_with_indices',
]



