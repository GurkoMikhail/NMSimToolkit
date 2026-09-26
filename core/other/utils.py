"""
Вспомогательные геометрические и математические утилиты расчетного ядра.
"""
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


def compute_translation_matrix(translation: Union[NDArray[Float], Sequence[Float]]) -> NDArray[Float]:
    """
    Вычисляет матрицу аффинного переноса 4x4 по заданному 3-вектору смещения.
    """
    translation_x, translation_y, translation_z = translation
    translation_matrix = np.array([
        [1., 0., 0., translation_x],
        [0., 1., 0., translation_y],
        [0., 0., 1., translation_z],
        [0., 0., 0., 1.]
    ])
    return translation_matrix


def compute_rotation_matrix(angles: Union[NDArray[Float], Sequence[Float]]) -> NDArray[Float]:
    """
    Вычисляет матрицу 4x4 трехмерного поворота по углам Эйлера (alpha, beta, gamma).
    """
    alpha, beta, gamma = angles
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

    rotation_matrix = np.array([
        [cos_alpha * cos_beta, cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma, cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma, 0.],
        [sin_alpha * cos_beta, sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma, sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma, 0.],
        [-sin_beta,           cos_beta * sin_gamma,                                      cos_beta * cos_gamma,                                      0.],
        [0.,                  0.,                                                        0.,                                                        1.]
    ])
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



