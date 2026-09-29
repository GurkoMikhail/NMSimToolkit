"""
Модуль поворотной станины томографа (GantryNode) в графе сцены.
Представляет ротор аппарата (ОФЭКТ / ПЭТ / КТ), вращающийся вокруг продольной оси Z в изоцентре.
"""

from typing import Optional
import numpy as np

from core.other.typing_definitions import Float
from core.scene.nodes import CompositeNode


class GantryNode(CompositeNode):
    """
    Узел поворотной станины (ротора томографа) в графе сцены.

    Является составным узлом (CompositeNode), дочерними элементами которого выступают
    детекторы (GammaCamera, PetScanner и т.д.).
    Вращение станины задает ориентацию вокруг продольной оси стола Z в изоцентре (0, 0, 0).
    Локальные трансформации установленных детекторов остаются неизменными, а их глобальные
    матрицы пересчитываются автоматически по формуле M_global = M_parent @ M_local.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name or "Gantry")

    @property
    def gantry_angle(self) -> float:
        """
        Текущий угол поворота ротора вокруг оси Z в радианах.
        Извлекается из локальной матрицы трансформации.
        """
        return float(np.arctan2(self.local_matrix[1, 0], self.local_matrix[0, 0]))

    def set_rotation_angle(self, angle_rad: float) -> None:
        """
        Устанавливает угол поворота станины вокруг оси Z в радианах, сохраняя изоцентрическое вращение.
        """
        angle_value = float(angle_rad)
        cos_val = np.cos(angle_value)
        sin_val = np.sin(angle_value)
        self.local_matrix = np.array([
            [cos_val, -sin_val, 0.0, self.local_matrix[0, 3]],
            [sin_val,  cos_val, 0.0, self.local_matrix[1, 3]],
            [0.0,      0.0,     1.0, self.local_matrix[2, 3]],
            [0.0,      0.0,     0.0, 1.0],
        ], dtype=Float)
        self.invalidate_matrix_cache()


__all__ = ["GantryNode"]
