"""
Модуль пространственного узла регулярной трехмерной сетки накопления дозы (DoseGridNode).
Относится к расчетному графу сцены (наследник CompositeNode) и определяет положение,
ориентацию, габариты и дискретизацию вокселей для подсчета энерговыделения.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float
from core.scene.nodes import CompositeNode


class DoseGridNode(CompositeNode):
    """
    Узел графа сцены для сеточного накопления дозы (Dose Scorer).
    Является пространственным узлом сцены (наследник SpatialNode / CompositeNode),
    положение и ориентация которого однозначно определяются иерархией графа сцены
    (локальной и глобальной матрицами трансформации 4x4).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        size: Sequence[Float | float] | NDArray[Float] = (100.0, 100.0, 100.0),
        dose_voxel_size: Float | float = 5.0,
        is_active: bool = True,
    ) -> None:
        super().__init__(name=name)
        size_array = np.asarray(size, dtype=Float)
        if size_array.shape != (3,) or any(dim <= 0.0 for dim in size_array):
            raise ValueError(
                f"Габаритные размеры сетки дозы должны состоять из 3 строго положительных чисел, получено {size}."
            )
        voxel_step = float(dose_voxel_size)
        if voxel_step <= 0.0:
            raise ValueError("Размер вокселя дозы должен быть строго положительным.")

        self._size: NDArray[Float] = size_array
        self._dose_voxel_size: float = voxel_step
        self.is_active: bool = bool(is_active)

    @property
    def size(self) -> NDArray[Float]:
        """Габаритные размеры параллелепипеда сетки дозы (Lx, Ly, Lz) в мм."""
        return self._size

    @size.setter
    def size(self, value: Sequence[Float | float] | NDArray[Float]) -> None:
        size_array = np.asarray(value, dtype=Float)
        if size_array.shape != (3,) or any(dim <= 0.0 for dim in size_array):
            raise ValueError(
                f"Габаритные размеры сетки дозы должны состоять из 3 строго положительных чисел, получено {value}."
            )
        self._size = size_array

    @property
    def dose_voxel_size(self) -> float:
        """Шаг регулярной сетки вокселей в мм."""
        return self._dose_voxel_size

    @dose_voxel_size.setter
    def dose_voxel_size(self, value: Float | float) -> None:
        voxel_step = float(value)
        if voxel_step <= 0.0:
            raise ValueError("Размер вокселя дозы должен быть строго положительным.")
        self._dose_voxel_size = voxel_step

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """
        Разрешение трехмерной воксельной сетки (Nx, Ny, Nz),
        рассчитываемое автоматически как ceil(size / dose_voxel_size).
        """
        voxel_step = self._dose_voxel_size
        grid_counts = [int(max(1, np.ceil(float(dim_size) / voxel_step))) for dim_size in self._size]
        return (int(grid_counts[0]), int(grid_counts[1]), int(grid_counts[2]))

    @property
    def origin(self) -> Tuple[float, float, float]:
        """
        Локальные координаты нижнего угла параллелепипеда (-Lx/2, -Ly/2, -Lz/2),
        так как центр узла в локальной системе координат находится в (0, 0, 0).
        """
        return (
            -float(self._size[0]) / 2.0,
            -float(self._size[1]) / 2.0,
            -float(self._size[2]) / 2.0,
        )

    @property
    def memory_mb(self) -> float:
        """Оценка расхода оперативной памяти для сетки типа float64 в МБ."""
        n_elements = int(np.prod(self.grid_shape))
        return float(n_elements * 8 / (1024 * 1024))


__all__ = ["DoseGridNode"]

