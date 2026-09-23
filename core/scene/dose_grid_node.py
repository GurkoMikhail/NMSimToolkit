"""
Модуль пространственного узла регулярной трехмерной сетки накопления дозы (DoseGridNode).
Относится к расчетному графу сцены (наследник CompositeNode) и определяет положение,
ориентацию, габариты и дискретизацию вокселей для подсчета энерговыделения.
"""
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
        size: Sequence[Float] = (100.0, 100.0, 100.0),
        dose_voxel_size: Float = 5.0,
        is_active: bool = True,
    ) -> None:
        super().__init__(name=name)
        self._size = np.asarray(size, dtype=Float)
        self._dose_voxel_size = float(dose_voxel_size)
        self.is_active = bool(is_active)
        self.dose_data: Optional[np.ndarray] = None

    @property
    def size(self) -> NDArray[Float]:
        """Габаритные размеры параллелепипеда сетки дозы (Lx, Ly, Lz) в мм."""
        return self._size

    @size.setter
    def size(self, value: Sequence[Float]) -> None:
        self._size = np.asarray(value, dtype=Float)

    @property
    def dose_voxel_size(self) -> float:
        """Шаг регулярной сетки вокселей в мм."""
        return self._dose_voxel_size

    @dose_voxel_size.setter
    def dose_voxel_size(self, value: Float) -> None:
        val = float(value)
        if val <= 0.0:
            raise ValueError("Размер вокселя дозы должен быть строго положительным.")
        self._dose_voxel_size = val

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """
        Разрешение трехмерной воксельной сетки (Nx, Ny, Nz),
        рассчитываемое автоматически как ceil(size / dose_voxel_size).
        """
        vs = self._dose_voxel_size
        return tuple(int(max(1, np.ceil(float(s) / vs))) for s in self._size)

    @property
    def origin(self) -> Tuple[float, float, float]:
        """
        Локальные координаты нижнего угла параллелепипеда (-Lx/2, -Ly/2, -Lz/2),
        так как центр узла в локальной системе координат находится в (0, 0, 0).
        """
        return tuple(-float(s) / 2.0 for s in self._size)

    @property
    def memory_mb(self) -> float:
        """Оценка расхода оперативной памяти для сетки типа float64 в МБ."""
        n_elements = int(np.prod(self.grid_shape))
        return float(n_elements * 8 / (1024 * 1024))

    def clear(self) -> None:
        """Сброс накопленной дозы в ноль."""
        if self.dose_data is not None:
            self.dose_data.fill(0.0)
