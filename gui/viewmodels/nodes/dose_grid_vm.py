import logging
from typing import Optional, Sequence, Tuple
import numpy as np

from core.scene.dose_grid_node import DoseGridNode
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel

_logger = logging.getLogger(__name__)


class DoseGridViewModel(NodeViewModel):
    """
    Модель представления узла сетки дозы DoseGridNode (паттерн MVVM).
    Предоставляет реактивные свойства size, dose_voxel_size, grid_shape, memory_mb,
    а также сигналы обновления параметров для PropertyInspector и VTKViewport.
    """

    color = gui_field(default=(0.2, 0.9, 0.2, 0.8))
    wireframe_visible = gui_field(default=True)
    dose_visible = gui_field(default=True)

    def __init__(self, core_node: DoseGridNode, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)

    @property
    def size(self) -> np.ndarray:
        """Габаритные размеры сетки (Lx, Ly, Lz) в мм."""
        return np.asarray(self.core_node.size, dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        size_array = np.asarray(new_size, dtype=float)
        self.core_node.size = size_array
        self.property_changed.emit('size', size_array)
        self.property_changed.emit('grid_shape', self.grid_shape)
        self.property_changed.emit('memory_mb', self.memory_mb)

    @property
    def dose_voxel_size(self) -> float:
        """Размер стороны вокселя в мм."""
        return float(self.core_node.dose_voxel_size)

    @dose_voxel_size.setter
    def dose_voxel_size(self, val: float) -> None:
        voxel_size_float = float(val)
        self.core_node.dose_voxel_size = voxel_size_float
        self.property_changed.emit('dose_voxel_size', voxel_size_float)
        self.property_changed.emit('grid_shape', self.grid_shape)
        self.property_changed.emit('memory_mb', self.memory_mb)

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Количество вокселей по осям (Nx, Ny, Nz)."""
        return self.core_node.grid_shape

    @property
    def memory_mb(self) -> float:
        """Расход памяти RAM для сетки типа float64 в МБ."""
        return float(self.core_node.memory_mb)

    @property
    def origin(self) -> Tuple[float, float, float]:
        """Локальные координаты нижнего угла параллелепипеда сетки дозы (-Lx/2, -Ly/2, -Lz/2)."""
        return self.core_node.origin

    @property
    def is_active(self) -> bool:
        """Флаг активности сетки для накопления дозы."""
        return self.core_node.is_active

    @is_active.setter
    def is_active(self, val: bool) -> None:
        active_bool = bool(val)
        self.core_node.is_active = active_bool
        self.property_changed.emit('is_active', active_bool)
