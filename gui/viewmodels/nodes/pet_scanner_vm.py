import logging
from typing import Optional
import numpy as np

from core.geometry.pet_scanners import PetScanner
from gui.viewmodels.decorators import core_field, gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel

_logger = logging.getLogger(__name__)


class PetScannerViewModel(NodeViewModel):
    """
    ViewModel для ПЭТ-сканера с кольцевой цилиндрической геометрией детекторов.
    Наследуется строго от NodeViewModel (поскольку PetScanner является CompositeNode, а не Volume).
    Свойства диаметра, аксиальной длины и числа секторов привязаны к core_node через core_field.
    """
    diameter = core_field(default=600.0)
    axial_length = core_field(default=200.0)
    num_sectors = core_field(default=32)
    color = gui_field(default=(0.2, 0.7, 0.9, 0.5))

    def __init__(self, core_node: PetScanner, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)

    @property
    def size(self) -> np.ndarray:
        """Габаритные размеры цилиндрического кольца сканера (D, D, L) в мм."""
        d = float(self.diameter)
        l = float(self.axial_length)
        return np.array([d, d, l], dtype=float)
