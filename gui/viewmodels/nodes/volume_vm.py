import logging
from typing import Optional, Sequence
import numpy as np

from core.geometry.volumes import Volume
from core.materials.materials import Material
import settings.database_setting as database_setting
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewport_3d.material_palette import get_material_rgba

_logger = logging.getLogger(__name__)


class VolumeViewModel(NodeViewModel):
    """
    ViewModel для геометрического объема Volume.
    """
    color = gui_field(default=(0.5, 0.7, 1.0, 0.4))

    def __init__(self, core_node: Volume, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        initial_material_name = self.material_name
        self.color = get_material_rgba(initial_material_name)

    @property
    def local_bound(self) -> np.ndarray:
        """Локальные габариты геометрии объема [Lx, Ly, Lz]."""
        return np.asarray(self.core_node.local_bound, dtype=float)

    @property
    def material_name(self) -> str:
        current_material = self.core_node.material if isinstance(self.core_node, Volume) else None
        return current_material.name if current_material is not None else "Vacuum"

    @material_name.setter
    def material_name(self, new_material_name: str) -> None:
        if new_material_name == "Vacuum":
            self.core_node.material = Material(name="Vacuum")
        elif new_material_name in database_setting.material_database:
            self.core_node.material = database_setting.material_database[new_material_name]
        else:
            raise KeyError(f"Material '{new_material_name}' is not found in the material database.")
        self.core_node.invalidate_geometry()
        self.color = get_material_rgba(new_material_name)
        self.property_changed.emit('material_name', new_material_name)

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.core_node.size, dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        new_size_arr = np.asarray(new_size, dtype=float)
        self.core_node.size = new_size_arr
        self.property_changed.emit('size', new_size_arr)


__all__ = [
    "VolumeViewModel",
]
