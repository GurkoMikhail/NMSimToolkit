import logging
from typing import Optional

from core.scene.nodes import SpatialNode
from core.scene.dose_grid_node import DoseGridNode
from core.scene.gantry_node import GantryNode
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.scene.gamma_camera_node import GammaCameraNode
from core.geometry.parametric_collimators import (
    ParametricParallelCollimator,
)
from core.geometry.direct_collimators import DirectParallelCollimator
from core.geometry.pet_scanners import PetScanner
from core.source.sources import Source

from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.nodes.pet_scanner_vm import PetScannerViewModel

_logger = logging.getLogger(__name__)


def create_node_viewmodel(core_node: SpatialNode, parent_vm: Optional[NodeViewModel] = None) -> NodeViewModel:
    """
    Строгая фабрика для создания специализированных моделей представления по типу узла расчетного ядра.
    Использует явный механизм isinstance без обращения к строковым именам классов.
    """
    if isinstance(core_node, DoseGridNode):
        return DoseGridViewModel(core_node, parent_vm)
    if isinstance(core_node, GantryNode):
        return GantryViewModel(core_node, parent_vm)
    if isinstance(core_node, (ParametricParallelCollimator, DirectParallelCollimator)):
        return CollimatorViewModel(core_node, parent_vm)
    if isinstance(core_node, WoodcockVoxelVolume):
        return VoxelVolumeViewModel(core_node, parent_vm)
    if isinstance(core_node, GammaCameraNode):
        return GammaCameraViewModel(core_node, parent_vm)
    if isinstance(core_node, PetScanner):
        return PetScannerViewModel(core_node, parent_vm)
    if isinstance(core_node, Source):
        return SourceViewModel(core_node, parent_vm)
    if isinstance(core_node, Volume):
        return VolumeViewModel(core_node, parent_vm)
    return NodeViewModel(core_node, parent_vm)


# Регистрация фабрики в базовом классе для исключения циклических импортов
NodeViewModel._factory = create_node_viewmodel
