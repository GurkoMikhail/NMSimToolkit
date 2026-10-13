"""
Пакет компонентов секций инспектора свойств.
"""
from gui.views.inspectors.base_section import BaseInspectorSection
from gui.views.inspectors.collimator_section import CollimatorInspectorSection
from gui.views.inspectors.data_handler_section import DataHandlerInspectorSection
from gui.views.inspectors.data_manager_section import DataManagerInspectorSection
from gui.views.inspectors.dose_grid_section import DoseGridInspectorSection
from gui.views.inspectors.gamma_camera_section import GammaCameraInspectorSection
from gui.views.inspectors.gantry_section import GantryInspectorSection
from gui.views.inspectors.general_section import GeneralInspectorSection
from gui.views.inspectors.inspector_registry import InspectorRegistry
from gui.views.inspectors.procedure_section import ProcedureInspectorSection
from gui.views.inspectors.source_section import SourceInspectorSection
from gui.views.inspectors.transform_section import TransformInspectorSection
from gui.views.inspectors.volume_section import VolumeInspectorSection
from gui.views.inspectors.voxel_volume_section import VoxelVolumeInspectorSection

__all__ = [
    "BaseInspectorSection",
    "InspectorRegistry",
    "GeneralInspectorSection",
    "TransformInspectorSection",
    "VolumeInspectorSection",
    "DoseGridInspectorSection",
    "CollimatorInspectorSection",
    "VoxelVolumeInspectorSection",
    "SourceInspectorSection",
    "GammaCameraInspectorSection",
    "GantryInspectorSection",
    "ProcedureInspectorSection",
    "DataHandlerInspectorSection",
    "DataManagerInspectorSection",
]
