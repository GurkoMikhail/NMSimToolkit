"""
Подпакет управления обработчиками данных симуляции, метаданными и загрузкой воксельных распределений.
"""

from core.data.data_handlers import (
    BaseDataHandler,
    DirectStreamHandler,
    HistoryAssemblerHandler,
    SensitiveVolumeHandler,
)
from core.data.data_manager import DataManager
from core.data.dose_map_handler import DoseGridEntry, DoseMapHandler
from core.data.distribution_loader import DistributionLoader
from core.data.metadata_collector import (
    ProtocolMetadataProvider,
    KinematicsMetadataProvider,
    DetectorMetadataProvider,
    GeometryMetadataProvider,
    GantryMetadataProvider,
    SourceMetadataProvider,
    SimulationMetadataProvider,
    ProcedureMetadataCollector,
)
from core.physics.interaction_buffers import ChunkType

__all__ = [
    'ChunkType',
    'BaseDataHandler',
    'DirectStreamHandler',
    'SensitiveVolumeHandler',
    'HistoryAssemblerHandler',
    'DataManager',
    'DoseGridEntry',
    'DoseMapHandler',
    'DistributionLoader',
    'ProtocolMetadataProvider',
    'KinematicsMetadataProvider',
    'DetectorMetadataProvider',
    'GeometryMetadataProvider',
    'GantryMetadataProvider',
    'SourceMetadataProvider',
    'SimulationMetadataProvider',
    'ProcedureMetadataCollector',
]

