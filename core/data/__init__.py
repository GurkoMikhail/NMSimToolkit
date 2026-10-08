"""
Подпакет управления обработчиками данных симуляции, метаданными и загрузкой воксельных распределений.
"""

from core.data.data_handlers import BaseDataHandler
from core.data.data_manager import DataManager
from core.data.dose_map_handler import DoseMapHandler
from core.data.distribution_loader import DistributionLoader
from core.data.metadata_collector import (
    ProtocolMetadataProvider,
    KinematicsMetadataProvider,
    DetectorMetadataProvider,
    ProcedureMetadataCollector,
)

__all__ = [
    'BaseDataHandler',
    'DataManager',
    'DoseMapHandler',
    'DistributionLoader',
    'ProtocolMetadataProvider',
    'KinematicsMetadataProvider',
    'DetectorMetadataProvider',
    'ProcedureMetadataCollector',
]
