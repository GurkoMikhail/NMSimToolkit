from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np


class ImportTargetKind(Enum):
    """
    Тип целевого узла сцены для импорта воксельного распределения.
    """
    PHANTOM = "phantom"
    SOURCE = "source"


@dataclass
class DistributionImportParameters:
    """
    Параметры импорта воксельного распределения из внешнего файла данных.
    Используется диалогом DistributionImportDialog для унифицированной передачи настроек
    в методы reload_distribution моделей представления VoxelVolumeViewModel и SourceViewModel.
    """
    target_kind: ImportTargetKind
    file_path: Union[str, Path]
    shape: Tuple[int, int, int]
    order: str = "F"
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))
    encoding: str = "text"
    voxel_size: Union[float, Tuple[float, float, float]] = 1.0
    mapping: Optional[Dict[float, str]] = None
    material_mapping: Optional[Dict[float, str]] = None
    fill_value: str = "Air, Dry (near sea level)"
    total_activity: Optional[float] = None
    noise_threshold: Optional[float] = None
    is_npy: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        if self.mapping is None and self.material_mapping is not None:
            self.mapping = self.material_mapping
        elif self.material_mapping is None and self.mapping is not None:
            self.material_mapping = self.mapping
        if any(dimension_step <= 0 for dimension_step in self.shape):
            raise ValueError(f"Размеры сетки должны быть положительными: {self.shape}")
        if self.order not in ("F", "C"):
            raise ValueError(f"Недопустимый порядок развертки: {self.order} (допустимы 'F' или 'C')")
        if self.encoding not in ("binary", "text"):
            raise ValueError(f"Недопустимый режим кодирования: {self.encoding} (допустимы 'binary' или 'text')")
        if isinstance(self.voxel_size, (int, float)):
            if self.voxel_size <= 0:
                raise ValueError(f"Шаг вокселя должен быть положительным: {self.voxel_size}")
        elif isinstance(self.voxel_size, (tuple, list)):
            if any(step_val <= 0 for step_val in self.voxel_size):
                raise ValueError(f"Компоненты шага вокселей должны быть положительными: {self.voxel_size}")
        else:
            raise TypeError(f"Некорректный тип voxel_size: {type(self.voxel_size)}")
