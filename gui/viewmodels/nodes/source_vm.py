import logging
from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np
import hepunits as units

from core.other.typing_definitions import Float
from core.source.sources import Source, PointSource
from core.data.distribution_loader import DistributionLoader
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel

_logger = logging.getLogger(__name__)


def _on_source_activity_change(instance: Any, value: float) -> None:
    if isinstance(instance.core_node, Source):
        instance.core_node.initial_activity = Float(float(value) * 1e6 * units.Bq)


def _on_source_energy_change(instance: Any, value: float) -> None:
    if isinstance(instance.core_node, Source):
        energy_val = float(value) * units.keV
        instance.core_node.energy = np.zeros(1, dtype=[("energy", Float), ("probability", Float)])
        instance.core_node.energy["energy"] = Float(energy_val)
        instance.core_node.energy["probability"] = Float(1.0)


def _on_source_radiation_type_change(instance: Any, value: str) -> None:
    if isinstance(instance.core_node, Source):
        instance.core_node.radiation_type = str(value)


def _on_source_half_life_change(instance: Any, value: float) -> None:
    if isinstance(instance.core_node, Source):
        hours_value = float(value)
        instance.core_node.half_life = Float(hours_value * 3600.0 * units.second) if hours_value > 0 else Float(np.inf)


class SourceViewModel(NodeViewModel):
    """
    ViewModel для источника излучения (Source, PointSource).
    """
    activity = gui_field(default=100.0, on_change=_on_source_activity_change)      # МБк
    energy = gui_field(default=140.5, on_change=_on_source_energy_change)          # кэВ
    radiation_type = gui_field(default='Gamma', on_change=_on_source_radiation_type_change)
    half_life = gui_field(default=6.0, on_change=_on_source_half_life_change)     # часы
    file_path = gui_field(default='')

    def __init__(self, core_node: Any, parent_vm: Optional[NodeViewModel] = None, file_path: str = "") -> None:
        super().__init__(core_node, parent_vm)
        if file_path:
            self.file_path = str(file_path)
        self._sync_from_core()

    def _sync_from_core(self) -> None:
        """
        Синхронизация параметров ViewModel с атрибутами расчетного ядра Source без скрытого подавления исключений.
        """
        if isinstance(self.core_node, Source):
            if self.core_node.initial_activity is not None:
                total_activity = float(np.sum(self.core_node.initial_activity))
                activity_bq = total_activity if total_activity >= 1e3 else (total_activity / units.Bq)
                self.activity = float(activity_bq / 1e6)

            energy_field = self.core_node.energy
            if energy_field is not None:
                if isinstance(energy_field, np.ndarray) and energy_field.dtype.names is not None and 'energy' in energy_field.dtype.names:
                    raw_energy = float(energy_field['energy'][0])
                elif isinstance(energy_field, (int, float, np.floating, np.integer)):
                    raw_energy = float(energy_field)
                else:
                    raw_energy = 140.5 * units.keV
                self.energy = float(raw_energy / units.keV)

            if self.core_node.radiation_type is not None:
                self.radiation_type = str(self.core_node.radiation_type)

            half_life_field = self.core_node.half_life
            if half_life_field is not None:
                half_life_float = float(half_life_field)
                self.half_life = (half_life_float / 3600.0) if (half_life_float > 0 and not np.isinf(half_life_float)) else 0.0

    @property
    def is_point_source(self) -> bool:
        return isinstance(self.core_node, PointSource)

    @property
    def dimensions(self) -> Tuple[int, ...]:
        if isinstance(self.core_node, Source) and self.core_node.distribution is not None:
            return tuple(self.core_node.distribution.shape)
        return (1, 1, 1)

    @property
    def size(self) -> np.ndarray:
        if isinstance(self.core_node, Source):
            return np.asarray(self.core_node.size, dtype=float)
        return np.array([20.0, 20.0, 20.0], dtype=float)

    @property
    def voxel_size(self) -> float:
        """Шаг вокселей источника в мм."""
        if isinstance(self.core_node, Source):
            return float(self.core_node.voxel_size)
        return 4.0

    @voxel_size.setter
    def voxel_size(self, val: float) -> None:
        if isinstance(self.core_node, Source):
            voxel_val = float(val)
            self.core_node.voxel_size = Float(voxel_val)
            self.property_changed.emit('voxel_size', voxel_val)
            self.property_changed.emit('size', self.size)

    def reload_distribution(self, path: str, shape: Optional[Tuple[int, ...]] = None, order: str = 'F') -> bool:
        """
        Перезагрузка матрицы активности источника из файла (.npy, .dat, .raw) с использованием DistributionLoader.
        """
        target_path = Path(path)
        if not target_path.is_file():
            return False
        try:
            target_shape = shape or self.dimensions
            data = DistributionLoader.load(target_path, target_shape=target_shape, order=order)
            if isinstance(self.core_node, Source):
                self.core_node.distribution = data.astype(float)
            self.file_path = str(target_path)
            self.property_changed.emit('file_path', self.file_path)
            self.property_changed.emit('size', self.size)
            self.property_changed.emit('distribution', data)
            return True
        except (OSError, ValueError, TypeError, KeyError) as err:
            _logger.warning(f"Ошибка загрузки распределения источника: {err}")
            return False
