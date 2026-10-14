import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple
import numpy as np
import hepunits as units

from core.other.typing_definitions import Float
from core.source.sources import Source, PointSource
from core.data.distribution_loader import DistributionLoader
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel

_logger = logging.getLogger(__name__)


def _on_source_activity_change(instance: Any, value: float) -> None:
    if isinstance(instance, SourceViewModel) and instance._mutation_blocked:
        return
    if isinstance(instance.core_node, Source):
        instance.core_node.initial_activity = Float(float(value) * float(units.MBq))


def _on_source_radiation_type_change(instance: Any, value: str) -> None:
    if isinstance(instance, SourceViewModel) and instance._mutation_blocked:
        return
    if isinstance(instance.core_node, Source):
        instance.core_node.radiation_type = str(value)


def _on_source_half_life_change(instance: Any, value: float) -> None:
    if isinstance(instance, SourceViewModel) and instance._mutation_blocked:
        return
    if isinstance(instance.core_node, Source):
        hours_value = float(value)
        instance.core_node.half_life = (
            Float(hours_value * float(units.hour)) if hours_value > 0.0 else Float(np.inf)
        )


class SourceViewModel(NodeViewModel):
    """
    ViewModel для источника излучения (Source, PointSource).
    """
    core_node: Source

    activity = gui_field(default=100.0, on_change=_on_source_activity_change)      # МБк
    radiation_type = gui_field(default='Gamma', on_change=_on_source_radiation_type_change)
    half_life = gui_field(default=6.0, on_change=_on_source_half_life_change)     # часы
    file_path = gui_field(default='')

    def __init__(self, core_node: Source, parent_vm: Optional[NodeViewModel] = None, file_path: str = "") -> None:
        self._mutation_blocked: bool = False
        super().__init__(core_node, parent_vm)
        self.core_node = core_node
        if file_path:
            self.file_path = str(file_path)
        self._sync_from_core()

    @contextmanager
    def _block_mutation(self) -> Iterator[None]:
        """Контекстный менеджер для предотвращения обратной feedback-мутации ядра при синхронизации."""
        old_val = self._mutation_blocked
        self._mutation_blocked = True
        try:
            yield
        finally:
            self._mutation_blocked = old_val

    @property
    def spectrum(self) -> List[List[float]]:
        """
        Спектр излучения источника в виде списка пар [энергия (кэВ), вес / относительный выход].
        """
        if isinstance(self.core_node, Source) and self.core_node.energy is not None:
            return [
                [float(energy_val) / float(units.keV), float(prob_val)]
                for energy_val, prob_val in zip(
                    self.core_node.energy["energy"],
                    self.core_node.energy["probability"],
                )
            ]
        return [[140.5, 1.0]]

    @spectrum.setter
    def spectrum(self, pairs: List[List[float]]) -> None:
        """
        Установка спектра излучения с автоматической передачей в расчетное ядро Source.
        """
        if self._mutation_blocked:
            return
        if not pairs:
            pairs = [[140.5, 1.0]]
        if isinstance(self.core_node, Source):
            core_pairs = [
                [float(energy_val) * float(units.keV), float(weight_val)]
                for energy_val, weight_val in pairs
            ]
            self.core_node.energy = core_pairs
        self.property_changed.emit('spectrum', pairs)
        if len(pairs) > 0:
            self.property_changed.emit('energy', pairs[0][0])

    @property
    def energy(self) -> float:
        """
        Энергия основного пика излучения в кэВ.
        """
        spec = self.spectrum
        return float(spec[0][0]) if spec else 140.5

    @energy.setter
    def energy(self, value: float) -> None:
        """
        Установка моноэнергетического значения (или первой линии спектра) в кэВ.
        """
        if self._mutation_blocked:
            return
        val = float(value)
        spec = self.spectrum
        if len(spec) <= 1:
            self.spectrum = [[val, 1.0]]
        else:
            new_spec = [list(line) for line in spec]
            new_spec[0][0] = val
            self.spectrum = new_spec

    def _sync_from_core(self) -> None:
        """
        Синхронизация параметров ViewModel с атрибутами расчетного ядра Source без обратной мутации.
        """
        with self._block_mutation():
            if isinstance(self.core_node, Source):
                if self.core_node.initial_activity is not None:
                    total_activity = float(np.sum(self.core_node.initial_activity))
                    self.activity = float(total_activity / float(units.MBq))

                if self.core_node.radiation_type is not None:
                    self.radiation_type = str(self.core_node.radiation_type)

                half_life_field = self.core_node.half_life
                if half_life_field is not None:
                    half_life_float = float(half_life_field)
                    self.half_life = (
                        half_life_float / float(units.hour)
                        if (half_life_float > 0.0 and not np.isinf(half_life_float))
                        else 0.0
                    )

        self.property_changed.emit('spectrum', self.spectrum)
        self.property_changed.emit('energy', self.energy)

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

    def reload_distribution(
        self,
        path: str,
        shape: Optional[Tuple[int, ...]] = None,
        order: str = 'F',
        dtype: Any = np.float32,
        encoding: Optional[str] = None,
        voxel_size: Optional[float] = None,
        total_activity: Optional[float] = None,
        noise_threshold: Optional[float] = None,
    ) -> bool:
        """
        Перезагрузка матрицы активности источника из файла (.npy, .dat, .raw) с использованием DistributionLoader.
        """
        target_path = Path(path)
        if not target_path.is_file():
            return False
        try:
            target_shape = shape or self.dimensions
            data = DistributionLoader.load(
                target_path,
                target_shape=target_shape,
                order=order,
                dtype=dtype,
                encoding=encoding,
            )
            data_floats = np.asarray(data, dtype=float)
            if noise_threshold is not None and noise_threshold > 0.0:
                data_floats[data_floats < noise_threshold] = 0.0

            if voxel_size is not None:
                self.voxel_size = float(voxel_size)

            if isinstance(self.core_node, Source):
                self.core_node.distribution = data_floats

            if total_activity is not None and total_activity > 0.0:
                self.activity = float(total_activity)

            self.file_path = str(target_path)
            self.property_changed.emit('file_path', self.file_path)
            self.property_changed.emit('size', self.size)
            self.property_changed.emit('distribution', data_floats)
            return True
        except (OSError, ValueError, TypeError, KeyError) as err:
            _logger.warning(f"Ошибка загрузки распределения источника: {err}")
            return False


__all__ = [
    "SourceViewModel",
]
