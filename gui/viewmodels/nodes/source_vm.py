import logging
from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple
import numpy as np
import hepunits as units

from core.other.typing_definitions import Float, ParticleSpecies
from core.source.sources import (
    ContinuousSpectrum,
    DiscreteSpectrum,
    PointDistribution,
    RadiationChannel,
    Source,
    SpectralLine,
    VoxelDistribution,
)
from core.data.distribution_loader import DistributionLoader
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel

_logger = logging.getLogger(__name__)

_SPECIES_TO_RAD_TYPE = {
    ParticleSpecies.PHOTON: "Gamma",
    ParticleSpecies.ELECTRON: "Beta",
    ParticleSpecies.POSITRON: "Positron",
    ParticleSpecies.ALPHA: "Alpha",
}

_RAD_TYPE_TO_SPECIES = {
    "gamma": ParticleSpecies.PHOTON,
    "photon": ParticleSpecies.PHOTON,
    "beta": ParticleSpecies.ELECTRON,
    "electron": ParticleSpecies.ELECTRON,
    "positron": ParticleSpecies.POSITRON,
    "alpha": ParticleSpecies.ALPHA,
}


def _on_source_activity_change(instance: Any, value: float) -> None:
    if isinstance(instance, SourceViewModel) and instance._mutation_blocked:
        return
    instance.core_node.initial_activity = Float(float(value) * float(units.MBq))


def _on_source_radiation_type_change(instance: Any, value: str) -> None:
    if isinstance(instance, SourceViewModel) and instance._mutation_blocked:
        return
    if instance.core_node.channels:
        new_species = _RAD_TYPE_TO_SPECIES.get(str(value).strip().lower(), ParticleSpecies.PHOTON)
        ch0 = instance.core_node.channels[0]
        updated_ch0 = replace(ch0, species=new_species)
        instance.core_node.channels = (updated_ch0, *instance.core_node.channels[1:])


def _on_source_half_life_change(instance: Any, value: float) -> None:
    if isinstance(instance, SourceViewModel) and instance._mutation_blocked:
        return
    hours_value = float(value)
    instance.core_node.half_life = (
        Float(hours_value * float(units.hour)) if hours_value > 0.0 else Float(np.inf)
    )


class SourceViewModel(NodeViewModel):
    """
    ViewModel для источника излучения (Source).
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
        if self.core_node.channels:
            ch0_spec = self.core_node.channels[0].spectrum
            if isinstance(ch0_spec, DiscreteSpectrum):
                return [
                    [float(line.energy) / float(units.keV), float(line.probability)]
                    for line in ch0_spec.lines
                ]
            if isinstance(ch0_spec, ContinuousSpectrum):
                return [
                    [float(energy_val) / float(units.keV), float(pdf_val)]
                    for energy_val, pdf_val in zip(ch0_spec.energy_grid, ch0_spec.pdf_values)
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
        lines_tuple = tuple(
            SpectralLine(
                energy=Float(float(energy_val) * float(units.keV)),
                probability=Float(weight_val),
            )
            for energy_val, weight_val in pairs
        )
        new_spec = DiscreteSpectrum(lines=lines_tuple)
        if self.core_node.channels:
            ch0 = self.core_node.channels[0]
            updated_ch0 = replace(ch0, spectrum=new_spec)
            self.core_node.channels = (updated_ch0, *self.core_node.channels[1:])
        else:
            self.core_node.channels = (
                RadiationChannel(species=ParticleSpecies.PHOTON, spectrum=new_spec),
            )
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
            total_activity = float(self.core_node.initial_activity)
            self.activity = float(total_activity / float(units.MBq))

            if self.core_node.channels:
                ch0_species = self.core_node.channels[0].species
                self.radiation_type = _SPECIES_TO_RAD_TYPE.get(ch0_species, "Gamma")

            half_life_float = float(self.core_node.half_life)
            self.half_life = (
                half_life_float / float(units.hour)
                if (half_life_float > 0.0 and not np.isinf(half_life_float))
                else 0.0
            )

        self.property_changed.emit('spectrum', self.spectrum)
        self.property_changed.emit('energy', self.energy)

    @property
    def is_point_source(self) -> bool:
        return isinstance(self.core_node.spatial, PointDistribution)

    @property
    def dimensions(self) -> Tuple[int, ...]:
        if isinstance(self.core_node.spatial, VoxelDistribution):
            return tuple(self.core_node.spatial.distribution.shape)
        return (1, 1, 1)

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.core_node.size, dtype=float)

    @property
    def voxel_size(self) -> float:
        """Шаг вокселей источника в мм."""
        if isinstance(self.core_node.spatial, VoxelDistribution):
            return float(np.mean(self.core_node.spatial.voxel_size))
        return 0.0

    @voxel_size.setter
    def voxel_size(self, val: float) -> None:
        if isinstance(self.core_node.spatial, VoxelDistribution):
            voxel_val = float(val)
            if voxel_val > 0.0:
                self.core_node.spatial = VoxelDistribution(
                    distribution=self.core_node.spatial.distribution,
                    voxel_size=Float(voxel_val),
                )
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

            effective_voxel_size = (
                float(voxel_size)
                if (voxel_size is not None and voxel_size > 0.0)
                else (self.voxel_size if self.voxel_size > 0.0 else 4.0)
            )

            self.core_node.spatial = VoxelDistribution(
                distribution=data_floats,
                voxel_size=Float(effective_voxel_size),
            )
            self.property_changed.emit('voxel_size', effective_voxel_size)

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
