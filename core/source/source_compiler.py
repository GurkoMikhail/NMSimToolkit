"""
Модуль компилятора источников излучения для обхода графа сцены и уплощения
активных источников по каналам эмиссии в плоский буфер SourceBuffer с таблицами метода Воза.
"""

from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index, Species
from core.scene.nodes import CompositeNode, SpatialNode
from core.source.source_buffer import (
    SourceBuffer,
    SpatialType,
    SpectrumType,
)
from core.source.sources import (
    ContinuousSpectrum,
    DiscreteSpectrum,
    PointDistribution,
    Source,
    VoxelDistribution,
)


class SourceCompiler:
    """
    Компилятор, отвечающий за извлечение плоского списка источников частиц из графа сцены
    и уплощение их каналов излучения (RadiationChannel) в плоский SoA/CSR-буфер SourceBuffer.
    """

    active_sources: List[Source]

    def __init__(self) -> None:
        self.active_sources = []

    def compile_scene(self, root_node: SpatialNode) -> List[Source]:
        """
        Обходит граф сцены и извлекает все активные источники излучения.
        Извлекает только листовые источники (не имеющие других источников в качестве потомков).
        """
        self.active_sources = []
        self._extract_sources(root_node)
        return self.active_sources

    @classmethod
    def compile(cls, active_sources: Sequence[Source]) -> SourceBuffer:
        """
        Компилирует список активных источников в плоский SoA/CSR-буфер SourceBuffer на уровне каналов эмиссии.

        Все каналы одного воксельного источника разделяют единый срез [vox_start, vox_end) в таблицах
        Воза активных вокселей без дублирования памяти. Точечные источники (PointDistribution)
        не выделяют элементов в воксельных таблицах (vox_start == vox_end).
        """
        if not active_sources:
            return SourceBuffer.empty()

        total_channels = sum(len(source.channels) for source in active_sources)
        if total_channels == 0:
            return SourceBuffer.empty()

        spatial_type = np.empty(total_channels, dtype=Index)
        local_to_global = np.empty((total_channels, 4, 4), dtype=Float)
        voxel_range = np.empty((total_channels, 2), dtype=Index)

        voxel_index_parts: List[NDArray[np.int32]] = []
        voxel_prob_parts: List[NDArray[Float]] = []
        voxel_alias_parts: List[NDArray[np.int32]] = []

        grid_shape = np.zeros((total_channels, 3), dtype=Index)
        voxel_size = np.zeros(total_channels, dtype=Float)
        half_size = np.zeros((total_channels, 3), dtype=Float)
        cell_jitter = np.zeros((total_channels, 2), dtype=Float)

        initial_emission_rate = np.empty(total_channels, dtype=Float)
        decay_constant = np.empty(total_channels, dtype=Float)
        source_id = np.empty(total_channels, dtype=Index)
        species = np.empty(total_channels, dtype=Species)

        spectrum_type = np.empty(total_channels, dtype=Index)
        spectrum_offsets = np.zeros(total_channels + 1, dtype=Index)

        line_energy_parts: List[NDArray[Float]] = []
        line_grid_high_parts: List[NDArray[Float]] = []
        line_pdf_ratio_parts: List[NDArray[Float]] = []
        line_prob_parts: List[NDArray[Float]] = []
        line_alias_parts: List[NDArray[np.int32]] = []

        cos_theta_min = np.empty(total_channels, dtype=Float)

        ch_cursor = 0
        vox_cursor = Index(0)

        for src_idx, source in enumerate(active_sources):
            g_mat = np.asarray(source.global_matrix, dtype=Float)
            if g_mat.shape != (4, 4) or np.any(~np.isfinite(g_mat)):
                raise ValueError(
                    f"Матрица global_matrix источника '{source.name}' имеет некорректную форму {g_mat.shape} или NaN/Inf."
                )

            spatial = source.spatial
            if isinstance(spatial, PointDistribution):
                src_spatial_type = Index(SpatialType.POINT)
                src_vox_start = vox_cursor
                src_vox_end = vox_cursor
                src_grid_shape = (0, 0, 0)
                src_vox_size = Float(0.0)
                src_half_size = np.zeros(3, dtype=Float)
                src_jitter = spatial.cell_jitter
            elif isinstance(spatial, VoxelDistribution):
                src_spatial_type = Index(SpatialType.VOXEL)
                src_vox_idx = spatial.voxel_indices
                src_vox_start = vox_cursor
                src_vox_end = vox_cursor + Index(src_vox_idx.shape[0])
                vox_cursor = src_vox_end

                voxel_index_parts.append(src_vox_idx)
                voxel_prob_parts.append(spatial.voxel_prob)
                voxel_alias_parts.append(spatial.voxel_alias)

                src_grid_shape = spatial.grid_shape
                src_vox_size = Float(spatial.voxel_size)
                src_half_size = 0.5 * np.asarray(src_grid_shape, dtype=Float) * src_vox_size
                src_jitter = (Float(0.0), Float(0.0))
            else:
                raise TypeError(f"Неподдерживаемый тип пространственного распределения: {type(spatial)}.")

            src_act = Float(source.initial_activity)
            src_decay = Float(source.decay_constant)

            for channel in source.channels:
                spatial_type[ch_cursor] = src_spatial_type
                local_to_global[ch_cursor, :, :] = g_mat
                voxel_range[ch_cursor, 0] = src_vox_start
                voxel_range[ch_cursor, 1] = src_vox_end
                grid_shape[ch_cursor, :] = src_grid_shape
                voxel_size[ch_cursor] = src_vox_size
                half_size[ch_cursor, :] = src_half_size
                cell_jitter[ch_cursor, 0] = src_jitter[0]
                cell_jitter[ch_cursor, 1] = src_jitter[1]

                initial_emission_rate[ch_cursor] = Float(src_act * Float(channel.branching_ratio))
                decay_constant[ch_cursor] = src_decay
                source_id[ch_cursor] = Index(src_idx)
                species[ch_cursor] = Species(int(channel.species))
                cos_theta_min[ch_cursor] = Float(channel.angular.cos_theta_min)

                spec = channel.spectrum
                if isinstance(spec, DiscreteSpectrum):
                    spectrum_type[ch_cursor] = Index(SpectrumType.DISCRETE)
                    num_lines = Index(spec.line_energy.shape[0])
                    spectrum_offsets[ch_cursor + 1] = spectrum_offsets[ch_cursor] + num_lines
                    line_energy_parts.append(spec.line_energy)
                    line_grid_high_parts.append(spec.line_energy)
                    line_pdf_ratio_parts.append(np.ones(int(num_lines), dtype=Float))
                    line_prob_parts.append(spec.line_prob)
                    line_alias_parts.append(spec.line_alias)
                elif isinstance(spec, ContinuousSpectrum):
                    spectrum_type[ch_cursor] = Index(SpectrumType.CONTINUOUS_TRAPEZOID)
                    num_bins = Index(spec.bin_energy_low.shape[0])
                    spectrum_offsets[ch_cursor + 1] = spectrum_offsets[ch_cursor] + num_bins
                    line_energy_parts.append(spec.bin_energy_low)
                    line_grid_high_parts.append(spec.bin_energy_high)
                    line_pdf_ratio_parts.append(spec.bin_pdf_ratio)
                    line_prob_parts.append(spec.bin_prob)
                    line_alias_parts.append(spec.bin_alias)
                else:
                    raise TypeError(f"Неподдерживаемый тип энергетического спектра: {type(spec)}.")

                ch_cursor += 1

        if voxel_index_parts:
            flat_vox_index = np.concatenate(voxel_index_parts)
            flat_vox_prob = np.concatenate(voxel_prob_parts)
            flat_vox_alias = np.concatenate(voxel_alias_parts)
        else:
            flat_vox_index = np.empty(0, dtype=np.int32)
            flat_vox_prob = np.empty(0, dtype=Float)
            flat_vox_alias = np.empty(0, dtype=np.int32)

        return SourceBuffer(
            spatial_type=spatial_type,
            local_to_global=local_to_global,
            voxel_range=voxel_range,
            voxel_index=flat_vox_index,
            voxel_prob=flat_vox_prob,
            voxel_alias=flat_vox_alias,
            grid_shape=grid_shape,
            voxel_size=voxel_size,
            half_size=half_size,
            cell_jitter=cell_jitter,
            initial_emission_rate=initial_emission_rate,
            decay_constant=decay_constant,
            source_id=source_id,
            species=species,
            spectrum_type=spectrum_type,
            spectrum_offsets=spectrum_offsets,
            line_energy=np.concatenate(line_energy_parts),
            line_grid_high=np.concatenate(line_grid_high_parts),
            line_pdf_ratio=np.concatenate(line_pdf_ratio_parts),
            line_prob=np.concatenate(line_prob_parts),
            line_alias=np.concatenate(line_alias_parts),
            cos_theta_min=cos_theta_min,
        )

    def _extract_sources(self, node: SpatialNode) -> bool:
        """
        Возвращает True, если текущий узел является источником или содержит дочерние источники.
        """
        has_source_children = False

        if isinstance(node, CompositeNode):
            for child_node in node.childs:
                is_child_source = self._extract_sources(child_node)
                if is_child_source:
                    has_source_children = True

        if isinstance(node, Source):
            if not has_source_children:
                self.active_sources.append(node)
            return True

        return has_source_children


__all__ = [
    "SourceCompiler",
]
