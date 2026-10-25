"""
Модуль компилятора источников излучения для обхода графа сцены и упаковки
активных источников в плоский буфер SourceBuffer с таблицами метода Воза.
"""

from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.scene.nodes import CompositeNode, SpatialNode
from core.source.source_buffer import SourceBuffer
from core.source.sources import Source


class SourceCompiler:
    """
    Компилятор, отвечающий за извлечение плоского списка источников частиц из графа сцены
    и компиляцию их пространственных и спектральных распределений в плоский SourceBuffer.
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
        Компилирует список активных источников в плоский SoA/CSR-буфер SourceBuffer.

        Выполняет проверку предусловий LBYL: непустые распределения и спектры,
        неотрицательные вероятности, строго положительная сумма вероятностей,
        корректная нормировка (сумма равна 1.0) и согласованность таблиц алиасов Воза.
        """
        num_sources = len(active_sources)
        if num_sources == 0:
            return SourceBuffer.empty()

        voxel_offsets = np.zeros(num_sources + 1, dtype=Index)
        line_offsets = np.zeros(num_sources + 1, dtype=Index)

        voxel_index_parts: List[NDArray[np.int32]] = []
        voxel_prob_parts: List[NDArray[Float]] = []
        voxel_alias_parts: List[NDArray[np.int32]] = []

        line_energy_parts: List[NDArray[Float]] = []
        line_prob_parts: List[NDArray[Float]] = []
        line_alias_parts: List[NDArray[np.int32]] = []

        grid_shape = np.empty((num_sources, 3), dtype=Index)
        voxel_size = np.empty(num_sources, dtype=Float)
        half_size = np.empty((num_sources, 3), dtype=Float)
        local_to_global = np.empty((num_sources, 4, 4), dtype=Float)
        initial_activity = np.empty(num_sources, dtype=Float)
        decay_constant = np.empty(num_sources, dtype=Float)
        source_id = np.arange(num_sources, dtype=Index)
        cos_theta_min = np.empty(num_sources, dtype=Float)
        cell_jitter = np.zeros((num_sources, 2), dtype=Float)

        for src_idx, source in enumerate(active_sources):
            dist = source.distribution
            src_vox_idx = source.voxel_indices
            src_energies = np.asarray(source.energy["energy"], dtype=Float)
            g_mat = np.asarray(source.global_matrix, dtype=Float)
            if g_mat.shape != (4, 4) or np.any(~np.isfinite(g_mat)):
                raise ValueError(
                    f"Матрица global_matrix источника '{source.name}' имеет некорректную форму {g_mat.shape} или NaN/Inf."
                )

            voxel_offsets[src_idx + 1] = voxel_offsets[src_idx] + Index(src_vox_idx.shape[0])
            line_offsets[src_idx + 1] = line_offsets[src_idx] + Index(src_energies.shape[0])

            voxel_index_parts.append(np.ascontiguousarray(src_vox_idx, dtype=np.int32))
            voxel_prob_parts.append(np.ascontiguousarray(source.voxel_prob, dtype=Float))
            voxel_alias_parts.append(np.ascontiguousarray(source.voxel_alias, dtype=np.int32))

            line_energy_parts.append(np.ascontiguousarray(src_energies, dtype=Float))
            line_prob_parts.append(np.ascontiguousarray(source.line_prob, dtype=Float))
            line_alias_parts.append(np.ascontiguousarray(source.line_alias, dtype=np.int32))

            grid_shape[src_idx, :] = dist.shape
            vox_sz_val = Float(source.voxel_size)
            voxel_size[src_idx] = vox_sz_val
            half_size[src_idx, :] = 0.5 * np.asarray(dist.shape, dtype=Float) * vox_sz_val
            local_to_global[src_idx, :, :] = g_mat
            initial_activity[src_idx] = Float(source.initial_activity)
            decay_constant[src_idx] = Float(source.decay_constant)
            cos_theta_min[src_idx] = Float(source.cos_theta_min)
            if source.cell_jitter_span is not None:
                cell_jitter[src_idx, 0] = Float(source.cell_jitter_span[0])
                cell_jitter[src_idx, 1] = Float(source.cell_jitter_span[1])

        buf = SourceBuffer(
            voxel_offsets=np.ascontiguousarray(voxel_offsets, dtype=Index),
            voxel_index=np.ascontiguousarray(np.concatenate(voxel_index_parts), dtype=np.int32),
            voxel_prob=np.ascontiguousarray(np.concatenate(voxel_prob_parts), dtype=Float),
            voxel_alias=np.ascontiguousarray(np.concatenate(voxel_alias_parts), dtype=np.int32),
            grid_shape=np.ascontiguousarray(grid_shape, dtype=Index),
            voxel_size=np.ascontiguousarray(voxel_size, dtype=Float),
            half_size=np.ascontiguousarray(half_size, dtype=Float),
            line_offsets=np.ascontiguousarray(line_offsets, dtype=Index),
            line_energy=np.ascontiguousarray(np.concatenate(line_energy_parts), dtype=Float),
            line_prob=np.ascontiguousarray(np.concatenate(line_prob_parts), dtype=Float),
            line_alias=np.ascontiguousarray(np.concatenate(line_alias_parts), dtype=np.int32),
            local_to_global=np.ascontiguousarray(local_to_global, dtype=Float),
            initial_activity=np.ascontiguousarray(initial_activity, dtype=Float),
            decay_constant=np.ascontiguousarray(decay_constant, dtype=Float),
            source_id=np.ascontiguousarray(source_id, dtype=Index),
            cos_theta_min=np.ascontiguousarray(cos_theta_min, dtype=Float),
            cell_jitter=np.ascontiguousarray(cell_jitter, dtype=Float),
        )
        buf.validate()
        return buf

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
