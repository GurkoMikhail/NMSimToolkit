"""
Модуль структуры плоского буфера скомпилированных каналов первичных источников излучения (SourceBuffer)
в формате Structure of Arrays / CSR для высокопроизводительных Numba JIT-ядер.
"""

from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index, Species


class SpatialType(IntEnum):
    """Код типа пространственного распределения первичного источника в SourceBuffer."""

    POINT = 0
    VOXEL = 1
    ANALYTIC = 2


class SpectrumType(IntEnum):
    """Код типа энергетического спектра канала излучения в SourceBuffer."""

    DISCRETE = 0
    CONTINUOUS_TRAPEZOID = 1


class SourceBuffer(NamedTuple):
    """
    Плоский SoA/CSR-буфер скомпилированных каналов эмиссии первичных источников для Numba JIT-ядер.

    Уплощает источники непосредственно на уровне каналов эмиссии (N_ch записей) по теореме о
    суперпозиции пуассоновских потоков: I_{s,c}(0) = A_{0,s} * branching_ratio_{s,c}.
    Все каналы одного воксельного источника ссылаются на общий диапазон [vox_start, vox_end)
    в voxel_range без дублирования воксельных таблиц Воза, а точечные источники (SpatialType.POINT)
    занимают 0 элементов в воксельных таблицах.
    """

    spatial_type: NDArray[Index]
    local_to_global: NDArray[Float]

    voxel_range: NDArray[Index]
    voxel_index: NDArray[np.int32]
    voxel_prob: NDArray[Float]
    voxel_alias: NDArray[np.int32]

    grid_shape: NDArray[Index]
    voxel_size: NDArray[Float]
    half_size: NDArray[Float]
    cell_jitter: NDArray[Float]

    initial_emission_rate: NDArray[Float]
    decay_constant: NDArray[Float]
    source_id: NDArray[Index]
    species: NDArray[Species]

    spectrum_type: NDArray[Index]
    spectrum_offsets: NDArray[Index]
    line_energy: NDArray[Float]
    line_grid_high: NDArray[Float]
    line_pdf_ratio: NDArray[Float]
    line_prob: NDArray[Float]
    line_alias: NDArray[np.int32]

    cos_theta_min: NDArray[Float]

    @property
    def num_channels(self) -> int:
        """Количество скомпилированных каналов эмиссии в буфере."""
        return int(self.initial_emission_rate.shape[0])

    @classmethod
    def empty(cls) -> "SourceBuffer":
        """Создает пустой валидный экземпляр SourceBuffer для сцен без активных источников."""
        return cls(
            spatial_type=np.empty(0, dtype=Index),
            local_to_global=np.empty((0, 4, 4), dtype=Float),
            voxel_range=np.empty((0, 2), dtype=Index),
            voxel_index=np.empty(0, dtype=np.int32),
            voxel_prob=np.empty(0, dtype=Float),
            voxel_alias=np.empty(0, dtype=np.int32),
            grid_shape=np.empty((0, 3), dtype=Index),
            voxel_size=np.empty(0, dtype=Float),
            half_size=np.empty((0, 3), dtype=Float),
            cell_jitter=np.empty((0, 2), dtype=Float),
            initial_emission_rate=np.empty(0, dtype=Float),
            decay_constant=np.empty(0, dtype=Float),
            source_id=np.empty(0, dtype=Index),
            species=np.empty(0, dtype=Species),
            spectrum_type=np.empty(0, dtype=Index),
            spectrum_offsets=np.zeros(1, dtype=Index),
            line_energy=np.empty(0, dtype=Float),
            line_grid_high=np.empty(0, dtype=Float),
            line_pdf_ratio=np.empty(0, dtype=Float),
            line_prob=np.empty(0, dtype=Float),
            line_alias=np.empty(0, dtype=np.int32),
            cos_theta_min=np.empty(0, dtype=Float),
        )


__all__ = [
    "SpatialType",
    "SpectrumType",
    "SourceBuffer",
]
