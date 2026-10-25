"""
Модуль структуры плоского буфера скомпилированных источников излучения (SourceBuffer)
в формате Structure of Arrays / CSR для высокопроизводительных Numba JIT-ядер.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index


class SourceBuffer(NamedTuple):
    """
    Плоский SoA/CSR-буфер скомпилированных активных источников излучения для Numba JIT-ядер.

    Хранит разреженные индексы активных вокселей, таблицы метода алиасов Воза (Vose's Alias)
    для пространственных распределений и энергетических спектров, геометрические параметры
    сеток, матрицы трансформации local_to_global и параметры радиоактивного распада.
    """

    voxel_offsets: NDArray[Index]
    voxel_index: NDArray[np.int32]
    voxel_prob: NDArray[Float]
    voxel_alias: NDArray[np.int32]

    grid_shape: NDArray[Index]
    voxel_size: NDArray[Float]
    half_size: NDArray[Float]

    line_offsets: NDArray[Index]
    line_energy: NDArray[Float]
    line_prob: NDArray[Float]
    line_alias: NDArray[np.int32]

    local_to_global: NDArray[Float]

    initial_activity: NDArray[Float]
    decay_constant: NDArray[Float]
    source_id: NDArray[Index]

    cos_theta_min: NDArray[Float]
    cell_jitter: NDArray[Float]

    @property
    def num_sources(self) -> int:
        """Количество скомпилированных источников в буфере."""
        return int(self.initial_activity.shape[0])

    @classmethod
    def empty(cls) -> "SourceBuffer":
        """Создает пустой валидный экземпляр SourceBuffer для сцен без активных источников."""
        buf = cls(
            voxel_offsets=np.zeros(1, dtype=Index),
            voxel_index=np.empty(0, dtype=np.int32),
            voxel_prob=np.empty(0, dtype=Float),
            voxel_alias=np.empty(0, dtype=np.int32),
            grid_shape=np.empty((0, 3), dtype=Index),
            voxel_size=np.empty(0, dtype=Float),
            half_size=np.empty((0, 3), dtype=Float),
            line_offsets=np.zeros(1, dtype=Index),
            line_energy=np.empty(0, dtype=Float),
            line_prob=np.empty(0, dtype=Float),
            line_alias=np.empty(0, dtype=np.int32),
            local_to_global=np.empty((0, 4, 4), dtype=Float),
            initial_activity=np.empty(0, dtype=Float),
            decay_constant=np.empty(0, dtype=Float),
            source_id=np.empty(0, dtype=Index),
            cos_theta_min=np.empty(0, dtype=Float),
            cell_jitter=np.empty((0, 2), dtype=Float),
        )
        buf.validate()
        return buf

    def validate(self) -> None:
        """Проверяет структурные, размерные и численные инварианты SourceBuffer."""
        num_src = int(self.initial_activity.shape[0])

        if self.voxel_offsets.shape != (num_src + 1,) or self.line_offsets.shape != (num_src + 1,):
            raise ValueError("Размеры voxel_offsets и line_offsets должны быть равны (num_sources + 1,).")
        if self.voxel_offsets[0] != 0 or self.line_offsets[0] != 0:
            raise ValueError("Начальные смещения voxel_offsets[0] и line_offsets[0] должны быть равны 0.")

        total_voxels = int(self.voxel_offsets[-1])
        for vox_arr in (self.voxel_index, self.voxel_prob, self.voxel_alias):
            if vox_arr.shape != (total_voxels,):
                raise ValueError(f"Несогласованные размеры воксельных таблиц: ожидалось ({total_voxels},).")

        total_lines = int(self.line_offsets[-1])
        for line_arr in (self.line_energy, self.line_prob, self.line_alias):
            if line_arr.shape != (total_lines,):
                raise ValueError(f"Несогласованные размеры спектральных таблиц: ожидалось ({total_lines},).")

        for arr_1d in (self.voxel_size, self.decay_constant, self.source_id, self.cos_theta_min):
            if arr_1d.shape != (num_src,):
                raise ValueError(f"Ожидалась форма ({num_src},), получено {arr_1d.shape}.")
        if (
            self.grid_shape.shape != (num_src, 3)
            or self.half_size.shape != (num_src, 3)
            or self.local_to_global.shape != (num_src, 4, 4)
            or self.cell_jitter.shape != (num_src, 2)
        ):
            raise ValueError("Некорректные размерности многомерных полей SourceBuffer.")

        if num_src == 0:
            return

        if np.any(np.diff(self.voxel_offsets) <= 0):
            raise ValueError("Каждый источник в SourceBuffer должен содержать хотя бы один активный воксель.")
        if np.any(np.diff(self.line_offsets) <= 0):
            raise ValueError("Каждый источник в SourceBuffer должен содержать хотя бы одну линию спектра.")
        if np.any(self.initial_activity <= 0.0) or np.any(~np.isfinite(self.initial_activity)):
            raise ValueError("Начальная активность всех источников в SourceBuffer должна быть > 0 и конечной.")
        if np.any(self.decay_constant < 0.0) or np.any(~np.isfinite(self.decay_constant)):
            raise ValueError("Постоянная распада всех источников в SourceBuffer должна быть >= 0 и конечной.")
        if np.any(self.voxel_size < 0.0) or np.any(~np.isfinite(self.voxel_size)):
            raise ValueError("Размер вокселя всех источников в SourceBuffer должен быть >= 0 и конечным.")
        if np.any(self.line_energy <= 0.0) or np.any(~np.isfinite(self.line_energy)):
            raise ValueError("Энергии всех спектральных линий в SourceBuffer должны быть > 0 и конечными.")
        if np.any(self.grid_shape <= 0):
            raise ValueError("Все размерности сетки grid_shape в SourceBuffer должны быть строго положительными (> 0).")
        if np.any((self.voxel_prob < 0.0) | (self.voxel_prob > 1.0)) or np.any(~np.isfinite(self.voxel_prob)):
            raise ValueError("Значения voxel_prob должны принадлежать интервалу [0.0, 1.0].")
        if np.any((self.line_prob < 0.0) | (self.line_prob > 1.0)) or np.any(~np.isfinite(self.line_prob)):
            raise ValueError("Значения line_prob должны принадлежать интервалу [0.0, 1.0].")
        if np.any((self.cos_theta_min < -1.0) | (self.cos_theta_min >= 1.0)) or np.any(~np.isfinite(self.cos_theta_min)):
            raise ValueError("Значения cos_theta_min должны принадлежать полуинтервалу [-1.0, 1.0).")
        if np.any(self.cell_jitter < 0.0) or np.any(~np.isfinite(self.cell_jitter)):
            raise ValueError("Значения cell_jitter должны быть неотрицательными (>= 0.0) и конечными.")

        for src_idx in range(num_src):
            vox_start = int(self.voxel_offsets[src_idx])
            vox_end = int(self.voxel_offsets[src_idx + 1])
            vox_count = vox_end - vox_start
            max_grid_voxels = (
                int(self.grid_shape[src_idx, 0])
                * int(self.grid_shape[src_idx, 1])
                * int(self.grid_shape[src_idx, 2])
            )
            src_vox_idx = self.voxel_index[vox_start:vox_end]
            if np.any((src_vox_idx < 0) | (src_vox_idx >= max_grid_voxels)):
                raise ValueError(
                    f"Плоские индексы voxel_index источника {src_idx} выходят за пределы [0, {max_grid_voxels})."
                )
            src_vox_alias = self.voxel_alias[vox_start:vox_end]
            if np.any((src_vox_alias < 0) | (src_vox_alias >= vox_count)):
                raise ValueError(
                    f"Индексы voxel_alias источника {src_idx} выходят за пределы [0, {vox_count})."
                )

            line_start = int(self.line_offsets[src_idx])
            line_end = int(self.line_offsets[src_idx + 1])
            line_count = line_end - line_start
            src_line_alias = self.line_alias[line_start:line_end]
            if np.any((src_line_alias < 0) | (src_line_alias >= line_count)):
                raise ValueError(
                    f"Индексы line_alias источника {src_idx} выходят за пределы [0, {line_count})."
                )


__all__ = [
    "SourceBuffer",
]

