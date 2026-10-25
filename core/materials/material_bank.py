"""
Модуль скомпилированного банка материалов (MaterialBank) для быстрых обращений из Numba-кернелов.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index

MaterialInfoDType = np.dtype([
    ('density', Float),
    ('Z', Float),
    ('A', Float),
])

MaterialPointerDType = np.dtype([
    ('start_idx', Index),
    ('length', Index),
    ('lookup_offset', Index),
])


class MaterialBank(NamedTuple):
    """
    Плоские массивы свойств материалов, таблиц линейных коэффициентов ослабления (LAC),
    предвычисленных натуральных логарифмов и двухуровневого логарифмического индекса
    поиска интервала, скомпилированные для используемых в сцене материалов.
    """

    mat_info_buffer: NDArray[np.void]
    mat_pointers: NDArray[np.void]
    physics_energy_grid: NDArray[Float]
    physics_lac_table: NDArray[Float]
    physics_log_energy_grid: NDArray[Float]
    physics_log_lac_table: NDArray[Float]
    lookup_bin_start: NDArray[Index]
    lookup_log_e_min: Float
    lookup_inv_dlog_e: Float
    lookup_num_bins: int
    lookup_max_advance: int

    def _validate(self) -> None:
        """
        Проверяет целостность типов, согласованность размерностей всех буферов
        и монотонность индекса корзин lookup_bin_start внутри каждого материала.
        """
        if self.mat_info_buffer.dtype != MaterialInfoDType:
            raise ValueError("Некорректный тип данных mat_info_buffer.")
        if self.mat_pointers.dtype != MaterialPointerDType:
            raise ValueError("Некорректный тип данных mat_pointers.")
        if self.mat_info_buffer.ndim != 1 or self.mat_pointers.ndim != 1:
            raise ValueError("Массивы mat_info_buffer и mat_pointers должны быть одномерными.")
        if self.mat_info_buffer.shape[0] != self.mat_pointers.shape[0]:
            raise ValueError("Размеры массивов mat_info_buffer и mat_pointers должны совпадать.")

        if self.physics_energy_grid.ndim != 1 or self.physics_energy_grid.dtype != Float:
            raise ValueError("Массив physics_energy_grid должен быть одномерным массивом типа Float.")
        if self.physics_lac_table.ndim != 2 or self.physics_lac_table.dtype != Float:
            raise ValueError("Таблица physics_lac_table должна быть двумерным массивом типа Float.")
        if self.physics_lac_table.shape[0] != self.physics_energy_grid.shape[0]:
            raise ValueError(
                "Число строк physics_lac_table должно совпадать с длиной physics_energy_grid."
            )

        if (
            self.physics_log_energy_grid.ndim != 1
            or self.physics_log_energy_grid.dtype != Float
            or self.physics_log_energy_grid.shape != self.physics_energy_grid.shape
        ):
            raise ValueError(
                "Массив physics_log_energy_grid должен быть одномерным массивом Float "
                "того же размера, что и physics_energy_grid."
            )

        if (
            self.physics_log_lac_table.ndim != 2
            or self.physics_log_lac_table.dtype != Float
            or self.physics_log_lac_table.shape != self.physics_lac_table.shape
        ):
            raise ValueError(
                "Таблица physics_log_lac_table должна быть двумерным массивом Float "
                "той же формы, что и physics_lac_table."
            )

        if self.lookup_bin_start.ndim != 1 or self.lookup_bin_start.dtype != Index:
            raise ValueError("Массив lookup_bin_start должен быть одномерным массивом типа Index.")
        if self.lookup_num_bins <= 0:
            raise ValueError("Число корзин lookup_num_bins должно быть строго положительным.")
        if self.lookup_max_advance < 0:
            raise ValueError("Максимальное число шагов досмотра lookup_max_advance не может быть отрицательным.")
        if not np.isfinite(self.lookup_log_e_min) or not np.isfinite(self.lookup_inv_dlog_e) or self.lookup_inv_dlog_e <= 0.0:
            raise ValueError(
                "Параметры логарифмической сетки lookup_log_e_min и lookup_inv_dlog_e "
                "должны быть конечными, при этом lookup_inv_dlog_e > 0."
            )

        total_grid_len = int(self.physics_energy_grid.shape[0])
        active_materials_count = 0
        seen_lookup_offsets = set()

        for mat_id in range(self.mat_pointers.shape[0]):
            pointer = self.mat_pointers[mat_id]
            start_idx = int(pointer['start_idx'])
            length = int(pointer['length'])
            lookup_offset = int(pointer['lookup_offset'])

            if start_idx < 0 or length < 0 or start_idx + length > total_grid_len:
                raise ValueError(
                    f"Некорректный диапазон сетки (start_idx={start_idx}, length={length}) "
                    f"для материала ID={mat_id} при общем размере сетки {total_grid_len}."
                )

            if length == 0:
                continue

            active_materials_count += 1
            if (
                lookup_offset < 0
                or lookup_offset + self.lookup_num_bins > self.lookup_bin_start.shape[0]
                or (lookup_offset % self.lookup_num_bins) != 0
            ):
                raise ValueError(
                    f"Некорректное смещение lookup_offset={lookup_offset} для материала ID={mat_id} "
                    f"при lookup_num_bins={self.lookup_num_bins} и размере буфера {self.lookup_bin_start.shape[0]}."
                )
            if lookup_offset in seen_lookup_offsets:
                raise ValueError(
                    f"Дублирующееся смещение lookup_offset={lookup_offset} для материала ID={mat_id}."
                )
            seen_lookup_offsets.add(lookup_offset)

            mat_grid = self.physics_energy_grid[start_idx : start_idx + length]
            if np.any(~np.isfinite(mat_grid)) or np.any(mat_grid <= 0.0):
                raise ValueError(
                    f"Энергетическая сетка материала ID={mat_id} должна содержать только конечные "
                    f"строго положительные значения."
                )
            if length > 1 and np.any(np.diff(mat_grid) < 0.0):
                raise ValueError(
                    f"Нарушена монотонность энергетической сетки physics_energy_grid внутри материала ID={mat_id}."
                )

            mat_bins = self.lookup_bin_start[lookup_offset : lookup_offset + self.lookup_num_bins]
            max_valid_idx = max(length - 2, 0)
            if np.any(mat_bins < 0) or np.any(mat_bins > max_valid_idx):
                raise ValueError(
                    f"Значения lookup_bin_start материала ID={mat_id} выходят за допустимый диапазон "
                    f"[0, {max_valid_idx}]."
                )
            if self.lookup_num_bins > 1 and np.any(np.diff(mat_bins) < 0):
                raise ValueError(
                    f"Нарушена монотонность индекса lookup_bin_start внутри материала ID={mat_id}."
                )

        expected_lookup_size = active_materials_count * self.lookup_num_bins
        if self.lookup_bin_start.shape[0] != expected_lookup_size:
            raise ValueError(
                f"Размер lookup_bin_start ({self.lookup_bin_start.shape[0]}) не совпадает с "
                f"произведением числа активных материалов ({active_materials_count}) "
                f"на lookup_num_bins ({self.lookup_num_bins}): ожидалось {expected_lookup_size}."
            )


__all__ = [
    "MaterialInfoDType",
    "MaterialPointerDType",
    "MaterialBank",
]


