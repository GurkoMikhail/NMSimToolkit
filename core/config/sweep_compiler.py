"""
Модуль компилятора протоколов сканирования и генератора матрицы параметров (SweepCompiler).
Отвечает за трансляцию протоколов (StepAndShoot, SPECT) в чистый CustomSweepProtocolConfig,
построение пространства задач (Grid/Zipped) и внедрение контекстных переменных в конфигурацию сцены.
"""

import itertools
import re
from typing import Any, Dict, List, Optional
import numpy as np

from core.config.models import (
    CustomSweepProtocolConfig,
    StepAndShootProtocolConfig,
    SpectProtocolConfig,
)


class SweepCompiler:
    """
    Компилятор протоколов сканирования и диспетчер матрицы параметров симуляции.
    """

    @classmethod
    def compile_protocol(cls, protocol: Optional[Any]) -> CustomSweepProtocolConfig:
        """
        Компилирует высокоуровневый протокол исследования в базовый CustomSweepProtocolConfig.
        Если протокол не задан, возвращает единичную пустую задачу.
        """
        if protocol is None:
            return CustomSweepProtocolConfig(grid_variables={}, zipped_variables={})

        if isinstance(protocol, CustomSweepProtocolConfig):
            return protocol

        if isinstance(protocol, (StepAndShootProtocolConfig, SpectProtocolConfig)):
            cameras_count = max(1, int(protocol.gamma_cameras))
            total_views = max(1, int(protocol.views))
            positions_count = max(1, total_views // cameras_count)
            endpoint = bool(protocol.endpoint)
            angles_list = np.linspace(
                float(protocol.start_angle),
                float(protocol.end_angle),
                positions_count,
                endpoint=endpoint,
            ).tolist()
            time_per_view_val = float(protocol.time_per_view)

            zipped_variables: Dict[str, List[float]] = {
                "gantry_angle": angles_list,
                "current_angle": angles_list,
                "current_time": [time_per_view_val] * positions_count,
            }

            if protocol.head_angles is not None:
                for head_index, head_offset in enumerate(protocol.head_angles):
                    zipped_variables[f"head_{head_index}_angle"] = [
                        float(angle_value + head_offset) for angle_value in angles_list
                    ]
            elif cameras_count > 1:
                step_offset = (
                    float(protocol.end_angle - protocol.start_angle) / max(1, cameras_count - 1)
                    if endpoint
                    else float(protocol.end_angle - protocol.start_angle) / cameras_count
                )
                for head_index in range(cameras_count):
                    zipped_variables[f"head_{head_index}_angle"] = [
                        float(angle_value + step_offset * head_index)
                        for angle_value in angles_list
                    ]

            return CustomSweepProtocolConfig(
                grid_variables={},
                zipped_variables=zipped_variables,
            )

        raise ValueError(f"Неизвестный тип протокола исследования: {type(protocol)}")

    @staticmethod
    def generate_job_matrix(sweep_config: CustomSweepProtocolConfig) -> List[Dict[str, float]]:
        """
        Генерирует плоский список словарей параметров для каждой задачи симуляции.
        Вычисляет декартово произведение по grid_variables и синхронную сшивку по zipped_variables.
        """
        # 1. Сетка параметров (декартово произведение)
        if sweep_config.grid_variables:
            grid_keys = list(sweep_config.grid_variables.keys())
            grid_combinations = [
                dict(zip(grid_keys, combination_values))
                for combination_values in itertools.product(*sweep_config.grid_variables.values())
            ]
        else:
            grid_combinations = [{}]

        # 2. Связанные параметры (zipped)
        if sweep_config.zipped_variables:
            zip_keys = list(sweep_config.zipped_variables.keys())
            zip_combinations = [
                dict(zip(zip_keys, combination_values))
                for combination_values in zip(*sweep_config.zipped_variables.values())
            ]
        else:
            zip_combinations = [{}]

        # 3. Декартово произведение пространств параметров
        job_matrix: List[Dict[str, float]] = []
        for grid_item in grid_combinations:
            for zipped_item in zip_combinations:
                job_matrix.append({**grid_item, **zipped_item})
        return job_matrix

    @classmethod
    def inject_variables(cls, data: Any, context: Dict[str, float]) -> Any:
        """
        Рекурсивно обходит структуры данных, подставляя строковые плейсхолдеры
        вида "${variable_name}" реальными значениями из контекста выполнения.
        """
        if isinstance(data, dict):
            resolved_dict = {}
            for dictionary_key, dictionary_value in data.items():
                resolved_dict[dictionary_key] = cls.inject_variables(dictionary_value, context)
            return resolved_dict
        elif isinstance(data, list):
            return [cls.inject_variables(list_element, context) for list_element in data]
        elif isinstance(data, str):
            # Точное совпадение с шаблоном "${var}" возвращает типизированное числовое значение
            if data.startswith("${") and data.endswith("}"):
                variable_name = data[2:-1]
                if variable_name in context:
                    return context[variable_name]

            # Составные строковые выражения с шаблонами
            if "${" in data:
                def replace_pattern(match_object: re.Match) -> str:
                    variable_name = match_object.group(1)
                    if variable_name in context:
                        return str(context[variable_name])
                    return match_object.group(0)

                return re.sub(r'\$\{([^}]+)\}', replace_pattern, data)

            return data
        else:
            return data


__all__ = ["SweepCompiler"]
