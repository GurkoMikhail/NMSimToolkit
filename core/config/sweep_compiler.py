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

        if isinstance(protocol, StepAndShootProtocolConfig):
            steps = max(1, int(protocol.steps))
            endpoint = bool(protocol.endpoint)
            start_angle_val = float(protocol.start_angle)
            end_angle_val = float(protocol.end_angle)
            if protocol.rotation_direction == "CCW":
                angle_span = end_angle_val - start_angle_val
                target_end_angle = start_angle_val - angle_span
            else:
                target_end_angle = end_angle_val
            angles_list = np.linspace(
                start_angle_val,
                target_end_angle,
                steps,
                endpoint=endpoint,
            ).tolist()
            time_per_step_val = float(protocol.time_per_step)

            zipped_variables: Dict[str, List[float]] = {
                "gantry_angle": angles_list,
                "current_angle": angles_list,
                "step_time": [time_per_step_val] * steps,
                "current_time": [time_per_step_val] * steps,
            }

            if protocol.radius is not None:
                if isinstance(protocol.radius, list):
                    radius_list = [float(rad_val) for rad_val in protocol.radius]
                else:
                    radius_list = [float(protocol.radius)] * steps
                zipped_variables["radius"] = radius_list

            if protocol.zipped_variables:
                for var_key, var_vals in protocol.zipped_variables.items():
                    if var_key not in zipped_variables:
                        zipped_variables[var_key] = [float(val) for val in var_vals]

            return CustomSweepProtocolConfig(
                grid_variables=dict(protocol.grid_variables),
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
            # Точное совпадение с одиночным шаблоном "${var}" возвращает типизированное числовое значение
            exact_match = re.fullmatch(r"\$\{([^{}]+)\}", data.strip())
            if exact_match is not None:
                variable_name = exact_match.group(1)
                if variable_name not in context:
                    raise ValueError(
                        f"Неразрешенная переменная шаблона '${{{variable_name}}}' отсутствует в контексте задач: {list(context.keys())}"
                    )
                return context[variable_name]

            # Составные строковые выражения с шаблонами
            if "${" in data:
                def replace_pattern(match_object: re.Match[str]) -> str:
                    variable_name = match_object.group(1)
                    if variable_name not in context:
                        raise ValueError(
                            f"Неразрешенная переменная шаблона '${{{variable_name}}}' в выражении '{data}' отсутствует в контексте задач: {list(context.keys())}"
                        )
                    return str(context[variable_name])

                return re.sub(r"\$\{([^{}]+)\}", replace_pattern, data)

            return data
        else:
            return data


__all__ = ["SweepCompiler"]
