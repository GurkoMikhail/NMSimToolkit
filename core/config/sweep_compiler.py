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
from core.config.units import convert_from_hepunits


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
            initial_time_val = float(protocol.initial_time)
            if protocol.sequential_time:
                start_times = [
                    initial_time_val + step_idx * time_per_step_val
                    for step_idx in range(steps)
                ]
                stop_times = [
                    initial_time_val + (step_idx + 1) * time_per_step_val
                    for step_idx in range(steps)
                ]
            else:
                start_times = [initial_time_val] * steps
                stop_times = [initial_time_val + time_per_step_val] * steps

            zipped_variables: Dict[str, List[float]] = {
                "gantry_angle": angles_list,
                "current_angle": angles_list,
                "start_time": start_times,
                "stop_time": stop_times,
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
                metadata=protocol.metadata,
                grid_variables=dict(protocol.grid_variables),
                zipped_variables=zipped_variables,
            )

        if isinstance(protocol, CustomSweepProtocolConfig):
            return protocol

        raise ValueError(f"Неизвестный тип протокола исследования: {type(protocol)}")

    @classmethod
    def apply_protocol_defaults(
        cls,
        raw_config_dict: Dict[str, Any],
        protocol: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Автоматически подставляет шаблонные переменные ${start_time} и ${stop_time}
        в секцию simulation_manager для протокола StepAndShoot, если пользователь
        не задал их явно в конфигурации.
        """
        proto_val = protocol if protocol is not None else raw_config_dict.get("protocol")
        is_step_and_shoot = (
            isinstance(proto_val, StepAndShootProtocolConfig)
            or (isinstance(proto_val, dict) and proto_val.get("type") == "StepAndShoot")
        )
        if not is_step_and_shoot:
            return raw_config_dict

        sim_mgr = raw_config_dict.get("simulation_manager")
        if isinstance(sim_mgr, dict):
            if "stop_time" not in sim_mgr or sim_mgr["stop_time"] is None:
                sim_mgr["stop_time"] = "${stop_time}"
            if "start_time" not in sim_mgr or sim_mgr["start_time"] is None:
                stop_val_str = str(sim_mgr.get("stop_time", "")).strip()
                if stop_val_str == "${stop_time}":
                    sim_mgr["start_time"] = "${start_time}"

        return raw_config_dict

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
        вида "${variable_name}" или "${variable_name unit}" реальными значениями из контекста выполнения.
        """
        if isinstance(data, dict):
            resolved_dict = {}
            for dictionary_key, dictionary_value in data.items():
                resolved_dict[dictionary_key] = cls.inject_variables(dictionary_value, context)
            return resolved_dict
        elif isinstance(data, list):
            return [cls.inject_variables(list_element, context) for list_element in data]
        elif isinstance(data, str):
            # Точное совпадение с одиночным шаблоном "${var}" или "${var unit}"
            exact_match = re.fullmatch(r"\$\{([^{}]+)\}", data.strip())
            if exact_match is not None:
                inner_expression = exact_match.group(1).strip()
                tokens = inner_expression.split(None, 1)
                variable_name = tokens[0]
                if variable_name not in context:
                    raise ValueError(
                        f"Неразрешенная переменная шаблона '${{{variable_name}}}' отсутствует в контексте задач: {list(context.keys())}"
                    )
                if len(tokens) == 1:
                    return context[variable_name]
                target_unit = tokens[1].strip()
                return convert_from_hepunits(float(context[variable_name]), target_unit)

            # Составные строковые выражения с шаблонами
            if "${" in data:
                def replace_pattern(match_object: re.Match[str]) -> str:
                    inner_expression = match_object.group(1).strip()
                    tokens = inner_expression.split(None, 1)
                    variable_name = tokens[0]
                    if variable_name not in context:
                        raise ValueError(
                            f"Неразрешенная переменная шаблона '${{{variable_name}}}' в выражении '{data}' отсутствует в контексте задач: {list(context.keys())}"
                        )
                    if len(tokens) == 1:
                        return str(context[variable_name])
                    target_unit = tokens[1].strip()
                    converted_value = convert_from_hepunits(float(context[variable_name]), target_unit)
                    return str(converted_value)

                return re.sub(r"\$\{([^{}]+)\}", replace_pattern, data)

            return data
        else:
            return data


__all__ = ["SweepCompiler"]
