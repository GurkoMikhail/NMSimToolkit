from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from core.config.models import SimulationConfig
from core.config.sweep_compiler import SweepCompiler


def load_raw_config(filepath: str | Path) -> Dict[str, Any]:
    """Загружает YAML-файл и возвращает сырой словарь конфигурации без секции YAML-якорей Materials."""
    with open(filepath, 'r', encoding='utf-8') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    if not isinstance(config_dict, dict):
        raise ValueError(f"Некорректный формат YAML-конфигурации в '{filepath}': ожидался словарь.")

    if 'Materials' in config_dict:
        del config_dict['Materials']

    return dict(config_dict)


def load_simulation_config(
    filepath: str | Path,
    context: Optional[Dict[str, float]] = None,
    resolve_protocol: bool = True
) -> SimulationConfig:
    """Загружает YAML-файл и парсит его в строго типизированную модель SimulationConfig.
    
    Если resolve_protocol=True и в конфигурации задан протокол сканирования (или передан context),
    выполняет подстановку шаблонных переменных (${var}) начальными значениями.
    """
    config_dict = load_raw_config(filepath)
    SweepCompiler.apply_protocol_defaults(config_dict)

    if resolve_protocol:
        has_protocol = bool(config_dict.get('protocol'))
        if context is not None:
            config_dict = SweepCompiler.inject_variables(deepcopy(config_dict), context)
        elif has_protocol:
            protocol_obj = SimulationConfig.model_validate(config_dict).protocol
            if protocol_obj is not None:
                sweep_proto = SweepCompiler.compile_protocol(protocol_obj)
                tasks = SweepCompiler.generate_job_matrix(sweep_proto)
                if tasks:
                    config_dict = SweepCompiler.inject_variables(deepcopy(config_dict), tasks[0])

    return SimulationConfig.model_validate(config_dict)


__all__ = ["load_raw_config", "load_simulation_config"]
