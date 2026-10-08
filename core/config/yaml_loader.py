from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from core.config.models import SimulationConfig
from core.config.sweep_compiler import SweepCompiler


def load_raw_config(filepath: str | Path) -> dict:
    """Loads a YAML file and returns the raw dictionary, stripping YAML anchors."""
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    if 'Materials' in config_dict:
        del config_dict['Materials']
        
    return config_dict


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
