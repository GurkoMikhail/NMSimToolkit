import yaml
from pathlib import Path
from core.config.models import SimulationConfig

def load_raw_config(filepath: str | Path) -> dict:
    """Loads a YAML file and returns the raw dictionary, stripping YAML anchors."""
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    if 'Materials' in config_dict:
        del config_dict['Materials']
        
    return config_dict

def load_simulation_config(filepath: str | Path) -> SimulationConfig:
    """Loads a YAML file and parses it into a strictly typed SimulationConfig Pydantic model."""
    config_dict = load_raw_config(filepath)
    return SimulationConfig.model_validate(config_dict)
