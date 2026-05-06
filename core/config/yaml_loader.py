import yaml
from pathlib import Path
from core.config.models import SimulationConfig

def load_simulation_config(filepath: str | Path) -> SimulationConfig:
    """Loads a YAML file and parses it into a strictly typed SimulationConfig Pydantic model."""
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # We strip 'Materials' if present as PyYAML resolves anchors natively
    if 'Materials' in config_dict:
        del config_dict['Materials']

    return SimulationConfig.model_validate(config_dict)
