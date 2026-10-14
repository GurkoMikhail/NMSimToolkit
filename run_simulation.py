import os

os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['NUMEXPR_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import yaml
from core.config.orchestrator import Orchestrator

def main() -> None:
    """Точка входа CLI для запуска параллельного оркестратора симуляции по YAML-конфигурации."""
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
    else:
        config_file = "nema_1_cam.yaml"
    print(f"Loading configuration from {config_file}...")

    with open(config_file, 'r', encoding='utf-8') as config_stream:
        raw_dict = yaml.safe_load(config_stream)

    print("Initializing orchestrator...")
    orch = Orchestrator(raw_dict)

    print("Starting simulation...")
    orch.run()

    print("Simulation completed successfully.")


__all__ = ["main"]

if __name__ == '__main__':
    main()