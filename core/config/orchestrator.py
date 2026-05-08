import itertools
from copy import deepcopy
from typing import Dict, List, Any
import numpy as np

from core.config.models import (
    SimulationConfig,
    CustomSweepProtocolConfig,
    StepAndShootProtocolConfig,
)

class Orchestrator:
    def __init__(self, raw_config_dict: Dict[str, Any]):
        """
        Initializes the orchestrator with a raw dictionary read from YAML.
        """
        self.raw_config_dict = raw_config_dict
        # We validate the initial schema to catch protocol errors and static scene structure
        # But we do not use this fully parsed SimulationConfig for the final run,
        # as it may contain unresolved string templates.
        self.parsed_config = SimulationConfig.model_validate(raw_config_dict)

    def compile_protocol(self) -> CustomSweepProtocolConfig:
        """
        Converts any high-level protocol into the base CustomSweepProtocolConfig.
        If no protocol is provided, it returns a single task dummy sweep.
        """
        protocol = self.parsed_config.protocol

        if protocol is None:
            return CustomSweepProtocolConfig(grid_variables={}, zipped_variables={})

        if isinstance(protocol, CustomSweepProtocolConfig):
            return protocol

        if isinstance(protocol, StepAndShootProtocolConfig):
            angles = np.linspace(protocol.start_angle, protocol.end_angle, protocol.views).tolist()
            # In Step and Shoot, time steps and rotation steps are tied synchronously.
            zipped_vars = {
                "current_angle": angles,
                "current_time": [float(protocol.time_per_view)] * protocol.views
            }
            return CustomSweepProtocolConfig(grid_variables={}, zipped_variables=zipped_vars)

        raise ValueError(f"Unknown protocol type: {type(protocol)}")

    def _generate_job_list(self, sweep_config: CustomSweepProtocolConfig) -> List[Dict[str, float]]:
        """
        Returns a flat list of dictionaries representing every simulation task.
        Performs a Cartesian product over `grid_variables` and concurrent iteration over `zipped_variables`.
        """
        # 1. Grid Sweep Space
        if sweep_config.grid_variables:
            grid_keys = list(sweep_config.grid_variables.keys())
            grid_combos = [dict(zip(grid_keys, combo)) for combo in itertools.product(*sweep_config.grid_variables.values())]
        else:
            grid_combos = [{}]

        # 2. Zipped Sweep Space
        if sweep_config.zipped_variables:
            zip_keys = list(sweep_config.zipped_variables.keys())
            zip_combos = [dict(zip(zip_keys, combo)) for combo in zip(*sweep_config.zipped_variables.values())]
        else:
            zip_combos = [{}]

        # 3. Final Merge (The Cross)
        return [{**g, **z} for g in grid_combos for z in zip_combos]

    def inject_variables(self, node: Any, context: Dict[str, float]) -> Any:
        """
        Recursively traverse dictionaries and lists, replacing string templates
        like "${current_angle}" with actual float values from the context.
        """
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                new_dict[k] = self.inject_variables(v, context)
            return new_dict
        elif isinstance(node, list):
            return [self.inject_variables(v, context) for v in node]
        elif isinstance(node, str):
            # Check if this string is EXACTLY a template.
            # E.g. "${current_angle}" -> matches 'current_angle'
            if node.startswith("${") and node.endswith("}"):
                var_name = node[2:-1]
                if var_name in context:
                    return context[var_name]
            return node
        else:
            return node

    def run(self):
        """
        Example of the orchestrator run loop.
        In a real scenario, this would use multiprocessing.
        """
        sweep_config = self.compile_protocol()
        tasks = self._generate_job_list(sweep_config)

        results = []
        for i, context in enumerate(tasks):
            # Deepcopy the original dictionary to avoid mutations across tasks
            task_dict = deepcopy(self.raw_config_dict)

            # Inject numerical values
            injected_dict = self.inject_variables(task_dict, context)

            # Now validate the injected dictionary.
            # This triggers all Pydantic validators, Pint conversions, etc.
            # and will fail if templates weren't properly resolved into numbers.
            final_config = SimulationConfig.model_validate(injected_dict)

            # In a real implementation:
            # engine = SimulationManager(...)
            # engine.run()
            # results.append(output)
            results.append((context, final_config))

        return results
