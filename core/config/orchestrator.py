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
            return CustomSweepProtocolConfig(variables={})

        if isinstance(protocol, CustomSweepProtocolConfig):
            return protocol

        if isinstance(protocol, StepAndShootProtocolConfig):
            angles = np.linspace(protocol.start_angle, protocol.end_angle, protocol.views).tolist()
            # If multiple cameras, we might want to interleave or just rotate them via templating.
            # But the primary sweep is over views.
            variables = {
                "current_angle": angles,
                "current_time": [float(protocol.time_per_view)] * protocol.views
            }
            return CustomSweepProtocolConfig(variables=variables)

        raise ValueError(f"Unknown protocol type: {type(protocol)}")

    def build_task_matrix(self, sweep_config: CustomSweepProtocolConfig) -> List[Dict[str, float]]:
        """
        Returns a flat list of dictionaries, where each dict represents
        a single permutation of variables for a simulation task.
        In StepAndShoot or synchronous sweeps, variables iterate together (zip).
        If true combinatorial sweeps are needed, we can expand this later.
        For now, we assume all lists in variables have the same length and we zip them.
        """
        if not sweep_config.variables:
            return [{}]

        keys = list(sweep_config.variables.keys())
        # Make sure all lists are the same length
        lengths = [len(v) for v in sweep_config.variables.values()]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All variables in the protocol sweep must have the same number of steps.")

        values_lists = [sweep_config.variables[k] for k in keys]

        tasks = []
        for combo in zip(*values_lists):
            task_context = dict(zip(keys, combo))
            tasks.append(task_context)

        return tasks

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
        tasks = self.build_task_matrix(sweep_config)

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
