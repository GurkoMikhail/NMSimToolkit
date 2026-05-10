import itertools
from copy import deepcopy
from typing import Dict, List, Any, Tuple
import numpy as np
from multiprocessing import Manager, Pool
from numpy.random import SeedSequence

from core.config.models import (
    SimulationConfig,
    CustomSweepProtocolConfig,
    StepAndShootProtocolConfig,
)
from core.config.builder import SceneBuilder
from core.transport.simulation_managers import SimulationManager
from core.transport.propagator import ParticlePropagator
from core.data.data_manager import DataManager
from core.data.data_handlers import DirectStreamHandler, SensitiveVolumeHandler, HistoryAssemblerHandler


def _find_nodes_by_names(root: Any, names: List[str]) -> List[Any]:
    from core.scene.nodes import SpatialNode, CompositeNode
    found = []
    def traverse(node):
        if isinstance(node, SpatialNode) and getattr(node, 'name', None) in names:
            found.append(node)
        if isinstance(node, CompositeNode):
            for child in node.childs:
                traverse(child)
    traverse(root)
    return found

def _worker_function(payload: Tuple[Dict[str, Any], int, Any]) -> None:
    task_dict, seed, file_lock = payload
    
    # 1. Validate Config
    final_config = SimulationConfig.model_validate(task_dict)
    
    # 2. Build Scene
    builder = SceneBuilder()
    root_scene = builder.build_scene(final_config.scene)
    
    # 3. Instantiate Propagator with seed
    rng = np.random.default_rng(seed)
    propagator = ParticlePropagator(rng=rng)
    
    # Ensure sources use the same rng.
    def set_rng_for_sources(node):
        from core.source.sources import Source
        from core.scene.nodes import CompositeNode

        if isinstance(node, Source):
            node.rng = rng

        if isinstance(node, CompositeNode):
            for child in node.childs:
                set_rng_for_sources(child)
    
    set_rng_for_sources(root_scene)

    # 4. Build Data Handlers
    handlers = []
    for h_config in final_config.data_manager.handlers:
        if h_config.type == 'DirectStreamHandler':
            handlers.append(DirectStreamHandler())
        elif h_config.type == 'SensitiveVolumeHandler':
            vols = _find_nodes_by_names(root_scene, h_config.sensitive_volumes)
            handlers.append(SensitiveVolumeHandler(sensitive_volumes=vols))
        elif h_config.type == 'HistoryAssemblerHandler':
            vols = _find_nodes_by_names(root_scene, h_config.sensitive_volumes)
            handlers.append(HistoryAssemblerHandler(sensitive_volumes=vols, save_initial_states=h_config.save_initial_states))

    # 5. Instantiate Managers
    sim_config = final_config.simulation_manager
    manager = SimulationManager(
        scene=root_scene,
        propagator=propagator,
        stop_time=sim_config.stop_time,
        particles_number=sim_config.particles_number,
        buffer_capacity=final_config.data_manager.buffer_capacity
    )
    manager.global_timer = sim_config.start_time
    manager.min_energy = sim_config.min_energy
    manager.name = f"Task_seed_{seed}"

    data_manager = DataManager(
        filename=final_config.data_manager.filename,
        handlers=handlers,
        queue=manager.queue,
        lock=file_lock
    )

    # 6. Run
    manager.start()
    data_manager.start()
    manager.join()
    data_manager.join()


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
            # Check if this string is EXACTLY a template to return original type (e.g. float)
            if node.startswith("${") and node.endswith("}"):
                var_name = node[2:-1]
                if var_name in context:
                    return context[var_name]
            
            # Otherwise, do string substitution for any embedded templates
            if "${" in node:
                import re
                def replace_vars(match):
                    var_name = match.group(1)
                    if var_name in context:
                        return str(context[var_name])
                    return match.group(0)
                return re.sub(r'\$\{([^}]+)\}', replace_vars, node)
                
            return node
        else:
            return node

    def run(self):
        """
        Executes the orchestrator run loop, distributing tasks across a multiprocessing pool.
        """
        pool_size = self.parsed_config.pool_size
        sweep_config = self.compile_protocol()
        tasks = self._generate_job_list(sweep_config)

        mp_manager = Manager()
        locks = {}
        
        seed_seq = SeedSequence()
        seeds = seed_seq.spawn(len(tasks))

        payloads = []
        for context, seed in zip(tasks, seeds):
            task_dict = deepcopy(self.raw_config_dict)
            injected_dict = self.inject_variables(task_dict, context)
            
            # File grouping for locking
            filename = injected_dict.get('data_manager', {}).get('filename', 'default.hdf')
            if filename not in locks:
                locks[filename] = mp_manager.Lock()
                
            # Extract just the integer value from SeedSequence spawn for simplicity
            seed_val = seed.generate_state(1)[0]
            
            payloads.append((injected_dict, seed_val, locks[filename]))

        if pool_size > 1:
            with Pool(pool_size) as pool:
                pool.map(_worker_function, payloads)
        else:
            for payload in payloads:
                _worker_function(payload)
