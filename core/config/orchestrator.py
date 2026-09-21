import itertools
import re
from copy import deepcopy
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from multiprocessing import Manager, Pool
from numpy.random import SeedSequence

from core.config.models import (
    SimulationConfig,
    CustomSweepProtocolConfig,
    StepAndShootProtocolConfig,
)
from core.config.builder import SceneBuilder
from core.scene.nodes import SpatialNode, CompositeNode
from core.source.sources import Source
from core.transport.simulation_managers import SimulationManager
from core.transport.propagator import ParticlePropagator
from core.data.data_manager import DataManager
from core.data.data_handlers import DirectStreamHandler, SensitiveVolumeHandler, HistoryAssemblerHandler


def _find_nodes_by_names(root: Any, names: List[str]) -> List[Any]:
    found = []
    def traverse(node):
        if isinstance(node, SpatialNode) and node.name in names:
            found.append(node)
        if isinstance(node, CompositeNode):
            for child in node.childs:
                traverse(child)
    traverse(root)
    return found


def _worker_function(payload: Tuple[Dict[str, Any], int, Any]) -> None:
    task_dict, seed, file_lock = payload
    
    # 1. Валидация конфигурации
    final_config = SimulationConfig.model_validate(task_dict)
    
    # 2. Построение дерева сцены
    builder = SceneBuilder()
    root_scene = builder.build_scene(final_config.scene)
    
    # 3. Инициализация генератора случайных чисел и пропогатора
    rng = np.random.default_rng(seed)
    propagator = ParticlePropagator(rng=rng)
    
    # Гарантируем использование того же rng всеми источниками
    def set_rng_for_sources(node):
        if isinstance(node, Source):
            node.rng = rng

        if isinstance(node, CompositeNode):
            for child in node.childs:
                set_rng_for_sources(child)
    
    set_rng_for_sources(root_scene)

    # 4. Построение обработчиков данных
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

    # 5. Инициализация менеджеров выполнения
    sim_config = final_config.simulation_manager
    manager = SimulationManager(
        scene=root_scene,
        propagator=propagator,
        stop_time=sim_config.stop_time,
        start_time=sim_config.start_time,
        particles_number=sim_config.particles_number,
        min_energy=sim_config.min_energy,
        buffer_capacity=final_config.data_manager.buffer_capacity,
        name=f"Task_seed_{seed}",
        seed=seed
    )

    data_manager = DataManager(
        filename=final_config.data_manager.filename,
        handlers=handlers,
        queue=manager.queue,
        lock=file_lock
    )

    # 6. Запуск и ожидание завершения потоков
    manager.start()
    data_manager.start()
    manager.join()
    data_manager.join()


class Orchestrator:
    def __init__(self, raw_config_dict: Any):
        """
        Инициализирует оркестратор словарем конфигурации или объектом SimulationConfig.
        """
        if isinstance(raw_config_dict, SimulationConfig):
            self.raw_config_dict = raw_config_dict.model_dump()
            self.parsed_config = raw_config_dict
        else:
            self.raw_config_dict = dict(raw_config_dict)
            self.parsed_config = SimulationConfig.model_validate(raw_config_dict)

    def compile_protocol(self) -> CustomSweepProtocolConfig:
        """
        Компилирует высокоуровневый протокол в CustomSweepProtocolConfig.
        """
        protocol = self.parsed_config.protocol

        if protocol is None:
            return CustomSweepProtocolConfig(grid_variables={}, zipped_variables={})

        if isinstance(protocol, CustomSweepProtocolConfig):
            return protocol

        if isinstance(protocol, StepAndShootProtocolConfig):
            angles = np.linspace(protocol.start_angle, protocol.end_angle, protocol.views).tolist()
            zipped_vars = {
                "current_angle": angles,
                "current_time": [float(protocol.time_per_view)] * protocol.views
            }
            return CustomSweepProtocolConfig(grid_variables={}, zipped_variables=zipped_vars)

        raise ValueError(f"Неизвестный тип протокола: {type(protocol)}")

    def _generate_job_list(self, sweep_config: CustomSweepProtocolConfig) -> List[Dict[str, float]]:
        """
        Генерирует плоский список словарей параметров для каждой задачи симуляции.
        """
        # 1. Декартово произведение по grid_variables
        if sweep_config.grid_variables:
            grid_keys = list(sweep_config.grid_variables.keys())
            grid_combos = [dict(zip(grid_keys, combo)) for combo in itertools.product(*sweep_config.grid_variables.values())]
        else:
            grid_combos = [{}]

        # 2. Синхронная итерация по zipped_variables
        if sweep_config.zipped_variables:
            zip_keys = list(sweep_config.zipped_variables.keys())
            zip_combos = [dict(zip(zip_keys, combo)) for combo in zip(*sweep_config.zipped_variables.values())]
        else:
            zip_combos = [{}]

        # 3. Объединение пространств
        return [{**g, **z} for g in grid_combos for z in zip_combos]

    def inject_variables(self, node: Any, context: Dict[str, float]) -> Any:
        """
        Рекурсивно подставляет значения переменных контекста в шаблоны строк ${var}.
        """
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                new_dict[k] = self.inject_variables(v, context)
            return new_dict
        elif isinstance(node, list):
            return [self.inject_variables(v, context) for v in node]
        elif isinstance(node, str):
            if node.startswith("${") and node.endswith("}"):
                var_name = node[2:-1]
                if var_name in context:
                    return context[var_name]
            
            if "${" in node:
                def replace_vars(match):
                    var_name = match.group(1)
                    if var_name in context:
                        return str(context[var_name])
                    return match.group(0)
                return re.sub(r'\$\{([^}]+)\}', replace_vars, node)
                
            return node
        else:
            return node

    def generate_tasks(self) -> List[Dict[str, float]]:
        """
        Генерирует плоский список словарей параметров для каждой задачи симуляции согласно скомпилированному протоколу.
        """
        return self._generate_job_list(self.compile_protocol())

    def run(self) -> None:
        """
        Выполняет параллельный цикл задач симуляции через пул процессов
        с гарантированным закрытием mp_manager в finally блоке.
        """
        pool_size = self.parsed_config.pool_size
        tasks = self.generate_tasks()

        mp_manager = Manager()
        try:
            locks = {}
            
            seed_seq = SeedSequence()
            seeds = seed_seq.spawn(len(tasks))

            payloads = []
            for context, seed in zip(tasks, seeds):
                task_dict = deepcopy(self.raw_config_dict)
                injected_dict = self.inject_variables(task_dict, context)
                
                filename = injected_dict.get('data_manager', {}).get('filename', 'default.hdf')
                if filename not in locks:
                    locks[filename] = mp_manager.Lock()
                    
                seed_val = seed.generate_state(1)[0]
                payloads.append((injected_dict, seed_val, locks[filename]))

            if pool_size > 1:
                with Pool(pool_size) as pool:
                    pool.map(_worker_function, payloads)
            else:
                for payload in payloads:
                    _worker_function(payload)
        finally:
            try:
                mp_manager.shutdown()
            except (OSError, ValueError):
                pass
