"""
Модуль изолированного воркера симуляции (simulation_worker).
Выполняет единичную задачу моделирования в отдельном процессе,
полностью изолирован от типов медицинских устройств и специфических протоколов.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from core.config.builder import SceneBuilder
from core.config.models import SimulationConfig
from core.data.data_handlers import (
    BaseDataHandler,
    DirectStreamHandler,
    HistoryAssemblerHandler,
    SensitiveVolumeHandler,
)
from core.data.data_manager import DataManager
from core.data.dose_map_handler import DoseMapHandler
from core.scene.dose_grid_node import DoseGridNode
from core.scene.nodes import CompositeNode, SpatialNode
from core.source.sources import Source
from core.transport.ipc_pause_bridge import IpcPauseBridge
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager


def _find_nodes_by_names(root_node: Any, target_names: List[str]) -> List[Any]:
    """Рекурсивный поиск узлов с указанными именами в поддереве сцены."""
    found_nodes: List[Any] = []

    def traverse(current_node: Any) -> None:
        if isinstance(current_node, SpatialNode) and current_node.name in target_names:
            found_nodes.append(current_node)
        if isinstance(current_node, CompositeNode):
            for child_node in current_node.childs:
                traverse(child_node)

    traverse(root_node)
    return found_nodes


def _find_dose_grids(root_node: Any) -> List[DoseGridNode]:
    """Рекурсивный поиск всех узлов DoseGridNode в поддереве сцены."""
    grids_list: List[DoseGridNode] = []
    if isinstance(root_node, DoseGridNode):
        grids_list.append(root_node)
    if isinstance(root_node, CompositeNode):
        for child_node in root_node.childs:
            grids_list.extend(_find_dose_grids(child_node))
    return grids_list


def simulation_worker_task(payload: Tuple[Any, ...]) -> Tuple[Dict[str, float], SimulationConfig]:
    """
    Точка входа единичной задачи симуляции для процесса из пула multiprocessing.
    Принимает кортеж аргументов payload, собирает сцену, настраивает сборщики данных
    и запускает перенос частиц.
    """
    task_dict = payload[0]
    random_seed = payload[1]
    file_lock = payload[2]
    telemetry_queue = payload[3] if len(payload) > 3 else None
    extra_handlers = payload[4] if len(payload) > 4 else None
    extra_handler_factory = payload[5] if len(payload) > 5 else None
    extra_handler_args = payload[6] if len(payload) > 6 else ()
    task_id = payload[7] if len(payload) > 7 else task_dict.get("_task_id", 0)

    # 1. Валидация конфигурации симуляции
    final_config = SimulationConfig.model_validate(task_dict)

    # 2. Сборка сцены через декларативный строитель
    builder = SceneBuilder()
    root_scene = builder.build_scene(final_config.scene)
    context_data: Dict[str, float] = task_dict.get("_context", {})

    # 3. Инициализация пропагатора с изолированным генератором случайных чисел
    random_generator = np.random.default_rng(random_seed)
    propagator = ParticlePropagator(rng=random_generator)

    def set_rng_for_sources(scene_node: Any) -> None:
        if isinstance(scene_node, Source):
            scene_node.rng = random_generator
        if isinstance(scene_node, CompositeNode):
            for child_node in scene_node.childs:
                set_rng_for_sources(child_node)

    set_rng_for_sources(root_scene)

    # 4. Построение конвейера обработчиков данных
    handlers: List[BaseDataHandler] = []
    for handler_config in final_config.data_manager.handlers:
        if handler_config.type == "DirectStreamHandler":
            handlers.append(DirectStreamHandler())
        elif handler_config.type == "SensitiveVolumeHandler":
            volumes = _find_nodes_by_names(root_scene, handler_config.sensitive_volumes)
            handlers.append(SensitiveVolumeHandler(sensitive_volumes=volumes))
        elif handler_config.type == "HistoryAssemblerHandler":
            volumes = _find_nodes_by_names(root_scene, handler_config.sensitive_volumes)
            handlers.append(
                HistoryAssemblerHandler(
                    sensitive_volumes=volumes,
                    save_initial_states=handler_config.save_initial_states,
                )
            )
        elif handler_config.type == "DoseMapHandler":
            grid_nodes = _find_dose_grids(root_scene)
            if handler_config.grid_names:
                grid_nodes = [
                    grid_node for grid_node in grid_nodes if grid_node.name in handler_config.grid_names
                ]
            handlers.append(DoseMapHandler(grid_nodes=grid_nodes, shm_name=handler_config.shm_name))

    # Внедрение внешних обработчиков (Dependency Injection)
    if extra_handlers:
        handlers.extend(extra_handlers)

    if extra_handler_factory is not None:
        created_handlers = extra_handler_factory(
            root_scene, task_dict, task_id, *extra_handler_args
        )
        if created_handlers:
            handlers.extend(created_handlers)

    # 5. Инициализация менеджера симуляции
    sim_config = final_config.simulation_manager
    manager = SimulationManager(
        scene=root_scene,
        propagator=propagator,
        stop_time=sim_config.stop_time,
        start_time=sim_config.start_time,
        particles_number=sim_config.particles_number,
        min_energy=sim_config.min_energy,
        buffer_capacity=final_config.data_manager.buffer_capacity,
        name=f"Task_seed_{random_seed}",
        seed=random_seed,
    )

    ipc_pause_event = task_dict.get("_run_event", task_dict.get("_pause_event"))
    pause_bridge: Optional[IpcPauseBridge] = None
    if ipc_pause_event is not None:
        if not bool(ipc_pause_event.is_set()):
            manager.pause()
        pause_bridge = IpcPauseBridge(
            manager=manager,
            ipc_event=ipc_pause_event,
            poll_interval_seconds=0.02,
        )
        pause_bridge.start()

    task_metadata: Dict[str, Any] = {
        "task_id": task_id,
        "context": context_data,
        "protocol_type": final_config.protocol.type if hasattr(final_config.protocol, "type") else "CustomSweep",
    }

    data_manager = DataManager(
        filename=final_config.data_manager.filename,
        handlers=handlers,
        queue=manager.queue,
        lock=file_lock,
        swmr=False,
        metadata=task_metadata,
    )

    # 6. Запуск вычислений с передачей статусов в телеметрию
    if telemetry_queue is not None:
        try:
            telemetry_queue.put({"type": "task_started", "task_id": task_id, "seed": random_seed})
        except (ValueError, OSError):
            pass

    try:
        manager.start()
        data_manager.start()
        manager.join()
        data_manager.join()
        if telemetry_queue is not None:
            try:
                telemetry_queue.put({"type": "task_finished", "task_id": task_id, "seed": random_seed})
            except (ValueError, OSError):
                pass
        return (context_data, final_config)
    except Exception as simulation_exception:
        if telemetry_queue is not None:
            try:
                telemetry_queue.put({"type": "task_error", "task_id": task_id, "error": str(simulation_exception)})
            except (ValueError, OSError):
                pass
        raise
    finally:
        if pause_bridge is not None:
            pause_bridge.stop_bridge()
            pause_bridge.join(timeout=1.0)


__all__ = ["simulation_worker_task"]
