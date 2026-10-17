"""
Модуль изолированного воркера симуляции (simulation_worker).
Выполняет единичную задачу моделирования в отдельном процессе,
полностью изолирован от типов медицинских устройств и специфических протоколов.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import hepunits as units
import numpy as np

from core.config.logging_setup import (
    setup_task_logging,
    close_logging_handlers,
)
from core.config.builder import SceneBuilder
from core.config.models import SimulationConfig
from core.data.metadata_collector import ProcedureMetadataCollector
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
from core.transport.ipc_pause_bridge import IpcPauseBridge
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager

_logger = logging.getLogger(__name__)


def _find_nodes_by_names(root_node: SpatialNode, target_names: List[str]) -> List[SpatialNode]:
    """Рекурсивный поиск узлов с указанными именами в поддереве сцены."""
    found_nodes: List[SpatialNode] = []

    def traverse(current_node: SpatialNode) -> None:
        if current_node.name in target_names:
            found_nodes.append(current_node)
        if isinstance(current_node, CompositeNode):
            for child_node in current_node.childs:
                traverse(child_node)

    traverse(root_node)
    return found_nodes


def _find_dose_grids(root_node: SpatialNode) -> List[DoseGridNode]:
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
    worker_log_handlers = setup_task_logging(final_config.logging, task_id=int(task_id))
    task_start_time = time.perf_counter()
    _logger.info(f"Воркер {task_id}: запуск задачи симуляции (seed={random_seed}).")

    data_manager: Optional[DataManager] = None
    pause_bridge: Optional[IpcPauseBridge] = None
    handlers: List[BaseDataHandler] = []

    try:
        # 2. Сборка сцены через декларативный строитель
        builder = SceneBuilder()
        built_scene = builder.build_scene(final_config.scene)
        if isinstance(built_scene, CompositeNode):
            root_scene: CompositeNode = built_scene
        else:
            root_scene = CompositeNode(name="World")
            root_scene.add_child(built_scene)
        context_data: Dict[str, float] = task_dict.get("_context", {})

        # 3. Проверка параметров менеджера симуляции и предварительное сканирование чекпоинта HDF5
        sim_config = final_config.simulation_manager
        if isinstance(sim_config.particles_number, str):
            raise ValueError(
                f"Неразрешенный шаблон интерполяции в particles_number: '{sim_config.particles_number}'. "
                f"Все макросы обязаны быть разрешены протоколом перед запуском воркера."
            )
        if isinstance(sim_config.min_energy, str):
            raise ValueError(
                f"Неразрешенный шаблон интерполяции в min_energy: '{sim_config.min_energy}'. "
                f"Все макросы обязаны быть разрешены протоколом перед запуском воркера."
            )
        if isinstance(sim_config.stop_time, str):
            raise ValueError(
                f"Неразрешенный шаблон интерполяции в stop_time: '{sim_config.stop_time}'. "
                f"Все макросы обязаны быть разрешены протоколом перед запуском воркера."
            )

        requested_start_time = float(sim_config.start_time)
        requested_stop_time = float(sim_config.stop_time)
        effective_start_time = requested_start_time
        effective_stop_time = requested_stop_time
        effective_seed = int(random_seed)
        initial_particle_id = 0
        rng_state: Optional[Dict[str, Any]] = None
        effective_task_id: Any = task_id

        resume_state = DataManager.inspect_resume_state(
            final_config.data_manager.filename,
            lock=file_lock,
            context=context_data,
        )
        if resume_state is not None:
            initial_particle_id = int(resume_state["next_particle_id"])
            effective_task_id = int(resume_state["next_task_id"])
            if resume_state.get("actual_stop_time") is not None:
                saved_stop_time = float(resume_state["actual_stop_time"])
                if saved_stop_time >= requested_stop_time:
                    _logger.info(
                        f"Воркер {task_id}: задача уже полностью рассчитана "
                        f"(actual_stop_time={saved_stop_time} >= stop_time={requested_stop_time}), пропуск."
                    )
                    if telemetry_queue is not None:
                        try:
                            telemetry_queue.put({"type": "task_finished", "task_id": task_id, "seed": random_seed})
                        except (ValueError, OSError) as queue_error:
                            _logger.debug(
                                f"Не удалось отправить статус task_finished в очередь телеметрии: {queue_error}"
                            )
                    return (context_data, final_config)

                effective_start_time = saved_stop_time
                effective_stop_time = requested_stop_time
                effective_seed = int(resume_state["seed"])
                rng_state = resume_state["rng_state"]
                _logger.info(
                    f"Воркер {task_id}: обнаружен чекпоинт в '{final_config.data_manager.filename}'. "
                    f"Бесшовный досчёт на интервале [{effective_start_time}, {effective_stop_time}] нс "
                    f"(seed={effective_seed}, initial_particle_id={initial_particle_id}, task_id={effective_task_id})."
                )

        # 4. Построение конвейера обработчиков данных
        dm_cfg = final_config.data_manager
        for handler_config in dm_cfg.handlers:
            if handler_config.type == "DirectStreamHandler":
                handlers.append(
                    DirectStreamHandler(
                        floating_point_precision=dm_cfg.floating_point_precision,
                        compression=dm_cfg.compression,
                        compression_opts=dm_cfg.compression_opts,
                    )
                )
            elif handler_config.type == "SensitiveVolumeHandler":
                volumes = _find_nodes_by_names(root_scene, handler_config.sensitive_volumes)
                handlers.append(
                    SensitiveVolumeHandler(
                        sensitive_volumes=volumes,
                        floating_point_precision=dm_cfg.floating_point_precision,
                        compression=dm_cfg.compression,
                        compression_opts=dm_cfg.compression_opts,
                    )
                )
            elif handler_config.type == "HistoryAssemblerHandler":
                volumes = _find_nodes_by_names(root_scene, handler_config.sensitive_volumes)
                handlers.append(
                    HistoryAssemblerHandler(
                        sensitive_volumes=volumes,
                        save_initial_states=handler_config.save_initial_states,
                        floating_point_precision=dm_cfg.floating_point_precision,
                        compression=dm_cfg.compression,
                        compression_opts=dm_cfg.compression_opts,
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

        # 5. Инициализация пропагатора и менеджера симуляции с единым детерминированным ГСЧ
        consumed_chunk_types = frozenset().union(*(handler.consumed_chunk_types for handler in handlers))
        propagator = ParticlePropagator(rng=np.random.default_rng(effective_seed))
        manager = SimulationManager(
            scene=root_scene,
            propagator=propagator,
            stop_time=effective_stop_time,
            start_time=effective_start_time,
            particles_number=int(sim_config.particles_number),
            min_energy=float(sim_config.min_energy),
            buffer_capacity=final_config.data_manager.buffer_capacity,
            name=f"Task_seed_{effective_seed}",
            seed=effective_seed,
            initial_particle_id=initial_particle_id,
            rng_state=rng_state,
            consumed_chunk_types=consumed_chunk_types,
        )

        ipc_pause_event = task_dict.get("_run_event", task_dict.get("_pause_event"))
        if ipc_pause_event is not None:
            if not bool(ipc_pause_event.is_set()):
                manager.pause()
            pause_bridge = IpcPauseBridge(
                manager=manager,
                ipc_event=ipc_pause_event,
                poll_interval=0.02,
            )
            pause_bridge.start()

        precision_mode_val = "single" if dm_cfg.floating_point_precision == "float32" else "double"
        metadata_collector = ProcedureMetadataCollector()
        collected_metadata = metadata_collector.collect(
            root_scene=root_scene,
            protocol=final_config.protocol,
            context=context_data,
            task_id=effective_task_id,
            sim_config=sim_config,
            scene_config=final_config.scene,
            random_seed=effective_seed,
            precision_mode=precision_mode_val,
            active_processes=propagator.processes,
        )

        task_metadata: Dict[str, Any] = {
            "task_id": effective_task_id,
            "seed": int(effective_seed),
            "status": "in_progress",
            "simulation": collected_metadata["simulation"],
            "gantry": collected_metadata["gantry"],
            "geometry": collected_metadata["geometry"],
            "source": collected_metadata["source"],
            "context": context_data,
            "protocol": collected_metadata["protocol"],
            "kinematics": collected_metadata["kinematics"],
            "detectors": collected_metadata["detectors"],
        }

        data_manager = DataManager(
            filename=final_config.data_manager.filename,
            handlers=handlers,
            queue=manager.queue,
            lock=file_lock,
            swmr=False,
            metadata=task_metadata,
            task_id=effective_task_id,
        )

        # 6. Ранняя фиксация метаданных задачи в HDF5 ДО запуска моделирования
        data_manager.initialize_metadata()

        # 7. Запуск вычислений с передачей статусов в телеметрию
        if telemetry_queue is not None:
            try:
                telemetry_queue.put({"type": "task_started", "task_id": task_id, "seed": effective_seed})
            except (ValueError, OSError) as queue_error:
                _logger.debug(f"Не удалось отправить статус task_started в очередь телеметрии: {queue_error}")

        manager.start()
        data_manager.start()
        manager.join()
        data_manager.join()

        actual_start_time = float(effective_start_time)
        actual_stop_time = float(manager.global_timer)
        runtime_summary: Dict[str, Any] = {
            "seed": int(effective_seed),
            "actual_start_time": actual_start_time,
            "actual_stop_time": actual_stop_time,
            "frame_duration": round((actual_stop_time - actual_start_time) / float(units.s), 6),
            "particle_id_start": int(initial_particle_id),
            "particle_id_end": int(manager.bank.count),
            "simulated_histories": int(manager.bank.count) - int(initial_particle_id),
            "final_rng_state": json.dumps(manager.rng.bit_generator.state),
        }
        data_manager.finalize_metadata(status="completed", runtime_summary=runtime_summary)

        if telemetry_queue is not None:
            try:
                telemetry_queue.put({"type": "task_finished", "task_id": task_id, "seed": effective_seed})
            except (ValueError, OSError) as queue_error:
                _logger.debug(f"Не удалось отправить статус task_finished в очередь телеметрии: {queue_error}")
        task_duration = time.perf_counter() - task_start_time
        _logger.info(
            f"Воркер {task_id}: симуляция успешно завершена за {task_duration:.2f} с. "
            f"HDF5: {final_config.data_manager.filename}"
        )
        return (context_data, final_config)
    except BaseException as simulation_exception:
        _logger.error(
            f"Воркер {task_id}: ошибка моделирования: {simulation_exception}",
            exc_info=True,
        )
        if data_manager is not None:
            data_manager.stop(timeout=1.0)
            try:
                data_manager.finalize_metadata(status="failed", error=str(simulation_exception))
            except (OSError, RuntimeError, ValueError, KeyError) as metadata_write_error:
                _logger.error(
                    f"Не удалось зафиксировать статус ошибки в HDF5 для задачи {task_id}: {metadata_write_error}",
                    exc_info=True,
                )
        if telemetry_queue is not None:
            try:
                telemetry_queue.put({"type": "task_error", "task_id": task_id, "error": str(simulation_exception)})
            except (ValueError, OSError) as queue_error:
                _logger.debug(f"Не удалось отправить статус task_error в очередь телеметрии: {queue_error}")
        raise
    finally:
        for handler_instance in handlers:
            try:
                handler_instance.close()
            except (OSError, RuntimeError, ValueError, KeyError) as close_error:
                _logger.debug(f"Ошибка при закрытии обработчика {handler_instance}: {close_error}")
        if pause_bridge is not None:
            pause_bridge.stop_bridge()
            pause_bridge.join(timeout=1.0)
        close_logging_handlers(worker_log_handlers)


__all__ = ["simulation_worker_task"]

