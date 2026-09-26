import itertools
import re
from copy import deepcopy
from typing import Dict, List, Any, Tuple, Optional, Callable
import numpy as np
from multiprocessing import Manager, Pool
from numpy.random import SeedSequence

from core.config.models import (
    SimulationConfig,
    CustomSweepProtocolConfig,
    StepAndShootProtocolConfig,
    SpectProtocolConfig,
)
from core.config.builder import SceneBuilder
from core.scene.nodes import SpatialNode, CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from core.source.sources import Source
from core.transport.simulation_managers import SimulationManager
from core.transport.propagator import ParticlePropagator
from core.data.data_manager import DataManager
from core.data.data_handlers import (
    BaseDataHandler,
    DirectStreamHandler,
    SensitiveVolumeHandler,
    HistoryAssemblerHandler,
)
from core.data.dose_map_handler import DoseMapHandler
from core.geometry.gamma_cameras import GammaCamera


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


def _find_dose_grids(node: Any) -> List[DoseGridNode]:
    """Рекурсивный поиск всех узлов DoseGridNode в поддереве сцены."""
    grids: List[DoseGridNode] = []
    if isinstance(node, DoseGridNode):
        grids.append(node)
    if isinstance(node, CompositeNode):
        for child in node.childs:
            grids.extend(_find_dose_grids(child))
    return grids


def _find_gamma_cameras(node: Any) -> List[GammaCamera]:
    """Рекурсивный поиск всех узлов GammaCamera в поддереве сцены."""
    cameras: List[GammaCamera] = []
    if isinstance(node, GammaCamera):
        cameras.append(node)
    if isinstance(node, CompositeNode):
        for child in node.childs:
            cameras.extend(_find_gamma_cameras(child))
    return cameras

def _worker_function(payload: Tuple[Any, ...]) -> Any:
    task_dict = payload[0]
    seed = payload[1]
    file_lock = payload[2]
    telemetry_queue = payload[3] if len(payload) > 3 else None
    extra_handlers = payload[4] if len(payload) > 4 else None
    extra_handler_factory = payload[5] if len(payload) > 5 else None
    extra_handler_args = payload[6] if len(payload) > 6 else ()
    task_id = payload[7] if len(payload) > 7 else task_dict.get('_task_id', 0)
    
    # 1. Validate Config
    final_config = SimulationConfig.model_validate(task_dict)
    
    # 2. Build Scene
    builder = SceneBuilder()
    root_scene = builder.build_scene(final_config.scene)
    
    # 2.1. Позиционирование гамма-камер по круговой орбите для текущего ракурса
    context = task_dict.get('_context', {})
    if context:
        cameras = _find_gamma_cameras(root_scene)
        if cameras:
            radius = 250.0
            if isinstance(final_config.protocol, (StepAndShootProtocolConfig, SpectProtocolConfig)):
                if final_config.protocol.radius is not None:
                    radius = float(final_config.protocol.radius)
            for i, cam in enumerate(cameras):
                angle = context.get(f"head_{i}_angle")
                if angle is None:
                    angle = context.get("current_angle")
                if angle is not None:
                    cam.set_orbit_position(radius=radius, angle_deg=float(angle), z=0.0)
    
    # 3. Instantiate Propagator with seed
    rng = np.random.default_rng(seed)
    propagator = ParticlePropagator(rng=rng)
    
    # Ensure sources use the same rng.
    def set_rng_for_sources(node):
        if isinstance(node, Source):
            node.rng = rng

        if isinstance(node, CompositeNode):
            for child in node.childs:
                set_rng_for_sources(child)
    
    set_rng_for_sources(root_scene)

    # 4. Build Data Handlers
    handlers: List[BaseDataHandler] = []
    for h_config in final_config.data_manager.handlers:
        if h_config.type == 'DirectStreamHandler':
            handlers.append(DirectStreamHandler())
        elif h_config.type == 'SensitiveVolumeHandler':
            vols = _find_nodes_by_names(root_scene, h_config.sensitive_volumes)
            handlers.append(SensitiveVolumeHandler(sensitive_volumes=vols))
        elif h_config.type == 'HistoryAssemblerHandler':
            vols = _find_nodes_by_names(root_scene, h_config.sensitive_volumes)
            handlers.append(HistoryAssemblerHandler(sensitive_volumes=vols, save_initial_states=h_config.save_initial_states))
        elif h_config.type == 'DoseMapHandler':
            grids = _find_dose_grids(root_scene)
            if h_config.grid_names:
                grids = [g for g in grids if g.name in h_config.grid_names]
            handlers.append(DoseMapHandler(grid_nodes=grids, shm_name=h_config.shm_name))

    # Внедрение внешних обработчиков данных (Dependency Injection)
    if extra_handlers:
        handlers.extend(extra_handlers)

    if extra_handler_factory is not None:
        created_handlers = extra_handler_factory(
            root_scene, task_dict, task_id, *extra_handler_args
        )
        if created_handlers:
            handlers.extend(created_handlers)

    # 5. Instantiate Managers
    sim_config = final_config.simulation_manager
    pause_event = task_dict.get('_pause_event')
    manager = SimulationManager(
        scene=root_scene,
        propagator=propagator,
        stop_time=sim_config.stop_time,
        start_time=sim_config.start_time,
        particles_number=sim_config.particles_number,
        min_energy=sim_config.min_energy,
        buffer_capacity=final_config.data_manager.buffer_capacity,
        name=f"Task_seed_{seed}",
        seed=seed,
        pause_event=pause_event,
    )

    data_manager = DataManager(
        filename=final_config.data_manager.filename,
        handlers=handlers,
        queue=manager.queue,
        lock=file_lock,
        swmr=False
    )

    # 6. Run с оповещением телеметрии
    if telemetry_queue is not None:
        try:
            telemetry_queue.put({'type': 'task_started', 'task_id': task_id, 'seed': seed})
        except (ValueError, OSError):
            pass

    try:
        manager.start()
        data_manager.start()
        manager.join()
        data_manager.join()
        if telemetry_queue is not None:
            try:
                telemetry_queue.put({'type': 'task_finished', 'task_id': task_id, 'seed': seed})
            except (ValueError, OSError):
                pass
        return (context, final_config)
    except Exception as exc:
        if telemetry_queue is not None:
            try:
                telemetry_queue.put({'type': 'task_error', 'task_id': task_id, 'error': str(exc)})
            except (ValueError, OSError):
                pass
        raise


class Orchestrator:
    def __init__(self, raw_config_dict: Any):
        """
        Инициализирует оркестратор сырым словарем конфигурации или экземпляром SimulationConfig.
        """
        if isinstance(raw_config_dict, SimulationConfig):
            self.raw_config_dict = raw_config_dict.model_dump()
            self.parsed_config = raw_config_dict
        else:
            self.raw_config_dict = dict(raw_config_dict)
            # Валидация начальной схемы для раннего выявления ошибок протокола и структуры сцены
            self.parsed_config = SimulationConfig.model_validate(raw_config_dict)

    def compile_protocol(self) -> CustomSweepProtocolConfig:
        """
        Компилирует высокоуровневый протокол в базовый CustomSweepProtocolConfig.
        Если протокол не задан, возвращает фиктивный свип на одну задачу.
        """
        protocol = self.parsed_config.protocol

        if protocol is None:
            return CustomSweepProtocolConfig(grid_variables={}, zipped_variables={})

        if isinstance(protocol, CustomSweepProtocolConfig):
            return protocol

        if isinstance(protocol, (StepAndShootProtocolConfig, SpectProtocolConfig)):
            gamma_cameras = max(1, protocol.gamma_cameras)
            views = max(1, protocol.views)
            n_positions = max(1, views // gamma_cameras)
            endpoint = protocol.endpoint
            angles = np.linspace(protocol.start_angle, protocol.end_angle, n_positions, endpoint=endpoint).tolist()
            zipped_vars = {
                "current_angle": angles,
                "current_time": [float(protocol.time_per_view)] * n_positions
            }
            if protocol.head_angles is not None:
                for idx, head_ang in enumerate(protocol.head_angles):
                    zipped_vars[f"head_{idx}_angle"] = [float(a + head_ang) for a in angles]
            elif gamma_cameras > 1:
                step_off = float(protocol.end_angle - protocol.start_angle) / gamma_cameras if not endpoint else float(protocol.end_angle - protocol.start_angle) / max(1, gamma_cameras - 1)
                for idx in range(gamma_cameras):
                    zipped_vars[f"head_{idx}_angle"] = [float(a + step_off * idx) for a in angles]
            return CustomSweepProtocolConfig(grid_variables={}, zipped_variables=zipped_vars)

        raise ValueError(f"Unknown protocol type: {type(protocol)}")

    @staticmethod
    def compute_spect_poses(
        views_or_protocol: Any,
        gamma_cameras: int = 1,
        start_angle_deg: float = 0.0,
        end_angle_deg: float = 360.0,
        head_angle_offsets: Optional[List[float]] = None,
        endpoint: bool = False
    ) -> List[List[float]]:
        """
        Вычисляет список углов для всех N головок на каждом шаге гантри ОФЭКТ.
        Возвращает список позиций гантри, где каждая позиция — список углов [head_0, head_1, ..., head_{N-1}].
        Поддерживает передачу как объекта SpectProtocolConfig, так и скалярных параметров.
        """
        if isinstance(views_or_protocol, SpectProtocolConfig):
            views = int(views_or_protocol.views)
            gc = max(1, int(views_or_protocol.gamma_cameras))
            start_angle_deg = float(np.degrees(views_or_protocol.start_angle))
            end_angle_deg = float(np.degrees(views_or_protocol.end_angle))
            head_angle_offsets = [float(np.degrees(a)) for a in views_or_protocol.head_angles] if views_or_protocol.head_angles else None
            endpoint = bool(views_or_protocol.endpoint)
        else:
            views = int(views_or_protocol)
            gc = max(1, int(gamma_cameras))

        n_positions = max(1, views // gc)
        base_angles = np.linspace(start_angle_deg, end_angle_deg, n_positions, endpoint=endpoint)
        if head_angle_offsets is not None and len(head_angle_offsets) == gc:
            offsets = head_angle_offsets
        else:
            step_offset = 360.0 / gc if gc > 0 else 0.0
            offsets = [step_offset * i for i in range(gc)]

        poses = []
        for base in base_angles:
            pose = [(float(base) + float(off)) % 360.0 for off in offsets]
            poses.append(pose)
        return poses

    def _generate_job_list(self, sweep_config: CustomSweepProtocolConfig) -> List[Dict[str, float]]:
        """
        Генерирует плоский список словарей параметров для каждой задачи симуляции.
        Вычисляет декартово произведение по grid_variables и синхронную итерацию по zipped_variables.
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
        Рекурсивно обходит словари и списки, подставляя строковые шаблоны
        вида "${current_angle}" реальными числовыми значениями из контекста.
        """
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                new_dict[k] = self.inject_variables(v, context)
            return new_dict
        elif isinstance(node, list):
            return [self.inject_variables(v, context) for v in node]
        elif isinstance(node, str):
            # Если строка в точности шаблон вида "${var}", возвращаем типизированное значение (например, float)
            if node.startswith("${") and node.endswith("}"):
                var_name = node[2:-1]
                if var_name in context:
                    return context[var_name]
            
            # Иначе строковая подстановка для составных выражений
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

    def run(
        self,
        telemetry_queue: Optional[Any] = None,
        extra_handlers: Optional[List[BaseDataHandler]] = None,
        extra_handler_factory: Optional[Callable[..., List[BaseDataHandler]]] = None,
        extra_handler_args: Tuple[Any, ...] = (),
    ) -> List[Any]:
        """
        Запускает параллельный цикл вычислений оркестратора, распределяя задачи по пулу процессов.
        Поддерживает передачу общей очереди телеметрии и внедрение внешних обработчиков данных
        (extra_handlers / extra_handler_factory) для динамического расширения контура сбора результатов.
        """
        pool_size = self.parsed_config.pool_size
        tasks = self.generate_tasks()

        mp_manager = Manager()
        try:
            locks = {}
            
            seed_seq = SeedSequence()
            seeds = seed_seq.spawn(len(tasks))

            payloads = []
            for idx, (context, seed) in enumerate(zip(tasks, seeds)):
                task_dict = deepcopy(self.raw_config_dict)
                task_dict['_task_id'] = idx
                task_dict['_context'] = context
                injected_dict = self.inject_variables(task_dict, context)
                
                # Группировка файловых блокировок по имени HDF5-файла
                filename = injected_dict.get('data_manager', {}).get('filename', 'default.hdf')
                if filename not in locks:
                    locks[filename] = mp_manager.Lock()
                    
                seed_val = seed.generate_state(1)[0]
                
                payloads.append((
                    injected_dict,
                    seed_val,
                    locks[filename],
                    telemetry_queue,
                    extra_handlers,
                    extra_handler_factory,
                    extra_handler_args,
                    idx,
                ))

            if pool_size > 1:
                with Pool(pool_size) as pool:
                    return pool.map(_worker_function, payloads)
            else:
                results = []
                for payload in payloads:
                    results.append(_worker_function(payload))
                return results
        finally:
            try:
                mp_manager.shutdown()
            except (OSError, ValueError):
                pass
