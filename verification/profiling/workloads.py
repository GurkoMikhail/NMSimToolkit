"""
Модуль эталонных сцен и рабочих нагрузок (Workloads) для комплексного профилирования NMSimToolkit.

Поддерживает сборку идентичных физических конфигураций с двумя видами коллиматоров:
- ParametricParallelCollimator (дельта-трекинг Вудкока через @cfunc);
- DirectParallelCollimator (аналитический рейкастинг _hex_prism_intersect).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.data.distribution_loader import DistributionLoader
from core.geometry.direct_collimators import CollimatorHoleShape, DirectParallelCollimator
from core.geometry.geometries import Box
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.geometry_kernels import cast_path_kernel
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import MaterialArray
from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext, SimulationDataBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.scene.dose_grid_node import DoseGridNode
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.gantry_node import GantryNode
from core.scene.nodes import CompositeNode, SpatialNode
from core.source.source_compiler import SourceCompiler
from core.source.source_kernels import inject_poisson_sources_kernel
from core.source.sources import Source
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from core.transport.transport_buffer import TransportBuffer
from core.transport.transport_kernels import (
    _push_first_interacting_to_initial_state_kernel,
    _push_to_dead_particles_kernel,
    _push_to_escaped_particles_kernel,
    _push_to_initial_state_kernel,
    _push_uncollided_to_initial_state_kernel,
)
from settings.database_setting import attenuation_database, material_database
from settings.processes_settings import processes_list
from examples.tb161_nema_simulation import TB161_ENERGY_SPECTRUM, TB161_HALF_LIFE

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CollimatorKind = Literal["parametric", "direct"]

_VOXEL_CACHE: Dict[str, NDArray[Any]] = {}


@dataclass(frozen=True)
class ProfilingWorkload:
    """
    Контейнер собранной сцены для профилирования со ссылками на ключевые узлы.
    Поле particles_number задает полное число моделируемых историй за stop_time,
    а bank_capacity — емкость банка частиц ParticleBank (размер батча).
    """

    workload_id: str
    title: str
    collimator_kind: CollimatorKind
    root_scene: CompositeNode
    sensitive_detectors: List[SpatialNode]
    dose_grids: List[DoseGridNode]
    source_node: Source
    particles_number: int
    stop_time: Float
    min_energy: Float
    scene_build_duration: float
    bank_capacity: int = 0

    @property
    def effective_bank_capacity(self) -> int:
        """Емкость ParticleBank (если bank_capacity <= 0, равна particles_number)."""
        return self.bank_capacity if self.bank_capacity > 0 else self.particles_number


def _load_cached_raw_grid(relative_path: str, shape: Tuple[int, int, int]) -> NDArray[Any]:
    """Загружает воксельный массив из файла с кэшированием в оперативной памяти."""
    if relative_path not in _VOXEL_CACHE:
        full_path = PROJECT_ROOT / relative_path
        _VOXEL_CACHE[relative_path] = DistributionLoader.load(
            full_path,
            target_shape=shape,
            order="F",
            dtype=np.float32,
        )
    return _VOXEL_CACHE[relative_path].copy()


def _load_cached_npy(relative_path: str) -> NDArray[Any]:
    """Загружает массив .npy с кэшированием в оперативной памяти."""
    if relative_path not in _VOXEL_CACHE:
        full_path = PROJECT_ROOT / relative_path
        _VOXEL_CACHE[relative_path] = np.load(full_path)
    return _VOXEL_CACHE[relative_path].copy()


def build_ge_discovery_camera_head(
    head_index: int,
    collimator_kind: CollimatorKind,
    relative_angle: float,
    orbit_radius: float,
) -> Tuple[GammaCameraNode, Volume]:
    """
    Создает детекторную головку GE Discovery NM/CT 670 с коллиматором LEHR
    выбранного типа ('parametric' или 'direct').
    """
    camera_node = GammaCameraNode(name=f"GammaCamera_{head_index}")

    casing = Volume(
        geometry=Box(Float(58.0 * units.cm), Float(44.0 * units.cm), Float(14.06 * units.cm)),
        material=material_database["Pb"],
        name=f"Casing_{head_index}",
    )

    detector_box = Volume(
        geometry=Box(Float(54.0 * units.cm), Float(40.0 * units.cm), Float(12.06 * units.cm)),
        material=material_database["Air, Dry (near sea level)"],
        name=f"Detector_box_{head_index}",
    )
    detector_box.translate(z=Float(1.0 * units.cm))

    collimator_size = (Float(54.0 * units.cm), Float(40.0 * units.cm), Float(3.5 * units.cm))
    collimator_node: SpatialNode
    if collimator_kind == "parametric":
        collimator_node = ParametricParallelCollimator(
            size=collimator_size,
            hole_diameter=Float(1.5 * units.mm),
            septa=Float(0.2 * units.mm),
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            material=material_database["Pb"],
            name=f"Collimator_{head_index}",
        )
    elif collimator_kind == "direct":
        collimator_node = DirectParallelCollimator(
            size=collimator_size,
            hole_diameter=Float(1.5 * units.mm),
            septa=Float(0.2 * units.mm),
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            material=material_database["Pb"],
            hole_material=material_database["Air, Dry (near sea level)"],
            name=f"Collimator_{head_index}",
        )
    else:
        raise ValueError(f"Неизвестный тип коллиматора: {collimator_kind}")

    collimator_node.translate(z=Float(4.28 * units.cm))

    crystal = Volume(
        geometry=Box(Float(54.0 * units.cm), Float(40.0 * units.cm), Float(9.5 * units.mm)),
        material=material_database["Sodium Iodide"],
        name=f"Detector_{head_index}",
    )
    crystal.translate(z=Float(2.045 * units.cm))

    glass_backend = Volume(
        geometry=Box(Float(54.0 * units.cm), Float(40.0 * units.cm), Float(7.6 * units.cm)),
        material=material_database["Glass, Borosilicate (Pyrex)"],
        name=f"Glass_backend_{head_index}",
    )
    glass_backend.translate(z=Float(-2.23 * units.cm))

    detector_box.add_child(collimator_node)
    detector_box.add_child(crystal)
    detector_box.add_child(glass_backend)
    casing.add_child(detector_box)
    camera_node.add_child(casing)

    camera_node.rotate(Float(0.0), Float(0.0), Float(90.0 * units.deg))
    camera_node.translate(y=Float(orbit_radius))
    if relative_angle != 0.0:
        camera_node.rotate(Float(relative_angle), Float(0.0), Float(0.0))

    camera_node.slots = {
        "casing": casing.name or "",
        "detector_box": detector_box.name or "",
        "collimator": collimator_node.name or "",
        "crystal": crystal.name or "",
        "glass_backend": glass_backend.name or "",
    }
    return camera_node, crystal


def build_point_source_workload(
    collimator_kind: CollimatorKind,
    particles_number: int = 100_000,
    seed: int = 42,
    bank_capacity: Optional[int] = None,
) -> ProfilingWorkload:
    """
    Сценарий W1: Точечный источник Tc-99m (140.5 кэВ) + 1 головка гамма-камеры GE Discovery LEHR.
    """
    build_start = time.perf_counter()
    root_scene = Volume(
        geometry=Box(Float(120.0 * units.cm), Float(120.0 * units.cm), Float(80.0 * units.cm)),
        material=material_database["Air, Dry (near sea level)"],
        name="Simulation_volume",
    )

    camera_head, crystal = build_ge_discovery_camera_head(
        head_index=1,
        collimator_kind=collimator_kind,
        relative_angle=0.0,
        orbit_radius=float(250.0 * units.mm),
    )
    root_scene.add_child(camera_head)

    stop_time = Float(1.0 * units.s)
    activity_val = Float(float(particles_number) * units.Bq)
    source_node = Source.from_point(
        activity=activity_val,
        energy=Float(140.5 * units.keV),
        half_life=Float(np.inf),
        name="PointSource_Tc99m",
    )
    root_scene.add_child(source_node)

    dose_grid = DoseGridNode(
        name="DetectorDoseGrid",
        size=(Float(540.0 * units.mm), Float(400.0 * units.mm), Float(120.0 * units.mm)),
        dose_voxel_size=Float(8.0 * units.mm),
    )
    dose_grid.translate(y=Float(250.0 * units.mm))
    root_scene.add_child(dose_grid)

    build_elapsed = time.perf_counter() - build_start
    suffix = "Parametric" if collimator_kind == "parametric" else "Direct"
    eff_bank = int(particles_number if bank_capacity is None else bank_capacity)
    return ProfilingWorkload(
        workload_id=f"W1_PointSource_{suffix}",
        title=f"W1: Точечный источник Tc-99m + GE LEHR ({suffix})",
        collimator_kind=collimator_kind,
        root_scene=root_scene,
        sensitive_detectors=[crystal],
        dose_grids=[dose_grid],
        source_node=source_node,
        particles_number=particles_number,
        stop_time=stop_time,
        min_energy=Float(1.0 * units.keV),
        scene_build_duration=build_elapsed,
        bank_capacity=eff_bank,
    )


def build_nema_tb161_workload(
    collimator_kind: CollimatorKind,
    particles_number: int = 100_000,
    seed: int = 42,
    use_cached_files: bool = True,
    bank_capacity: Optional[int] = None,
) -> ProfilingWorkload:
    """
    Сценарий W2: Воксельный фантом NEMA IEC (128x128x92) с многолинейным спектром 161Tb
    и двухголовочным гантри GE Discovery LEHR (Parametric vs Direct).
    """
    build_start = time.perf_counter()
    root_scene = Volume(
        geometry=Box(Float(120.0 * units.cm), Float(120.0 * units.cm), Float(80.0 * units.cm)),
        material=material_database["Air, Dry (near sea level)"],
        name="Simulation_volume",
    )

    if use_cached_files:
        raw_attenuation = _load_cached_raw_grid("phantoms/nema/anema_voxel_size_4.2_mm.dat", (128, 128, 92))
        raw_activity = _load_cached_raw_grid("phantoms/nema/nema_voxel_size_4.2_mm.dat", (128, 128, 92))
    else:
        raw_attenuation = DistributionLoader.load(
            PROJECT_ROOT / "phantoms/nema/anema_voxel_size_4.2_mm.dat",
            target_shape=(128, 128, 92),
            order="F",
            dtype=np.float32,
        )
        raw_activity = DistributionLoader.load(
            PROJECT_ROOT / "phantoms/nema/nema_voxel_size_4.2_mm.dat",
            target_shape=(128, 128, 92),
            order="F",
            dtype=np.float32,
        )

    material_distribution = MaterialArray((128, 128, 92))
    material_distribution[np.isclose(raw_attenuation, 0.04)] = material_database["Air, Dry (near sea level)"]
    material_distribution[np.isclose(raw_attenuation, 0.15)] = material_database["Water, Liquid"]

    phantom = WoodcockVoxelVolume(
        voxel_size=Float(4.2 * units.mm),
        material_distribution=material_distribution,
        name="Phantom",
    )
    root_scene.add_child(phantom)

    stop_time = Float(1.0 * units.s)
    activity_val = Float(float(particles_number) * units.Bq)
    source_node = Source.from_voxels(
        distribution=raw_activity,
        activity=activity_val,
        voxel_size=Float(4.2 * units.mm),
        energy=TB161_ENERGY_SPECTRUM,
        half_life=Float(TB161_HALF_LIFE),
        name="Source",
    )
    phantom.add_child(source_node)

    gantry = GantryNode(name="Gantry")
    head_1, crystal_1 = build_ge_discovery_camera_head(
        head_index=1,
        collimator_kind=collimator_kind,
        relative_angle=0.0,
        orbit_radius=float(329.05 * units.mm),
    )
    head_2, crystal_2 = build_ge_discovery_camera_head(
        head_index=2,
        collimator_kind=collimator_kind,
        relative_angle=float(180.0 * units.deg),
        orbit_radius=float(329.05 * units.mm),
    )
    gantry.add_child(head_1)
    gantry.add_child(head_2)
    root_scene.add_child(gantry)

    dose_grid = DoseGridNode(
        name="PhantomDoseGrid",
        size=(Float(537.6 * units.mm), Float(537.6 * units.mm), Float(386.4 * units.mm)),
        dose_voxel_size=Float(8.4 * units.mm),
    )
    root_scene.add_child(dose_grid)

    build_elapsed = time.perf_counter() - build_start
    suffix = "Parametric" if collimator_kind == "parametric" else "Direct"
    eff_bank = int(particles_number if bank_capacity is None else bank_capacity)
    return ProfilingWorkload(
        workload_id=f"W2_NEMA_Tb161_{suffix}",
        title=f"W2: Фантом NEMA 161Tb + 2x GE LEHR ({suffix})",
        collimator_kind=collimator_kind,
        root_scene=root_scene,
        sensitive_detectors=[crystal_1, crystal_2],
        dose_grids=[dose_grid],
        source_node=source_node,
        particles_number=particles_number,
        stop_time=stop_time,
        min_energy=Float(1.0 * units.keV),
        scene_build_duration=build_elapsed,
        bank_capacity=eff_bank,
    )


def build_heart_spect_workload(
    collimator_kind: CollimatorKind,
    particles_number: int = 100_000,
    seed: int = 42,
    bank_capacity: Optional[int] = None,
) -> ProfilingWorkload:
    """
    Сценарий W3: 5-тканевый воксельный фантом миокарда (воздух, легкие, жир, мягкая ткань, кость)
    с источником 99mTc и двухголовочной системой GE Discovery LEHR.
    """
    build_start = time.perf_counter()
    root_scene = Volume(
        geometry=Box(Float(120.0 * units.cm), Float(120.0 * units.cm), Float(80.0 * units.cm)),
        material=material_database["Air, Dry (near sea level)"],
        name="Simulation_volume",
    )

    material_id_distribution = _load_cached_npy("phantoms/material_map.npy")
    material_distribution = MaterialArray(material_id_distribution.shape)
    material_distribution[material_id_distribution == 0] = material_database["Air, Dry (near sea level)"]
    material_distribution[material_id_distribution == 1] = material_database["Lung"]
    material_distribution[material_id_distribution == 2] = material_database["Adipose Tissue (ICRU-44)"]
    material_distribution[material_id_distribution == 3] = material_database["Tissue, Soft (ICRU-44)"]
    material_distribution[material_id_distribution == 4] = material_database["Bone, Cortical (ICRU-44)"]

    phantom = WoodcockVoxelVolume(
        voxel_size=Float(4.0 * units.mm),
        material_distribution=material_distribution,
        name="HeartPhantom",
    )
    root_scene.add_child(phantom)

    activity_distribution = _load_cached_npy("phantoms/source_function.npy")
    activity_distribution[activity_distribution == 40] = 10
    activity_distribution[activity_distribution == 30] = 20
    activity_distribution[activity_distribution == 70] = 40
    activity_distribution[activity_distribution == 80] = 40
    activity_distribution[activity_distribution == 89] = 50
    activity_distribution[activity_distribution == 140] = 40
    activity_distribution[activity_distribution == 1200] = 1000
    activity_distribution[activity_distribution == 700] = 550
    activity_distribution[activity_distribution == 10000] = 7000

    stop_time = Float(1.0 * units.s)
    activity_val = Float(float(particles_number) * units.Bq)
    source_node = Source.from_voxels(
        distribution=activity_distribution,
        activity=activity_val,
        voxel_size=Float(4.0 * units.mm),
        energy=[[Float(140.511 * units.keV), 1.0]],
        half_life=Float(np.inf),
        name="HeartSource",
    )
    phantom.add_child(source_node)

    gantry = GantryNode(name="Gantry")
    head_1, crystal_1 = build_ge_discovery_camera_head(
        head_index=1,
        collimator_kind=collimator_kind,
        relative_angle=0.0,
        orbit_radius=float(303.3 * units.mm),
    )
    head_2, crystal_2 = build_ge_discovery_camera_head(
        head_index=2,
        collimator_kind=collimator_kind,
        relative_angle=float(90.0 * units.deg),
        orbit_radius=float(303.3 * units.mm),
    )
    gantry.add_child(head_1)
    gantry.add_child(head_2)
    root_scene.add_child(gantry)

    dose_grid = DoseGridNode(
        name="HeartDoseGrid",
        size=(Float(512.0 * units.mm), Float(512.0 * units.mm), Float(256.0 * units.mm)),
        dose_voxel_size=Float(8.0 * units.mm),
    )
    root_scene.add_child(dose_grid)

    build_elapsed = time.perf_counter() - build_start
    suffix = "Parametric" if collimator_kind == "parametric" else "Direct"
    eff_bank = int(particles_number if bank_capacity is None else bank_capacity)
    return ProfilingWorkload(
        workload_id=f"W3_Heart_Tc99m_{suffix}",
        title=f"W3: Фантом миокарда (5 тканей, кость) + 2x GE LEHR ({suffix})",
        collimator_kind=collimator_kind,
        root_scene=root_scene,
        sensitive_detectors=[crystal_1, crystal_2],
        dose_grids=[dose_grid],
        source_node=source_node,
        particles_number=particles_number,
        stop_time=stop_time,
        min_energy=Float(1.0 * units.keV),
        scene_build_duration=build_elapsed,
        bank_capacity=eff_bank,
    )


ALL_COLLIMATOR_KINDS: Tuple[CollimatorKind, ...] = ("parametric", "direct")


def build_all_workloads_matrix(
    w1_particles: int = 50_000,
    w2_particles: int = 35_000,
    w3_particles: int = 35_000,
    seed: int = 42,
    bank_capacity: Optional[int] = None,
) -> List[ProfilingWorkload]:
    """Строит полную матрицу из 6 эталонных нагрузок (W1, W2, W3 для обоих типов коллиматоров)."""
    return [
        build_point_source_workload("parametric", particles_number=w1_particles, seed=seed, bank_capacity=bank_capacity),
        build_point_source_workload("direct", particles_number=w1_particles, seed=seed, bank_capacity=bank_capacity),
        build_nema_tb161_workload("parametric", particles_number=w2_particles, seed=seed, bank_capacity=bank_capacity),
        build_nema_tb161_workload("direct", particles_number=w2_particles, seed=seed, bank_capacity=bank_capacity),
        build_heart_spect_workload("parametric", particles_number=w3_particles, seed=seed, bank_capacity=bank_capacity),
        build_heart_spect_workload("direct", particles_number=w3_particles, seed=seed, bank_capacity=bank_capacity),
    ]


def warmup_propagator_instance(
    propagator: ParticlePropagator,
    scene: CompositeNode,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Выполняет принудительную первичную JIT-компиляцию всех локальных замыканий Numba
    конкретного экземпляра ParticlePropagator (propagator._transport_kernel и proc._kernel
    для PhotoelectricEffect, ComptonScattering, CoherentScattering) на мини-батче частиц.

    Возвращает измеренное время JIT-компиляции (cold_compile_time_sec) и оставляет
    экземпляр propagator в полностью прогретом состоянии (Warm Steady-State).
    """
    warm_rng = np.random.default_rng(77777) if rng is None else rng
    geom_buffer = GeometryCompiler().compile_scene(scene)
    physics_buffer = PhysicsCompiler().compile_scene(scene, propagator.processes)
    active_sources = SourceCompiler().compile_scene(scene)
    source_buffer = SourceCompiler().compile(active_sources)

    batch_n = 16
    warm_bank = ParticleBank.allocate(batch_n)
    warm_data_buf = SimulationDataBuffer.allocate(64, 64, 64, 64)
    warm_rng_ctx = RNGContext.from_numpy_rng(warm_rng)

    inject_poisson_sources_kernel(
        warm_bank,
        source_buffer,
        Float(np.inf),
        Float(0.0),
        Float(0.0),
        Float(-1.0),
        warm_rng_ctx,
        Index(batch_n),
        Index(0),
    )
    active_indices = warm_bank.active_indices

    # Предварительный рейкастинг (cast_path_kernel — модульное ядро, уже скомпилировано)
    cast_path_kernel(
        warm_bank.state.position,
        warm_bank.state.direction,
        active_indices,
        geom_buffer,
        warm_bank.navigation_state,
    )

    propagator.transport_buffer = TransportBuffer.allocate(batch_n, len(propagator.processes))
    target_proc_indices = np.arange(min(4, batch_n), dtype=Index)
    warm_material_ids = np.zeros(batch_n, dtype=Index)

    # Предварительный прогрев модульных ядер записи буферов
    _push_first_interacting_to_initial_state_kernel(
        warm_bank.initial_state,
        warm_bank.state,
        active_indices,
        propagator.transport_buffer.process_ids,
        warm_data_buf.initial_states,
    )
    _push_uncollided_to_initial_state_kernel(
        warm_bank.initial_state,
        warm_bank.state,
        target_proc_indices,
        warm_data_buf.initial_states,
    )
    _push_to_initial_state_kernel(warm_bank.initial_state, target_proc_indices, warm_data_buf.initial_states)
    _push_to_dead_particles_kernel(warm_bank.initial_state, target_proc_indices, warm_data_buf.dead_particles)
    _push_to_escaped_particles_kernel(
        warm_bank.initial_state,
        warm_bank.state,
        target_proc_indices,
        warm_data_buf.escaped_particles,
    )
    warm_data_buf.initial_states.cursor[0] = 0
    warm_data_buf.dead_particles.cursor[0] = 0
    warm_data_buf.escaped_particles.cursor[0] = 0

    compile_start = time.perf_counter()

    # 1. Компиляция замыкания propagator._transport_kernel
    propagator._transport_kernel(
        warm_bank.state,
        warm_bank.navigation_state,
        active_indices,
        physics_buffer,
        propagator.transport_buffer,
        geom_buffer,
        warm_rng_ctx,
    )

    # 2. Гарантированная компиляция замыканий proc._kernel каждого физического процесса
    for process_obj in propagator.processes:
        warm_bank.state.energy[:4] = Float(0.1405)
        warm_bank.state.is_active[:4] = True
        process_obj.apply(
            warm_bank,
            target_proc_indices,
            warm_data_buf.interactions,
            physics_buffer,
            warm_material_ids,
            warm_rng_ctx,
        )

    cold_compile_time_sec = float(time.perf_counter() - compile_start)
    propagator.transport_buffer = TransportBuffer.allocate(0, len(propagator.processes))
    return cold_compile_time_sec


def warmup_all_profiling_kernels() -> float:
    """
    Выполняет предварительный прогрев всех @njit и @cfunc ядер Numba
    на сценах с воксельным фантомом и обоими типами коллиматоров.
    Возвращает суммарное время JIT-прогрева в секундах.
    """
    warmup_start = time.perf_counter()
    for coll_kind in ALL_COLLIMATOR_KINDS:
        workload = build_nema_tb161_workload(
            collimator_kind=coll_kind,
            particles_number=2_000,
            seed=777,
        )
        propagator = ParticlePropagator(
            processes_list=processes_list,
            attenuation_database=attenuation_database,
            rng=np.random.default_rng(777),
        )
        warmup_propagator_instance(propagator, workload.root_scene, np.random.default_rng(777))
        manager = SimulationManager(
            scene=workload.root_scene,
            propagator=propagator,
            stop_time=workload.stop_time,
            particles_number=workload.particles_number,
            min_energy=workload.min_energy,
            buffer_capacity=10_000,
            seed=777,
        )
        while manager.has_active_particles or manager.has_pending_sources:
            manager.next_step()
            while not manager.queue.empty():
                manager.queue.get_nowait()

    return float(time.perf_counter() - warmup_start)


__all__ = [
    "ALL_COLLIMATOR_KINDS",
    "CollimatorKind",
    "ProfilingWorkload",
    "build_all_workloads_matrix",
    "build_ge_discovery_camera_head",
    "build_heart_spect_workload",
    "build_nema_tb161_workload",
    "build_point_source_workload",
    "warmup_all_profiling_kernels",
    "warmup_propagator_instance",
]
