"""
Модуль сборки изолированных сцен верификации коллиматоров и запуска парных симуляций
(ParametricParallelCollimator vs DirectParallelCollimator) без искажающих эффектов детектора.
"""
from __future__ import annotations

from dataclasses import dataclass
import tempfile
import time
from typing import Any, Literal, Optional, Sequence
import os
import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.data.data_manager import DataManager
from core.geometry.direct_collimators import DirectParallelCollimator
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.other.typing_definitions import Float, Index
from core.source.source_kernels import inject_poisson_sources_kernel
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
import settings.database_setting as database_setting
from verification.common.history_classifier import (
    ClassifiedExitPhotons,
    CollimatorExitPlaneHandler,
)
from verification.common.presets import CollimatorPreset
from verification.common.sources import ConePointSource

CollimatorImplementationKind = Literal["parametric", "direct"]


@dataclass(frozen=True)
class CollimatorSceneSetup:
    """
    Контейнер собранной сцены верификации коллиматора с геометрическими координатами торцов.
    """

    world_volume: Volume
    collimator_node: ParametricParallelCollimator | DirectParallelCollimator
    source_node: ConePointSource
    z_source: Float
    z_enter_face: Float
    z_exit_face: Float
    preset: CollimatorPreset
    implementation_kind: CollimatorImplementationKind


@dataclass(frozen=True)
class SimulationRunSummary:
    """
    Сводный результат одного прогона верификационной симуляции коллиматора.
    """

    preset: CollimatorPreset
    implementation_kind: CollimatorImplementationKind
    source_distance: Float
    energy: Float
    particles_emitted: int
    elapsed_time: float
    particle_throughput: float
    simulation_steps: int
    photons: ClassifiedExitPhotons


def build_collimator_verification_scene(
    preset: CollimatorPreset,
    implementation_kind: CollimatorImplementationKind,
    source_distance: Float | float,
    energy: Float | float | Sequence[Any] | NDArray[Any],
    particles_number: int,
    max_polar_angle: Float | float = Float(np.pi),
    cell_averaging: bool = False,
    source_offset_xy: tuple[Float | float, Float | float] = (Float(0.0), Float(0.0)),
    seed: int = 42,
) -> CollimatorSceneSetup:
    """
    Строит изолированную сцену верификации коллиматора в вакууме без детектора, световода и кожуха.

    Геометрическая компоновка вдоль оси Z:
    - Выходной торец коллиматора z_exit_face строго совпадает с верхней гранью WorldVolume (+world_half_z),
      благодаря чему каждый фотон, вышедший через выходную апертуру коллиматора (dir_z > 0),
      мгновенно регистрируется на плоскости z = z_exit_face без обратного рассеяния и параллакса.
    - Входной торец коллиматора расположен при z_enter_face = z_exit_face - thickness.
    - Источник расположен при z_source = z_enter_face - source_distance.
    """
    distance_val = Float(source_distance)
    if distance_val <= 0.0:
        raise ValueError(f"Расстояние от источника до коллиматора должно быть > 0, получено {source_distance}")
    if particles_number <= 0:
        raise ValueError(f"Число испускаемых частиц должно быть > 0, получено {particles_number}")

    collimator_x, collimator_y, collimator_z = preset.size
    back_margin_z = Float(20.0 * units.mm)
    world_size_z = Float(collimator_z + distance_val + back_margin_z)
    world_half_z = Float(0.5 * world_size_z)

    vacuum_material = database_setting.material_database[preset.hole_material_name]
    lead_material = database_setting.material_database[preset.material_name]

    world_volume = Volume(
        geometry=Box(collimator_x, collimator_y, world_size_z),
        material=vacuum_material,
        name="VerificationWorld",
    )

    collimator_center_z = Float(world_half_z - 0.5 * collimator_z)
    z_exit_face = world_half_z
    z_enter_face = Float(world_half_z - collimator_z)
    z_source = Float(z_enter_face - distance_val)

    collimator_node: ParametricParallelCollimator | DirectParallelCollimator
    if implementation_kind == "parametric":
        collimator_node = ParametricParallelCollimator(
            size=preset.size,
            hole_diameter=preset.hole_diameter,
            septa=preset.septa,
            material=lead_material,
            hole_shape=preset.hole_shape,
            name=f"Parametric_{preset.code}",
        )
    elif implementation_kind == "direct":
        collimator_node = DirectParallelCollimator(
            size=preset.size,
            hole_diameter=preset.hole_diameter,
            septa=preset.septa,
            material=lead_material,
            hole_material=vacuum_material,
            hole_shape=preset.hole_shape,
            name=f"Direct_{preset.code}",
        )
    else:
        raise ValueError(
            f"Неизвестный тип реализации коллиматора '{implementation_kind}'. "
            f"Ожидается 'parametric' или 'direct'."
        )

    collimator_node.translate(z=collimator_center_z)
    world_volume.add_child(collimator_node)

    # 3. Конический источник излучения с заданной активностью
    # Активность выбирается так, чтобы за 1 секунду распалось ровно particles_number частиц
    activity_val = Float(float(particles_number) * units.Bq)
    jitter_span = (preset.pitch_x, preset.pitch_y) if cell_averaging else None

    source_node = ConePointSource(
        activity=activity_val,
        energy=energy,
        max_polar_angle=max_polar_angle,
        cell_jitter_span=jitter_span,
        half_life=Float(np.inf),
    )
    source_node.translate(
        x=Float(source_offset_xy[0]),
        y=Float(source_offset_xy[1]),
        z=z_source,
    )
    world_volume.add_child(source_node)

    return CollimatorSceneSetup(
        world_volume=world_volume,
        collimator_node=collimator_node,
        source_node=source_node,
        z_source=z_source,
        z_enter_face=z_enter_face,
        z_exit_face=z_exit_face,
        preset=preset,
        implementation_kind=implementation_kind,
    )


_SHARED_PROPAGATOR: Optional[ParticlePropagator] = None


class QuotaBoundedSimulationManager(SimulationManager):
    """
    Менеджер симуляции с точным соблюдением верхней границы квоты испускаемых частиц
    при многошаговом пополнении банка (предотвращает переполнение квоты за stop_time
    на последнем неполном батче, см. docs/architecture_backlog.md п. 24).
    """

    def __init__(self, *args: Any, max_emitted_particles: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._max_emitted_particles = int(max_emitted_particles)

    @property
    def has_pending_sources(self) -> bool:
        """Проверяет наличие неисчерпанной квоты частиц и активных источников."""
        if not self.active_sources or self._emission_finished:
            return False
        emitted_so_far = int(self.bank.count_array[0]) - self.initial_particle_id
        if emitted_so_far >= self._max_emitted_particles:
            return False
        return True

    def _replenish_bank(self) -> None:
        """Докачивает в банк не более оставшейся квоты max_emitted_particles частиц."""
        if not self.has_pending_sources:
            return

        emitted_so_far = int(self.bank.count_array[0]) - self.initial_particle_id
        remaining_quota = self._max_emitted_particles - emitted_so_far
        if remaining_quota <= 0:
            self._emission_finished = True
            self.global_timer = self.stop_time
            return

        injected_cnt, _, new_t_last, new_pending = inject_poisson_sources_kernel(
            self.bank,
            self.source_buffer,
            Float(np.inf),
            self.t_ref,
            self.t_last,
            self.pending_arrival_time,
            self.rng_ctx,
            Index(remaining_quota),
            Index(0),
        )
        self.t_last = Float(new_t_last)
        self.pending_arrival_time = Float(new_pending)
        self.global_timer = self.t_last

        if int(self.bank.count_array[0]) - self.initial_particle_id >= self._max_emitted_particles:
            self._emission_finished = True
            self.global_timer = self.stop_time
        elif bool(np.isinf(self.pending_arrival_time)) or (
            int(injected_cnt) == 0 and int(np.count_nonzero(self.bank.state.is_active)) == 0
        ):
            self._emission_finished = True


def _get_shared_propagator() -> ParticlePropagator:
    """Возвращает общий экземпляр ParticlePropagator со скомпилированным Numba transport_kernel."""
    global _SHARED_PROPAGATOR
    if _SHARED_PROPAGATOR is None:
        _SHARED_PROPAGATOR = ParticlePropagator(rng=np.random.default_rng(0))
    return _SHARED_PROPAGATOR


def run_collimator_simulation(
    preset: CollimatorPreset,
    implementation_kind: CollimatorImplementationKind,
    source_distance: Float | float,
    energy: Float | float | Sequence[Any] | NDArray[Any],
    particles_number: int,
    max_polar_angle: Float | float = Float(np.pi),
    cell_averaging: bool = False,
    source_offset_xy: tuple[Float | float, Float | float] = (Float(0.0), Float(0.0)),
    seed: int = 42,
    batch_capacity: Optional[int] = None,
) -> SimulationRunSummary:
    """
    Запускает изолированную симуляцию прохождения излучения через коллиматор,
    собирает статистику на выходном торце и измеряет время счета.
    """
    scene_setup = build_collimator_verification_scene(
        preset=preset,
        implementation_kind=implementation_kind,
        source_distance=source_distance,
        energy=energy,
        particles_number=particles_number,
        max_polar_angle=max_polar_angle,
        cell_averaging=cell_averaging,
        source_offset_xy=source_offset_xy,
        seed=seed,
    )

    primary_energy_scalar: Float
    if isinstance(energy, (int, float, np.floating)):
        primary_energy_scalar = Float(energy)
    else:
        energy_arr = scene_setup.source_node.energy
        max_prob_idx = int(np.argmax(energy_arr["probability"]))
        primary_energy_scalar = Float(energy_arr["energy"][max_prob_idx])

    effective_batch = (
        min(particles_number, 500_000) if batch_capacity is None else min(int(batch_capacity), particles_number)
    )
    buffer_cap = max(effective_batch * 2, 100_000)

    propagator = _get_shared_propagator()
    exit_handler = CollimatorExitPlaneHandler(
        z_enter_face=scene_setup.z_enter_face,
        z_exit_face=scene_setup.z_exit_face,
        aperture_size_x=preset.size[0],
        aperture_size_y=preset.size[1],
        hole_diameter=preset.hole_diameter,
        septa=preset.septa,
        primary_energy=primary_energy_scalar,
        total_emitted=particles_number,
        solid_angle_fraction=scene_setup.source_node.solid_angle_fraction,
    )

    sim_manager = QuotaBoundedSimulationManager(
        scene=scene_setup.world_volume,
        propagator=propagator,
        stop_time=Float(1.0 * units.s),
        particles_number=effective_batch,
        min_energy=Float(1.0 * units.keV),
        buffer_capacity=buffer_cap,
        seed=seed,
        consumed_chunk_types=exit_handler.consumed_chunk_types,
        max_emitted_particles=particles_number,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_h5 = os.path.join(tmp_dir, "collimator_verify_run.hdf5")
        data_manager = DataManager(
            filename=dummy_h5,
            handlers=[exit_handler],
            queue=sim_manager.queue,
        )
        data_manager.start()

        start_perf = time.perf_counter()
        sim_manager.start()
        sim_manager.join()
        data_manager.join()
        elapsed_duration = float(max(time.perf_counter() - start_perf, 1e-9))

    classified = exit_handler.classified_result
    throughput = float(particles_number) / elapsed_duration

    return SimulationRunSummary(
        preset=preset,
        implementation_kind=implementation_kind,
        source_distance=Float(source_distance),
        energy=primary_energy_scalar,
        particles_emitted=particles_number,
        elapsed_time=elapsed_duration,
        particle_throughput=throughput,
        simulation_steps=int(sim_manager.step_count),
        photons=classified,
    )


__all__ = [
    "CollimatorImplementationKind",
    "CollimatorSceneSetup",
    "QuotaBoundedSimulationManager",
    "SimulationRunSummary",
    "build_collimator_verification_scene",
    "run_collimator_simulation",
]
