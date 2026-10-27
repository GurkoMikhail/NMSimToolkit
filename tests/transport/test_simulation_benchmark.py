"""
Интеграционный бенчмарк полной симуляции переноса излучения в архитектуре SoA.
"""

from pathlib import Path
import os
import tempfile
import time
import unittest

import hepunits as units
import numpy as np

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from core.data.data_handlers import HistoryAssemblerHandler
from core.data.data_manager import DataManager
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.other.typing_definitions import Float
from core.source.sources import Source
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from gui.factories.gamma_camera_factory import create_default_gamma_camera
from settings.database_setting import material_database
from verification.profiling import (
    build_point_source_workload,
    collect_transport_telemetry,
    run_end_to_end_cprofile,
    run_single_batch_size_benchmark,
    run_single_handler_benchmark,
    run_step_phase_profiler,
    warmup_propagator_instance,
)


class TestFullBenchmark(unittest.TestCase):
    """Проверка полного цикла симуляции ОФЭКТ и компонентов профилировочного стенда."""

    def test_run_benchmark(self) -> None:
        root_scene = Volume(
            geometry=Box(120 * units.cm, 120 * units.cm, 80 * units.cm),
            material=material_database["Air, Dry (near sea level)"],
            name="Simulation_volume",
        )

        detector = Volume(
            geometry=Box(54.0 * units.cm, 40 * units.cm, 0.95 * units.cm),
            material=material_database["Sodium Iodide"],
            name="Detector",
        )

        collimator = ParametricParallelCollimator(
            size=(detector.size[0], detector.size[1], 3.5 * units.cm),
            hole_diameter=1.5 * units.mm,
            septa=0.2 * units.mm,
            material=material_database["Pb"],
            name="Collimator",
        )

        spect_head, _ = create_default_gamma_camera(
            collimator=collimator,
            detector=detector,
            shielding_material=material_database["Pb"],
            internal_medium=material_database["Air, Dry (near sea level)"],
            glass_material=material_database["Glass, Borosilicate (Pyrex)"],
            shielding_thickness=2 * units.cm,
            glass_backend_thickness=7.6 * units.cm,
            name="Gamma_camera",
        )

        casing_node = next(
            child for child in spect_head.childs if child.name == spect_head.slots["casing"]
        )
        assert isinstance(casing_node, Volume)
        radius = 23.3 * units.cm
        spect_head.rotate(gamma=units.pi / 2)
        spect_head.translate(y=radius + casing_node.size[2] / 2)

        root_scene.add_child(spect_head)

        activity = Float(1000 * units.Bq)
        energy = Float(140.5 * units.keV)
        source = Source.from_point(
            activity=activity,
            energy=energy,
        )
        root_scene.add_child(source)

        propagator = ParticlePropagator(rng=np.random.default_rng(42))
        stop_time = Float(0.1 * units.second)
        particles_number = 10000

        manager = SimulationManager(
            scene=root_scene,
            propagator=propagator,
            stop_time=stop_time,
            particles_number=particles_number,
            buffer_capacity=100000,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = os.path.join(tmp_dir, "benchmark_soa.hdf5")
            handler = HistoryAssemblerHandler(sensitive_volumes=[detector])
            data_manager = DataManager(
                filename=h5_path,
                handlers=[handler],
                queue=manager.queue,
            )
            data_manager.start()

            start_time = time.perf_counter()
            manager.start()
            manager.join()
            data_manager.join()
            end_time = time.perf_counter()

            print(f"Simulation finished in {end_time - start_time:.4f} seconds.")

    def test_profiling_harness_components(self) -> None:
        """Смоук-проверка модулей профилировочного стенда (cProfile, DataHandlers, StepProfiler, Telemetry, BatchSize)."""
        prim_tests_by_collimator = {}
        for coll_kind in ("parametric", "direct"):
            workload = build_point_source_workload(
                coll_kind,
                particles_number=6_000,
                bank_capacity=2_000,
                seed=101,
            )
            self.assertEqual(workload.collimator_kind, coll_kind)
            self.assertEqual(workload.effective_bank_capacity, 2_000)

            step_res = run_step_phase_profiler(workload, buffer_capacity=15_000, seed=101)
            self.assertGreater(step_res.total_wall_time, 0.0)
            self.assertGreater(step_res.propagator_cold_compile_time, 0.0)
            self.assertGreater(step_res.warm_vs_cold_speedup, 0.0)
            self.assertGreater(step_res.throughput_histories_per_sec, 0.0)
            self.assertGreater(step_res.full_bank_steps, 1)
            self.assertGreaterEqual(step_res.depletion_steps, 1)
            self.assertGreater(step_res.full_bank_tracks_per_sec, 0.0)
            self.assertGreater(step_res.depletion_tracks_per_sec, 0.0)
            self.assertGreater(step_res.depletion_track_slowdown_factor, 0.0)
            pcts = step_res.phase_percentages()
            self.assertAlmostEqual(sum(pcts.values()), 100.0, places=3)
            # После прогрева замыканий ParticlePropagator фаза P8 больше не раздута компиляцией LLVM (~75%),
            # а P4_Geometry_Raycast занимает > 5% от суммарного времени JIT-ядер (P4 + P5 + P8)
            self.assertGreater(pcts["P4_Geometry_Raycast"], 0.25)
            self.assertLess(pcts["P8_Physics_Processes"], 50.0)
            jit_sum = (
                pcts["P4_Geometry_Raycast"]
                + pcts["P5_Woodcock_Transport"]
                + pcts["P8_Physics_Processes"]
            )
            p4_jit_share = pcts["P4_Geometry_Raycast"] / max(jit_sum, 1e-12)
            self.assertGreater(p4_jit_share, 0.05)

            telem = collect_transport_telemetry(workload, seed=101)
            self.assertEqual(telem.histories_count, 6_000)
            self.assertGreater(telem.total_raycast_traces, 0)
            if coll_kind == "direct":
                self.assertGreater(telem.total_hex_prism_intersections, 0)
            else:
                self.assertEqual(telem.total_hex_prism_intersections, 0)
            prim_tests_by_collimator[coll_kind] = telem.primitive_tests_per_raycast

        # Для DirectParallelCollimator число проверок примитивов на луч выше, чем для Parametric
        self.assertGreater(prim_tests_by_collimator["direct"], prim_tests_by_collimator["parametric"])

        # Проверка warmup_propagator_instance
        w_warm = build_point_source_workload("parametric", particles_number=1_500, seed=101)
        prop_test = ParticlePropagator(rng=np.random.default_rng(101))
        cold_sec = warmup_propagator_instance(prop_test, w_warm.root_scene)
        warm_sec = warmup_propagator_instance(prop_test, w_warm.root_scene)
        self.assertGreater(cold_sec, 0.0)
        self.assertGreater(warm_sec, 0.0)

        # Проверка бенчмарка размера батча (Batch Size Sweep) и разделения Full-Bank / Depletion фаз
        bs_small = run_single_batch_size_benchmark(
            workload=build_point_source_workload("parametric", particles_number=3_000, seed=101),
            batch_size=500,
            total_particles=3_000,
            propagator=prop_test,
        )
        bs_large = run_single_batch_size_benchmark(
            workload=build_point_source_workload("parametric", particles_number=3_000, seed=101),
            batch_size=2_000,
            total_particles=3_000,
            propagator=prop_test,
        )
        self.assertEqual(bs_small.emitted_particles, 3_000)
        self.assertEqual(bs_large.emitted_particles, 3_000)
        self.assertLess(bs_large.steps_count, bs_small.steps_count)
        self.assertGreater(bs_small.full_bank_steps_count, 2)
        self.assertGreaterEqual(bs_small.depletion_steps_count, 1)
        self.assertGreater(bs_small.full_bank_histories_per_sec, 0.0)
        self.assertGreater(bs_small.depletion_track_slowdown_factor, 0.0)

        with tempfile.TemporaryDirectory() as tmp_stats:
            w_dir = build_point_source_workload("direct", particles_number=1_500, seed=101)
            cprof_res = run_end_to_end_cprofile(
                w_dir,
                stats_dir=Path(tmp_stats),
                buffer_capacity=5_000,
                propagator=prop_test,
            )
            self.assertTrue(Path(cprof_res.end_to_end_prof_path).is_file())
            self.assertTrue(Path(cprof_res.dual_thread_metrics.sim_prof_path).is_file())
            self.assertTrue(Path(cprof_res.dual_thread_metrics.data_prof_path).is_file())
            self.assertGreater(cprof_res.domain_breakdown.total_profiled_time, 0.0)

            h_rec = run_single_handler_benchmark(
                workload=w_dir,
                handler_code="H3_HistoryAssembler_Full_Gzip",
                handler_label="HistoryAssembler (gzip)",
                compression="gzip",
                enable_escaped=True,
                buffer_capacity=5_000,
                propagator=prop_test,
            )
            self.assertGreater(h_rec.total_sync_time, 0.0)
            self.assertGreaterEqual(h_rec.chunks_dispatched, 1)


if __name__ == "__main__":
    unittest.main()
