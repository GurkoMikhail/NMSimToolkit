from core.geometry.geometries import Box
from core.geometry.geometry_compiler import GeometryBufferDType
from core.geometry.volumes import Volume
import os
import queue
import sys
import threading
import time
from typing import Any
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.materials.materials import Material
from core.other.typing_definitions import CFuncAddress, Charge, Energy, Float, ID, Index, ProcessID, Species
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import ChunkType, InteractionBuffer, RNGContext
from core.physics.physics_buffer import ElementCSR, PhysicsBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.processes import CoherentScattering, ComptonScattering, PhotoelectricEffect
from core.physics.processes_kernels import _push_to_interaction_buffer
from core.scene.nodes import CompositeNode
from core.source.sources import Source
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from core.transport.transport_buffer import TransportBuffer
from core.transport.transport_kernels import make_transport_kernel
import hepunits as units
from settings.database_setting import material_database


class TestStep5TransportLogic(unittest.TestCase):
    def test_transport_kernel_execution(self):
        bank = ParticleBank.allocate(10)
        bank.state.is_active[:] = True
        bank.state.energy[:] = 1.0
        bank.state.position.x[:] = 0.0
        bank.state.position.y[:] = 0.0
        bank.state.position.z[:] = 0.0
        bank.state.direction.x[:] = 1.0
        bank.state.direction.y[:] = 0.0
        bank.state.direction.z[:] = 0.0

        bank.navigation_state.current_volume[:] = 0
        bank.navigation_state.boundary_distance[:] = 1.0

        mat_info = np.zeros(2, dtype=MaterialInfoDType)
        mat_pointers = np.zeros(2, dtype=MaterialPointerDType)
        mat_pointers[0]['length'] = 2
        mat_pointers[0]['lookup_offset'] = 0
        mat_pointers[1]['start_idx'] = 2
        mat_pointers[1]['length'] = 2
        mat_pointers[1]['lookup_offset'] = 4

        physics_energy_grid = np.array([0.1, 10.0, 0.1, 10.0], dtype=Float)
        physics_lac_table = np.array([
            [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]
        ], dtype=Float)

        log_e_min = Float(np.log(0.1))
        log_e_max = Float(np.log(10.0))
        lookup_num_bins = 4
        material_bank = MaterialBank(
            mat_info_buffer=mat_info,
            mat_pointers=mat_pointers,
            physics_energy_grid=physics_energy_grid,
            physics_lac_table=physics_lac_table,
            physics_log_energy_grid=np.log(physics_energy_grid).astype(Float),
            physics_log_lac_table=np.log(physics_lac_table).astype(Float),
            lookup_bin_start=np.zeros(2 * lookup_num_bins, dtype=Index),
            lookup_log_e_min=log_e_min,
            lookup_inv_dlog_e=Float(lookup_num_bins / (log_e_max - log_e_min)),
            lookup_num_bins=lookup_num_bins,
            lookup_max_advance=0,
        )
        material_bank._validate()
        majorant_material_map = np.array([0, 1], dtype=Index)
        woodcock_function_pointers = np.zeros(2, dtype=CFuncAddress)

        element_partial_fractions = np.zeros((4, 3, 2), dtype=Float)
        element_partial_fractions[0:2, :, 0] = 1.0
        element_partial_fractions[2:4, :, 0] = 0.5
        element_partial_fractions[2:4, :, 1] = 0.5

        element_csr = ElementCSR(
            element_offsets=np.array([0, 1, 3], dtype=Index),
            element_Z=np.array([1, 6, 8], dtype=Charge),
            element_partial_fractions=element_partial_fractions,
        )

        physics_buffer = PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=majorant_material_map,
            woodcock_function_pointers=woodcock_function_pointers,
            element_csr=element_csr,
            relaxation_bank=PhysicsCompiler()._build_atomic_relaxation_bank(
                [],
                unique_atomic_numbers=element_csr.element_Z.tolist(),
            ),
        )

        rng = np.random.default_rng(42)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        mapped_process_ids = np.array([0, 1, 2], dtype=Index)
        transport_kernel = make_transport_kernel(mapped_process_ids)
        transport_buffer = TransportBuffer.allocate(10)
        self.assertEqual(transport_buffer.process_ids.shape, (10,))
        self.assertEqual(transport_buffer.material_ids.shape, (10,))
        self.assertEqual(transport_buffer.process_lacs.ndim, 1)
        active_indices = np.arange(10, dtype=Index)

        transport_kernel(
            bank.state,
            bank.navigation_state,
            active_indices,
            physics_buffer,
            transport_buffer,
            np.zeros(1, dtype=GeometryBufferDType), rng_ctx
        )

        self.assertTrue(np.all(bank.state.position.x > 0.0))

    def test_simulation_manager_buffer_capacity_invariant(self):
        """Проверка инварианта: buffer_capacity тихо поднимается до particles_number при передаче меньшего значения."""
        scene = CompositeNode()
        # buffer_capacity < particles_number: тихий clamp, исключения нет
        mgr_clamped = SimulationManager(scene=scene, particles_number=5000, buffer_capacity=1000)
        self.assertEqual(mgr_clamped.data_buffer.interactions.capacity, 5000)

        # buffer_capacity >= particles_number: значение сохраняется как есть
        mgr = SimulationManager(scene=scene, particles_number=5000, buffer_capacity=5000)
        self.assertEqual(mgr.data_buffer.interactions.capacity, 5000)

        # При buffer_capacity=None ёмкость автоматически равна particles_number
        mgr_auto = SimulationManager(scene=scene, particles_number=150000, buffer_capacity=None)
        self.assertEqual(mgr_auto.data_buffer.interactions.capacity, 150000)

    def test_causal_flush_ordering(self):
        """Проверка причинно-следственного порядка сброса через _flush_through: initial_states -> interactions -> dead_particles."""
        scene = CompositeNode()
        received_chunks = []
        test_queue = queue.Queue()
        mgr = SimulationManager(scene=scene, particles_number=10, buffer_capacity=10, queue=test_queue)

        # Имитируем накопление данных во всех трех буферах
        mgr.data_buffer.initial_states.particle_ID[0] = 1
        mgr.data_buffer.initial_states.emission_time[0] = 0.0
        mgr.data_buffer.initial_states.emission_energy[0] = 140.0
        mgr.data_buffer.initial_states.emission_position.x[0] = 0.0
        mgr.data_buffer.initial_states.emission_position.y[0] = 0.0
        mgr.data_buffer.initial_states.emission_position.z[0] = 0.0
        mgr.data_buffer.initial_states.emission_direction.x[0] = 0.0
        mgr.data_buffer.initial_states.emission_direction.y[0] = 0.0
        mgr.data_buffer.initial_states.emission_direction.z[0] = 1.0
        mgr.data_buffer.initial_states.cursor[0] = 1

        mgr.data_buffer.interactions.process_id[0] = 1
        mgr.data_buffer.interactions.volume_id[0] = 0
        mgr.data_buffer.interactions.material_id[0] = 1
        mgr.data_buffer.interactions.particle_ID[0] = 1
        mgr.data_buffer.interactions.energy_deposit[0] = 20.0
        mgr.data_buffer.interactions.scattering_theta[0] = 0.1
        mgr.data_buffer.interactions.scattering_phi[0] = 0.2
        mgr.data_buffer.interactions.distance_traveled[0] = 5.0
        mgr.data_buffer.interactions.species[0] = 0
        mgr.data_buffer.interactions.Z[0] = 7.4
        mgr.data_buffer.interactions.position.x[0] = 0.0
        mgr.data_buffer.interactions.position.y[0] = 0.0
        mgr.data_buffer.interactions.position.z[0] = 5.0
        mgr.data_buffer.interactions.direction.x[0] = 0.0
        mgr.data_buffer.interactions.direction.y[0] = 0.0
        mgr.data_buffer.interactions.direction.z[0] = 1.0
        mgr.data_buffer.interactions.cursor[0] = 1

        mgr.data_buffer.dead_particles.particle_ID[0] = 1
        mgr.data_buffer.dead_particles.cursor[0] = 1

        # Вызов _flush_through(DEAD_PARTICLES) должен сначала сбросить initial_states и interactions
        mgr._flush_through(ChunkType.DEAD_PARTICLES)

        while not test_queue.empty():
            received_chunks.append(test_queue.get())

        chunk_types = [c['type'] for c in received_chunks if isinstance(c, dict)]
        self.assertEqual(
            chunk_types,
            [ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS, ChunkType.DEAD_PARTICLES],
        )

    def test_flush_all_ordering(self):
        """Проверка причинно-следственного порядка сброса через flush_all: initial_states -> interactions -> dead_particles."""
        scene = CompositeNode()
        received_chunks = []
        test_queue = queue.Queue()
        mgr = SimulationManager(scene=scene, particles_number=10, buffer_capacity=10, queue=test_queue)

        # Заполняем буферы данными
        mgr.data_buffer.initial_states.particle_ID[0] = 42
        mgr.data_buffer.initial_states.cursor[0] = 1

        mgr.data_buffer.interactions.particle_ID[0] = 42
        mgr.data_buffer.interactions.cursor[0] = 1

        mgr.data_buffer.dead_particles.particle_ID[0] = 42
        mgr.data_buffer.dead_particles.cursor[0] = 1

        # Вызов flush_all должен произвести сброс всех трех буферов в правильном порядке
        mgr.flush_all()

        while not test_queue.empty():
            received_chunks.append(test_queue.get())

        chunk_types = [c['type'] for c in received_chunks if isinstance(c, dict)]
        self.assertEqual(
            chunk_types,
            [ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS, ChunkType.DEAD_PARTICLES],
        )

    def test_buffer_overflow_raises_runtime_error_without_ring_overwrite(self):
        """
        Проверка детекции переполнения буфера телеметрии:
        1. Запись сверх capacity в Numba-ядре не перезаписывает начало буфера по модулю.
        2. Счетчик cursor[0] инкрементируется сверх capacity, а _check_no_overflow() выбрасывает RuntimeError.
        """
        scene = CompositeNode()
        mgr = SimulationManager(scene=scene, particles_number=2, buffer_capacity=2)
        ibuf = mgr.data_buffer.interactions

        # Записываем 3 взаимодействия при capacity == 2 (имитация процесса, пишущего > 1 записи на частицу)
        _push_to_interaction_buffer(
            ibuf, mgr.bank.state, Index(0), ProcessID(1), Index(0), Index(0), ID(101), Energy(0.05),
            Float(0.1), Float(0.2), Charge(82),
        )
        _push_to_interaction_buffer(
            ibuf, mgr.bank.state, Index(0), ProcessID(1), Index(0), Index(0), ID(202), Energy(0.06),
            Float(0.3), Float(0.4), Charge(82),
        )
        # Третья запись не должна затереть нулевой слот (ID=101)
        _push_to_interaction_buffer(
            ibuf, mgr.bank.state, Index(0), ProcessID(2), Index(0), Index(0), ID(999), Energy(0.99),
            Float(0.5), Float(0.6), Charge(82),
        )

        self.assertEqual(ibuf.cursor_value, 3)
        np.testing.assert_array_equal(ibuf.particle_ID, np.array([101, 202], dtype=ID))

        with self.assertRaisesRegex(RuntimeError, r"Переполнение канала interactions: 3 > 2"):
            mgr._check_no_overflow()

    def test_uncollided_initial_states_recorded_without_escaped_subscription(self):
        """
        Проверка дозаписи INITIAL_STATES для непрореагировавших частиц без подписки на ESCAPED_PARTICLES:
        как при вылете за границу мира (current_volume < 0), так и при гибели по порогу min_energy внутри мира.
        """
        world = Volume(geometry=Box(100.0, 100.0, 100.0), material=Material(name="Air"), name="World")
        test_queue = queue.Queue()
        mgr = SimulationManager(
            scene=world,
            particles_number=3,
            buffer_capacity=4,
            min_energy=Float(0.05),
            queue=test_queue,
            consumed_chunk_types=frozenset({ChunkType.INITIAL_STATES, ChunkType.DEAD_PARTICLES}),
        )
        self.assertFalse(mgr.data_buffer.escaped_particles.is_enabled)
        self.assertTrue(mgr.data_buffer.initial_states.is_enabled)
        self.assertTrue(mgr.data_buffer.dead_particles.is_enabled)

        # Настраиваем 3 частицы:
        # 0: вылетела из мира без взаимодействий (current_volume = -1, state == initial_state)
        # 1: убита по порогу min_energy внутри мира без взаимодействий (energy = 0.01 < min_energy, emission_energy = 0.01)
        # 2: провзаимодействовала ранее (direction != emission_direction) и погибла
        mgr.bank.state.is_active[:3] = True
        mgr.bank.initial_state.ID[:3] = np.array([10, 20, 30], dtype=ID)
        mgr.bank.initial_state.emission_energy[:3] = np.array([0.14, 0.01, 0.14], dtype=Float)
        mgr.bank.state.energy[:3] = np.array([0.14, 0.01, 0.02], dtype=Float)
        mgr.bank.initial_state.emission_direction.x[:3] = 0.0
        mgr.bank.initial_state.emission_direction.y[:3] = 0.0
        mgr.bank.initial_state.emission_direction.z[:3] = 1.0
        mgr.bank.state.direction.x[:3] = np.array([0.0, 0.0, 1.0], dtype=Float)
        mgr.bank.state.direction.y[:3] = 0.0
        mgr.bank.state.direction.z[:3] = np.array([1.0, 1.0, 0.0], dtype=Float)
        mgr.bank.navigation_state.current_volume[:3] = np.array([-1, 0, 0], dtype=Index)
        mgr.bank.navigation_state.boundary_distance[:3] = 1.0

        dead_indices = mgr._apply_invalidators(np.array([0, 1, 2], dtype=Index))
        np.testing.assert_array_equal(dead_indices, np.array([0, 1, 2], dtype=Index))

        mgr._record_finished_histories(dead_indices)
        mgr.flush_all()

        chunks = []
        while not test_queue.empty():
            chunks.append(test_queue.get_nowait())

        self.assertEqual([c["type"] for c in chunks], [ChunkType.INITIAL_STATES, ChunkType.DEAD_PARTICLES])
        # Обе непрореагировавшие частицы (10 и 20) попали в INITIAL_STATES перед их сбросом в DEAD_PARTICLES
        np.testing.assert_array_equal(chunks[0]["data"]["particle_ID"], np.array([10, 20], dtype=ID))
        np.testing.assert_array_equal(chunks[1]["data"], np.array([10, 20, 30], dtype=ID))

    def test_stop_unblocks_full_queue_send_data(self):
        """Проверка того, что вызов stop() разблокирует поток SimulationManager при переполненной очереди."""
        scene = CompositeNode()
        bounded_queue = queue.Queue(maxsize=1)
        bounded_queue.put({"type": "dummy_blocker"})

        mgr = SimulationManager(scene=scene, particles_number=4, buffer_capacity=4, queue=bounded_queue)
        mgr.resume()

        finished_event = threading.Event()

        def _blocked_sender():
            mgr.send_data({"type": ChunkType.DEAD_PARTICLES, "data": np.array([1], dtype=ID)})
            finished_event.set()

        sender_thread = threading.Thread(target=_blocked_sender, daemon=True)
        sender_thread.start()

        # Убеждаемся, что поток действительно заблокирован на полной очереди
        self.assertFalse(finished_event.wait(timeout=0.15))

        # Вызов stop() должен вывести _put_to_queue из цикла ожидания на следующем тике таймаута
        mgr.stop()
        self.assertTrue(finished_event.wait(timeout=1.0))
        sender_thread.join(timeout=1.0)
        self.assertFalse(sender_thread.is_alive())

    def test_batch_capacity_and_bank_consistency(self):
        """Проверка совпадения batch_capacity с емкостью ParticleBank и выброса ValueError при расхождении."""
        scene = CompositeNode()
        bank = ParticleBank.allocate(32)
        mgr = SimulationManager(scene=scene, particles_number=100, batch_capacity=32, bank=bank)
        self.assertEqual(mgr.batch_capacity, 32)
        self.assertEqual(mgr.bank.capacity, 32)

        with self.assertRaises(ValueError):
            SimulationManager(scene=scene, particles_number=100, batch_capacity=64, bank=bank)
        with self.assertRaises(ValueError):
            SimulationManager(scene=scene, particles_number=100, batch_capacity=0)

    def test_nai_crystal_iodine_k_escape_peak_and_full_absorption_recovery(self) -> None:
        """
        Физический тест Фазы 4 на кристаллическом детекторе NaI(Tl) при облучении 99mTc (140.5 кэВ):
        1. Когда вторичный K_alpha / K_beta фотон иода (Z=53, E_K ~ 28.3-33.0 кэВ) покидает
           тонкий кристалл NaI без взаимодействия внутри кристалла, энерговыделение первичной
           вершины в кристалле формирует K-escape пик иода:
               E_dep_primary == E_gamma - E_K(I) ~ 107.5-112.5 кэВ (~108-112 кэВ).
        2. Когда вторичный характеристический фотон иода поглощается внутри того же кристалла NaI
           (без вылета вторичных/третичных квантов), сумма энерговыделений родителя и потомка
           (связанных через parent_ID == primary_particle_ID) в точности восстанавливает пик
           полного поглощения 140.5 кэВ.
        """
        primary_energy = Float(140.5 * units.keV)
        air_mat = material_database["Air, Dry (near sea level)"]
        nai_mat = material_database["Sodium Iodide"]

        world = Volume(
            geometry=Box(400.0 * units.mm, 400.0 * units.mm, 400.0 * units.mm),
            material=air_mat,
            name="World",
        )
        # Кристалл NaI умеренной толщины (1.0 мм), чтобы наблюдались как вылеты K-фотонов иода
        # наружу (K-escape пик), так и их полное поглощение внутри кристалла (Photopeak 140.5 кэВ).
        crystal = Volume(
            geometry=Box(200.0 * units.mm, 200.0 * units.mm, 1.0 * units.mm),
            material=nai_mat,
            name="NaICrystal",
        )
        world.add_child(crystal)
        point_src = Source.from_point(
            activity=Float(1.0e5 * units.Bq),
            energy=primary_energy,
            half_life=Float(np.inf),
            name="Tc99m_Source",
        )
        world.add_child(point_src)

        # Порог photon_production_cut = 10 кэВ пропускает все K-линии иода (28.3 - 33.0 кэВ),
        # оставляя мягкие L-переходы (< 5.2 кэВ) в локальном энерговыделении для чистого двухколенного каскада.
        mgr = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(
                processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
            ),
            start_time=Float(0.0),
            stop_time=None,
            particles_number=4000,
            batch_capacity=512,
            buffer_capacity=16384,
            min_energy=Float(1.0 * units.keV),
            seed=2026420,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(10.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )
        mgr.start()
        mgr.join()

        init_list: list[dict[str, np.ndarray]] = []
        inter_list: list[dict[str, np.ndarray]] = []
        while True:
            item = mgr.queue.get_nowait()
            if item is None or item == "stop":
                break
            if item["type"] == ChunkType.INITIAL_STATES:
                init_list.append(item["data"])
            elif item["type"] == ChunkType.INTERACTIONS:
                inter_list.append(item["data"])

        init_data = {
            key: np.concatenate([chunk[key] for chunk in init_list])
            for key in init_list[0]
        }
        inter_data = {
            key: np.concatenate([chunk[key] for chunk in inter_list])
            for key in inter_list[0]
        }

        crystal_mat_id = Index(nai_mat.ID)
        crystal_inter_mask = inter_data["material_id"] == crystal_mat_id

        # Суммарное энерговыделение каждой частицы (по её particle_ID) внутри кристалла NaI
        max_pid = int(np.max(init_data["particle_ID"])) + 1
        crystal_deposit_by_pid = np.zeros(max_pid, dtype=Float)
        np.add.at(
            crystal_deposit_by_pid,
            inter_data["particle_ID"][crystal_inter_mask],
            inter_data["energy_deposit"][crystal_inter_mask],
        )

        # Общее количество взаимодействий каждой частицы во всей сцене и в кристалле
        # (чтобы выделить чистый фотоэффект в кристалле на первом шаге без предварительного рассеяния в воздухе)
        total_inter_count_by_pid = np.zeros(max_pid, dtype=np.int32)
        np.add.at(
            total_inter_count_by_pid,
            inter_data["particle_ID"],
            1,
        )
        crystal_inter_count_by_pid = np.zeros(max_pid, dtype=np.int32)
        np.add.at(
            crystal_inter_count_by_pid,
            inter_data["particle_ID"][crystal_inter_mask],
            1,
        )

        # Находим все вторичные K-фотоны иода (parent_ID >= 0, E_sec in [27.0 кэВ, 33.2 кэВ])
        sec_mask = (init_data["parent_ID"] >= 0) & (
            init_data["emission_energy"] >= Float(27.0 * units.keV)
        ) & (init_data["emission_energy"] <= Float(33.2 * units.keV))
        sec_pids = init_data["particle_ID"][sec_mask]
        sec_parent_pids = init_data["parent_ID"][sec_mask]
        sec_emit_energies = init_data["emission_energy"][sec_mask]

        self.assertGreater(len(sec_pids), 200)

        escape_primary_deposits: list[float] = []
        absorbed_pair_deposits: list[float] = []

        for idx in range(len(sec_pids)):
            child_pid = int(sec_pids[idx])
            parent_pid = int(sec_parent_pids[idx])
            child_emit_energy = float(sec_emit_energies[idx])

            # Рассматриваем первичные фотоны, испытавшие прямое фотопоглощение в кристалле на первом шаге
            if crystal_inter_count_by_pid[parent_pid] != 1 or total_inter_count_by_pid[parent_pid] != 1:
                continue

            parent_dep = float(crystal_deposit_by_pid[parent_pid])
            child_dep = float(crystal_deposit_by_pid[child_pid])

            # Энерговыделение первичной вершины обязано равняться E_gamma - E_K_line
            self.assertAlmostEqual(
                parent_dep + child_emit_energy,
                float(primary_energy),
                places=12,
            )

            if child_dep == 0.0:
                # Вторичный K-фотон иода покинул кристалл NaI без взаимодействия -> K-escape пик
                escape_primary_deposits.append(parent_dep)
            elif abs(child_dep - child_emit_energy) <= 1e-12:
                # Вторичный K-фотон иода полностью поглотился внутри того же кристалла NaI -> пик 140.5 кэВ
                absorbed_pair_deposits.append(parent_dep + child_dep)

        self.assertGreater(len(escape_primary_deposits), 30)
        self.assertGreater(len(absorbed_pair_deposits), 50)

        escape_arr = np.array(escape_primary_deposits, dtype=Float)
        # K-escape пик иода для 140.5 кэВ лежит в диапазоне 140.5 - (28.3..33.0) = 107.5..112.2 кэВ (~108-112 кэВ)
        self.assertTrue(np.all((escape_arr >= Float(107.3 * units.keV)) & (escape_arr <= Float(112.5 * units.keV))))
        # Проверяем наличие основной K_alpha escape линии (~111.9 - 112.2 кэВ) и K_beta escape линии (~107.5 - 108.2 кэВ)
        self.assertGreater(
            int(np.sum((escape_arr >= Float(111.5 * units.keV)) & (escape_arr <= Float(112.5 * units.keV)))),
            20,
        )
        self.assertGreater(
            int(np.sum((escape_arr >= Float(107.4 * units.keV)) & (escape_arr <= Float(108.5 * units.keV)))),
            3,
        )

        # Сумма энерговыделений родителя и поглощённого потомка (parent_ID == primary_particle_ID)
        # в точности восстанавливает пик полного поглощения 140.5 кэВ
        absorbed_arr = np.array(absorbed_pair_deposits, dtype=Float)
        np.testing.assert_allclose(
            absorbed_arr,
            np.full(len(absorbed_arr), primary_energy, dtype=Float),
            rtol=0.0,
            atol=1e-12,
        )

    def test_stateful_resume_with_active_secondary_source_and_restore_checkpoint(self) -> None:
        """
        Сквозной тест Фазы 4 (Stateful Resume при активном SecondarySource и restore_checkpoint):
        1. Сохранение состояния в середине расчёта, когда в SecondaryActivationBuffer накоплены
           невыгруженные точки активации (secondary_source.has_pending == True) и уже выпущена
           часть первичных частиц; восстановление в новый SimulationManager через restore_checkpoint
           и проверка 100% побитовой идентичности продолжения расчёта и отсутствия коллизий
           particle_ID / parent_ID.
        2. Двухэтапный расчёт по времени [0, T1] + [T1, T2] против непрерывного [0, T2]
           при включённой атомной релаксации.
        """
        lead_mat = material_database["Pb"]
        world = Volume(
            geometry=Box(500.0 * units.mm, 500.0 * units.mm, 500.0 * units.mm),
            material=lead_mat,
            name="World",
        )
        world.add_child(
            Source.from_point(
                activity=Float(8000.0 * units.Bq),
                energy=Float(140.5 * units.keV),
                half_life=Float(2.0 * units.s),
                name="Tc99m_Pb",
            )
        )

        def _drain_all_telemetry(mgr: SimulationManager) -> dict[str, Any]:
            init_chunks: list[dict[str, np.ndarray]] = []
            inter_chunks: list[dict[str, np.ndarray]] = []
            dead_chunks: list[np.ndarray] = []
            while not mgr.queue.empty():
                item = mgr.queue.get_nowait()
                if item is None or item == "stop":
                    continue
                if item["type"] == ChunkType.INITIAL_STATES:
                    init_chunks.append(item["data"])
                elif item["type"] == ChunkType.INTERACTIONS:
                    inter_chunks.append(item["data"])
                elif item["type"] == ChunkType.DEAD_PARTICLES:
                    dead_chunks.append(item["data"])
            init_merged = {
                key: np.concatenate([chunk[key] for chunk in init_chunks])
                for key in init_chunks[0]
            }
            inter_merged = {
                key: np.concatenate([chunk[key] for chunk in inter_chunks])
                for key in inter_chunks[0]
            }
            dead_merged = np.concatenate(dead_chunks)
            return {
                "initial_states": init_merged,
                "interactions": inter_merged,
                "dead_particles": dead_merged,
            }

        # Часть 1: Сохранение чекпоинта прямо в середине расчёта при непустом SecondaryActivationBuffer
        # Используем PhotoelectricEffect в свинце: на первом шаге первичные фотоны поглощаются (банк пустеет),
        # а в SecondaryActivationBuffer накапливаются живые точки активации K/L-флуоресценции!
        target_primaries = 60
        batch_cap = 15
        mgr_interrupted = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(processes_list=[PhotoelectricEffect]),
            start_time=Float(0.0),
            stop_time=None,
            particles_number=target_primaries,
            batch_capacity=batch_cap,
            buffer_capacity=2000,
            min_energy=Float(1.0 * units.keV),
            seed=2026430,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )

        # Выполняем 1 шаг: первичные частицы первого батча поглотились и создали вторичные активации
        mgr_interrupted.step_once()
        self.assertFalse(mgr_interrupted.has_active_particles)
        self.assertTrue(mgr_interrupted.secondary_source.has_pending)
        self.assertGreater(mgr_interrupted.secondary_source.activation_buffer.cursor_value, 8)
        self.assertEqual(mgr_interrupted.emitted_primaries, batch_cap)

        # Снимаем полный чекпоинт состояния в середине расчёта (с активным SecondarySource!)
        mid_checkpoint: dict[str, Any] = {
            "next_particle_id": int(mgr_interrupted.bank.count_array[0]),
            "rng_state": mgr_interrupted.rng.bit_generator.state,
            "secondary_activations": mgr_interrupted.secondary_source.export_state(),
        }
        prim_exported = mgr_interrupted.primary_source.export_state()
        if prim_exported is not None:
            mid_checkpoint.update(prim_exported)

        telemetry_before = _drain_all_telemetry(mgr_interrupted)

        # Восстанавливаем чекпоинт в совершенно новый экземпляр SimulationManager через restore_checkpoint
        mgr_resumed = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(processes_list=[PhotoelectricEffect]),
            start_time=Float(0.0),
            stop_time=None,
            particles_number=target_primaries,
            batch_capacity=batch_cap,
            buffer_capacity=2000,
            min_energy=Float(1.0 * units.keV),
            seed=1,  # Намеренно другой начальный сид, который обязан перезаписаться через restore_checkpoint
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )
        mgr_resumed.restore_checkpoint(mid_checkpoint)
        self.assertTrue(mgr_resumed.secondary_source.has_pending)
        self.assertEqual(
            mgr_resumed.secondary_source.activation_buffer.cursor_value,
            mgr_interrupted.secondary_source.activation_buffer.cursor_value,
        )

        # Доводим оба менеджера (прерванный и восстановленный) до завершения
        mgr_interrupted.start()
        mgr_interrupted.join()
        mgr_resumed.start()
        mgr_resumed.join()

        telemetry_cont_tail = _drain_all_telemetry(mgr_interrupted)
        telemetry_resumed_tail = _drain_all_telemetry(mgr_resumed)

        # Продолжение расчёта из чекпоинта обязано быть на 100% побитово идентичным!
        for key in telemetry_cont_tail["initial_states"]:
            np.testing.assert_array_equal(
                telemetry_resumed_tail["initial_states"][key],
                telemetry_cont_tail["initial_states"][key],
            )
        for key in telemetry_cont_tail["interactions"]:
            np.testing.assert_array_equal(
                telemetry_resumed_tail["interactions"][key],
                telemetry_cont_tail["interactions"][key],
            )
        np.testing.assert_array_equal(
            telemetry_resumed_tail["dead_particles"],
            telemetry_cont_tail["dead_particles"],
        )

        # Проверяем отсутствие коллизий particle_ID и корректность дерева parent_ID во всём объединённом расчёте
        all_pids = np.concatenate([
            telemetry_before["initial_states"]["particle_ID"],
            telemetry_resumed_tail["initial_states"]["particle_ID"],
        ])
        all_parents = np.concatenate([
            telemetry_before["initial_states"]["parent_ID"],
            telemetry_resumed_tail["initial_states"]["parent_ID"],
        ])
        total_histories = int(mgr_resumed.bank.count_array[0])
        self.assertEqual(len(all_pids), total_histories)
        self.assertEqual(len(np.unique(all_pids)), total_histories)
        self.assertEqual(int(np.sum(all_parents == -1)), target_primaries)
        sec_parents = all_parents[all_parents >= 0]
        self.assertGreater(len(sec_parents), 30)
        self.assertTrue(np.all(np.isin(sec_parents, all_pids)))

        # Часть 2: Двухэтапный расчёт [0, T1] + [T1, T2] == [0, T2] с полным набором процессов и релаксацией
        time_split = Float(0.010 * units.s)
        time_end = Float(0.020 * units.s)

        mgr_stage1 = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(
                processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
            ),
            start_time=Float(0.0),
            stop_time=time_split,
            particles_number=500,
            batch_capacity=1,
            buffer_capacity=2000,
            min_energy=Float(1.0 * units.keV),
            seed=2026431,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )
        mgr_stage1.start()
        mgr_stage1.join()
        tel_s1 = _drain_all_telemetry(mgr_stage1)

        stage1_checkpoint: dict[str, Any] = {
            "next_particle_id": int(mgr_stage1.bank.count_array[0]),
            "rng_state": mgr_stage1.rng.bit_generator.state,
            "secondary_activations": mgr_stage1.secondary_source.export_state(),
        }
        s1_prim_state = mgr_stage1.primary_source.export_state()
        if s1_prim_state is not None:
            stage1_checkpoint.update(s1_prim_state)

        mgr_stage2 = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(
                processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
            ),
            start_time=time_split,
            stop_time=time_end,
            particles_number=500,
            batch_capacity=1,
            buffer_capacity=2000,
            min_energy=Float(1.0 * units.keV),
            seed=999,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )
        mgr_stage2.restore_checkpoint(stage1_checkpoint)
        mgr_stage2.start()
        mgr_stage2.join()
        tel_s2 = _drain_all_telemetry(mgr_stage2)

        mgr_full = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(
                processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
            ),
            start_time=Float(0.0),
            stop_time=time_end,
            particles_number=500,
            batch_capacity=1,
            buffer_capacity=2000,
            min_energy=Float(1.0 * units.keV),
            seed=2026431,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )
        mgr_full.start()
        mgr_full.join()
        tel_full = _drain_all_telemetry(mgr_full)

        self.assertEqual(int(mgr_stage2.bank.count_array[0]), int(mgr_full.bank.count_array[0]))
        for key in tel_full["initial_states"]:
            merged_arr = np.concatenate([tel_s1["initial_states"][key], tel_s2["initial_states"][key]])
            np.testing.assert_array_equal(merged_arr, tel_full["initial_states"][key])
        for key in tel_full["interactions"]:
            merged_inter = np.concatenate([tel_s1["interactions"][key], tel_s2["interactions"][key]])
            np.testing.assert_array_equal(merged_inter, tel_full["interactions"][key])
        np.testing.assert_array_equal(
            np.concatenate([tel_s1["dead_particles"], tel_s2["dead_particles"]]),
            tel_full["dead_particles"],
        )


if __name__ == '__main__':
    unittest.main()
