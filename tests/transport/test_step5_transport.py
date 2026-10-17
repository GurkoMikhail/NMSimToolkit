from core.geometry.geometries import Box
from core.geometry.geometry_compiler import GeometryBufferDType
from core.geometry.volumes import Volume
import os
import queue
import sys
import threading
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.materials.materials import Material
from core.other.typing_definitions import CFuncAddress, Charge, Energy, Float, ID, Index, ProcessID, Species
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import ChunkType, InteractionBuffer, RNGContext
from core.physics.physics_buffer import ElementCSR, PhysicsBuffer
from core.physics.processes_kernels import _push_to_interaction_buffer
from core.scene.nodes import CompositeNode
from core.transport.simulation_managers import SimulationManager
from core.transport.transport_buffer import TransportBuffer
from core.transport.transport_kernels import make_transport_kernel


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
        mat_pointers[1]['start_idx'] = 2
        mat_pointers[1]['length'] = 2

        physics_energy_grid = np.array([0.1, 10.0, 0.1, 10.0], dtype=Float)
        physics_lac_table = np.array([
            [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]
        ], dtype=Float)

        material_bank = MaterialBank(mat_info, mat_pointers, physics_energy_grid, physics_lac_table)
        majorant_material_map = np.array([0, 1], dtype=Index)
        woodcock_function_pointers = np.zeros(2, dtype=CFuncAddress)

        element_csr = ElementCSR(
            element_offsets=np.array([0, 1, 3], dtype=Index),
            element_Z=np.array([1, 6, 8], dtype=Charge),
            element_fraction=np.array([1.0, 0.5, 0.5], dtype=Float)
        )

        physics_buffer = PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=majorant_material_map,
            woodcock_function_pointers=woodcock_function_pointers,
            element_csr=element_csr
        )

        rng = np.random.default_rng(42)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        mapped_process_ids = np.array([0, 1, 2], dtype=Index)
        transport_kernel = make_transport_kernel(mapped_process_ids)
        transport_buffer = TransportBuffer.allocate(10)
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
            ibuf, ProcessID(1), Index(0), Index(0), ID(101), Energy(0.05),
            Float(0.1), Float(0.2), Float(1.0), Species(0), Charge(82),
            Float(0.0), Float(0.0), Float(1.0), Float(0.0), Float(0.0), Float(1.0),
        )
        _push_to_interaction_buffer(
            ibuf, ProcessID(1), Index(0), Index(0), ID(202), Energy(0.06),
            Float(0.3), Float(0.4), Float(2.0), Species(0), Charge(82),
            Float(0.0), Float(0.0), Float(2.0), Float(0.0), Float(0.0), Float(1.0),
        )
        # Третья запись не должна затереть нулевой слот (ID=101)
        _push_to_interaction_buffer(
            ibuf, ProcessID(2), Index(0), Index(0), ID(999), Energy(0.99),
            Float(0.5), Float(0.6), Float(3.0), Species(0), Charge(82),
            Float(0.0), Float(0.0), Float(3.0), Float(0.0), Float(0.0), Float(1.0),
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
        # 1: убита по min_energy внутри мира без взаимодействий (energy = 0.01 < min_energy, emission_energy = 0.01)
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


if __name__ == '__main__':
    unittest.main()

