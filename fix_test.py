with open("tests/transport/test_transport.py", "w") as f:
    f.write("""import sys
import os
import unittest
import numpy as np
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.transport.transport_kernels import make_transport_kernel
from core.particles.particles_soa import ParticleBank
from core.geometry.navigation_state import NavigationState
from core.physics.physics_buffer import PhysicsBuffer, ElementCSR
from core.physics.interaction_soa import RNGContext
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.other.typing_definitions import Index, Float, Charge, CFuncAddress
from core.transport.transport_buffer import TransportBuffer

@ctypes.CFUNCTYPE(np.ctypeslib.as_ctypes_type(Index), np.ctypeslib.as_ctypes_type(Float), np.ctypeslib.as_ctypes_type(Float), np.ctypeslib.as_ctypes_type(Float))
def mock_woodcock_cfunc(x, y, z):
    return 1 # return material 1

class TestStep5TransportLogic(unittest.TestCase):
    def setUp(self):
        self.capacity = 5
        self.bank = ParticleBank(self.capacity)
        self.bank.state.is_active[:] = True
        self.bank.state.energy[:] = 1.0 # 1 MeV
        self.bank.state.position.x[:] = 0.0
        self.bank.state.position.y[:] = 0.0
        self.bank.state.position.z[:] = 0.0
        self.bank.state.direction.x[:] = 1.0
        self.bank.state.direction.y[:] = 0.0
        self.bank.state.direction.z[:] = 0.0

        self.bank.navigation_state.current_volume[:] = 0
        self.bank.navigation_state.next_volume[:] = -1
        self.bank.navigation_state.boundary_distance[:] = 10.0

        mat_info = np.zeros(2, dtype=MaterialInfoDType)
        mat_pointers = np.zeros(2, dtype=MaterialPointerDType)
        mat_pointers[0]['length'] = 2
        mat_pointers[1]['start_idx'] = 2
        mat_pointers[1]['length'] = 2

        physics_energy_grid = np.array([0.1, 10.0, 0.1, 10.0], dtype=Float)
        physics_lac_table = np.array([
            [0.5, 0.5], [0.5, 0.5], # mat 0: total 1.0
            [0.1, 0.4], [0.1, 0.4]  # mat 1: total 0.5
        ], dtype=Float)

        material_bank = MaterialBank(mat_info, mat_pointers, physics_energy_grid, physics_lac_table)
        majorant_material_map = np.array([0, 1], dtype=Index)
        self.woodcock_function_pointers = np.zeros(2, dtype=CFuncAddress)

        element_csr = ElementCSR(
            element_offsets=np.array([0, 1, 3], dtype=Index),
            element_Z=np.array([1, 6, 8], dtype=Charge),
            element_fraction=np.array([1.0, 0.5, 0.5], dtype=Float)
        )

        self.physics_buffer = PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=majorant_material_map,
            woodcock_function_pointers=self.woodcock_function_pointers,
            element_csr=element_csr
        )

        rng = np.random.default_rng(42)
        self.rng_ctx = RNGContext.from_numpy_rng(rng)

        self.mapped_process_ids = np.array([0, 1], dtype=Index)
        self.transport_kernel = make_transport_kernel(self.mapped_process_ids)
        self.transport_buffer = TransportBuffer.allocate(self.capacity)
        self.active_indices = np.array([0], dtype=Index)

    def test_boundary_crossing(self):
        self.bank.navigation_state.boundary_distance[:] = 0.001
        self.physics_buffer.woodcock_function_pointers[0] = 0

        self.transport_kernel(
            self.bank.state,
            self.bank.navigation_state,
            self.active_indices,
            self.physics_buffer,
            self.transport_buffer,
            self.rng_ctx
        )

        np.testing.assert_allclose(self.bank.state.position.x[0], 0.001001, rtol=1e-5)
        self.assertEqual(self.transport_buffer.process_ids[0], -1)
        self.assertEqual(self.bank.navigation_state.boundary_distance[0], 0.0)

    def test_real_interaction(self):
        self.bank.navigation_state.boundary_distance[:] = 1000.0
        self.physics_buffer.woodcock_function_pointers[0] = 0

        self.transport_kernel(
            self.bank.state,
            self.bank.navigation_state,
            self.active_indices,
            self.physics_buffer,
            self.transport_buffer,
            self.rng_ctx
        )

        proc_id = self.transport_buffer.process_ids[0]
        self.assertIn(proc_id, [0, 1])
        self.assertEqual(self.transport_buffer.material_ids[0], 0)
        self.assertTrue(self.bank.state.position.x[0] > 0.0)
        self.assertTrue(self.bank.navigation_state.boundary_distance[0] < 1000.0)

    def test_delta_scattering_and_dirty_state(self):
        self.bank.navigation_state.boundary_distance[:] = 1000.0
        # Set woodcock function
        self.physics_buffer.woodcock_function_pointers[0] = ctypes.cast(mock_woodcock_cfunc, ctypes.c_void_p).value

        self.transport_kernel(
            self.bank.state,
            self.bank.navigation_state,
            self.active_indices,
            self.physics_buffer,
            self.transport_buffer,
            self.rng_ctx
        )

        proc_id = self.transport_buffer.process_ids[0]
        self.assertIn(proc_id, [0, 1])
        self.assertEqual(self.transport_buffer.material_ids[0], 1)
        self.assertTrue(self.bank.state.position.x[0] > 0.0)

if __name__ == '__main__':
    unittest.main()
""")
