import sys
import os
import unittest
import numpy as np
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.transport.transport_kernels import make_transport_kernel
from core.transport.transport_buffer import TransportBuffer
from core.particles.particles_soa import ParticleBank
from core.physics.physics_buffer import PhysicsBuffer, ElementCSR
from core.physics.interaction_soa import RNGContext
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.other.typing_definitions import Index, Float, Charge, CFuncAddress

class TestStep5TransportLogic(unittest.TestCase):
    def test_transport_kernel_execution(self):
        bank = ParticleBank(10)
        bank.state.is_active[:] = True
        bank.state.energy[:] = 1.0
        bank.state.position.x[:] = 0.0
        bank.state.position.y[:] = 0.0
        bank.state.position.z[:] = 0.0
        bank.state.direction.x[:] = 1.0
        bank.state.direction.y[:] = 0.0
        bank.state.direction.z[:] = 0.0

        bank.navigation_state.current_volume[:] = 0
        bank.navigation_state.next_volume[:] = -1
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
            rng_ctx
        )

        self.assertTrue(np.all(bank.state.position.x > 0.0))

if __name__ == '__main__':
    unittest.main()
