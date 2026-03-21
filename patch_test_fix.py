import re

with open("tests/transport/test_step5_transport.py", "r") as f:
    content = f.read()

# We discovered earlier that ParticleBank is actually instantiated via ParticleBank.allocate(capacity).
# Wait, no, earlier when I changed allocate to __init__, it worked.
# But ParticleBank might not have zero initialization by default in the test unless we make sure it works.
# The `run_dbg.py` succeeded with ParticleBank(10).
# What if it's the `testMain.py` imports messing up or something with `unittest` runners?
# Let's write the test cleanly.

new_test = """import sys
import os
import unittest
import numpy as np
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.transport.transport_kernels import make_transport_kernel
from core.particles.particles_soa import ParticleBank, ParticleState
from core.geometry.navigation_state import NavigationState
from core.physics.physics_buffer import PhysicsBuffer, ElementCSR
from core.physics.interaction_soa import InteractionBuffer, RNGContext
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.other.typing_definitions import Index, Float, Charge, CFuncAddress, ProcessID

class TestStep5TransportLogic(unittest.TestCase):
    def setUp(self):
        self.capacity = 10
        self.bank = ParticleBank(self.capacity)
        self.bank.state.is_active[:] = True
        self.bank.state.energy[:] = 1.0
        self.bank.state.position.x[:] = 0.0
        self.bank.state.position.y[:] = 0.0
        self.bank.state.position.z[:] = 0.0
        self.bank.state.direction.x[:] = 1.0
        self.bank.state.direction.y[:] = 0.0
        self.bank.state.direction.z[:] = 0.0

        self.bank.navigation_state.current_volume[:] = 0
        self.bank.navigation_state.next_volume[:] = -1
        self.bank.navigation_state.boundary_distance[:] = 1.0

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

        self.physics_buffer = PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=majorant_material_map,
            woodcock_function_pointers=woodcock_function_pointers,
            element_csr=element_csr
        )

        rng = np.random.default_rng(42)
        self.rng_ctx = RNGContext.from_numpy_rng(rng)

    def test_transport_kernel_compilation_and_execution(self):
        num_processes = 3
        transport_kernel = make_transport_kernel(num_processes)

        process_ids = np.empty(self.capacity, dtype=Index)
        materials_buffer = np.empty(self.capacity, dtype=Index)
        mapped_process_ids = np.array([0, 1, 2], dtype=Index)

        active_indices = np.arange(self.capacity, dtype=Index)

        try:
            transport_kernel(
                self.bank.state,
                self.bank.navigation_state,
                active_indices,
                self.physics_buffer,
                self.rng_ctx,
                process_ids,
                materials_buffer,
                mapped_process_ids
            )
        except Exception as e:
            self.fail(f"Kernel execution failed with: {e}")

        self.assertTrue(np.all(self.bank.state.position.x > 0.0))

if __name__ == '__main__':
    unittest.main()
"""

with open("tests/transport/test_step5_transport.py", "w") as f:
    f.write(new_test)
