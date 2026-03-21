import sys
import os
import ctypes
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

import faulthandler
faulthandler.enable()

from core.particles.particles_soa import ParticleBank
from core.physics.physics_buffer import PhysicsBuffer, ElementCSR
from core.physics.interaction_soa import InteractionBuffer, RNGContext
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.other.typing_definitions import Index, Float, Charge, CFuncAddress
from core.transport.transport_buffer import TransportBuffer

def get_rng_ctx():
    rng = np.random.default_rng(seed=42)
    return RNGContext.from_numpy_rng(rng)

def get_particle_bank():
    capacity = 5
    bank = ParticleBank(capacity)
    bank.state.is_active[:] = True
    bank.state.energy[:] = 1.0 # 1 MeV
    bank.state.position.x[:] = 0.0
    bank.state.position.y[:] = 0.0
    bank.state.position.z[:] = 0.0
    bank.state.direction.x[:] = 1.0
    bank.state.direction.y[:] = 0.0
    bank.state.direction.z[:] = 0.0

    bank.navigation_state.current_volume[:] = 0
    bank.navigation_state.next_volume[:] = -1
    bank.navigation_state.boundary_distance[:] = 10.0
    return bank

@ctypes.CFUNCTYPE(np.ctypeslib.as_ctypes_type(Index), np.ctypeslib.as_ctypes_type(Float), np.ctypeslib.as_ctypes_type(Float), np.ctypeslib.as_ctypes_type(Float))
def mock_woodcock_cfunc(x, y, z):
    return 1 # return material 1

def get_physics_buffer():
    mat_info = np.zeros(2, dtype=MaterialInfoDType)
    mat_pointers = np.zeros(2, dtype=MaterialPointerDType)
    mat_pointers[0]['length'] = 2
    mat_pointers[1]['start_idx'] = 2
    mat_pointers[1]['length'] = 2

    physics_energy_grid = np.array([0.1, 10.0, 0.1, 10.0], dtype=Float)
    physics_lac_table = np.array([
        [0.5, 0.5], [0.5, 0.5], # mat 0
        [0.1, 0.4], [0.1, 0.4]  # mat 1
    ], dtype=Float)

    material_bank = MaterialBank(mat_info, mat_pointers, physics_energy_grid, physics_lac_table)
    majorant_material_map = np.array([0, 1], dtype=Index)
    woodcock_function_pointers = np.zeros(2, dtype=CFuncAddress)
    woodcock_function_pointers[0] = ctypes.cast(mock_woodcock_cfunc, ctypes.c_void_p).value

    element_csr = ElementCSR(
        element_offsets=np.array([0, 1, 3], dtype=Index),
        element_Z=np.array([1, 6, 8], dtype=Charge),
        element_fraction=np.array([1.0, 0.5, 0.5], dtype=Float)
    )

    return PhysicsBuffer(
        material_bank=material_bank,
        majorant_material_map=majorant_material_map,
        woodcock_function_pointers=woodcock_function_pointers,
        element_csr=element_csr
    )

def get_transport_buffer():
    return TransportBuffer.allocate(5)

from tests.transport.test_transport_kernel import test_boundary_crossing, test_real_interaction, test_delta_scattering_and_dirty_state

print("Running test_boundary_crossing...")
test_boundary_crossing(get_rng_ctx(), get_particle_bank(), get_physics_buffer(), get_transport_buffer())

print("Running test_real_interaction...")
test_real_interaction(get_rng_ctx(), get_particle_bank(), get_physics_buffer(), get_transport_buffer())

print("Running test_delta_scattering_and_dirty_state...")
test_delta_scattering_and_dirty_state(get_rng_ctx(), get_particle_bank(), get_physics_buffer(), get_transport_buffer())

print("All tests passed.")
