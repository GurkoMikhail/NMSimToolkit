import pytest
import numpy as np
import ctypes
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.particles.particles_soa import ParticleBank
from core.physics.physics_buffer import PhysicsBuffer, ElementCSR
from core.physics.interaction_soa import InteractionBuffer, RNGContext
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.other.typing_definitions import Index, Float, Charge, CFuncAddress, CMaterialFunc
from core.transport.transport_buffer import TransportBuffer

@pytest.fixture
def mock_rng_ctx():
    # Predictable RNG sequence
    rng = np.random.default_rng(seed=42)
    return RNGContext.from_numpy_rng(rng)

@pytest.fixture
def mock_particle_bank():
    capacity = 5
    bank = ParticleBank.allocate(capacity)
    bank.state.is_active[:] = True
    bank.state.energy[:] = 1.0 # 1 MeV
    bank.state.position.x[:] = 0.0
    bank.state.position.y[:] = 0.0
    bank.state.position.z[:] = 0.0
    bank.state.direction.x[:] = 1.0
    bank.state.direction.y[:] = 0.0
    bank.state.direction.z[:] = 0.0

    bank.navigation_state.current_volume[:] = 0
    bank.navigation_state.boundary_distance[:] = 10.0

    return bank

# Define a real cfunc for woodcock tracking to return mat_id=1
@ctypes.CFUNCTYPE(np.ctypeslib.as_ctypes_type(Index), np.ctypeslib.as_ctypes_type(Float), np.ctypeslib.as_ctypes_type(Float), np.ctypeslib.as_ctypes_type(Float))
def mock_woodcock_cfunc(x, y, z):
    return 1 # return material 1

@pytest.fixture
def mock_physics_buffer():
    # 2 materials: 0 (Majorant/Water), 1 (Bone)
    mat_info = np.zeros(2, dtype=MaterialInfoDType)
    mat_info[0]['density'] = 1.0
    mat_info[1]['density'] = 2.0

    mat_pointers = np.zeros(2, dtype=MaterialPointerDType)
    mat_pointers[0]['start_idx'] = 0
    mat_pointers[0]['length'] = 2
    mat_pointers[1]['start_idx'] = 2
    mat_pointers[1]['length'] = 2

    # 2 energy grid points for interpolation (e.g. 0.1 and 10.0 MeV)
    physics_energy_grid = np.array([0.1, 10.0, 0.1, 10.0], dtype=Float)

    # 2 processes: Process 0 (Photo), Process 1 (Compton)
    # material 0 (Majorant) LACs: [0.5, 0.5] -> total 1.0
    # material 1 (Bone) LACs: [0.1, 0.4] -> total 0.5
    physics_lac_table = np.array([
        [0.5, 0.5], [0.5, 0.5], # mat 0
        [0.1, 0.4], [0.1, 0.4]  # mat 1
    ], dtype=Float)

    material_bank = MaterialBank(mat_info, mat_pointers, physics_energy_grid, physics_lac_table)

    majorant_material_map = np.array([0, 1], dtype=Index)

    woodcock_function_pointers = np.zeros(2, dtype=CFuncAddress)
    # Set cfunc for volume 0 to point to material 1
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

@pytest.fixture
def mock_interaction_buffer():
    return InteractionBuffer.allocate(100)

@pytest.fixture
def mock_transport_buffer():
    return TransportBuffer.allocate(5)
