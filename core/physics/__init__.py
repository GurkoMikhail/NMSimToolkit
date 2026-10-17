"""
Подпакет физических процессов, буферов взаимодействия и компилятора физики.
"""

from core.physics.processes import (
    Process,
    PhotoelectricEffect,
    CoherentScattering,
    ComptonScattering,
    PairProduction,
    get_process_mapping,
    get_species_mapping,
)
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_buffer import ElementCSR, PhysicsBuffer
from core.physics.interaction_buffers import (
    TELEMETRY_ORDER,
    ChunkType,
    TelemetryBuffer,
    TelemetryPayload,
    InteractionBuffer,
    InitialStateBuffer,
    DeadParticlesBuffer,
    EscapedParticlesBuffer,
    SimulationDataBuffer,
    RNGContext,
)

__all__ = [
    'Process',
    'PhotoelectricEffect',
    'CoherentScattering',
    'ComptonScattering',
    'PairProduction',
    'get_process_mapping',
    'get_species_mapping',
    'PhysicsCompiler',
    'ElementCSR',
    'PhysicsBuffer',
    'TELEMETRY_ORDER',
    'ChunkType',
    'TelemetryBuffer',
    'TelemetryPayload',
    'InteractionBuffer',
    'InitialStateBuffer',
    'DeadParticlesBuffer',
    'EscapedParticlesBuffer',
    'SimulationDataBuffer',
    'RNGContext',
]


