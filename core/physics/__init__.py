"""
Подпакет физических процессов, буферов взаимодействия и компилятора физики.
"""

from core.physics.processes import (
    Process,
    PhotoelectricEffect,
    CoherentScattering,
    ComptonScattering,
    PairProduction,
)
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.interaction_buffers import (
    InteractionBuffer,
    InitialStateBuffer,
    DeadParticlesBuffer,
    SimulationDataBuffer,
    RNGContext,
)

__all__ = [
    'Process',
    'PhotoelectricEffect',
    'CoherentScattering',
    'ComptonScattering',
    'PairProduction',
    'PhysicsCompiler',
    'PhysicsBuffer',
    'InteractionBuffer',
    'InitialStateBuffer',
    'DeadParticlesBuffer',
    'SimulationDataBuffer',
    'RNGContext',
]
