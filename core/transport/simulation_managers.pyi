import numpy as np
import threading as mt
import queue
from typing import List, Optional, Any, Union, Tuple, Callable
from core.transport.propagation_managers import PropagationWithInteraction
from core.particles.particles import ParticleArray
from core.geometry.volumes import ElementaryVolume
from core.other.typing_definitions import Float
from core.data.interaction_data import InteractionArray

Queue = queue.Queue

class SimulationManager(mt.Thread):
    source: Any
    simulation_volume: ElementaryVolume
    propagation_manager: PropagationWithInteraction
    stop_time: Float
    particles_number: int
    valid_filters: List[Callable[[ParticleArray], np.ndarray]]
    min_energy: Float
    queue: Queue
    particles: ParticleArray
    step: int
    profile: bool

    def __init__(self, source: Any, simulation_volume: ElementaryVolume, propagation_manager: Optional[PropagationWithInteraction] = None, stop_time: Float = ..., particles_number: Union[int, Float] = ..., queue: Optional[Queue] = None) -> None: ...
    def check_valid(self, particles: ParticleArray) -> np.ndarray: ...
    def sigint_handler(self, signal: Any, frame: Any) -> None: ...
    def send_data(self, data: Union[InteractionArray, str]) -> None: ...
    def next_step(self) -> None: ...
    def run(self) -> None: ...
    def run_profile(self) -> None: ...
    def _run(self) -> None: ...
