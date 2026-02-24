import logging
import queue
import threading as mt
from cProfile import runctx
from datetime import datetime
from signal import SIGINT, signal
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import hepunits as units
from numpy.typing import NDArray

from core.geometry.volumes import ElementaryVolume
from core.other.typing_definitions import Float
from core.other.utils import datetime_from_seconds
from core.particles.particles import ParticleArray
from core.transport.propagation_managers import PropagationWithInteraction

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

Queue = queue.Queue
Thread = mt.Thread


class SimulationManager(Thread):
    """ Класс менеджера симуляции """
    source: Any
    simulation_volume: ElementaryVolume
    propagation_manager: PropagationWithInteraction
    stop_time: Float
    particles_number: int
    valid_filters: List[Callable[[ParticleArray], NDArray[np.bool_]]] # type: ignore
    min_energy: Float
    queue: Queue
    particles: ParticleArray

    def __init__(self, source: Any, simulation_volume: ElementaryVolume, propagation_manager: Optional[PropagationWithInteraction] = None, stop_time: Float = 1*units.s, particles_number: Union[int, Float] = 10**3, queue: Optional[queue.Queue] = None) -> None:
        super().__init__()
        self.source = source
        self.simulation_volume = simulation_volume
        self.propagation_manager = PropagationWithInteraction() if propagation_manager is None else propagation_manager
        self.stop_time = stop_time
        self.particles_number = int(particles_number)
        self.valid_filters = []
        self.min_energy = 1*units.keV
        self.queue = Queue(maxsize=1) if queue is None else queue
        self.step = 1
        self.profile = False
        self.daemon = True
        signal(SIGINT, self.sigint_handler)

    def check_valid(self, particles: ParticleArray) -> NDArray[np.bool_]: # type: ignore
        result = particles.energy > self.min_energy
        result *= self.simulation_volume.check_inside(particles.position)
        for filter in self.valid_filters:
            result *= filter(particles)
        return result

    def sigint_handler(self, signal, frame):
        _logger.error(f'{self.name} interrupted at {datetime_from_seconds(self.source.timer/units.second)}')
        self.stop_time = 0

    def send_data(self, data):
        self.queue.put(data)

    def next_step(self):
        propagation_data = self.propagation_manager(self.particles, self.simulation_volume)
        invalid_particles = ~self.check_valid(self.particles)
        if self.source.timer <= self.stop_time:
            newParticles = self.source.generate_particles(np.count_nonzero(invalid_particles))
            self.particles[invalid_particles] = newParticles
        else:
            self.particles = self.particles[~invalid_particles]
        self.step += 1
        if propagation_data is not None:
            _logger.debug(f'{self.name} generated {propagation_data.size} events')
            self.send_data(propagation_data)

    def run(self):
        if self.profile:
            self.run_profile()
        else:
            self._run()

    def run_profile(self):
        runctx('self._run()', globals(), locals(), f'stats/{self.name}.txt')

    def _run(self):
        """ Реализация работы потока частиц """
        _logger.warning(f'{self.name} started from {datetime_from_seconds(self.source.timer/units.second)} to {datetime_from_seconds(self.stop_time/units.second)}')
        start_timepoint = datetime.now()
        self.particles = self.source.generate_particles(self.particles_number)
        while self.particles.size > 0:
                self.next_step()
                _logger.debug(f'Source timer of {self.name} at {datetime_from_seconds(self.source.timer/units.second)}')
        self.queue.put('stop')
        stop_timepoint = datetime.now()
        _logger.warning(f'{self.name} finished at {datetime_from_seconds(self.source.timer/units.second)}')
        _logger.info(f'The simulation of {self.name} took {stop_timepoint - start_timepoint}')
