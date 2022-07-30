from cProfile import runctx
from datetime import datetime
import logging
from signal import SIGINT, signal
from hepunits import*
import numpy as np
from core.other.utils import datetime_from_seconds
from core.transport.propagation_managers import PropagationWithInteraction
import threading as mt
import queue

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

Queue = queue.Queue
Thread = mt.Thread


class SimulationManager(Thread):
    """ Класс менеджера симуляции """

    def __init__(self, source, simulation_volume, propagation_manager=None, stop_time=1*s, particles_number=10**3, queue=None):
        super().__init__()
        self.source = source
        self.simulation_volume = simulation_volume
        self.propagation_manager = PropagationWithInteraction() if propagation_manager is None else propagation_manager
        self.stop_time = stop_time
        self.particles_number = int(particles_number)
        self.valid_filters = []
        self.min_energy = 1*keV
        self.queue = Queue(maxsize=1) if queue is None else queue
        self.step = 1
        self.profile = False
        self.daemon = True
        signal(SIGINT, self.sigint_handler)

    def check_valid(self, particles):
        result = particles.energy > self.min_energy
        result *= self.simulation_volume.check_inside(particles.position)
        for filter in self.valid_filters:
            result *= filter(particles)
        return result

    def sigint_handler(self, signal, frame):
        _logger.error(f'{self.name} interrupted at {datetime_from_seconds(self.source.timer/second)}')
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
        _logger.warning(f'{self.name} started from {datetime_from_seconds(self.source.timer/second)} to {datetime_from_seconds(self.stop_time/second)}')
        start_timepoint = datetime.now()
        self.particles = self.source.generate_particles(self.particles_number)
        while self.particles.size > 0:
                self.next_step()
                _logger.debug(f'Source timer of {self.name} at {datetime_from_seconds(self.source.timer/second)}')
        self.queue.put('stop')
        stop_timepoint = datetime.now()
        _logger.warning(f'{self.name} finished at {datetime_from_seconds(self.source.timer/second)}')
        _logger.info(f'The simulation of {self.name} took {stop_timepoint - start_timepoint}')

