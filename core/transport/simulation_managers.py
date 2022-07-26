from signal import SIGINT, signal
from hepunits import*
import numpy as np
from core.transport.propagation_managers import PropagationWithInteraction
import threading as mt
import queue
from time import time

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
        self.daemon = True
        signal(SIGINT, self.sigint_handler)

    def check_valid(self, particles):
        result = particles.energy > self.min_energy
        result *= self.simulation_volume.check_inside(particles.position)
        for filter in self.valid_filters:
            result *= filter(particles)
        return result

    def sigint_handler(self, signal, frame):
        print(f'\"{self.name}\" interrupted; '
            + f'Source timer: {round(self.source.timer/second, 3)} seconds')
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
        self.send_data(propagation_data)

    def run(self):
        """ Реализация работы потока частиц """
        print(f'\"{self.name}\" started; '
            + f'Source timer: {round(self.source.timer/second, 3)} seconds')
        start = time()
        self.particles = self.source.generate_particles(self.particles_number)
        while self.particles.size > 0:
                self.next_step()
                print(f'Source timer: {self.source.timer/s} seconds')
        self.queue.put('stop')
        print(f'{self.name} finished\n'
            + f'\tTime passed: {time() - start} seconds')

