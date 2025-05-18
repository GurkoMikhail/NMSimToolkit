import numpy as np
from numpy import inf
import settings.database_setting as database_setting
import settings.processes_settings as processes_settings
from core.geometry.woodcoock_volumes import WoodcockVolume
from hepunits import*
from core.numba_backend.fast_index import numba_fast_index, numba_fast_insert


class PropagationWithInteraction:
    """ Класс распространения частиц с взаимодействием """

    def __init__(self, processes_list=None, attenuation_database=None, rng=None):
        processes_list = processes_settings.processes_list if processes_list is None else processes_list
        self.attenuation_database = database_setting.attenuation_database if attenuation_database is None else attenuation_database
        self.rng = np.random.default_rng() if rng is None else rng
        self.processes = [process(self.attenuation_database, rng) for process in processes_list]

    def __call__(self, particles, volume):
        """ Сделать шаг """
        distance, current_volume = volume.cast_path(particles.position, particles.direction)
        materials = current_volume.material
        processes_LAC = self.get_processes_LAC(particles, materials)
        total_LAC = processes_LAC.sum(axis=0)
        free_path = self.rng.exponential(1/total_LAC)
        interacted = (free_path < distance).nonzero()[0]
        distance[interacted] = free_path[interacted]
        particles.move(distance)
        if interacted.size > 0:
            current_volume = current_volume[interacted]
            materials = materials[interacted]    
            interacted_particles = numba_fast_index(particles, interacted, np.empty(interacted.size, dtype=particles.dtype)).view(type(particles))
            processes_LAC = processes_LAC[:, interacted]
            total_LAC = total_LAC[interacted]
            woodcock_volume = current_volume.type_matching(WoodcockVolume)
            if woodcock_volume.any():
                materials[woodcock_volume] = volume.get_material_by_position(interacted_particles.position[woodcock_volume])
                processes_LAC[:, woodcock_volume] = self.get_processes_LAC(interacted_particles[woodcock_volume], materials[woodcock_volume])
            interaction_data = []

            for process, indices in self.choose_process(processes_LAC, total_LAC):
                processing_particles = interacted_particles[indices]
                interaction_data.append(process(processing_particles, materials[indices]))
                interacted_particles[indices] = processing_particles
    
            numba_fast_insert(particles, interacted, interacted_particles)

            return np.concatenate(interaction_data).view(np.recarray)
    
    def get_processes_LAC(self, particles, materials):
        results = []
        for process in self.processes:
            results.append(process.get_LAC(particles, materials))
        return np.stack(results)
    
    def get_total_LAC(self, particles, materials):
        total_LAC = np.zeros(particles.size, dtype=float)
        for process in self.processes:
            total_LAC += process.get_LAC(particles, materials)
        return total_LAC

    def generate_free_path(self, particles, materials):
        free_path = np.full((len(self.processes), particles.size), inf, dtype=float)
        for i, process in enumerate(self.processes):
            free_path[i] = process.generate_free_path(particles, materials)
        return free_path.min(axis=0)

    def choose_process(self, processes_LAC, total_LAC):
        probabilities = processes_LAC/total_LAC
        rnd = self.rng.random(total_LAC.size)
        chosen_process = []
        p0 = 0
        for i, process in enumerate(self.processes):
            p1 = p0 + probabilities[i]
            in_delta = (p0 <= rnd)
            in_delta *= (rnd <= p1)
            indices = in_delta.nonzero()[0]
            p0 = p1
            chosen_process.append((process, indices))
        return chosen_process

