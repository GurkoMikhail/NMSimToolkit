import numpy as np
import time

from core.other.typing_definitions import Float
from core.physics.processes import Process
from core.transport.propagation_managers import PropagationWithInteraction

class DummyMaterial:
    def __init__(self, size):
        self.size = size

class DummyProcess(Process):
    def __init__(self, factor, seed=42):
        self.factor = factor
        self.rng = np.random.default_rng(seed)
        self._name = f"Process_{factor}"

    @property
    def name(self):
        return self._name

    def get_LAC(self, particles, materials):
        return np.full(particles.size, self.factor, dtype=Float)

    def generate_free_path(self, particles, materials):
        lac = self.get_LAC(particles, materials)
        return self.rng.exponential(1/lac)

class DummyParticleArray:
    def __init__(self, size):
        self.size = size

class OldPropagationWithInteraction:
    def __init__(self, processes, rng):
        self.processes = processes
        self.rng = rng

    def get_processes_LAC(self, particles, materials):
        LAC = []
        for process in self.processes:
            LAC.append(process.get_LAC(particles, materials))
        return np.asarray(LAC)

    def get_total_LAC(self, particles, materials):
        total_LAC = np.zeros(particles.size, dtype=Float)
        for process in self.processes:
            total_LAC += process.get_LAC(particles, materials)
        return total_LAC

    def generate_free_path(self, particles, materials):
        free_path = np.full((len(self.processes), particles.size), np.inf, dtype=Float)
        for i, process in enumerate(self.processes):
            free_path[i] = process.generate_free_path(particles, materials)
        return free_path.min(axis=0)

    def choose_process(self, processes_LAC, total_LAC):
        probabilities = processes_LAC / total_LAC
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

def test_propagation_optimization_equivalence():
    N = 1000000
    particles = DummyParticleArray(N)
    materials = DummyMaterial(N)

    processes = [DummyProcess(1.0), DummyProcess(2.0), DummyProcess(3.0)]

    # Old Manager
    rng_old = np.random.default_rng(42)
    old_pm = OldPropagationWithInteraction(processes, rng_old)

    old_proc_lac = old_pm.get_processes_LAC(particles, materials)
    old_tot_lac = old_pm.get_total_LAC(particles, materials)

    # reset seeds for process rngs to ensure same free path generation
    for p in processes: p.rng = np.random.default_rng(42)
    old_free_path = old_pm.generate_free_path(particles, materials)
    old_chosen = old_pm.choose_process(old_proc_lac, old_tot_lac)

    # New Manager
    rng_new = np.random.default_rng(42)
    new_pm = PropagationWithInteraction(processes_list=[], rng=rng_new)
    new_pm.processes = processes

    new_proc_lac = new_pm.get_processes_LAC(particles, materials)
    new_tot_lac = new_pm.get_total_LAC(particles, materials)

    # reset seeds for process rngs to ensure same free path generation
    for p in processes: p.rng = np.random.default_rng(42)
    new_free_path = new_pm.generate_free_path(particles, materials)
    new_chosen = new_pm.choose_process(new_proc_lac, new_tot_lac)

    np.testing.assert_allclose(old_proc_lac, new_proc_lac)
    np.testing.assert_allclose(old_tot_lac, new_tot_lac)
    np.testing.assert_allclose(old_free_path, new_free_path)

    for i in range(len(processes)):
        assert old_chosen[i][0] is processes[i]
        assert new_chosen[i][0] is processes[i]
        # In the new code, `rnd < p1` instead of `rnd <= p1` but since randoms are floats,
        # probability of exact match is negligible, but we should verify exact match here.
        np.testing.assert_array_equal(old_chosen[i][1], new_chosen[i][1])

def benchmark_propagation():
    N = 5000000
    particles = DummyParticleArray(N)
    materials = DummyMaterial(N)
    processes = [DummyProcess(1.0), DummyProcess(2.0), DummyProcess(3.0)]

    rng_old = np.random.default_rng(42)
    old_pm = OldPropagationWithInteraction(processes, rng_old)

    rng_new = np.random.default_rng(42)
    new_pm = PropagationWithInteraction(processes_list=[], rng=rng_new)
    new_pm.processes = processes

    # Warmup
    old_pm.get_processes_LAC(particles, materials)
    new_pm.get_processes_LAC(particles, materials)

    t0 = time.time()
    old_proc_lac = old_pm.get_processes_LAC(particles, materials)
    old_tot_lac = old_pm.get_total_LAC(particles, materials)
    old_pm.choose_process(old_proc_lac, old_tot_lac)
    t1 = time.time()

    t2 = time.time()
    new_proc_lac = new_pm.get_processes_LAC(particles, materials)
    new_tot_lac = new_pm.get_total_LAC(particles, materials)
    new_pm.choose_process(new_proc_lac, new_tot_lac)
    t3 = time.time()

    print(f"Old approach time: {t1-t0:.4f}s")
    print(f"New approach time: {t3-t2:.4f}s")
    print(f"Speedup: {(t1-t0)/(t3-t2):.2f}x")

if __name__ == '__main__':
    test_propagation_optimization_equivalence()
    benchmark_propagation()
    print("Tests passed successfully.")
