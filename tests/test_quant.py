import numpy as np
import hepunits as units
import time
from core.particles.particles import ParticleArray
from core.materials.materials import Material
from core.physics.processes import Process
from core.other.typing_definitions import Float
from typing import Union, Any
from numpy.typing import NDArray

class DummyProcess(Process):
    def __init__(self, name="Dummy"):
        self._name = name
        # Override parent init to avoid full process loading
        pass
    @property
    def name(self): return self._name
    def get_LAC(self, particles, materials):
        return np.full(particles.size, 0.1, dtype=Float)
    def generate_free_path(self, particles, materials):
        return np.full(particles.size, 5.0, dtype=Float)
    def __call__(self, particles, materials):
        pass

# Original unoptimized methods
def original_get_processes_LAC(processes, particles: ParticleArray, materials: Union[Any, Any]) -> NDArray[Float]:
    LAC = []
    for process in processes:
        LAC.append(process.get_LAC(particles, materials))
    return np.asarray(LAC)

def original_generate_free_path(processes, particles: ParticleArray, materials: Union[Any, Any]) -> NDArray[Float]:
    free_path = np.full((len(processes), particles.size), np.inf, dtype=Float)
    for i, process in enumerate(processes):
        free_path[i] = process.generate_free_path(particles, materials)
    return free_path.min(axis=0)

# Optimized methods mapping directly to current codebase
def optimized_get_processes_LAC(processes, particles: ParticleArray, materials: Union[Any, Any]) -> NDArray[Float]:
    num_processes = len(processes)
    LAC = np.empty((num_processes, particles.size), dtype=Float)
    for i, process in enumerate(processes):
        LAC[i] = process.get_LAC(particles, materials)
    return LAC

def optimized_generate_free_path(processes, particles: ParticleArray, materials: Union[Any, Any]) -> NDArray[Float]:
    free_path = np.full(particles.size, np.inf, dtype=Float)
    for process in processes:
        np.minimum(free_path, process.generate_free_path(particles, materials), out=free_path)
    return free_path


def benchmark():
    print("Testing optimizations for Quant...")
    N = 10_000_000 # 10 million particles
    iterations = 5
    processes = [DummyProcess(name="P1"), DummyProcess(name="P2"), DummyProcess(name="P3")]

    pos = np.zeros((N, 3))
    dir = np.zeros((N, 3)); dir[:,0] = 1.0
    energy = np.full(N, 140*units.keV)
    particles = ParticleArray.create(np.zeros(N, dtype=np.uint8), pos, dir, energy, np.zeros(N))
    materials = Material()

    # 1. Correctness tests
    res_orig_lac = original_get_processes_LAC(processes, particles, materials)
    res_opt_lac = optimized_get_processes_LAC(processes, particles, materials)
    np.testing.assert_array_equal(res_orig_lac, res_opt_lac)
    print("Correctness check passed for get_processes_LAC.")

    res_orig_path = original_generate_free_path(processes, particles, materials)
    res_opt_path = optimized_generate_free_path(processes, particles, materials)
    np.testing.assert_array_equal(res_orig_path, res_opt_path)
    print("Correctness check passed for generate_free_path.\n")

    # 2. Performance tests
    # Benchmark get_processes_LAC
    t0 = time.time()
    for _ in range(iterations):
        original_get_processes_LAC(processes, particles, materials)
    t1 = time.time()
    print(f"Original get_processes_LAC: {t1-t0:.4f}s")

    t0 = time.time()
    for _ in range(iterations):
        optimized_get_processes_LAC(processes, particles, materials)
    t1 = time.time()
    print(f"Optimized get_processes_LAC: {t1-t0:.4f}s")

    # Benchmark generate_free_path
    t0 = time.time()
    for _ in range(iterations):
        original_generate_free_path(processes, particles, materials)
    t1 = time.time()
    print(f"Original generate_free_path: {t1-t0:.4f}s")

    t0 = time.time()
    for _ in range(iterations):
        optimized_generate_free_path(processes, particles, materials)
    t1 = time.time()
    print(f"Optimized generate_free_path: {t1-t0:.4f}s")

if __name__ == '__main__':
    benchmark()
