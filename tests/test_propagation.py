import numpy as np
import hepunits as units
from core.transport.propagation_managers import PropagationWithInteraction
from core.particles.particles import ParticleArray
from core.materials.materials import Material
from core.physics.processes import Process
from core.geometry.volumes import ElementaryVolume
from core.other.typing_definitions import Float

class DummyProcess(Process):
    def __init__(self, attenuation_database=None, rng=None, name="Dummy"):
        self._name = name
        super().__init__(attenuation_database, rng)

    @property
    def name(self): return self._name

    def get_LAC(self, particles, materials):
        return np.full(particles.size, 0.1, dtype=Float)

    def __call__(self, particles, materials):
        from core.data.interaction_data import InteractionArray
        ia = InteractionArray(particles.size)
        ia.process_name = self.name
        return ia

def test_get_processes_LAC():
    prop = PropagationWithInteraction(processes_list=[])
    prop.processes = [DummyProcess(name="Proc1", attenuation_database={}), DummyProcess(name="Proc2", attenuation_database={})]

    N = 100
    pos = np.zeros((N, 3))
    dir = np.zeros((N, 3)); dir[:,0] = 1.0
    energy = np.full(N, 140*units.keV)
    particles = ParticleArray.create(np.zeros(N, dtype=np.uint8), pos, dir, energy, np.zeros(N))
    materials = Material()

    # Call optimized
    result = prop.get_processes_LAC(particles, materials)
    expected = np.full((2, N), 0.1, dtype=Float)

    np.testing.assert_array_equal(result, expected)
    print("test_get_processes_LAC passed")

def test_get_total_LAC():
    prop = PropagationWithInteraction(processes_list=[])
    prop.processes = [DummyProcess(name="Proc1", attenuation_database={}), DummyProcess(name="Proc2", attenuation_database={})]

    N = 100
    pos = np.zeros((N, 3))
    dir = np.zeros((N, 3)); dir[:,0] = 1.0
    energy = np.full(N, 140*units.keV)
    particles = ParticleArray.create(np.zeros(N, dtype=np.uint8), pos, dir, energy, np.zeros(N))
    materials = Material()

    expected = np.full(N, 0.2, dtype=Float)
    result = prop.get_total_LAC(particles, materials)

    np.testing.assert_array_almost_equal(result, expected)
    print("test_get_total_LAC passed")

if __name__ == "__main__":
    test_get_processes_LAC()
    test_get_total_LAC()
