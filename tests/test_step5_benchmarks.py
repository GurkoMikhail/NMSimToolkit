import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestStep5Transport(unittest.TestCase):
    def test_imports(self):
        try:
            from core.transport.propagator_soa import ParticlePropagator
            from core.transport.simulation_managers_soa import SimulationManagerSOA
            from core.transport.transport_kernels import make_transport_kernel
            from core.physics.processes import PhotoelectricEffect, ComptonScattering, CoherentScattering
        except ImportError as e:
            self.fail(f"Import failed: {e}")

if __name__ == '__main__':
    unittest.main()
