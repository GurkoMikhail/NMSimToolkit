import os
import time
import unittest
import numpy as np
import hepunits as units

# Set env variables before importing to avoid multi-threading overheads in numpy
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from core.geometry.volumes import Volume
from core.scene.nodes import CompositeNode
from core.geometry.geometries import Box
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.parametric_collimators import ParametricParallelCollimator
from settings.database_setting import material_database
from core.source.sources import PointSource
from core.transport.simulation_managers import SimulationManager
from core.transport.propagator import ParticlePropagator
from core.physics.physics_compiler import PhysicsCompiler
from core.data.data_manager import DataManager
from core.data.data_handlers import HistoryAssemblerHandler

class TestFullBenchmark(unittest.TestCase):
    def test_run_benchmark(self):
        # 1. Geometry Setup
        print("Setting up geometry...")
        root_scene = CompositeNode()

        simulation_volume = Volume(
            geometry=Box(120*units.cm, 120*units.cm, 80*units.cm),
            material=material_database['Air, Dry (near sea level)'],
            name='Simulation_volume'
        )

        detector = Volume(
            geometry=Box(54.0*units.cm, 40*units.cm, 0.95*units.cm),
            material=material_database['Sodium Iodide'],
            name='Detector'
        )

        collimator = ParametricParallelCollimator(
            size=(detector.size[0], detector.size[1], 3.5*units.cm),
            hole_diameter=1.5*units.mm,
            septa=0.2*units.mm,
            material=material_database['Pb'],
            name='Collimator'
        )

        spect_head = GammaCamera(
            collimator=collimator,
            detector=detector,
            shielding_thickness=2*units.cm,
            glass_backend_thickness=7.6*units.cm,
            name='Gamma_camera'
        )

        # Position the camera at some radius from the center
        radius = 23.3 * units.cm
        spect_head.rotate(gamma=units.pi/2)
        spect_head.translate(y=radius + spect_head.size[2]/2)

        simulation_volume.add_child(spect_head)

        # 2. Source Setup
        print("Setting up source...")
        activity = 1000 * units.Bq
        energy = 140.5 * units.keV
        source = PointSource(
            activity=activity,
            energy=energy
        )

        root_scene.add_child(simulation_volume)
        root_scene.add_child(source)

        # 3. Simulation Manager Setup
        print("Setting up simulation manager...")
        propagator = ParticlePropagator()
        stop_time = 0.1 * units.second
        particles_number = 10000

        manager = SimulationManager(
            scene=root_scene,
            propagator=propagator,
            stop_time=stop_time,
            particles_number=particles_number,
            buffer_capacity=100000
        )

        # 6. Initialize DataManager
        print("Setting up data manager...")
        handler = HistoryAssemblerHandler(sensitive_volumes=[detector])
        data_manager = DataManager(
            filename="benchmark_soa.hdf5",
            handlers=[handler],
            queue=manager.queue
        )
        data_manager.start()

        # 7. Run Simulation
        print("Starting simulation...")
        start_time = time.perf_counter()
        manager.start()
        manager.join()
        data_manager.join()
        end_time = time.perf_counter()

        print(f"Simulation finished in {end_time - start_time:.4f} seconds.")

if __name__ == '__main__':
    unittest.main()
