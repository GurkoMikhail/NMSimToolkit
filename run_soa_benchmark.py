import os
import time

# Set env variables before importing to avoid multi-threading overheads in numpy
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import hepunits as units

from core.geometry.volumes import VolumeWithChilds, TransformableVolume
from core.geometry.geometries import Box
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.materials.materials import MaterialArray
from settings.database_setting import material_database
from core.source.sources_soa import PointSourceSoA
from core.transport.simulation_managers_soa import SimulationManagerSOA
from core.transport.propagator_soa import ParticlePropagator
from core.physics.physics_compiler import PhysicsCompiler
from core.data.data_manager_soa import DataManagerSoA

def run_benchmark():
    # 1. Geometry Setup
    print("Setting up geometry...")
    simulation_volume = VolumeWithChilds(
        geometry=Box(120*units.cm, 120*units.cm, 80*units.cm),
        material=material_database['Air, Dry (near sea level)'],
        name='Simulation_volume'
    )

    detector = TransformableVolume(
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
    source = PointSourceSoA(
        activity=activity,
        energy=energy
    )

    # 3. Compile Geometry
    print("Compiling geometry...")
    flat_scene = simulation_volume.flattened_scene
    geometry_buffer = simulation_volume.geometry_buffer

    # 4. Compile Physics
    print("Compiling physics...")
    propagator = ParticlePropagator()
    physics_compiler = PhysicsCompiler()
    physics_buffer = physics_compiler.compile_scene(simulation_volume, propagator.processes)

    # 5. Simulation Manager Setup
    print("Setting up simulation manager...")
    stop_time = 0.1 * units.second
    particles_number = 10000

    manager = SimulationManagerSOA(
        source=source,
        simulation_volume=simulation_volume,
        geometry_buffer=geometry_buffer,
        physics_buffer=physics_buffer,
        propagator=propagator,
        stop_time=stop_time,
        particles_number=particles_number,
        buffer_capacity=100000
    )

    # 6. Initialize DataManagerSoA
    print("Setting up data manager...")
    data_manager = DataManagerSoA(
        filename="benchmark_soa.hdf5",
        sensitive_volumes=[detector],
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
    run_benchmark()
