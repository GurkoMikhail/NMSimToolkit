import os

os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['NUMEXPR_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from hepunits import*


def modeling(filename, angle, radius, gamma_cameras, delta_angle, time_interval, seed, lock):
    import logging
    from pathlib import Path
    
    log_path = Path(f'logs/{filename}/{round(angle/degree, 1)} deg.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s: %(levelname)s] %(message)s'))
    
    logging.basicConfig(
        handlers=[file_handler, ]
    )
    
    from core.materials.materials import MaterialArray
    from core.geometry.gamma_cameras import GammaCamera
    from core.geometry.geometries import Box
    from core.geometry.parametric_collimators import ParametricParallelCollimator
    from core.geometry.volumes import TransformableVolume, VolumeWithChilds
    from core.transport.simulation_managers import SimulationManager
    from core.data.data_manager import SimulationDataManager
    from core.geometry.voxel_volumes import WoodcockVoxelVolume
    from core.transport.propagation_managers import PropagationWithInteraction
    from core.source.sources import Тс99m_MIBI
    from settings.database_setting import material_database, attenuation_database

    rng = np.random.default_rng(seed)

    start_time, stop_time = time_interval

    simulation_volume = VolumeWithChilds(
        geometry=Box(120*cm, 120*cm, 80*cm),
        material=material_database['Air, Dry (near sea level)'],
        name='Simulation_volume'
    )

    material_ID_distribution = np.load('phantoms/material_map.npy')
    material_distribution = MaterialArray(material_ID_distribution.shape)
    material_distribution[material_ID_distribution == 0] = material_database['Air, Dry (near sea level)']
    material_distribution[material_ID_distribution == 1] = material_database['Lung']
    material_distribution[material_ID_distribution == 2] = material_database['Adipose Tissue (ICRU-44)']
    material_distribution[material_ID_distribution == 3] = material_database['Tissue, Soft (ICRU-44)']
    material_distribution[material_ID_distribution == 4] = material_database['Bone, Cortical (ICRU-44)']

    phantom = WoodcockVoxelVolume(
        voxel_size=4*mm,
        material_distribution=material_distribution,
        name='Phantom'
    )
    phantom.set_parent(simulation_volume)
    
    detector_list = []

    for i in range(gamma_cameras):
        detector = TransformableVolume(
            geometry=Box(54.*cm, 40*cm, 0.95*cm),
            material=material_database['Sodium Iodide'],
            name=f'Detector at {round((angle + delta_angle*i)/degree, 1)} deg'
        )

        collimator = ParametricParallelCollimator(
            size=(detector.size[0], detector.size[1], 3.5*cm),
            hole_diameter=1.5*mm,
            septa=0.2*mm,
            material=material_database['Pb'],
            name=f'Collimator at {round((angle + delta_angle*i)/degree, 1)} deg'
        )

        spect_head = GammaCamera(
            collimator=collimator,
            detector=detector,
            shielding_thickness=2*cm,
            glass_backend_thickness=7.6*cm,
            name=f'Gamma_camera at {round((angle + delta_angle*i)/degree, 1)} deg'
        )
        spect_head.rotate(gamma=pi/2)
        spect_head.translate(y=radius + spect_head.size[2]/2)
        spect_head.rotate(alpha=angle + delta_angle*i)
    
        simulation_volume.add_child(spect_head)
        detector_list.append(detector)

    distribution = np.load(f'phantoms/source_function.npy')
    distribution[distribution==40] = 10
    distribution[distribution==30] = 20
    distribution[distribution==70] = 40
    distribution[distribution==80] = 40
    distribution[distribution==89] = 50
    distribution[distribution==140] = 40
    distribution[distribution==1200] = 1000
    distribution[distribution==700] = 550
    distribution[distribution==10000] = 7000
    source = Тс99m_MIBI(
        distribution=distribution,
        activity=300*MBq,
        voxel_size=4*mm
    )
    source.rng = rng
    source.set_state(start_time)

    propagation_manager = PropagationWithInteraction(
        attenuation_database=attenuation_database,
        rng=rng
    )

    simulation_manager = SimulationManager(
        source=source,
        simulation_volume=simulation_volume,
        propagation_manager=propagation_manager,
        particles_number=10**6,
        stop_time=stop_time
    )
    simulation_manager.name = f'{round(angle/degree, 1)} deg'
    simulation_manager.start()

    simulation_data_manager = SimulationDataManager(
        filename=f'{filename}/{simulation_manager.name}.hdf',
        sensitive_volumes=detector_list,
        lock=lock,
        iteraction_buffer_size=int(10**4)
    )
    
    while True:
        data = simulation_manager.queue.get()
        if isinstance(data, np.ndarray):
            simulation_data_manager.add_interaction_data(data)
        elif data == 'stop':
            simulation_manager.join()
            simulation_data_manager.save_interaction_data()
            break
        else:
            raise ValueError("Неверное значение Propagation Manager")


if __name__ == '__main__':
    
    filename = 'heart_32'
    
    views = 32
    gamma_cameras = 2
    radius = 233*mm
    
    angle_start = -pi/4 - pi/2
    angle_stop = pi/4
    endpoint = True
    
    time_start = 0.*second
    time_stop = 15*second
    steps = 5
    
    pool_size = 32


    from multiprocessing import Pool, Manager
    from numpy.random import SeedSequence


    angles = np.linspace(angle_start, angle_stop, views, endpoint=endpoint)[:views//gamma_cameras]
    angle_step = angles[1] - angles[0]
    delta_angle = pi/2 + (pi/2)%angle_step
    
    time_intervals = np.linspace(time_start, time_stop, steps + 1)
    time_intervals = np.column_stack([time_intervals[:-1], time_intervals[1:]])
    
    seed_sequence = SeedSequence()
    
    manager = Manager()
    locks = [manager.Lock() for _ in range(angles.size)]
    
    with Pool(pool_size) as pool:
        for time_interval in time_intervals:
            for angle, lock in zip(angles, locks):
                seed = seed_sequence.spawn(1)[0]
                pool.apply_async(modeling, (filename, angle, radius, gamma_cameras, delta_angle, time_interval, seed, lock))
        pool.close()
        pool.join()

