import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from core.geometry.volumes import Volume, TransformableVolume

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class DataManagerSoA(threading.Thread):
    """
    Consumer Thread for saving InteractionBuffer chunks from SoA engine
    to HDF5 file in legacy SimulationDataManager format.
    Computes local coordinates in background to not block physics thread.
    """

    PROCESS_MAP = {
        0: b'PhotoelectricEffect',
        1: b'ComptonScattering',
        2: b'CoherentScattering',
        3: b'PairProduction'
    }

    def __init__(self, filename: str, sensitive_volumes: List[Volume], queue: Any = None, lock: Optional[Any] = None) -> None:
        super().__init__()
        self.filename = Path(f'output data/{filename}')
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.sensitive_volumes = sensitive_volumes
        self.queue = queue
        self.lock = lock
        self.daemon = True

    def run(self):
        """
        Consumes chunks from the queue until 'stop' signal.
        """
        if self.queue is None:
            return

        while True:
            chunk = self.queue.get()
            if isinstance(chunk, str) and chunk == 'stop':
                break
            elif isinstance(chunk, dict):
                self.append_data(chunk)

    def _write_with_retry(self, write_func: Any) -> None:
        """
        Executes an HDF5 write function with retry logic and optional mutex locking.
        """
        import time
        retries = 100

        def do_write():
            for i in range(retries):
                try:
                    with h5py.File(self.filename, 'a') as f:
                        write_func(f)
                    return
                except (OSError, BlockingIOError):
                    if i == retries - 1:
                        raise
                    time.sleep(0.1)

        if self.lock is not None:
            with self.lock:
                do_write()
        else:
            do_write()

    def append_data(self, chunk: Dict[str, Any]) -> None:
        """
        Routes the chunk to the appropriate write method based on type.
        """
        chunk_type = chunk.get('type')
        if chunk_type == 'initial_states':
            self._write_initial_states(chunk['data'])
        elif chunk_type == 'interactions':
            self._write_interactions(chunk['data'])

    def _write_initial_states(self, initial_states: Dict[str, np.ndarray]) -> None:
        """
        Writes initial states to the /initial_states group in HDF5.
        """
        def write_func(f: h5py.File):
            if 'initial_states' not in f:
                group = f.create_group('initial_states')
            else:
                group = f['initial_states']

            for field, array in initial_states.items():
                if field not in group:
                    maxshape = list(array.shape)
                    maxshape[0] = None
                    group.create_dataset(
                        field,
                        data=array,
                        compression="gzip",
                        chunks=True,
                        maxshape=tuple(maxshape)
                    )
                else:
                    current_size = group[field].shape[0]
                    new_size = current_size + array.shape[0]
                    group[field].resize(new_size, axis=0)
                    group[field][current_size:] = array

        try:
            self._write_with_retry(write_func)
            _logger.info(f"{len(initial_states['particle_ID'])} initial states saved to {self.filename}")
        except OSError:
            _logger.exception(f'Failed to save initial states to {self.filename}!')

    def _write_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        """
        Filters the interactions by sensitive volumes using check_inside,
        formats data to legacy representation (without emission fields), and writes to HDF5.
        """
        pos_x = interactions['pos_x']
        pos_y = interactions['pos_y']
        pos_z = interactions['pos_z']

        dir_x = interactions['dir_x']
        dir_y = interactions['dir_y']
        dir_z = interactions['dir_z']

        global_position = np.column_stack((pos_x, pos_y, pos_z))
        global_direction = np.column_stack((dir_x, dir_y, dir_z))

        events_saved = 0

        # Create dictionary to hold data to be written
        volume_data_map = {}

        for volume in self.sensitive_volumes:
            # 1. Check inside for all particles
            mask = volume.check_inside(global_position)

            # Since check_inside might return a scalar bool (if empty) or single element,
            # ensure it's a 1D boolean array.
            if isinstance(mask, bool):
                if mask and len(global_position) > 0:
                    mask = np.ones(len(global_position), dtype=np.bool_)
                else:
                    mask = np.zeros(len(global_position), dtype=np.bool_)

            if not np.any(mask):
                continue

            # 2. Filter data
            vol_global_pos = global_position[mask]
            vol_global_dir = global_direction[mask]

            # 3. Local coordinates
            if isinstance(volume, TransformableVolume):
                local_position = volume.convert_to_local_position(vol_global_pos, as_parent=False)
                local_direction = volume.convert_to_local_direction(vol_global_dir, as_parent=False)
            else:
                local_position = vol_global_pos.copy()
                local_direction = vol_global_dir.copy()

            # 4. Filter scalar arrays
            process_id = interactions['process_id'][mask]
            particle_ID = interactions['particle_ID'][mask]
            energy_deposit = interactions['energy_deposit'][mask]
            scattering_theta = interactions['scattering_theta'][mask]
            scattering_phi = interactions['scattering_phi'][mask]

            n_events = len(process_id)
            events_saved += n_events

            # Map Process ID to Names
            process_name = np.empty(n_events, dtype='S30')
            for pid, name in self.PROCESS_MAP.items():
                process_name[process_id == pid] = name

            # Create dummy arrays for missing legacy fields
            particle_type = np.full(n_events, b'', dtype='S30')
            material_density = np.zeros(n_events, dtype=np.float64)
            distance_traveled = np.zeros(n_events, dtype=np.float64)

            scattering_angles = np.column_stack((scattering_theta, scattering_phi))

            # Store in map (emission fields excluded!)
            volume_data_map[volume.name] = {
                'global_position': vol_global_pos,
                'global_direction': vol_global_dir,
                'local_position': local_position,
                'local_direction': local_direction,
                'process_name': process_name,
                'particle_type': particle_type,
                'particle_ID': particle_ID,
                'energy_deposit': energy_deposit,
                'material_density': material_density,
                'scattering_angles': scattering_angles,
                'distance_traveled': distance_traveled
            }

        # 5. Write to HDF5
        if not volume_data_map:
            return

        def write_func(f: h5py.File):
            if 'interaction_data' not in f:
                group = f.create_group('interaction_data')
            else:
                group = f['interaction_data']

            for volume_name, data in volume_data_map.items():
                if volume_name not in group:
                    volume_group = group.create_group(volume_name)
                    for field, array in data.items():
                        maxshape = list(array.shape)
                        maxshape[0] = None
                        volume_group.create_dataset(
                            field,
                            data=array,
                            compression="gzip",
                            chunks=True,
                            maxshape=tuple(maxshape)
                        )
                else:
                    volume_group = group[volume_name]
                    for field, array in data.items():
                        if field in volume_group:
                            current_size = volume_group[field].shape[0]
                            new_size = current_size + array.shape[0]
                            volume_group[field].resize(new_size, axis=0)
                            volume_group[field][current_size:] = array

        try:
            self._write_with_retry(write_func)
            _logger.info(f'{events_saved} events saved to {self.filename}')
        except OSError:
            _logger.exception(f'Failed to save interactions to {self.filename}!')
            return
