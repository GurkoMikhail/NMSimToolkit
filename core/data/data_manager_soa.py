import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from core.geometry.volumes import Volume, TransformableVolume, VolumeWithChilds

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

    def __init__(self, filename: str, sensitive_volumes: List[Volume], simulation_volume: Optional[Volume] = None, queue: Any = None, lock: Optional[Any] = None) -> None:
        super().__init__()
        self.filename = Path(f'output data/{filename}')
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.sensitive_volumes = sensitive_volumes
        self.queue = queue
        self.lock = lock
        self.daemon = True

        self.simulation_volume = simulation_volume
        self.target_volume_ids = np.array(self._get_hierarchy_indices(sensitive_volumes), dtype=np.int64)

        self.volume_mapping = {}
        for vol in sensitive_volumes:
            self._build_volume_mapping(vol, vol)

        self.active_initial_states = {}
        self.active_interactions = {}
        self.scored_particles = set()

    def _get_root_volume(self, vol: Volume) -> Volume:
        if self.simulation_volume is not None:
            return self.simulation_volume
        if isinstance(vol, TransformableVolume):
            return vol.root_volume
        return vol

    def _build_volume_mapping(self, current_vol: Volume, root_vol: Volume) -> None:
        sim_vol = self._get_root_volume(current_vol)
        for i, (v, _, _) in enumerate(sim_vol.flattened_scene.flat_list):
            if v is current_vol:
                self.volume_mapping[i] = root_vol
                break

        if isinstance(current_vol, VolumeWithChilds):
            for child in current_vol.childs:
                self._build_volume_mapping(child, root_vol)

    def _get_hierarchy_indices(self, volumes: List[Volume]) -> List[int]:
        indices = []
        for vol in volumes:
            sim_vol = self._get_root_volume(vol)
            for i, (v, _, _) in enumerate(sim_vol.flattened_scene.flat_list):
                if self._is_descendant(v, vol):
                    indices.append(i)
        return indices

    def _is_descendant(self, query_vol: Volume, root_vol: Volume) -> bool:
        if query_vol is root_vol:
            return True
        if isinstance(root_vol, VolumeWithChilds):
            for child in root_vol.childs:
                if self._is_descendant(query_vol, child):
                    return True
        return False

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
            self._cache_initial_states(chunk['data'])
        elif chunk_type == 'interactions':
            self._cache_interactions(chunk['data'])
        elif chunk_type == 'dead_particles':
            self._flush_dead_particles(chunk['data'])

    def _cache_initial_states(self, initial_states: Dict[str, np.ndarray]) -> None:
        """
        Caches initial states into memory dictionaries.
        """
        p_ids = initial_states['particle_ID']
        for i, p_id in enumerate(p_ids):
            self.active_initial_states[p_id] = {k: v[i] for k, v in initial_states.items()}

    def _cache_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        """
        Filters the interactions by sensitive volume hierarchy and caches them by ID.
        """
        # Find scored IDs
        volume_ids = interactions['volume_id']
        mask_sensitive = np.isin(volume_ids, self.target_volume_ids)
        if np.any(mask_sensitive):
            sensitive_ids = interactions['particle_ID'][mask_sensitive]
            self.scored_particles.update(sensitive_ids)

        p_ids = interactions['particle_ID']
        unique_ids, inverse_indices = np.unique(p_ids, return_inverse=True)

        for i, p_id in enumerate(unique_ids):
            mask = inverse_indices == i
            chunk_slice = {k: v[mask] for k, v in interactions.items()}
            if p_id not in self.active_interactions:
                self.active_interactions[p_id] = [chunk_slice]
            else:
                self.active_interactions[p_id].append(chunk_slice)

    def _flush_dead_particles(self, dead_ids: np.ndarray) -> None:
        """
        Retrieves scored dead particles and triggers write_to_hdf5. Then discards them from RAM.
        """
        scored_dead_ids = [pid for pid in dead_ids if pid in self.scored_particles]

        if not scored_dead_ids:
            # Clean up cache for dead but unscored particles
            for pid in dead_ids:
                self.active_initial_states.pop(pid, None)
                self.active_interactions.pop(pid, None)
            return

        # Safely find initial keys
        initial_keys = None
        for pid in scored_dead_ids:
            if pid in self.active_initial_states:
                initial_keys = self.active_initial_states[pid].keys()
                break

        if initial_keys is None:
            # Fallback if no initial states are found for any scored dead particles
            initial_keys = ['particle_ID', 'emission_time', 'emission_energy', 'pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z']

        initial_states_to_write = {k: [] for k in initial_keys}
        interactions_to_write = []

        for pid in scored_dead_ids:
            # Collect initial state
            if pid in self.active_initial_states:
                for k, v in self.active_initial_states[pid].items():
                    initial_states_to_write[k].append(v)

            # Collect interactions
            if pid in self.active_interactions:
                interactions_to_write.extend(self.active_interactions[pid])

        # Convert initial states lists to numpy arrays
        for k, v in initial_states_to_write.items():
            initial_states_to_write[k] = np.array(v)

        # Apply column_stack for emission data here lazily before writing
        if 'pos_x' in initial_states_to_write:
            pos_x = initial_states_to_write.pop('pos_x')
            pos_y = initial_states_to_write.pop('pos_y')
            pos_z = initial_states_to_write.pop('pos_z')
            initial_states_to_write['emission_position'] = np.column_stack((pos_x, pos_y, pos_z))

            dir_x = initial_states_to_write.pop('dir_x')
            dir_y = initial_states_to_write.pop('dir_y')
            dir_z = initial_states_to_write.pop('dir_z')
            initial_states_to_write['emission_direction'] = np.column_stack((dir_x, dir_y, dir_z))

        # Merge interactions lists of dicts to a single dict of numpy arrays
        if interactions_to_write:
            merged_interactions = {k: np.concatenate([d[k] for d in interactions_to_write]) for k in interactions_to_write[0].keys()}
        else:
            merged_interactions = None

        # Discard from RAM
        for pid in dead_ids:
            self.active_initial_states.pop(pid, None)
            self.active_interactions.pop(pid, None)
            self.scored_particles.discard(pid)

        # Proceed to write these collected arrays
        self._write_initial_states(initial_states_to_write)
        if merged_interactions:
            self._write_interactions(merged_interactions)

    def _write_initial_states(self, initial_states: Dict[str, np.ndarray]) -> None:
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
        Groups the interactions by their top-level sensitive volume (using volume_mapping),
        translates to local coordinates, formats data, and writes to HDF5.
        """
        pos_x = interactions['pos_x']
        pos_y = interactions['pos_y']
        pos_z = interactions['pos_z']

        dir_x = interactions['dir_x']
        dir_y = interactions['dir_y']
        dir_z = interactions['dir_z']

        global_position = np.column_stack((pos_x, pos_y, pos_z))
        global_direction = np.column_stack((dir_x, dir_y, dir_z))
        volume_ids = interactions['volume_id']

        events_saved = 0
        volume_data_map = {}

        root_vol_names = []
        for vid in volume_ids:
            if vid in self.volume_mapping:
                root_vol_names.append(self.volume_mapping[vid].name)
            else:
                root_vol_names.append(None)

        root_vol_names = np.array(root_vol_names)

        for top_volume in self.sensitive_volumes:
            mask = root_vol_names == top_volume.name

            if not np.any(mask):
                continue

            vol_global_pos = global_position[mask]
            vol_global_dir = global_direction[mask]

            if isinstance(top_volume, TransformableVolume):
                local_position = top_volume.convert_to_local_position(vol_global_pos, as_parent=False)
                local_direction = top_volume.convert_to_local_direction(vol_global_dir, as_parent=False)
            else:
                local_position = vol_global_pos.copy()
                local_direction = vol_global_dir.copy()

            process_id = interactions['process_id'][mask]
            particle_ID = interactions['particle_ID'][mask]
            energy_deposit = interactions['energy_deposit'][mask]
            scattering_theta = interactions['scattering_theta'][mask]
            scattering_phi = interactions['scattering_phi'][mask]

            n_events = len(process_id)
            events_saved += n_events

            process_name = np.empty(n_events, dtype='S30')
            for pid, name in self.PROCESS_MAP.items():
                process_name[process_id == pid] = name

            species_int = interactions['species'][mask]
            species_str = np.empty(n_events, dtype='S30')
            species_str[species_int == 0] = b'Photon'
            species_str[species_int == 1] = b'Electron'
            species_str[species_int == 2] = b'Positron'

            distance_traveled = interactions['distance_traveled'][mask]
            volume_id = interactions['volume_id'][mask]

            scattering_angles = np.column_stack((scattering_theta, scattering_phi))

            volume_data_map[top_volume.name] = {
                'global_position': vol_global_pos,
                'global_direction': vol_global_dir,
                'local_position': local_position,
                'local_direction': local_direction,
                'process_name': process_name,
                'species': species_str,
                'particle_ID': particle_ID,
                'energy_deposit': energy_deposit,
                'volume_id': volume_id,
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
