import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Optional
import h5py
import numpy as np

from core.geometry.volumes import Volume, TransformableVolume, VolumeWithChilds

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class BaseDataHandler(ABC):
    writer_callback: Callable[[Callable[[h5py.File], None]], None]

    def set_writer_callback(self, callback: Callable[[Callable[[h5py.File], None]], None]) -> None:
        self.writer_callback = callback

    @abstractmethod
    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        pass

class SensitiveVolumeHandler(BaseDataHandler):
    """
    Base handler for caching interactions that hit specific sensitive volumes.
    By default, it drops external interactions from memory and writes only
    interactions that fall within target_volume_ids.
    """
    PROCESS_MAP = {
        0: b'PhotoelectricEffect',
        1: b'ComptonScattering',
        2: b'CoherentScattering',
        3: b'PairProduction'
    }

    def __init__(self, sensitive_volumes: List[Volume], simulation_volume: Optional[Volume] = None) -> None:
        self.sensitive_volumes = sensitive_volumes
        self.simulation_volume = simulation_volume

        if self.simulation_volume is None:
            self.simulation_volume = self._find_simulation_volume(sensitive_volumes)

        self.target_volume_ids = np.array(self._get_hierarchy_indices(sensitive_volumes), dtype=np.int64)

        self.volume_mapping = {}
        if self.simulation_volume is not None:
            self._build_volume_mapping(self.simulation_volume, self.simulation_volume)

        for vol in sensitive_volumes:
            self._build_volume_mapping(vol, vol)


    def _find_simulation_volume(self, sensitive_volumes: List[Volume]) -> Optional[Volume]:
        unique_roots = set()
        for vol in sensitive_volumes:
            if isinstance(vol, TransformableVolume):
                unique_roots.add(vol.root_volume)
            else:
                unique_roots.add(vol)

        if len(unique_roots) > 1:
            raise ValueError("All sensitive volumes must share the same root simulation volume!")
        elif len(unique_roots) == 1:
            return unique_roots.pop()
        return None

    def _build_volume_mapping(self, current_vol: Volume, root_vol: Volume) -> None:
        for i, (v, _, _) in enumerate(self.simulation_volume.flattened_scene.flat_list):
            if self._is_descendant(v, current_vol):
                self.volume_mapping[i] = root_vol

    def _get_hierarchy_indices(self, volumes: List[Volume]) -> List[int]:
        indices = []
        for vol in volumes:
            for i, (v, _, _) in enumerate(self.simulation_volume.flattened_scene.flat_list):
                if self._is_descendant(v, vol):
                    indices.append(i)
        return list(set(indices))

    def _is_descendant(self, query_vol: Volume, root_vol: Volume) -> bool:
        if query_vol is root_vol:
            return True
        if isinstance(root_vol, VolumeWithChilds):
            for child in root_vol.childs:
                if self._is_descendant(query_vol, child):
                    return True
        return False

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        if chunk_type == 'interactions':
            self._process_interactions(chunk['data'])

    def _process_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        volume_ids = interactions['volume_id']
        mask_sensitive = np.isin(volume_ids, self.target_volume_ids)

        if np.any(mask_sensitive):
            interactions_to_write = {k: v[mask_sensitive] for k, v in interactions.items()}
            self._write_interactions(interactions_to_write)

    def _get_volumes_to_write(self) -> List[Volume]:
        return list(self.sensitive_volumes)

    def _write_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
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

        if getattr(self, 'writer_callback', None):
            self.writer_callback(write_func)
            _logger.debug(f'Interactions write task forwarded for {events_saved} events.')


class HistoryAssemblerHandler(SensitiveVolumeHandler):
    """
    Inherits from SensitiveVolumeHandler but saves the ENTIRE history
    (including background tracks outside sensitive volumes) for any particle
    that scored a hit, including its initial emission state.
    """
    def __init__(self, sensitive_volumes: List[Volume], simulation_volume: Optional[Volume] = None) -> None:
        super().__init__(sensitive_volumes, simulation_volume)
        self.initial_states_chunks = []
        self.interactions_chunks = []
        self.scored_particles = set()

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        if chunk_type == 'initial_states':
            self._cache_initial_states(chunk['data'])
        elif chunk_type == 'interactions':
            self._cache_interactions(chunk['data'])
        elif chunk_type == 'dead_particles':
            self._flush_dead_particles(chunk['data'])

    def _cache_initial_states(self, initial_states: Dict[str, np.ndarray]) -> None:
        self.initial_states_chunks.append(initial_states)

    def _cache_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        volume_ids = interactions['volume_id']
        mask_sensitive = np.isin(volume_ids, self.target_volume_ids)
        if np.any(mask_sensitive):
            sensitive_ids = interactions['particle_ID'][mask_sensitive]
            self.scored_particles.update(sensitive_ids)

        self.interactions_chunks.append(interactions)

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

        if getattr(self, 'writer_callback', None):
            self.writer_callback(write_func)

    def _write_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
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

        for top_volume in self._get_volumes_to_write():
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

        if getattr(self, 'writer_callback', None):
            self.writer_callback(write_func)
            _logger.debug(f'Interactions write task forwarded for {events_saved} events.')

    def _flush_dead_particles(self, dead_ids: np.ndarray) -> None:
        scored_dead_ids = np.intersect1d(dead_ids, list(self.scored_particles))

        initial_states_to_write = None
        if self.initial_states_chunks:
            mega_init = {k: np.concatenate([c[k] for c in self.initial_states_chunks]) for k in self.initial_states_chunks[0].keys()}

            if len(scored_dead_ids) > 0:
                scored_mask = np.isin(mega_init['particle_ID'], scored_dead_ids)
                if np.any(scored_mask):
                    initial_states_to_write = {k: v[scored_mask] for k, v in mega_init.items()}

                    pos_x = initial_states_to_write.pop('pos_x')
                    pos_y = initial_states_to_write.pop('pos_y')
                    pos_z = initial_states_to_write.pop('pos_z')
                    initial_states_to_write['emission_position'] = np.column_stack((pos_x, pos_y, pos_z))

                    dir_x = initial_states_to_write.pop('dir_x')
                    dir_y = initial_states_to_write.pop('dir_y')
                    dir_z = initial_states_to_write.pop('dir_z')
                    initial_states_to_write['emission_direction'] = np.column_stack((dir_x, dir_y, dir_z))

            survivor_mask = ~np.isin(mega_init['particle_ID'], dead_ids)
            if np.any(survivor_mask):
                self.initial_states_chunks = [{k: v[survivor_mask] for k, v in mega_init.items()}]
            else:
                self.initial_states_chunks.clear()

        interactions_to_write = None
        if self.interactions_chunks:
            mega_inter = {k: np.concatenate([c[k] for c in self.interactions_chunks]) for k in self.interactions_chunks[0].keys()}

            if len(scored_dead_ids) > 0:
                # HistoryAssembler logic: Keep ALL interactions (entire history) for scored particles
                scored_mask = np.isin(mega_inter['particle_ID'], scored_dead_ids)
                if np.any(scored_mask):
                    interactions_to_write = {k: v[scored_mask] for k, v in mega_inter.items()}
                    sort_idx = np.argsort(interactions_to_write['particle_ID'])
                    interactions_to_write = {k: v[sort_idx] for k, v in interactions_to_write.items()}

            survivor_mask = ~np.isin(mega_inter['particle_ID'], dead_ids)
            if np.any(survivor_mask):
                self.interactions_chunks = [{k: v[survivor_mask] for k, v in mega_inter.items()}]
            else:
                self.interactions_chunks.clear()

        self.scored_particles.difference_update(dead_ids)

        if initial_states_to_write is not None:
            self._write_initial_states(initial_states_to_write)
        if interactions_to_write is not None:
            self._write_interactions(interactions_to_write)


class DirectStreamHandler(BaseDataHandler):
    """
    Directly streams 'initial_states' and 'interactions' arrays into HDF5
    without any caching, mapping, or tracking dead particles.
    """
    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        if chunk_type == 'initial_states':
            self._stream_initial_states(chunk['data'])
        elif chunk_type == 'interactions':
            self._stream_interactions(chunk['data'])
        elif chunk_type == 'dead_particles':
            pass

    def _stream_initial_states(self, initial_states: Dict[str, np.ndarray]) -> None:
        # Pre-process 3D vectors
        # Create a new dict to avoid modifying read-only MappingProxyType
        output_states = {}
        for k, v in initial_states.items():
            if k not in ('pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z'):
                output_states[k] = v

        output_states['emission_position'] = np.column_stack((
            initial_states['pos_x'], initial_states['pos_y'], initial_states['pos_z']
        ))

        output_states['emission_direction'] = np.column_stack((
            initial_states['dir_x'], initial_states['dir_y'], initial_states['dir_z']
        ))

        def write_func(f: h5py.File):
            if 'initial_states' not in f:
                group = f.create_group('initial_states')
            else:
                group = f['initial_states']

            for field, array in output_states.items():
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

        if getattr(self, 'writer_callback', None):
            self.writer_callback(write_func)

    def _stream_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        def write_func(f: h5py.File):
            if 'interaction_data' not in f:
                group = f.create_group('interaction_data')
            else:
                group = f['interaction_data']

            if 'raw_stream' not in group:
                volume_group = group.create_group('raw_stream')
            else:
                volume_group = group['raw_stream']

            for field, array in interactions.items():
                if field not in volume_group:
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
                    current_size = volume_group[field].shape[0]
                    new_size = current_size + array.shape[0]
                    volume_group[field].resize(new_size, axis=0)
                    volume_group[field][current_size:] = array

        if getattr(self, 'writer_callback', None):
            self.writer_callback(write_func)
