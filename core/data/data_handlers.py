from core.geometry.flattened_scene import FlattenedScene
import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Set

import h5py
import numpy as np

from core.geometry.volumes import Volume

_logger = logging.getLogger(__name__)

class BaseDataHandler(abc.ABC):
    def __init__(self):
        self.writer_callback: Optional[Callable] = None

    def set_writer_callback(self, callback: Callable) -> None:
        self.writer_callback = callback

    @abc.abstractmethod
    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        pass

class DirectStreamHandler(BaseDataHandler):
    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        data = chunk.get('data')

        if chunk_type == 'initial_states':
            self._write_initial_states(data)
        elif chunk_type == 'interactions':
            self._write_interactions({'All_Volumes': data})

    def _write_initial_states(self, initial_states: Dict[str, np.ndarray]) -> None:
        def write_func(f: h5py.File):
            if 'initial_states' not in f:
                group = f.create_group('initial_states')
                for k, v in initial_states.items():
                    group.create_dataset(k, data=v, maxshape=(None, *v.shape[1:]))
            else:
                group = f['initial_states']
                for k, v in initial_states.items():
                    if k in group:
                        curr_size = group[k].shape[0]
                        group[k].resize(curr_size + v.shape[0], axis=0)
                        group[k][curr_size:] = v
        if self.writer_callback is not None:
            self.writer_callback(write_func)

    def _write_interactions(self, volume_data_map: Dict[str, Dict[str, np.ndarray]]) -> None:
        def write_func(f: h5py.File):
            if 'interactions' not in f:
                group = f.create_group('interactions')
            else:
                group = f['interactions']

            for volume_name, data in volume_data_map.items():
                if volume_name not in group:
                    volume_group = group.create_group(volume_name)
                    for field, array in data.items():
                        volume_group.create_dataset(field, data=array, maxshape=(None, *array.shape[1:]))
                else:
                    volume_group = group[volume_name]
                    for field, array in data.items():
                        if field in volume_group:
                            current_size = volume_group[field].shape[0]
                            new_size = current_size + array.shape[0]
                            volume_group[field].resize(new_size, axis=0)
                            volume_group[field][current_size:] = array

        if self.writer_callback is not None:
            self.writer_callback(write_func)

class SensitiveVolumeHandler(DirectStreamHandler):
    PROCESS_MAP = {
        0: b'RayleighScattering',
        1: b'ComptonScattering',
        2: b'PhotoElectricAbsorption',
        3: b'PairProduction'
    }

    def __init__(self, sensitive_volumes: List[Volume]):
        super().__init__()
        self.sensitive_volumes = sensitive_volumes

        unique_roots = set()
        self.unique_top_volumes = set()
        for vol in sensitive_volumes:
            unique_roots.add(vol.root)
            self.unique_top_volumes.add(vol.top_volume)

        if len(unique_roots) > 1:
            raise ValueError("All sensitive volumes must share the same root simulation volume!")
        elif len(unique_roots) == 1:
            self.scene_root = unique_roots.pop()
        else:
            self.scene_root = None

        self.volume_mapping: Dict[int, Volume] = {}
        self.target_volumes: List[int] = []
        self._build_volume_mapping()

    def _build_volume_mapping(self) -> None:
        if self.scene_root is not None:
            flat_list = FlattenedScene(self.scene_root).flat_list
            
            sensitive_indices = set()
            for i, (v, _, _) in enumerate(flat_list):
                if v in self.sensitive_volumes:
                    sensitive_indices.add(i)
            
            target_ids = set(sensitive_indices)
            for i, (v, _, parent_idx) in enumerate(flat_list):
                top_vol = v.top_volume
                if top_vol in self.unique_top_volumes:
                    self.volume_mapping[i] = top_vol
                
                curr_idx = parent_idx
                while curr_idx != -1:
                    if curr_idx in sensitive_indices:
                        target_ids.add(i)
                        break
                    curr_idx = flat_list[curr_idx][2]
            
            self.target_volumes = list(target_ids)

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        if chunk_type == 'interactions':
            self._process_interactions(chunk['data'])

    def _process_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        volume_ids = interactions['volume_id']
        mask_sensitive = np.isin(volume_ids, self.target_volumes)

        if np.any(mask_sensitive):
            interactions_to_write = {k: v[mask_sensitive] for k, v in interactions.items()}
            self._format_and_write_interactions(interactions_to_write)

    def _get_volumes_to_write(self) -> List[Volume]:
        return list(self.sensitive_volumes)

    def _format_and_write_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
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

            local_position = top_volume.convert_to_local_position(vol_global_pos)
            local_direction = top_volume.convert_to_local_direction(vol_global_dir)

            process_id = interactions['process_id'][mask]
            particle_ID = interactions['particle_ID'][mask]
            energy_deposit = interactions['energy_deposit'][mask]
            scattering_theta = interactions['scattering_theta'][mask]
            scattering_phi = interactions['scattering_phi'][mask]
            material_id = interactions['material_id'][mask]
            Z = interactions['Z'][mask]

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
                'material_id': material_id,
                'Z': Z,
                'scattering_angles': scattering_angles,
                'distance_traveled': distance_traveled
            }

        if not volume_data_map:
            return

        self._write_interactions(volume_data_map)
        _logger.debug(f'Interactions write task forwarded for {events_saved} events.')

class HistoryAssemblerHandler(SensitiveVolumeHandler):
    def __init__(self, sensitive_volumes: List[Volume], save_initial_states: bool = True):
        super().__init__(sensitive_volumes)
        self.save_initial_states = save_initial_states

        self.initial_states_chunks: List[Dict[str, np.ndarray]] = []
        self.interactions_chunks: List[Dict[str, np.ndarray]] = []
        self.scored_particles: Set[int] = set()

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        data = chunk.get('data')

        if chunk_type == 'initial_states':
            if self.save_initial_states:
                self.initial_states_chunks.append({k: np.array(v, copy=False) for k, v in data.items()})
        elif chunk_type == 'interactions':
            vol_ids = data['volume_id']
            scored_mask = np.isin(vol_ids, self.target_volumes)
            if np.any(scored_mask):
                self.scored_particles.update(data['particle_ID'][scored_mask])
            self.interactions_chunks.append({k: np.array(v, copy=False) for k, v in data.items()})
        elif chunk_type == 'dead_particles':
            dead_ids = data
            if len(dead_ids) > 0:
                self._flush_dead_particles(dead_ids)

    def _get_volumes_to_write(self) -> List[Volume]:
        volumes_to_write = list(self.sensitive_volumes)
        for tv in self.unique_top_volumes:
            if tv not in volumes_to_write:
                volumes_to_write.append(tv)
        return volumes_to_write

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
            self._format_and_write_interactions(interactions_to_write)
