import json
from core.geometry.flattened_scene import FlattenedScene
import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Set

import h5py
import numpy as np

from core.geometry.volumes import Volume
from core.physics.processes import get_process_mapping, get_species_mapping
from core.scene.nodes import SpatialNode

_logger = logging.getLogger(__name__)


class BaseDataHandler(abc.ABC):
    """
    Базовый абстрактный класс потребителя потоковых данных симуляции (Data Handler).
    Задает единый контракт обработки чанков данных и финализации результатов.
    """

    def __init__(
        self,
        floating_point_precision: str = "float32",
        compression: Optional[str] = None,
        compression_opts: int = 4,
    ) -> None:
        self.writer_callback: Optional[Callable[[Callable[[h5py.File], None]], None]] = None
        self.floating_point_precision: str = floating_point_precision
        self.compression: Optional[str] = compression
        self.compression_opts: int = compression_opts

    def set_writer_callback(self, callback: Callable[[Callable[[h5py.File], None]], None]) -> None:
        """
        Регистрация функции обратного вызова для безопасной записи результатов в HDF5.
        """
        self.writer_callback = callback

    @abc.abstractmethod
    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Обработка порции данных моделирования (начальные состояния частиц, взаимодействия).

        :param chunk: Словарь с типом данных ('type') и пакетом массивов ('data').
        """
        pass

    def finalize(self) -> None:
        """
        Финализация записи данных в хранилище при завершении моделирования.
        По умолчанию не выполняет действий, переопределяется в специализированных подклассах.
        """
        pass

class DirectStreamHandler(BaseDataHandler):
    def __init__(
        self,
        floating_point_precision: str = "float32",
        compression: Optional[str] = None,
        compression_opts: int = 4,
    ) -> None:
        super().__init__(
            floating_point_precision=floating_point_precision,
            compression=compression,
            compression_opts=compression_opts,
        )

    def _format_raw_interactions(self, interactions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Приводит необработанный SoA-пакет взаимодействий к канонической схеме датасетов HDF5:
        векторы (N, 3), process_code/species_code (uint8), volume_id/material_id (uint32),
        scattering_angles (N, 2), плавающая точность float32/float64.
        """
        target_float_dtype = np.float32 if self.floating_point_precision == "float32" else np.float64
        result: Dict[str, np.ndarray] = {}

        if 'pos_x' in interactions and 'pos_y' in interactions and 'pos_z' in interactions:
            result['global_position'] = np.column_stack((
                interactions['pos_x'],
                interactions['pos_y'],
                interactions['pos_z'],
            )).astype(target_float_dtype)
        elif 'global_position' in interactions:
            result['global_position'] = interactions['global_position'].astype(target_float_dtype)

        if 'dir_x' in interactions and 'dir_y' in interactions and 'dir_z' in interactions:
            result['global_direction'] = np.column_stack((
                interactions['dir_x'],
                interactions['dir_y'],
                interactions['dir_z'],
            )).astype(target_float_dtype)
        elif 'global_direction' in interactions:
            result['global_direction'] = interactions['global_direction'].astype(target_float_dtype)

        if 'process_id' in interactions:
            result['process_code'] = interactions['process_id'].astype(np.uint8)
        elif 'process_code' in interactions:
            result['process_code'] = interactions['process_code'].astype(np.uint8)

        if 'species' in interactions:
            result['species_code'] = interactions['species'].astype(np.uint8)
        elif 'species_code' in interactions:
            result['species_code'] = interactions['species_code'].astype(np.uint8)

        if 'particle_ID' in interactions:
            result['particle_ID'] = interactions['particle_ID']

        if 'energy_deposit' in interactions:
            result['energy_deposit'] = interactions['energy_deposit'].astype(target_float_dtype)

        if 'distance_traveled' in interactions:
            result['distance_traveled'] = interactions['distance_traveled'].astype(target_float_dtype)

        if 'volume_id' in interactions:
            result['volume_id'] = interactions['volume_id'].astype(np.uint32)

        if 'material_id' in interactions:
            result['material_id'] = interactions['material_id'].astype(np.uint32)

        if 'Z' in interactions:
            result['Z'] = interactions['Z'].astype(np.int8)

        if 'scattering_theta' in interactions and 'scattering_phi' in interactions:
            result['scattering_angles'] = np.column_stack((
                interactions['scattering_theta'],
                interactions['scattering_phi'],
            )).astype(target_float_dtype)
        elif 'scattering_angles' in interactions:
            result['scattering_angles'] = interactions['scattering_angles'].astype(target_float_dtype)

        for key, arr in interactions.items():
            if key not in (
                'pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z',
                'process_id', 'process_code', 'species', 'species_code',
                'particle_ID', 'energy_deposit', 'distance_traveled',
                'volume_id', 'material_id', 'Z',
                'scattering_theta', 'scattering_phi', 'scattering_angles',
                'global_position', 'global_direction',
            ):
                result[key] = arr

        return result

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        data = chunk.get('data')

        if chunk_type == 'initial_states':
            self._write_initial_states(data)
        elif chunk_type == 'interactions':
            formatted_data = self._format_raw_interactions(data)
            self._write_interactions({'All_Volumes': formatted_data})

    def _write_initial_states(self, initial_states: Dict[str, np.ndarray]) -> None:
        target_float_dtype = np.float32 if self.floating_point_precision == "float32" else np.float64
        states_copy = dict(initial_states)

        if 'pos_x' in states_copy and 'pos_y' in states_copy and 'pos_z' in states_copy:
            pos_x = states_copy.pop('pos_x')
            pos_y = states_copy.pop('pos_y')
            pos_z = states_copy.pop('pos_z')
            states_copy['emission_position'] = np.column_stack((pos_x, pos_y, pos_z))

        if 'dir_x' in states_copy and 'dir_y' in states_copy and 'dir_z' in states_copy:
            dir_x = states_copy.pop('dir_x')
            dir_y = states_copy.pop('dir_y')
            dir_z = states_copy.pop('dir_z')
            states_copy['emission_direction'] = np.column_stack((dir_x, dir_y, dir_z))

        prepared_states: Dict[str, np.ndarray] = {}
        for state_key, state_arr in states_copy.items():
            if isinstance(state_arr, np.ndarray) and state_arr.dtype.kind == "f":
                prepared_states[state_key] = state_arr.astype(target_float_dtype)
            else:
                prepared_states[state_key] = state_arr

        def write_func(f: h5py.File) -> None:
            group = f.require_group('initial_states')
            for state_key, state_arr in prepared_states.items():
                if state_key not in group:
                    kwargs: Dict[str, Any] = {'maxshape': (None, *state_arr.shape[1:])}
                    if self.compression is not None:
                        kwargs['compression'] = self.compression
                        if self.compression == 'gzip':
                            kwargs['compression_opts'] = self.compression_opts
                        kwargs['chunks'] = True
                    dataset = group.create_dataset(state_key, data=state_arr, **kwargs)
                    if state_key == 'emission_position':
                        dataset.attrs['units'] = 'mm'
                    elif state_key in ('emission_energy', 'energy'):
                        dataset.attrs['units'] = 'MeV'
                    elif state_key in ('emission_time', 'time'):
                        dataset.attrs['units'] = 'ns'
                else:
                    dataset = group[state_key]
                    curr_size = dataset.shape[0]
                    new_size = curr_size + state_arr.shape[0]
                    dataset.resize(new_size, axis=0)
                    dataset[curr_size:] = state_arr
        if self.writer_callback is not None:
            self.writer_callback(write_func)

    def _write_interactions(self, volume_data_map: Dict[str, Dict[str, np.ndarray]]) -> None:
        def write_func(f: h5py.File) -> None:
            group = f.require_group('interactions')
            for volume_name, volume_data in volume_data_map.items():
                volume_group = group.require_group(volume_name)
                for field_name, field_array in volume_data.items():
                    if field_name not in volume_group:
                        kwargs: Dict[str, Any] = {'maxshape': (None, *field_array.shape[1:])}
                        if self.compression is not None:
                            kwargs['compression'] = self.compression
                            if self.compression == 'gzip':
                                kwargs['compression_opts'] = self.compression_opts
                            kwargs['chunks'] = True
                        dataset = volume_group.create_dataset(field_name, data=field_array, **kwargs)
                        if field_name == 'process_code':
                            dataset.attrs['mapping'] = json.dumps(get_process_mapping())
                        elif field_name == 'species_code':
                            dataset.attrs['mapping'] = json.dumps(get_species_mapping())
                        elif field_name == 'global_position':
                            dataset.attrs['units'] = 'mm'
                        elif field_name == 'energy_deposit':
                            dataset.attrs['units'] = 'MeV'
                        elif field_name == 'distance_traveled':
                            dataset.attrs['units'] = 'mm'
                        elif field_name == 'scattering_angles':
                            dataset.attrs['units'] = 'rad'
                    else:
                        dataset = volume_group[field_name]
                        curr_size = dataset.shape[0]
                        new_size = curr_size + field_array.shape[0]
                        dataset.resize(new_size, axis=0)
                        dataset[curr_size:] = field_array

        if self.writer_callback is not None:
            self.writer_callback(write_func)

class SensitiveVolumeHandler(DirectStreamHandler):
    def __init__(
        self,
        sensitive_volumes: List[Volume],
        floating_point_precision: str = "float32",
        compression: Optional[str] = None,
        compression_opts: int = 4,
    ) -> None:
        super().__init__(
            floating_point_precision=floating_point_precision,
            compression=compression,
            compression_opts=compression_opts,
        )
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

        self.volume_mapping: Dict[int, SpatialNode] = {}
        self.target_volumes: List[int] = []
        self._build_volume_mapping()

    def _build_volume_mapping(self) -> None:
        """
        Формирует маппинг индексов объемов FlattenedScene на целевые узлы записи.
        Чувствительные объемы и их дочерние подузлы связываются с соответствующим
        чувствительным объемом для сохранения в его локальных координатах.
        Все остальные фоновые объемы сцены связываются с корнем сцены (scene_root).
        """
        if self.scene_root is not None:
            flat_list = FlattenedScene(self.scene_root).flat_list

            sensitive_set = set(self.sensitive_volumes)
            target_ids = set()

            for volume_idx, (vol, _, _) in enumerate(flat_list):
                matched_sensitive: Optional[SpatialNode] = None
                if vol in sensitive_set:
                    matched_sensitive = vol
                else:
                    curr_node: Optional[SpatialNode] = vol.parent
                    while curr_node is not None:
                        if curr_node in sensitive_set:
                            matched_sensitive = curr_node
                            break
                        curr_node = curr_node.parent

                if matched_sensitive is not None:
                    target_ids.add(volume_idx)
                    self.volume_mapping[volume_idx] = matched_sensitive
                elif self.scene_root is not None:
                    self.volume_mapping[volume_idx] = self.scene_root

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

    def _get_volumes_to_write(self) -> List[SpatialNode]:
        return list(self.sensitive_volumes)

    def _format_and_write_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        target_float_dtype = np.float32 if self.floating_point_precision == "float32" else np.float64

        if 'pos_x' in interactions and 'pos_y' in interactions and 'pos_z' in interactions:
            pos_x = interactions['pos_x']
            pos_y = interactions['pos_y']
            pos_z = interactions['pos_z']
            global_position = np.column_stack((pos_x, pos_y, pos_z)).astype(target_float_dtype)
        elif 'global_position' in interactions:
            global_position = interactions['global_position'].astype(target_float_dtype)
        else:
            global_position = np.empty((0, 3), dtype=target_float_dtype)

        if 'dir_x' in interactions and 'dir_y' in interactions and 'dir_z' in interactions:
            dir_x = interactions['dir_x']
            dir_y = interactions['dir_y']
            dir_z = interactions['dir_z']
            global_direction = np.column_stack((dir_x, dir_y, dir_z)).astype(target_float_dtype)
        elif 'global_direction' in interactions:
            global_direction = interactions['global_direction'].astype(target_float_dtype)
        else:
            global_direction = np.empty((0, 3), dtype=target_float_dtype)

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

            if 'process_id' in interactions:
                process_code = interactions['process_id'][mask].astype(np.uint8)
            else:
                process_code = interactions['process_code'][mask].astype(np.uint8)

            if 'species' in interactions:
                species_code = interactions['species'][mask].astype(np.uint8)
            else:
                species_code = interactions['species_code'][mask].astype(np.uint8)

            particle_ID = interactions['particle_ID'][mask]
            energy_deposit = interactions['energy_deposit'][mask].astype(target_float_dtype)

            if 'scattering_theta' in interactions and 'scattering_phi' in interactions:
                scattering_theta = interactions['scattering_theta'][mask]
                scattering_phi = interactions['scattering_phi'][mask]
                scattering_angles = np.column_stack((scattering_theta, scattering_phi)).astype(target_float_dtype)
            elif 'scattering_angles' in interactions:
                scattering_angles = interactions['scattering_angles'][mask].astype(target_float_dtype)
            else:
                scattering_angles = np.zeros((len(particle_ID), 2), dtype=target_float_dtype)

            material_id = interactions['material_id'][mask].astype(np.uint32)
            volume_id = interactions['volume_id'][mask].astype(np.uint32)
            Z = interactions['Z'][mask].astype(np.int8)

            n_events = len(process_code)
            events_saved += n_events

            distance_traveled = interactions['distance_traveled'][mask].astype(target_float_dtype)

            volume_data_map[top_volume.name] = {
                'global_position': vol_global_pos,
                'global_direction': vol_global_dir,
                'process_code': process_code,
                'species_code': species_code,
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
    def __init__(
        self,
        sensitive_volumes: List[Volume],
        save_initial_states: bool = True,
        floating_point_precision: str = "float32",
        compression: Optional[str] = None,
        compression_opts: int = 4,
    ) -> None:
        super().__init__(
            sensitive_volumes=sensitive_volumes,
            floating_point_precision=floating_point_precision,
            compression=compression,
            compression_opts=compression_opts,
        )
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

    def _get_volumes_to_write(self) -> List[SpatialNode]:
        volumes_to_write: List[SpatialNode] = list(self.sensitive_volumes)
        if self.scene_root is not None and self.scene_root not in volumes_to_write:
            volumes_to_write.append(self.scene_root)
        for top_vol in self.unique_top_volumes:
            if top_vol not in volumes_to_write:
                volumes_to_write.append(top_vol)
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

                    if 'pos_x' in initial_states_to_write:
                        pos_x = initial_states_to_write.pop('pos_x')
                        pos_y = initial_states_to_write.pop('pos_y')
                        pos_z = initial_states_to_write.pop('pos_z')
                        initial_states_to_write['emission_position'] = np.column_stack((pos_x, pos_y, pos_z))

                    if 'dir_x' in initial_states_to_write:
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
