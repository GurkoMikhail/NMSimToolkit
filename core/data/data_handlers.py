"""
Модуль потоковых обработчиков данных симуляции (Data Handlers).
Реализует фильтрацию событий по чувствительным объемам, сборку полных историй частиц
и форматирование датасетов для записи в HDF5.
"""

import abc
import json
import logging
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set

import h5py
import numpy as np

from core.data.dataset_write_buffer import DEFAULT_BUFFER_SIZE, DatasetWriteBuffer
from core.geometry.flattened_scene import FlattenedScene
from core.geometry.volumes import Volume
from core.physics.interaction_buffers import ChunkType
from core.physics.processes import get_process_mapping, get_species_mapping
from core.scene.nodes import CompositeNode, SpatialNode

_logger = logging.getLogger(__name__)

# Атрибуты датасетов группы 'initial_states': имя поля -> {имя атрибута -> значение}.
_INITIAL_STATES_ATTRS: Mapping[str, Mapping[str, str]] = {
    'emission_position': {'units': 'mm'},
    'emission_energy': {'units': 'MeV'},
    'energy': {'units': 'MeV'},
    'emission_time': {'units': 'ns'},
    'time': {'units': 'ns'},
}


# Поля физического времени, всегда сохраняемые строго в float64 для исключения потери точности наносекунд
_TIME_FIELDS: FrozenSet[str] = frozenset({'emission_time', 'time'})


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

    @property
    @abc.abstractmethod
    def consumed_chunk_types(self) -> FrozenSet[ChunkType]:
        """
        Множество типов чанков данных (ChunkType), потребляемых данным обработчиком.
        """
        pass

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

    def close(self) -> None:
        """
        Освобождение внешних ресурсов обработчика (например, SharedMemory) при завершении работы.
        По умолчанию не выполняет действий, переопределяется в специализированных подклассах.
        """
        pass


class DirectStreamHandler(BaseDataHandler):
    """
    Обработчик прямой потоковой записи всех начальных состояний и взаимодействий частиц в HDF5.
    """

    _CONSUMED_CHUNK_TYPES: FrozenSet[ChunkType] = frozenset({
        ChunkType.INITIAL_STATES,
        ChunkType.INTERACTIONS,
    })

    def __init__(
        self,
        floating_point_precision: str = "float32",
        compression: Optional[str] = None,
        compression_opts: int = 4,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """
        :param buffer_size: Объём буфера записи на группу датасетов в байтах;
            задаёт и порог сброса в файл, и размер чанка HDF5.
        """
        super().__init__(
            floating_point_precision=floating_point_precision,
            compression=compression,
            compression_opts=compression_opts,
        )
        self._write_buffer = DatasetWriteBuffer(buffer_size, compression, compression_opts)
        self._interactions_attrs: Mapping[str, Mapping[str, str]] = self._build_interactions_attrs()

    @staticmethod
    def _build_interactions_attrs() -> Mapping[str, Mapping[str, str]]:
        """
        Схема атрибутов датасетов групп 'interactions/<объём>': имя поля -> {имя атрибута -> значение}.
        Вычисляется один раз при создании обработчика.
        """
        return {
            'process_code': {'mapping': json.dumps(get_process_mapping())},
            'species_code': {'mapping': json.dumps(get_species_mapping())},
            'global_position': {'units': 'mm'},
            'energy_deposit': {'units': 'MeV'},
            'distance_traveled': {'units': 'mm'},
            'scattering_angles': {'units': 'rad'},
        }

    @property
    def consumed_chunk_types(self) -> FrozenSet[ChunkType]:
        return self._CONSUMED_CHUNK_TYPES

    def _format_raw_interactions(
        self, interactions: Dict[str, np.ndarray], fill_missing: bool = False
    ) -> Dict[str, np.ndarray]:
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

        if fill_missing:
            n_events = len(next(iter(interactions.values()))) if interactions else 0
            for def_key, def_arr in (
                ('global_position', np.zeros((n_events, 3), dtype=target_float_dtype)),
                ('global_direction', np.zeros((n_events, 3), dtype=target_float_dtype)),
                ('process_code', np.zeros(n_events, dtype=np.uint8)),
                ('species_code', np.zeros(n_events, dtype=np.uint8)),
                ('energy_deposit', np.zeros(n_events, dtype=target_float_dtype)),
                ('scattering_angles', np.zeros((n_events, 2), dtype=target_float_dtype)),
                ('material_id', np.zeros(n_events, dtype=np.uint32)),
                ('Z', np.zeros(n_events, dtype=np.int8)),
                ('distance_traveled', np.zeros(n_events, dtype=target_float_dtype)),
            ):
                result.setdefault(def_key, def_arr)

        return result

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        data = chunk.get('data')
        if data is None:
            return

        if chunk_type == ChunkType.INITIAL_STATES:
            self._write_initial_states(data)
        elif chunk_type == ChunkType.INTERACTIONS:
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
                if state_key in _TIME_FIELDS:
                    prepared_states[state_key] = state_arr.astype(np.float64, copy=False)
                else:
                    prepared_states[state_key] = state_arr.astype(target_float_dtype, copy=False)
            else:
                prepared_states[state_key] = state_arr

        self._write_buffer.append('initial_states', prepared_states, _INITIAL_STATES_ATTRS)
        self._flush_full_chunks()

    def _write_interactions(self, volume_data_map: Dict[str, Dict[str, np.ndarray]]) -> None:
        for volume_name, volume_data in volume_data_map.items():
            self._write_buffer.append(f'interactions/{volume_name}', volume_data, self._interactions_attrs)
        self._flush_full_chunks()

    def _flush_full_chunks(self) -> None:
        """
        Передаёт в писатель сброс целых чанков, если хотя бы в одной группе накоплено их достаточно.
        Без зарегистрированного writer_callback запись не выполняется.
        """
        if self._write_buffer.has_full_chunks and self.writer_callback is not None:
            self.writer_callback(self._write_buffer.write_full_chunks)

    def finalize(self) -> None:
        """
        Сбрасывает в файл всё накопленное в буфере записи, включая неполные хвосты групп.
        """
        if self.writer_callback is not None:
            self.writer_callback(self._write_buffer.write_all)


class SensitiveVolumeHandler(DirectStreamHandler):
    """
    Обработчик записи взаимодействий частиц только в заданных чувствительных объемах (и их поддеревьях).
    """

    _CONSUMED_CHUNK_TYPES: FrozenSet[ChunkType] = frozenset({ChunkType.INTERACTIONS})

    def __init__(
        self,
        sensitive_volumes: List[SpatialNode],
        floating_point_precision: str = "float32",
        compression: Optional[str] = None,
        compression_opts: int = 4,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        super().__init__(
            floating_point_precision=floating_point_precision,
            compression=compression,
            compression_opts=compression_opts,
            buffer_size=buffer_size,
        )
        self.sensitive_volumes = self._filter_topmost_nodes(sensitive_volumes)

        unique_roots: Set[SpatialNode] = set()
        self.unique_top_volumes: Set[SpatialNode] = set()
        for node in self.sensitive_volumes:
            unique_roots.add(node.root)
            if isinstance(node, Volume):
                self.unique_top_volumes.add(node.top_volume)
            else:
                self.unique_top_volumes.add(node)

        self.scene_root: Optional[SpatialNode] = None
        if len(unique_roots) > 1:
            raise ValueError("All sensitive volumes must share the same root simulation volume!")
        elif len(unique_roots) == 1:
            self.scene_root = unique_roots.pop()

        self.volume_mapping: Dict[int, SpatialNode] = {}
        self.target_volumes: List[int] = []
        self._build_volume_mapping()

    @staticmethod
    def _filter_topmost_nodes(nodes: Sequence[SpatialNode]) -> List[SpatialNode]:
        """
        Отсекает дочерние узлы, если их родительские узлы (предки) уже присутствуют в списке.
        Сохраняет порядок первого вхождения верхних узлов.
        """
        unique_nodes: List[SpatialNode] = list(dict.fromkeys(nodes))
        node_set = set(unique_nodes)
        filtered_nodes: List[SpatialNode] = []
        for node in unique_nodes:
            is_child_of_selected = False
            current = node.parent
            while current is not None:
                if current in node_set:
                    is_child_of_selected = True
                    break
                current = current.parent
            if not is_child_of_selected:
                filtered_nodes.append(node)
        return filtered_nodes

    def _build_volume_mapping(self) -> None:
        """
        Формирует маппинг индексов объемов FlattenedScene на целевые узлы записи.
        Чувствительные узлы (Volume / CompositeNode) и их дочерние подузлы связываются
        с соответствующим верхним чувствительным узлом для сохранения.
        Все остальные фоновые объемы сцены связываются с корнем сцены (scene_root).
        """
        if self.scene_root is not None and isinstance(self.scene_root, CompositeNode):
            flat_list = FlattenedScene(self.scene_root).flat_list

            sensitive_set = set(self.sensitive_volumes)
            target_ids: Set[int] = set()

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
        data = chunk.get('data')
        if chunk_type == ChunkType.INTERACTIONS and data is not None:
            self._process_interactions(data)

    def _process_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        volume_ids = interactions['volume_id']
        mask_sensitive = np.isin(volume_ids, self.target_volumes)

        if np.any(mask_sensitive):
            interactions_to_write = {field_key: field_val[mask_sensitive] for field_key, field_val in interactions.items()}
            self._format_and_write_interactions(interactions_to_write)

    def _get_volumes_to_write(self) -> List[SpatialNode]:
        return list(self.sensitive_volumes)

    def _format_and_write_interactions(self, interactions: Dict[str, np.ndarray]) -> None:
        formatted = self._format_raw_interactions(interactions, fill_missing=True)
        volume_ids = interactions['volume_id']
        root_vol_names = np.array(
            [self.volume_mapping[vid].name if vid in self.volume_mapping else None for vid in volume_ids],
            dtype=object,
        )
        volume_data_map: Dict[str, Dict[str, np.ndarray]] = {}
        for top_volume in self._get_volumes_to_write():
            if not top_volume.name:
                continue
            mask = root_vol_names == top_volume.name
            if np.any(mask):
                volume_data_map[top_volume.name] = {field_key: field_arr[mask] for field_key, field_arr in formatted.items()}

        if volume_data_map:
            self._write_interactions(volume_data_map)


class HistoryAssemblerHandler(SensitiveVolumeHandler):
    """
    Обработчик сборки полных историй частиц, попавших в чувствительные объемы детекторов.
    Накапливает начальные состояния и взаимодействия до гибели частицы и сбрасывает в HDF5
    полную историю только для зарегистрированных частиц.
    """

    _CONSUMED_WITH_INITIAL: FrozenSet[ChunkType] = frozenset({
        ChunkType.INITIAL_STATES,
        ChunkType.INTERACTIONS,
        ChunkType.DEAD_PARTICLES,
    })
    _CONSUMED_WITHOUT_INITIAL: FrozenSet[ChunkType] = frozenset({
        ChunkType.INTERACTIONS,
        ChunkType.DEAD_PARTICLES,
    })

    def __init__(
        self,
        sensitive_volumes: List[SpatialNode],
        save_initial_states: bool = True,
        floating_point_precision: str = "float32",
        compression: Optional[str] = None,
        compression_opts: int = 4,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        super().__init__(
            sensitive_volumes=sensitive_volumes,
            floating_point_precision=floating_point_precision,
            compression=compression,
            compression_opts=compression_opts,
            buffer_size=buffer_size,
        )
        self.save_initial_states = save_initial_states

        self.initial_states_chunks: List[Dict[str, np.ndarray]] = []
        self.interactions_chunks: List[Dict[str, np.ndarray]] = []
        self.scored_particles: Set[int] = set()

    @property
    def consumed_chunk_types(self) -> FrozenSet[ChunkType]:
        return self._CONSUMED_WITH_INITIAL if self.save_initial_states else self._CONSUMED_WITHOUT_INITIAL

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_type = chunk.get('type')
        data = chunk.get('data')
        if data is None:
            return

        if chunk_type == ChunkType.INITIAL_STATES:
            if self.save_initial_states:
                self.initial_states_chunks.append({key: np.array(val, copy=False) for key, val in data.items()})
        elif chunk_type == ChunkType.INTERACTIONS:
            vol_ids = data['volume_id']
            scored_mask = np.isin(vol_ids, self.target_volumes)
            if np.any(scored_mask):
                self.scored_particles.update(data['particle_ID'][scored_mask].tolist())
            self.interactions_chunks.append({key: np.array(val, copy=False) for key, val in data.items()})
        elif chunk_type == ChunkType.DEAD_PARTICLES:
            dead_ids = np.asarray(data)
            if len(dead_ids) > 0:
                self._flush_dead_particles(dead_ids)

    def _get_volumes_to_write(self) -> List[SpatialNode]:
        volumes_to_write: List[SpatialNode] = list(self.sensitive_volumes)
        if self.scene_root is not None and self.scene_root not in volumes_to_write:
            volumes_to_write.append(self.scene_root)
        unique_nodes: List[SpatialNode] = []
        seen_names: Set[Optional[str]] = set()
        for node in volumes_to_write:
            if node.name not in seen_names:
                unique_nodes.append(node)
                seen_names.add(node.name)
        return unique_nodes

    def _flush_dead_particles(self, dead_ids: np.ndarray) -> None:
        scored_dead_ids = np.intersect1d(dead_ids, list(self.scored_particles))

        initial_states_to_write = None
        if self.initial_states_chunks:
            mega_init = {
                key: np.concatenate([chunk_item[key] for chunk_item in self.initial_states_chunks])
                for key in self.initial_states_chunks[0].keys()
            }

            if len(scored_dead_ids) > 0:
                scored_mask = np.isin(mega_init['particle_ID'], scored_dead_ids)
                if np.any(scored_mask):
                    initial_states_to_write = {key: val[scored_mask] for key, val in mega_init.items()}

            survivor_mask = ~np.isin(mega_init['particle_ID'], dead_ids)
            if np.any(survivor_mask):
                self.initial_states_chunks = [{key: val[survivor_mask] for key, val in mega_init.items()}]
            else:
                self.initial_states_chunks.clear()

        interactions_to_write = None
        if self.interactions_chunks:
            mega_inter = {
                key: np.concatenate([chunk_item[key] for chunk_item in self.interactions_chunks])
                for key in self.interactions_chunks[0].keys()
            }

            if len(scored_dead_ids) > 0:
                scored_mask = np.isin(mega_inter['particle_ID'], scored_dead_ids)
                if np.any(scored_mask):
                    interactions_to_write = {key: val[scored_mask] for key, val in mega_inter.items()}
                    sort_idx = np.argsort(interactions_to_write['particle_ID'])
                    interactions_to_write = {key: val[sort_idx] for key, val in interactions_to_write.items()}

            survivor_mask = ~np.isin(mega_inter['particle_ID'], dead_ids)
            if np.any(survivor_mask):
                self.interactions_chunks = [{key: val[survivor_mask] for key, val in mega_inter.items()}]
            else:
                self.interactions_chunks.clear()

        self.scored_particles.difference_update(dead_ids.tolist())

        if initial_states_to_write is not None:
            self._write_initial_states(initial_states_to_write)
        if interactions_to_write is not None:
            self._format_and_write_interactions(interactions_to_write)

    def finalize(self) -> None:
        """
        Принудительный сброс оставшихся в буферах историй частиц при завершении симуляции
        и очистка внутренних накопителей; затем запись всего накопленного в буфере записи в файл.
        """
        if self.scored_particles or self.initial_states_chunks or self.interactions_chunks:
            id_arrays: List[np.ndarray] = []
            if self.scored_particles:
                id_arrays.append(np.fromiter(self.scored_particles, dtype=np.int64))
            for init_chunk in self.initial_states_chunks:
                if "particle_ID" in init_chunk and len(init_chunk["particle_ID"]) > 0:
                    id_arrays.append(np.asarray(init_chunk["particle_ID"], dtype=np.int64))
            for inter_chunk in self.interactions_chunks:
                if "particle_ID" in inter_chunk and len(inter_chunk["particle_ID"]) > 0:
                    id_arrays.append(np.asarray(inter_chunk["particle_ID"], dtype=np.int64))

            if id_arrays:
                all_remaining_ids = np.unique(np.concatenate(id_arrays))
                if len(all_remaining_ids) > 0:
                    self._flush_dead_particles(all_remaining_ids)

        self.initial_states_chunks.clear()
        self.interactions_chunks.clear()
        self.scored_particles.clear()

        super().finalize()


__all__ = [
    "BaseDataHandler",
    "DirectStreamHandler",
    "SensitiveVolumeHandler",
    "HistoryAssemblerHandler",
]

