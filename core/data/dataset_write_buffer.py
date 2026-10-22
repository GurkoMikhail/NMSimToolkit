"""
Буфер пакетной дозаписи строк в группы датасетов HDF5.

Модуль накапливает порции данных в оперативной памяти и сбрасывает их в файл
только целыми чанками, поэтому чанк HDF5 определяется настройкой ``buffer_size``,
а не размером первой порции. Модуль не зависит от Qt/VTK и не содержит
рефлексии над объектами (LBYL, Fail-Fast).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import h5py
import numpy as np

# Объём буфера на группу датасетов по умолчанию, байт (3 МиБ);
# значение совпадает с DataManagerConfig.buffer_size.
DEFAULT_BUFFER_SIZE: int = 3 * 1024 ** 2


@dataclass(frozen=True)
class DatasetLayout:
    """
    Неизменяемая схема датасета, фиксируемая при первой непустой порции группы.

    Attributes:
        dtype: Тип элементов датасета.
        row_shape: Форма одной строки (shape без первой оси); пустой кортеж для одномерного датасета.
        attrs: Атрибуты, выставляемые датасету при его создании в файле.
    """

    dtype: np.dtype
    row_shape: Tuple[int, ...]
    attrs: Mapping[str, str]

    @property
    def row_bytes(self) -> int:
        """Размер одной строки датасета в байтах (произведение пустого кортежа равно 1)."""
        return int(self.dtype.itemsize * int(np.prod(self.row_shape, dtype=np.int64)))


@dataclass
class _GroupState:
    """
    Изменяемое состояние одной группы датасетов внутри буфера.

    Attributes:
        layouts: Схема каждого поля группы (в порядке первой порции).
        chunk_rows: Эффективное число строк в чанке. Изначально ``max(1, buffer_size // row_bytes)``;
            после первого обращения к существующим в файле датасетам заменяется на их ``chunks[0]``.
        pending_parts: Накопленные порции по полям (списки массивов; конкатенируются только при сбросе).
        pending_rows: Число накопленных, ещё не записанных строк (одинаково для всех полей группы).
    """

    layouts: Dict[str, DatasetLayout]
    chunk_rows: int
    pending_parts: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    pending_rows: int = 0


class DatasetWriteBuffer:
    """
    Буфер дозаписи строк в группы датасетов HDF5 ('initial_states', 'interactions/<объём>').

    Для каждой группы ``chunk_rows = max(1, buffer_size // row_bytes)``, где ``row_bytes`` —
    суммарный размер одной строки по всем полям группы. Сбрасываются только целые чанки
    (с учётом длины уже существующих в файле датасетов), неполный хвост дописывается
    в ``write_all``.

    Договорённости:
        - Схема группы (набор полей, ``dtype`` и ``row_shape`` каждого поля) фиксируется первой
          непустой порцией; последующие порции обязаны ей соответствовать (иначе ``ValueError``).
        - Пустая порция (0 строк) проверяется на равенство длин полей и, если группа уже известна,
          на соответствие схеме; в остальном она ничего не меняет: схему новой группы не фиксирует
          и датасеты в файле не создаёт.
        - Атрибуты датасетов берутся из первой непустой порции группы и выставляются только при
          создании датасета в файле; существующие датасеты не изменяются. Атрибуты полей, которых
          нет в порции, игнорируются; атрибуты последующих порций игнорируются.
        - Буфер хранит ссылки на переданные массивы без копирования: вызывающий НЕ должен
          модифицировать массивы после ``append`` (read-only-вью допустимы).
        - Все поля группы сбрасываются одинаковым числом строк, поэтому их длины в файле всегда равны.
    """

    def __init__(self, buffer_size: int, compression: Optional[str], compression_opts: int) -> None:
        """
        Args:
            buffer_size: Объём буфера на группу в байтах (>= 1); задаёт размер чанка и порог сброса.
            compression: Имя фильтра сжатия HDF5 (например, 'gzip') либо ``None`` без сжатия.
            compression_opts: Параметр фильтра; передаётся в h5py только для ``compression == 'gzip'``.

        Raises:
            ValueError: ``buffer_size < 1``.
        """
        if buffer_size < 1:
            raise ValueError(f"buffer_size должен быть >= 1 байта, получено {buffer_size}.")
        self._buffer_size: int = int(buffer_size)
        self._compression: Optional[str] = compression
        self._compression_opts: int = compression_opts
        self._groups: Dict[str, _GroupState] = {}

    def append(
        self,
        group_path: str,
        fields: Mapping[str, np.ndarray],
        attrs: Mapping[str, Mapping[str, str]],
    ) -> None:
        """
        Добавляет порцию строк в буфер группы.

        Args:
            group_path: Путь группы в файле ('initial_states', 'interactions/Detector_1').
            fields: Поля порции: имя датасета -> массив, первая ось — строки.
            attrs: Атрибуты датасетов: имя поля -> {имя атрибута -> значение}.

        Raises:
            ValueError: Нет полей; массив нульмерный; длины полей по оси 0 различаются; набор полей,
                ``dtype`` или ``row_shape`` не совпадают со схемой группы; строка поля нулевого размера.
        """
        if not fields:
            raise ValueError(f"Порция группы '{group_path}' не содержит ни одного поля.")
        for field_name, array in fields.items():
            if array.ndim < 1:
                raise ValueError(
                    f"Поле '{field_name}' группы '{group_path}' нульмерное: первая ось обязана нумеровать строки."
                )

        portion_rows = {array.shape[0] for array in fields.values()}
        if len(portion_rows) != 1:
            lengths = {field_name: array.shape[0] for field_name, array in fields.items()}
            raise ValueError(f"Длины полей порции группы '{group_path}' различаются: {lengths}.")
        row_count = portion_rows.pop()

        state = self._groups.get(group_path)
        if state is not None:
            self._validate_against_layouts(group_path, state, fields)
        if row_count == 0:
            return

        if state is None:
            state = self._create_group_state(group_path, fields, attrs)
            self._groups[group_path] = state

        for field_name, array in fields.items():
            state.pending_parts.setdefault(field_name, []).append(array)
        state.pending_rows += row_count

    @property
    def has_full_chunks(self) -> bool:
        """
        Истина, если хотя бы в одной группе накоплено не меньше ``chunk_rows`` строк.

        Критерий использует эффективный ``chunk_rows`` группы: до первого обращения к файлу это
        значение по ``buffer_size``, после — ``chunks[0]`` существующих датасетов. Фактическое число
        записываемых строк определяет ``write_full_chunks`` по реальной длине датасетов в файле.
        """
        return any(state.pending_rows >= state.chunk_rows for state in self._groups.values())

    def pending_rows(self, group_path: str) -> int:
        """Число накопленных, ещё не записанных строк группы (0 для неизвестной группы)."""
        state = self._groups.get(group_path)
        if state is None:
            return 0
        return state.pending_rows

    def chunk_rows(self, group_path: str) -> int:
        """
        Эффективное число строк в чанке группы.

        Raises:
            KeyError: Группа ещё не получила ни одной непустой порции.
        """
        if group_path not in self._groups:
            raise KeyError(f"Группа '{group_path}' неизвестна буферу: схема не зафиксирована.")
        return self._groups[group_path].chunk_rows

    def write_full_chunks(self, h5_file: h5py.File) -> None:
        """
        Сбрасывает во всех группах максимальное число строк, завершающих целые чанки.

        Для группы с длиной датасетов в файле ``file_rows`` и накопленными ``pending_rows``
        записывается ``n = ((file_rows + pending_rows) // chunk_rows) * chunk_rows - file_rows``
        первых строк (если ``n > 0``); остаток сохраняется в буфере.

        Raises:
            RuntimeError: Датасеты группы в файле неконсистентны (разная длина, частичный набор, нет чанкирования).
            ValueError: ``dtype`` или ``row_shape`` существующих датасетов не совпадают со схемой.
        """
        for group_path, state in self._groups.items():
            if state.pending_rows == 0:
                continue
            self._flush_group(h5_file, group_path, state, flush_everything=False)

    def write_all(self, h5_file: h5py.File) -> None:
        """
        Сбрасывает всё накопленное во всех группах, включая неполные хвосты (используется в finalize).

        Raises:
            RuntimeError: Датасеты группы в файле неконсистентны (разная длина, частичный набор, нет чанкирования).
            ValueError: ``dtype`` или ``row_shape`` существующих датасетов не совпадают со схемой.
        """
        for group_path, state in self._groups.items():
            if state.pending_rows == 0:
                continue
            self._flush_group(h5_file, group_path, state, flush_everything=True)

    def _create_group_state(
        self,
        group_path: str,
        fields: Mapping[str, np.ndarray],
        attrs: Mapping[str, Mapping[str, str]],
    ) -> _GroupState:
        """Фиксирует схему новой группы по первой непустой порции и вычисляет ``chunk_rows``."""
        layouts: Dict[str, DatasetLayout] = {}
        for field_name, array in fields.items():
            row_shape = tuple(int(dim) for dim in array.shape[1:])
            if any(dim < 1 for dim in row_shape):
                raise ValueError(
                    f"Поле '{field_name}' группы '{group_path}' имеет вырожденную форму строки {row_shape}."
                )
            field_attrs = dict(attrs[field_name]) if field_name in attrs else {}
            layouts[field_name] = DatasetLayout(dtype=array.dtype, row_shape=row_shape, attrs=field_attrs)

        row_bytes = sum(layout.row_bytes for layout in layouts.values())
        chunk_rows = max(1, self._buffer_size // row_bytes)
        return _GroupState(layouts=layouts, chunk_rows=chunk_rows)

    @staticmethod
    def _validate_against_layouts(
        group_path: str,
        state: _GroupState,
        fields: Mapping[str, np.ndarray],
    ) -> None:
        """Проверяет, что порция совпадает со схемой группы (набор полей, dtype, row_shape)."""
        if set(fields) != set(state.layouts):
            raise ValueError(
                f"Набор полей порции группы '{group_path}' {sorted(fields)} "
                f"не совпадает со схемой {sorted(state.layouts)}."
            )
        for field_name, array in fields.items():
            layout = state.layouts[field_name]
            if array.dtype != layout.dtype:
                raise ValueError(
                    f"dtype поля '{field_name}' группы '{group_path}': {array.dtype} вместо {layout.dtype}."
                )
            row_shape = tuple(int(dim) for dim in array.shape[1:])
            if row_shape != layout.row_shape:
                raise ValueError(
                    f"row_shape поля '{field_name}' группы '{group_path}': {row_shape} вместо {layout.row_shape}."
                )

    @staticmethod
    def _dataset_path(group_path: str, field_name: str) -> str:
        """Полный путь датасета в файле."""
        return f"{group_path}/{field_name}"

    def _inspect_existing_datasets(
        self,
        h5_file: h5py.File,
        group_path: str,
        state: _GroupState,
    ) -> Tuple[int, Optional[int]]:
        """
        Определяет состояние группы в файле.

        Returns:
            Кортеж ``(file_rows, existing_chunk_rows)``: длина датасетов группы в файле (0, если их нет)
            и ``chunks[0]`` существующих датасетов (``None``, если датасетов нет).

        Raises:
            RuntimeError: Присутствует лишь часть полей, длины или чанкирование полей различаются,
                датасет не чанкирован.
            ValueError: ``dtype`` или ``row_shape`` существующего датасета не совпадают со схемой.
        """
        present = [
            field_name
            for field_name in state.layouts
            if self._dataset_path(group_path, field_name) in h5_file
        ]
        if not present:
            return 0, None
        if len(present) != len(state.layouts):
            missing = sorted(set(state.layouts) - set(present))
            raise RuntimeError(
                f"Группа '{group_path}' в файле неконсистентна: отсутствуют датасеты {missing}."
            )

        lengths: Dict[str, int] = {}
        chunk_heads: Dict[str, int] = {}
        for field_name, layout in state.layouts.items():
            dataset = h5_file[self._dataset_path(group_path, field_name)]
            if dataset.dtype != layout.dtype:
                raise ValueError(
                    f"dtype датасета '{group_path}/{field_name}' в файле: {dataset.dtype}, ожидается {layout.dtype}."
                )
            if tuple(dataset.shape[1:]) != layout.row_shape:
                raise ValueError(
                    f"row_shape датасета '{group_path}/{field_name}' в файле: {tuple(dataset.shape[1:])}, "
                    f"ожидается {layout.row_shape}."
                )
            if dataset.chunks is None:
                raise RuntimeError(f"Датасет '{group_path}/{field_name}' в файле не чанкирован: дозапись невозможна.")
            lengths[field_name] = int(dataset.shape[0])
            chunk_heads[field_name] = int(dataset.chunks[0])

        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Длины датасетов группы '{group_path}' в файле различаются: {lengths}.")
        if len(set(chunk_heads.values())) != 1:
            raise RuntimeError(f"Размеры чанков датасетов группы '{group_path}' в файле различаются: {chunk_heads}.")
        return next(iter(lengths.values())), next(iter(chunk_heads.values()))

    def _flush_group(
        self,
        h5_file: h5py.File,
        group_path: str,
        state: _GroupState,
        flush_everything: bool,
    ) -> None:
        """
        Записывает первые ``n`` накопленных строк группы во все её поля и оставляет остаток в буфере.

        Состояние буфера обновляется только после успешной записи всех полей.
        """
        file_rows, existing_chunk_rows = self._inspect_existing_datasets(h5_file, group_path, state)
        if existing_chunk_rows is not None:
            state.chunk_rows = existing_chunk_rows

        if flush_everything:
            rows_to_write = state.pending_rows
        else:
            aligned_end = ((file_rows + state.pending_rows) // state.chunk_rows) * state.chunk_rows
            rows_to_write = aligned_end - file_rows
        if rows_to_write <= 0:
            return

        remaining_parts: Dict[str, List[np.ndarray]] = {}
        for field_name, layout in state.layouts.items():
            parts = state.pending_parts[field_name]
            merged = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            head = merged[:rows_to_write]
            tail = merged[rows_to_write:]
            self._write_field(h5_file, group_path, field_name, layout, state.chunk_rows, file_rows, head)
            remaining_parts[field_name] = [tail] if tail.shape[0] > 0 else []

        state.pending_parts = remaining_parts
        state.pending_rows -= rows_to_write

    def _write_field(
        self,
        h5_file: h5py.File,
        group_path: str,
        field_name: str,
        layout: DatasetLayout,
        chunk_rows: int,
        file_rows: int,
        rows: np.ndarray,
    ) -> None:
        """Создаёт датасет поля (если его нет) либо один раз расширяет существующий и дописывает строки."""
        dataset_path = self._dataset_path(group_path, field_name)
        if dataset_path not in h5_file:
            group = h5_file.require_group(group_path)
            creation_kwargs: Dict[str, object] = {}
            if self._compression is not None:
                creation_kwargs["compression"] = self._compression
                if self._compression == "gzip":
                    creation_kwargs["compression_opts"] = self._compression_opts
            dataset = group.create_dataset(
                field_name,
                data=rows,
                dtype=layout.dtype,
                maxshape=(None, *layout.row_shape),
                chunks=(chunk_rows, *layout.row_shape),
                **creation_kwargs,
            )
            for attr_name, attr_value in layout.attrs.items():
                dataset.attrs[attr_name] = attr_value
            return

        dataset = h5_file[dataset_path]
        dataset.resize(file_rows + rows.shape[0], axis=0)
        dataset[file_rows:file_rows + rows.shape[0]] = rows


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DatasetLayout",
    "DatasetWriteBuffer",
]