"""
Модуль детерминированной классификации историй фотонов, прошедших через коллиматор,
на 3 физические компоненты:
- GEOMETRIC (0): чисто геометрическое прохождение внутри одного канала без взаимодействий;
- SEPTAL_PENETRATION (1): прямое проникновение сквозь свинцовые септы без рассеяния;
- COLLIMATOR_SCATTER (2): фотоны, испытавшие комптоновское или когерентное рассеяние в свинце коллиматора.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, FrozenSet, List, Tuple
import numpy as np
from numpy.typing import NDArray

from core.data.data_handlers import BaseDataHandler
from core.other.typing_definitions import Float, Index
from core.physics.interaction_buffers import ChunkType


class PhotonComponent(IntEnum):
    """Код физической компоненты прошедшего через коллиматор излучения."""

    GEOMETRIC = 0
    SEPTAL_PENETRATION = 1
    COLLIMATOR_SCATTER = 2


@dataclass(frozen=True)
class ClassifiedExitPhotons:
    """
    Структура классифицированных фотонов на выходной плоскости коллиматора z = z_exit.
    Все координаты выражены в мм, энергии — в МэВ, углы — в рад.
    """

    particle_id: NDArray[np.int64]
    exit_x: NDArray[Float]
    exit_y: NDArray[Float]
    relative_x: NDArray[Float]
    relative_y: NDArray[Float]
    radial_distance: NDArray[Float]
    azimuthal_angle: NDArray[Float]
    component: NDArray[np.int8]
    final_energy: NDArray[Float]
    compton_count: NDArray[np.int32]
    coherent_count: NDArray[np.int32]
    total_emitted: int
    solid_angle_fraction: Float

    @property
    def equivalent_4pi_emitted(self) -> Float:
        """Эквивалентное число распадов изотропного 4*pi источника."""
        if self.solid_angle_fraction <= 0.0:
            return Float(self.total_emitted)
        return Float(self.total_emitted / self.solid_angle_fraction)

    def mask_for(self, component_kind: PhotonComponent) -> NDArray[np.bool_]:
        """Возвращает булеву маску событий выбранной физической компоненты."""
        return np.asarray(self.component == int(component_kind), dtype=np.bool_)


def locate_hexagonal_cell_and_hole(
    coord_x: NDArray[Float],
    coord_y: NDArray[Float],
    hole_diameter: Float | float,
    septa: Float | float,
    margin_epsilon: Float | float = Float(1e-7),
) -> Tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64], NDArray[np.int8]]:
    """
    Векторизованно определяет для массива 2D-координат (coord_x, coord_y) в системе коллиматора:
    1. Принадлежность точки шестиугольному отверстию (is_in_hole);
    2. Индексы ближайшей ячейки Вороного центрированной гексагональной решетки:
       (cell_index_x, cell_index_y, sublattice_id), где sublattice_id in {0, 1}.

    Математика полностью идентична кернелу _hex_prism_intersect и ParametricParallelCollimator.
    """
    diameter_val = Float(hole_diameter)
    septa_val = Float(septa)
    eps_val = Float(margin_epsilon)

    period_x = Float(diameter_val + septa_val)
    period_y = Float(np.sqrt(3.0) * period_x)
    half_width = Float(0.5 * diameter_val + eps_val)
    side_limit = Float(diameter_val + 2.0 * eps_val)
    sqrt_three = Float(np.sqrt(3.0))

    # Подрешетка 0: узлы (i * period_x, j * period_y)
    idx_x_first = np.round(coord_x / period_x).astype(np.int64)
    idx_y_first = np.round(coord_y / period_y).astype(np.int64)
    center_x_first = idx_x_first.astype(Float) * period_x
    center_y_first = idx_y_first.astype(Float) * period_y
    delta_x_first = coord_x - center_x_first
    delta_y_first = coord_y - center_y_first
    dist_sq_first = delta_x_first * delta_x_first + delta_y_first * delta_y_first

    # Подрешетка 1: узлы ((i + 0.5) * period_x, (j + 0.5) * period_y)
    shift_x = Float(0.5 * period_x)
    shift_y = Float(0.5 * period_y)
    idx_x_second = np.round((coord_x - shift_x) / period_x).astype(np.int64)
    idx_y_second = np.round((coord_y - shift_y) / period_y).astype(np.int64)
    center_x_second = idx_x_second.astype(Float) * period_x + shift_x
    center_y_second = idx_y_second.astype(Float) * period_y + shift_y
    delta_x_second = coord_x - center_x_second
    delta_y_second = coord_y - center_y_second
    dist_sq_second = delta_x_second * delta_x_second + delta_y_second * delta_y_second

    choose_first = dist_sq_first <= dist_sq_second

    cell_x = np.where(choose_first, delta_x_first, delta_x_second)
    cell_y = np.where(choose_first, delta_y_first, delta_y_second)
    cell_idx_x = np.where(choose_first, idx_x_first, idx_x_second)
    cell_idx_y = np.where(choose_first, idx_y_first, idx_y_second)
    sublattice = np.where(choose_first, np.int8(0), np.int8(1))

    abs_x = np.abs(cell_x)
    abs_y = np.abs(cell_y)
    is_in_hole = (abs_x <= half_width) & ((abs_x + sqrt_three * abs_y) <= side_limit)

    return (
        np.asarray(is_in_hole, dtype=np.bool_),
        np.asarray(cell_idx_x, dtype=np.int64),
        np.asarray(cell_idx_y, dtype=np.int64),
        np.asarray(sublattice, dtype=np.int8),
    )


class CollimatorExitPlaneHandler(BaseDataHandler):
    """
    Потоковый обработчик регистрации и классификации фотонов на выходном торце коллиматора (z = z_exit).
    Перехватывает пакеты 'escaped_particles', 'initial_states' и 'interactions' из очереди SimulationManager,
    отбирает все фотоны, вышедшие через выходную апертуру коллиматора в сторону детектора,
    и классифицирует их на 3 физические компоненты (Geometric, Septal Penetration, Collimator Scatter).
    """

    _z_enter_face: Float
    _z_exit_face: Float
    _aperture_half_x: Float
    _aperture_half_y: Float
    _hole_diameter: Float
    _septa: Float
    _primary_energy: Float
    _total_emitted: int
    _solid_angle_fraction: Float

    _escaped_chunks: List[Dict[str, NDArray[Any]]]
    _initial_chunks: List[Dict[str, NDArray[Any]]]
    _scatter_chunks: List[Dict[str, NDArray[Any]]]
    _classified_result: ClassifiedExitPhotons | None

    _CONSUMED_CHUNK_TYPES: FrozenSet[ChunkType] = frozenset({
        ChunkType.INITIAL_STATES,
        ChunkType.INTERACTIONS,
        ChunkType.ESCAPED_PARTICLES,
    })

    @property
    def consumed_chunk_types(self) -> FrozenSet[ChunkType]:
        return self._CONSUMED_CHUNK_TYPES

    def __init__(
        self,
        z_enter_face: Float | float,
        z_exit_face: Float | float,
        aperture_size_x: Float | float,
        aperture_size_y: Float | float,
        hole_diameter: Float | float,
        septa: Float | float,
        primary_energy: Float | float,
        total_emitted: int,
        solid_angle_fraction: Float | float = Float(1.0),
    ) -> None:
        super().__init__(floating_point_precision="float64", compression=None)
        if Float(z_exit_face) <= Float(z_enter_face):
            raise ValueError(
                f"Координата выходного торца z_exit_face ({z_exit_face}) должна быть больше "
                f"входного торца z_enter_face ({z_enter_face})"
            )
        if Float(aperture_size_x) <= 0.0 or Float(aperture_size_y) <= 0.0:
            raise ValueError(
                f"Размеры апертуры должны быть положительными: ({aperture_size_x}, {aperture_size_y})"
            )
        if Float(hole_diameter) <= 0.0 or Float(septa) <= 0.0:
            raise ValueError(
                f"Параметры решетки hole_diameter ({hole_diameter}) и septa ({septa}) должны быть > 0"
            )
        if total_emitted <= 0:
            raise ValueError(f"Общее число испущенных частиц должно быть > 0, получено {total_emitted}")

        self._z_enter_face = Float(z_enter_face)
        self._z_exit_face = Float(z_exit_face)
        self._aperture_half_x = Float(0.5 * Float(aperture_size_x))
        self._aperture_half_y = Float(0.5 * Float(aperture_size_y))
        self._hole_diameter = Float(hole_diameter)
        self._septa = Float(septa)
        self._primary_energy = Float(primary_energy)
        self._total_emitted = int(total_emitted)
        self._solid_angle_fraction = Float(solid_angle_fraction)

        self._escaped_chunks = []
        self._initial_chunks = []
        self._scatter_chunks = []
        self._classified_result = None

    @property
    def classified_result(self) -> ClassifiedExitPhotons:
        """Возвращает финализированный результат классификации вышедших из коллиматора фотонов."""
        if self._classified_result is None:
            self.finalize()
        if self._classified_result is None:
            raise RuntimeError("Не удалось сформировать классифицированный результат на выходной плоскости коллиматора.")
        return self._classified_result

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        """Обрабатывает очередной чанк данных из очереди SimulationManager."""
        chunk_type = chunk.get("type")
        data = chunk.get("data")
        if data is None:
            return

        if chunk_type == ChunkType.ESCAPED_PARTICLES:
            pos_z = np.asarray(data["pos_z"], dtype=Float)
            pos_x = np.asarray(data["pos_x"], dtype=Float)
            pos_y = np.asarray(data["pos_y"], dtype=Float)
            raw_ids = data["particle_ID"]

            # Отбираем только фотоны, вышедшие через верхний (выходной) торец коллиматора z >= z_exit_face
            exit_mask = (
                (pos_z >= self._z_exit_face - 1e-4)
                & (np.abs(pos_x) <= self._aperture_half_x)
                & (np.abs(pos_y) <= self._aperture_half_y)
            )
            if np.any(exit_mask):
                self._escaped_chunks.append({
                    "particle_ID": np.asarray(raw_ids[exit_mask], dtype=np.int64).copy(),
                    "pos_x": pos_x[exit_mask].copy(),
                    "pos_y": pos_y[exit_mask].copy(),
                    "pos_z": pos_z[exit_mask].copy(),
                })

        elif chunk_type == ChunkType.INITIAL_STATES:
            self._initial_chunks.append({
                "particle_ID": np.asarray(data["particle_ID"], dtype=np.int64).copy(),
                "emission_energy": np.asarray(data["emission_energy"], dtype=Float).copy(),
                "pos_x": np.asarray(data["pos_x"], dtype=Float).copy(),
                "pos_y": np.asarray(data["pos_y"], dtype=Float).copy(),
                "pos_z": np.asarray(data["pos_z"], dtype=Float).copy(),
            })

        elif chunk_type == ChunkType.INTERACTIONS:
            # Сохраняем только акты рассеяния (process_id == 1 Compton, process_id == 2 Coherent);
            # фотоэффект (process_id == 0) поглощает частицу на месте.
            proc_ids = np.asarray(data["process_id"], dtype=np.int8)
            scatter_mask = proc_ids != 0
            if np.any(scatter_mask):
                self._scatter_chunks.append({
                    "particle_ID": np.asarray(data["particle_ID"][scatter_mask], dtype=np.int64).copy(),
                    "process_id": proc_ids[scatter_mask].copy(),
                    "energy_deposit": np.asarray(data["energy_deposit"][scatter_mask], dtype=Float).copy(),
                })

    def finalize(self) -> None:
        """Собирает и классифицирует все зарегистрированные на выходном торце коллиматора фотоны."""
        if not self._escaped_chunks:
            empty_f = np.zeros(0, dtype=Float)
            empty_i64 = np.zeros(0, dtype=np.int64)
            empty_i32 = np.zeros(0, dtype=np.int32)
            empty_i8 = np.zeros(0, dtype=np.int8)
            self._classified_result = ClassifiedExitPhotons(
                particle_id=empty_i64,
                exit_x=empty_f,
                exit_y=empty_f,
                relative_x=empty_f,
                relative_y=empty_f,
                radial_distance=empty_f,
                azimuthal_angle=empty_f,
                component=empty_i8,
                final_energy=empty_f,
                compton_count=empty_i32,
                coherent_count=empty_i32,
                total_emitted=self._total_emitted,
                solid_angle_fraction=self._solid_angle_fraction,
            )
            return

        particle_ids = np.concatenate([chunk["particle_ID"] for chunk in self._escaped_chunks])
        raw_exit_x = np.concatenate([chunk["pos_x"] for chunk in self._escaped_chunks])
        raw_exit_y = np.concatenate([chunk["pos_y"] for chunk in self._escaped_chunks])
        raw_exit_z = np.concatenate([chunk["pos_z"] for chunk in self._escaped_chunks])

        count_scored = len(particle_ids)
        birth_x = np.zeros(count_scored, dtype=Float)
        birth_y = np.zeros(count_scored, dtype=Float)
        birth_z = np.zeros(count_scored, dtype=Float)
        component = np.zeros(count_scored, dtype=np.int8)
        final_energy = np.full(count_scored, self._primary_energy, dtype=Float)
        compton_count = np.zeros(count_scored, dtype=np.int32)
        coherent_count = np.zeros(count_scored, dtype=np.int32)

        # Восстановление координат рождения и начальной энергии из initial_states
        if self._initial_chunks:
            init_ids = np.concatenate([chunk["particle_ID"] for chunk in self._initial_chunks])
            init_energies = np.concatenate([chunk["emission_energy"] for chunk in self._initial_chunks])
            init_bx = np.concatenate([chunk["pos_x"] for chunk in self._initial_chunks])
            init_by = np.concatenate([chunk["pos_y"] for chunk in self._initial_chunks])
            init_bz = np.concatenate([chunk["pos_z"] for chunk in self._initial_chunks])

            sort_init = np.argsort(init_ids)
            init_ids_sorted = init_ids[sort_init]

            found_pos = np.searchsorted(init_ids_sorted, particle_ids)
            found_pos = np.clip(found_pos, 0, max(len(init_ids_sorted) - 1, 0))
            valid_match = (len(init_ids_sorted) > 0) & (init_ids_sorted[found_pos] == particle_ids)

            matched_init_idx = sort_init[found_pos]
            birth_x = np.where(valid_match, init_bx[matched_init_idx], birth_x)
            birth_y = np.where(valid_match, init_by[matched_init_idx], birth_y)
            birth_z = np.where(valid_match, init_bz[matched_init_idx], birth_z)
            final_energy = np.where(valid_match, init_energies[matched_init_idx], final_energy)

        # Определение рассеянных частиц по наличию событий рассеяния в interactions
        if self._scatter_chunks:
            inter_ids = np.concatenate([chunk["particle_ID"] for chunk in self._scatter_chunks])
            inter_procs = np.concatenate([chunk["process_id"] for chunk in self._scatter_chunks])
            inter_edep = np.concatenate([chunk["energy_deposit"] for chunk in self._scatter_chunks])
            scattered_mask = np.isin(particle_ids, inter_ids)
        else:
            inter_ids = np.zeros(0, dtype=np.int64)
            inter_procs = np.zeros(0, dtype=np.int8)
            inter_edep = np.zeros(0, dtype=Float)
            scattered_mask = np.zeros(count_scored, dtype=np.bool_)

        unscattered_mask = ~scattered_mask

        # Точная проекция на плоскость выходного торца z = z_exit_face
        exit_x = raw_exit_x.copy()
        exit_y = raw_exit_y.copy()

        # 1. Обработка нерассеянных частиц (unscattered_mask): Geometric vs Septal Penetration
        if np.any(unscattered_mask):
            birth_x_unscattered = birth_x[unscattered_mask]
            birth_y_unscattered = birth_y[unscattered_mask]
            birth_z_unscattered = birth_z[unscattered_mask]
            exit_x_unscattered = raw_exit_x[unscattered_mask]
            exit_y_unscattered = raw_exit_y[unscattered_mask]
            exit_z_unscattered = raw_exit_z[unscattered_mask]

            delta_z = np.maximum(exit_z_unscattered - birth_z_unscattered, 1e-12)
            ratio_enter = (self._z_enter_face - birth_z_unscattered) / delta_z
            ratio_exit = (self._z_exit_face - birth_z_unscattered) / delta_z

            enter_plane_x = birth_x_unscattered + ratio_enter * (exit_x_unscattered - birth_x_unscattered)
            enter_plane_y = birth_y_unscattered + ratio_enter * (exit_y_unscattered - birth_y_unscattered)
            exit_plane_x = birth_x_unscattered + ratio_exit * (exit_x_unscattered - birth_x_unscattered)
            exit_plane_y = birth_y_unscattered + ratio_exit * (exit_y_unscattered - birth_y_unscattered)

            exit_x[unscattered_mask] = exit_plane_x
            exit_y[unscattered_mask] = exit_plane_y

            in_hole_enter, cell_x_enter, cell_y_enter, sub_enter = locate_hexagonal_cell_and_hole(
                enter_plane_x, enter_plane_y, self._hole_diameter, self._septa
            )
            in_hole_exit, cell_x_exit, cell_y_exit, sub_exit = locate_hexagonal_cell_and_hole(
                exit_plane_x, exit_plane_y, self._hole_diameter, self._septa
            )

            same_channel = (
                in_hole_enter
                & in_hole_exit
                & (cell_x_enter == cell_x_exit)
                & (cell_y_enter == cell_y_exit)
                & (sub_enter == sub_exit)
            )

            unscattered_components = np.where(
                same_channel,
                np.int8(PhotonComponent.GEOMETRIC),
                np.int8(PhotonComponent.SEPTAL_PENETRATION),
            )
            component[unscattered_mask] = unscattered_components

        # 2. Обработка рассеянных в коллиматоре частиц (scattered_mask)
        if np.any(scattered_mask):
            component[scattered_mask] = np.int8(PhotonComponent.COLLIMATOR_SCATTER)
            scattered_ids = particle_ids[scattered_mask]

            relevant_mask = np.isin(inter_ids, scattered_ids)
            if np.any(relevant_mask):
                rel_ids = inter_ids[relevant_mask]
                rel_procs = inter_procs[relevant_mask]
                rel_edep = inter_edep[relevant_mask]

                # Сопоставление с индексом внутри вектора scattered_ids
                sort_scat_order = np.argsort(scattered_ids)
                sorted_scat_ids = scattered_ids[sort_scat_order]
                target_positions = np.searchsorted(sorted_scat_ids, rel_ids)
                original_scat_indices = sort_scat_order[target_positions]

                edep_accum = np.zeros(len(scattered_ids), dtype=Float)
                compton_accum = np.zeros(len(scattered_ids), dtype=np.int32)
                coherent_accum = np.zeros(len(scattered_ids), dtype=np.int32)

                np.add.at(edep_accum, original_scat_indices, rel_edep)
                np.add.at(compton_accum, original_scat_indices, (rel_procs == 1).astype(np.int32))
                np.add.at(coherent_accum, original_scat_indices, (rel_procs == 2).astype(np.int32))

                final_energy[scattered_mask] = np.maximum(
                    0.0, final_energy[scattered_mask] - edep_accum
                )
                compton_count[scattered_mask] = compton_accum
                coherent_count[scattered_mask] = coherent_accum

        # Координаты относительно точки рождения источника (для shift-invariant PSF)
        relative_x = exit_x - birth_x
        relative_y = exit_y - birth_y
        radial_distance = np.hypot(relative_x, relative_y)
        azimuthal_angle = np.mod(np.arctan2(relative_y, relative_x), 2.0 * np.pi)

        self._classified_result = ClassifiedExitPhotons(
            particle_id=particle_ids,
            exit_x=exit_x,
            exit_y=exit_y,
            relative_x=relative_x,
            relative_y=relative_y,
            radial_distance=np.asarray(radial_distance, dtype=Float),
            azimuthal_angle=np.asarray(azimuthal_angle, dtype=Float),
            component=component,
            final_energy=final_energy,
            compton_count=compton_count,
            coherent_count=coherent_count,
            total_emitted=self._total_emitted,
            solid_angle_fraction=self._solid_angle_fraction,
        )


__all__ = [
    "PhotonComponent",
    "ClassifiedExitPhotons",
    "locate_hexagonal_cell_and_hole",
    "CollimatorExitPlaneHandler",
]
