"""Эксперимент 6: Сквозная (End-to-End) Монте-Карло верификация полной детекторной головки
гамма-камеры GE Discovery NM/CT 670 / Infinia по стандарту NEMA NU-1.

В отличие от Эксперимента 2, где регистрируются фотоны на выходном торце изолированного коллиматора,
данный эксперимент моделирует полный физический тракт реальной гамма-камеры Ангера:
1. Точечный источник в сухом воздухе ('Air, Dry (near sea level)') на расстоянии Z = 100 мм;
2. Свинцовый защитный кожух головки ('Pb', 580 x 440 мм) и воздушная полость ('Air, Dry (near sea level)');
3. Гексагональный коллиматор GE ('LEHR', 'LEUHR', 'MEGP', 'HEGP') в двух реализациях:
   - ParametricParallelCollimator (дельта-трекинг Вудкока);
   - DirectParallelCollimator (аналитический RayCasting);
4. Воздушный технологический зазор (2.0 мм) и сцинтилляционный кристалл NaI(Tl) толщиной 3/8 дюйма (9.5 мм,
   материал 'Sodium Iodide');
5. Боросиликатный стеклянный световод за кристаллом толщиной 76.0 мм ('Glass, Borosilicate (Pyrex)'),
   формирующий реальный вклад обратного комптоновского рассеяния (backscatter) в кристалл;
6. Формирование сцинтилляционного отклика камеры Ангера:
   - расчет взвешенного по энерговыделению 3D-центроида вспышки в кристалле NaI(Tl) с честным эффектом
     глубины взаимодействия (Depth of Interaction, DOI) и многократного комптоновского рассеяния;
   - гауссово энергетическое размытие тракта ФЭУ (9.8% FWHM на 140.5 кэВ, масштабируемое как 1/sqrt(E));
   - собственное пространственное разрешение позиционирования решетки ФЭУ Ангера (R_int);
   - амплитудная дискриминация в стандартном энергетическом окне фотопика NEMA NU-1 (+-10%).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.data.data_handlers import BaseDataHandler
from core.data.data_manager import DataManager
from core.geometry.direct_collimators import DirectParallelCollimator
from core.geometry.flattened_scene import FlattenedScene
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.other.typing_definitions import Float
from core.physics.interaction_buffers import ChunkType
from core.transport.propagator import ParticlePropagator
import settings.database_setting as database_setting
from verification.common.metrics import (
    ProfileResolutionMetrics,
    compute_psf_resolution_metrics,
)
from verification.common.plotting import plot_nema_detector_head_panel
from verification.common.presets import (
    GE_COLLIMATOR_PRESETS,
    RADIONUCLIDE_PRESETS,
    CollimatorPreset,
    RadionuclidePreset,
)
from verification.common.scene_builders import (
    CollimatorImplementationKind,
    QuotaBoundedSimulationManager,
)
from core.source.sources import Source


@dataclass(frozen=True)
class CrystalScintillationEvents:
    """Собранные сцинтилляционные события в кристалле NaI(Tl) после прохождения полной головки.

    Attributes:
        particle_id: Идентификаторы частиц, выделивших энергию в кристалле.
        deposited_energy: Истинная суммарная выделенная энергия в кристалле NaI(Tl) [МэВ].
        measured_energy: Измеренная энергия после размытия ФЭУ [МэВ].
        relative_x: Измеренная координата X относительно источника (с учетом R_int) [мм].
        relative_y: Измеренная координата Y относительно источника (с учетом R_int) [мм].
        interaction_depth: Глубина взаимодействия (DOI) от входной грани кристалла [мм].
        had_collimator_scatter: Флаг рассеяния фотона в свинце коллиматора до попадания в кристалл.
        had_pyrex_backscatter: Флаг обратного рассеяния фотона из световода Pyrex в кристалл.
        photopeak_mask: Булева маска попадания измеренной энергии в окно фотопика +-10%.
    """

    particle_id: NDArray[np.int64]
    deposited_energy: NDArray[np.float64]
    measured_energy: NDArray[np.float64]
    relative_x: NDArray[np.float64]
    relative_y: NDArray[np.float64]
    interaction_depth: NDArray[np.float64]
    had_collimator_scatter: NDArray[np.bool_]
    had_pyrex_backscatter: NDArray[np.bool_]
    photopeak_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class SingleHeadRunMetrics:
    """Метрики одного сквозного прогона полной детекторной головки гамма-камеры.

    Attributes:
        crystal_interacted_count: Общее число фотонов, провзаимодействовавших в кристалле NaI(Tl).
        photopeak_count: Число зарегистрированных импульсов в энергетическом окне фотопика +-10%.
        photopeak_fraction_in_crystal: Доля фотопиковых событий от всех взаимодействий в кристалле.
        system_fwhm: Сквозное системное пространственное разрешение FWHM в окне фотопика [мм].
        system_fwtm: Сквозное системное пространственное разрешение FWTM в окне фотопика [мм].
        system_activity_sensitivity: Сквозная системная чувствительность NEMA NU-1 [имп./с/МБк].
        system_activity_sensitivity_error: Пуассоновская погрешность системной чувствительности [имп./с/МБк].
        mean_interaction_depth: Средняя глубина взаимодействия (DOI) в кристалле для фотопика [мм].
        collimator_scatter_fraction_in_peak: Доля рассеянных в коллиматоре фотонов внутри фотопика.
        pyrex_backscatter_fraction_in_crystal: Доля событий в кристалле с участием обратного рассеяния от Pyrex.
        elapsed_time: Время выполнения Монте-Карло симуляции [с].
    """

    crystal_interacted_count: int
    photopeak_count: int
    photopeak_fraction_in_crystal: float
    system_fwhm: float
    system_fwtm: float
    system_activity_sensitivity: float
    system_activity_sensitivity_error: float
    mean_interaction_depth: float
    collimator_scatter_fraction_in_peak: float
    pyrex_backscatter_fraction_in_crystal: float
    elapsed_time: float


@dataclass(frozen=True)
class NemaDetectorHeadComparisonResult:
    """Сравнительный результат сквозной NEMA NU-1 верификации для одного коллиматора GE.

    Attributes:
        collimator_code: Код коллиматора ('LEHR', 'LEUHR', 'MEGP', 'HEGP').
        nuclide_name: Название радионуклида.
        photon_energy: Номинальная энергия гамма-линии [МэВ].
        num_particles: Число запущенных историй в конусе.
        passport_system_fwhm: Паспортное системное разрешение GE NEMA NU-1 [мм].
        passport_system_fwtm: Паспортное системное разрешение FWTM GE NEMA NU-1 [мм].
        passport_activity_sensitivity: Паспортная системная чувствительность GE NEMA NU-1 [имп./с/МБк].
        parametric: Метрики сквозного моделирования головки с ParametricParallelCollimator.
        direct: Метрики сквозного моделирования головки с DirectParallelCollimator.
        sensitivity_z_score_between_models: Z-score разницы чувствительности между Parametric и Direct.
        fwhm_rel_error_vs_passport_direct: Относительное отклонение системного FWHM Direct от паспорта GE.
        sensitivity_rel_error_vs_passport_direct: Относительное отклонение чувствительности Direct от паспорта GE.
        passed: Признак успешного прохождения сквозной валидации.
    """

    collimator_code: str
    nuclide_name: str
    photon_energy: float
    num_particles: int
    passport_system_fwhm: float
    passport_system_fwtm: float
    passport_activity_sensitivity: float
    parametric: SingleHeadRunMetrics
    direct: SingleHeadRunMetrics
    sensitivity_z_score_between_models: float
    fwhm_rel_error_vs_passport_direct: float
    sensitivity_rel_error_vs_passport_direct: float
    passed: bool


class NemaDetectorHeadHandler(BaseDataHandler):
    """Потоковый обработчик взаимодействий в полной детекторной головке гамма-камеры Ангера."""

    _crystal_volume_id: int
    _pyrex_volume_id: int
    _collimator_volume_ids: set[int]
    _z_crystal_front: float
    _nominal_energy: float
    _energy_resolution_ref: float
    _intrinsic_fwhm: float
    _window_half_fraction: float
    _rng: np.random.Generator

    _initial_chunks: list[dict[str, NDArray[Any]]]
    _crystal_chunks: list[dict[str, NDArray[Any]]]
    _collimator_scatter_ids: list[NDArray[np.int64]]
    _pyrex_scatter_ids: list[NDArray[np.int64]]
    _events: CrystalScintillationEvents | None

    _CONSUMED_CHUNK_TYPES: frozenset[ChunkType] = frozenset({
        ChunkType.INITIAL_STATES,
        ChunkType.INTERACTIONS,
    })

    @property
    def consumed_chunk_types(self) -> frozenset[ChunkType]:
        return self._CONSUMED_CHUNK_TYPES

    def __init__(
        self,
        crystal_volume_id: int,
        pyrex_volume_id: int,
        collimator_volume_ids: set[int],
        z_crystal_front: float,
        nominal_energy: float,
        energy_resolution_ref: float,
        intrinsic_fwhm: float,
        window_half_fraction: float = 0.10,
        seed: int = 12345,
    ) -> None:
        super().__init__(floating_point_precision="float64", compression=None)
        self._crystal_volume_id = int(crystal_volume_id)
        self._pyrex_volume_id = int(pyrex_volume_id)
        self._collimator_volume_ids = set(collimator_volume_ids)
        self._z_crystal_front = float(z_crystal_front)
        self._nominal_energy = float(nominal_energy)
        self._energy_resolution_ref = float(energy_resolution_ref)
        self._intrinsic_fwhm = float(intrinsic_fwhm)
        self._window_half_fraction = float(window_half_fraction)
        self._rng = np.random.default_rng(seed)

        self._initial_chunks = []
        self._crystal_chunks = []
        self._collimator_scatter_ids = []
        self._pyrex_scatter_ids = []
        self._events = None

    @property
    def events(self) -> CrystalScintillationEvents:
        """Возвращает реконструированные сцинтилляционные события в кристалле NaI(Tl)."""
        if self._events is None:
            self.finalize()
        if self._events is None:
            raise RuntimeError("Не удалось реконструировать события в кристалле детекторной головки.")
        return self._events

    def process_chunk(self, chunk: dict[str, Any]) -> None:
        """Перехватывает начальные координаты рождения и взаимодействия в кристалле, коллиматоре и Pyrex."""
        chunk_type = chunk.get("type")
        data = chunk.get("data")
        if data is None:
            return

        if chunk_type == ChunkType.INITIAL_STATES:
            self._initial_chunks.append({
                "particle_ID": np.asarray(data["particle_ID"], dtype=np.int64).copy(),
                "pos_x": np.asarray(data["pos_x"], dtype=np.float64).copy(),
                "pos_y": np.asarray(data["pos_y"], dtype=np.float64).copy(),
            })
        elif chunk_type == ChunkType.INTERACTIONS:
            vol_ids = np.asarray(data["volume_id"], dtype=np.int64)
            proc_ids = np.asarray(data["process_id"], dtype=np.int8)
            part_ids = np.asarray(data["particle_ID"], dtype=np.int64)

            crystal_mask = vol_ids == self._crystal_volume_id
            if np.any(crystal_mask):
                self._crystal_chunks.append({
                    "particle_ID": part_ids[crystal_mask].copy(),
                    "pos_x": np.asarray(data["pos_x"][crystal_mask], dtype=np.float64).copy(),
                    "pos_y": np.asarray(data["pos_y"][crystal_mask], dtype=np.float64).copy(),
                    "pos_z": np.asarray(data["pos_z"][crystal_mask], dtype=np.float64).copy(),
                    "energy_deposit": np.asarray(data["energy_deposit"][crystal_mask], dtype=np.float64).copy(),
                })

            scatter_mask = proc_ids != 0
            if np.any(scatter_mask):
                collimator_mask = scatter_mask & np.isin(vol_ids, list(self._collimator_volume_ids))
                if np.any(collimator_mask):
                    self._collimator_scatter_ids.append(part_ids[collimator_mask].copy())

                pyrex_mask = scatter_mask & (vol_ids == self._pyrex_volume_id)
                if np.any(pyrex_mask):
                    self._pyrex_scatter_ids.append(part_ids[pyrex_mask].copy())

    def finalize(self) -> None:
        """Агрегирует энерговыделения в кристалле NaI(Tl) и моделирует отклик ФЭУ камеры Ангера."""
        if not self._crystal_chunks:
            empty_f = np.zeros(0, dtype=np.float64)
            empty_i = np.zeros(0, dtype=np.int64)
            empty_b = np.zeros(0, dtype=np.bool_)
            self._events = CrystalScintillationEvents(
                particle_id=empty_i,
                deposited_energy=empty_f,
                measured_energy=empty_f,
                relative_x=empty_f,
                relative_y=empty_f,
                interaction_depth=empty_f,
                had_collimator_scatter=empty_b,
                had_pyrex_backscatter=empty_b,
                photopeak_mask=empty_b,
            )
            return

        cryst_ids = np.concatenate([chunk["particle_ID"] for chunk in self._crystal_chunks])
        cryst_x = np.concatenate([chunk["pos_x"] for chunk in self._crystal_chunks])
        cryst_y = np.concatenate([chunk["pos_y"] for chunk in self._crystal_chunks])
        cryst_z = np.concatenate([chunk["pos_z"] for chunk in self._crystal_chunks])
        cryst_edep = np.concatenate([chunk["energy_deposit"] for chunk in self._crystal_chunks])

        unique_ids, inverse_indices = np.unique(cryst_ids, return_inverse=True)
        deposited_energy = np.bincount(inverse_indices, weights=cryst_edep).astype(np.float64)

        valid_energy = deposited_energy > 1e-9
        unique_ids = unique_ids[valid_energy]
        deposited_energy = deposited_energy[valid_energy]

        weighted_x = np.bincount(inverse_indices, weights=cryst_edep * cryst_x)[valid_energy]
        weighted_y = np.bincount(inverse_indices, weights=cryst_edep * cryst_y)[valid_energy]
        weighted_z = np.bincount(inverse_indices, weights=cryst_edep * cryst_z)[valid_energy]

        scint_x = weighted_x / deposited_energy
        scint_y = weighted_y / deposited_energy
        scint_z = weighted_z / deposited_energy
        interaction_depth = np.maximum(0.0, scint_z - self._z_crystal_front)

        birth_x = np.zeros_like(scint_x)
        birth_y = np.zeros_like(scint_y)
        if self._initial_chunks:
            init_ids = np.concatenate([chunk["particle_ID"] for chunk in self._initial_chunks])
            init_x = np.concatenate([chunk["pos_x"] for chunk in self._initial_chunks])
            init_y = np.concatenate([chunk["pos_y"] for chunk in self._initial_chunks])
            sort_order = np.argsort(init_ids)
            sorted_init_ids = init_ids[sort_order]
            found_idx = np.searchsorted(sorted_init_ids, unique_ids)
            found_idx = np.clip(found_idx, 0, max(len(sorted_init_ids) - 1, 0))
            matched = (len(sorted_init_ids) > 0) & (sorted_init_ids[found_idx] == unique_ids)
            original_pos = sort_order[found_idx]
            birth_x = np.where(matched, init_x[original_pos], 0.0)
            birth_y = np.where(matched, init_y[original_pos], 0.0)

        if self._collimator_scatter_ids:
            all_coll_scat = np.unique(np.concatenate(self._collimator_scatter_ids))
            had_coll_scatter = np.isin(unique_ids, all_coll_scat)
        else:
            had_coll_scatter = np.zeros(len(unique_ids), dtype=np.bool_)

        if self._pyrex_scatter_ids:
            all_pyrex_scat = np.unique(np.concatenate(self._pyrex_scatter_ids))
            had_pyrex_back = np.isin(unique_ids, all_pyrex_scat)
        else:
            had_pyrex_back = np.zeros(len(unique_ids), dtype=np.bool_)

        # Размытие энергии в тракте ФЭУ: FWHM(E) = R_ref * sqrt(E_ref * E)
        fwhm_to_sigma = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        reference_tc_energy = float(0.1405 * units.MeV)
        sigma_energy = (
            self._energy_resolution_ref
            * np.sqrt(reference_tc_energy * np.maximum(deposited_energy, 1e-6))
            * fwhm_to_sigma
        )
        measured_energy = np.maximum(
            0.0,
            deposited_energy + self._rng.normal(0.0, sigma_energy, size=len(deposited_energy)),
        )

        # Собственное пространственное разрешение решетки ФЭУ Ангера R_int
        sigma_spatial = self._intrinsic_fwhm * fwhm_to_sigma
        relative_x = (
            (scint_x - birth_x)
            + self._rng.normal(0.0, sigma_spatial, size=len(scint_x))
        )
        relative_y = (
            (scint_y - birth_y)
            + self._rng.normal(0.0, sigma_spatial, size=len(scint_y))
        )

        low_bound = self._nominal_energy * (1.0 - self._window_half_fraction)
        high_bound = self._nominal_energy * (1.0 + self._window_half_fraction)
        photopeak_mask = np.asarray(
            (measured_energy >= low_bound) & (measured_energy <= high_bound),
            dtype=np.bool_,
        )

        self._events = CrystalScintillationEvents(
            particle_id=unique_ids,
            deposited_energy=deposited_energy,
            measured_energy=measured_energy,
            relative_x=relative_x,
            relative_y=relative_y,
            interaction_depth=interaction_depth,
            had_collimator_scatter=had_coll_scatter,
            had_pyrex_backscatter=had_pyrex_back,
            photopeak_mask=photopeak_mask,
        )


_HEAD_PROPAGATOR: ParticlePropagator | None = None


def _get_head_propagator() -> ParticlePropagator:
    """Возвращает общий экземпляр ParticlePropagator для симуляции детекторной головки."""
    global _HEAD_PROPAGATOR
    if _HEAD_PROPAGATOR is None:
        _HEAD_PROPAGATOR = ParticlePropagator(rng=np.random.default_rng(0))
    return _HEAD_PROPAGATOR


def run_single_detector_head_simulation(
    preset: CollimatorPreset,
    nuclide: RadionuclidePreset,
    implementation_kind: CollimatorImplementationKind,
    num_particles: int,
    source_distance: float = 100.0,
    max_polar_angle: float = 0.18,
    seed: int = 303000,
) -> tuple[SingleHeadRunMetrics, CrystalScintillationEvents]:
    """Строит 3D-сцену полной детекторной головки GE Discovery и выполняет сквозной Монте-Карло прогон."""
    air_material = database_setting.material_database["Air, Dry (near sea level)"]
    lead_material = database_setting.material_database[preset.material_name]
    aluminum_material = database_setting.material_database["Al"]
    nai_material = database_setting.material_database["Sodium Iodide"]
    pyrex_material = database_setting.material_database["Glass, Borosilicate (Pyrex)"]

    field_x, field_y, collimator_z = preset.size
    detector_gap = float(preset.nema_passport.detector_gap)
    crystal_z = float(preset.nema_passport.crystal_thickness)
    pyrex_z = 76.0
    back_shield_z = 20.0
    side_shield_thickness = 20.0

    cavity_size_z = float(collimator_z) + detector_gap + crystal_z + pyrex_z
    casing_size_z = cavity_size_z + back_shield_z
    casing_size_x = float(field_x) + 2.0 * side_shield_thickness
    casing_size_y = float(field_y) + 2.0 * side_shield_thickness

    world_margin_z = 40.0
    world_size_z = source_distance + casing_size_z + world_margin_z
    world_volume = Volume(
        geometry=Box(Float(casing_size_x + 40.0), Float(casing_size_y + 40.0), Float(world_size_z)),
        material=air_material,
        name="NemaHeadWorld",
    )

    # Размещаем входной торец коллиматора в плоскости z = 0.0
    z_enter_face = 0.0
    casing_center_z = 0.5 * casing_size_z
    casing_volume = Volume(
        geometry=Box(Float(casing_size_x), Float(casing_size_y), Float(casing_size_z)),
        material=lead_material,
        name="LeadCasing",
    )
    casing_volume.translate(z=Float(casing_center_z))
    world_volume.add_child(casing_volume)

    # Воздушная полость смещена к переднему краю кожуха так, чтобы входной торец коллиматора был открыт (z = 0)
    cavity_offset_in_casing = -0.5 * back_shield_z
    cavity_volume = Volume(
        geometry=Box(field_x, field_y, Float(cavity_size_z)),
        material=air_material,
        name="DetectorCavity",
    )
    cavity_volume.translate(z=Float(cavity_offset_in_casing))
    casing_volume.add_child(cavity_volume)

    # В локальной системе cavity_volume передняя грань находится при z_local = -0.5 * cavity_size_z
    cavity_front_local_z = -0.5 * cavity_size_z
    collimator_local_z = cavity_front_local_z + 0.5 * float(collimator_z)

    collimator_node: ParametricParallelCollimator | DirectParallelCollimator
    if implementation_kind == "parametric":
        collimator_node = ParametricParallelCollimator(
            size=preset.size,
            hole_diameter=preset.hole_diameter,
            septa=preset.septa,
            material=lead_material,
            hole_shape=preset.hole_shape,
            name=f"Head_Parametric_{preset.code}",
        )
    else:
        collimator_node = DirectParallelCollimator(
            size=preset.size,
            hole_diameter=preset.hole_diameter,
            septa=preset.septa,
            material=lead_material,
            hole_material=air_material,
            hole_shape=preset.hole_shape,
            name=f"Head_Direct_{preset.code}",
        )
    collimator_node.translate(z=Float(collimator_local_z))
    cavity_volume.add_child(collimator_node)

    # Входное герметизирующее алюминиевое окно кристалла NaI(Tl) (2.0 мм Al)
    window_local_z = cavity_front_local_z + float(collimator_z) + 0.5 * detector_gap
    aluminum_window_volume = Volume(
        geometry=Box(field_x, field_y, Float(detector_gap)),
        material=aluminum_material,
        name="AluminumCrystalWindow",
    )
    aluminum_window_volume.translate(z=Float(window_local_z))
    cavity_volume.add_child(aluminum_window_volume)

    crystal_local_z = cavity_front_local_z + float(collimator_z) + detector_gap + 0.5 * crystal_z
    crystal_volume = Volume(
        geometry=Box(field_x, field_y, Float(crystal_z)),
        material=nai_material,
        name="NaICrystal",
    )
    crystal_volume.translate(z=Float(crystal_local_z))
    cavity_volume.add_child(crystal_volume)

    pyrex_local_z = cavity_front_local_z + float(collimator_z) + detector_gap + crystal_z + 0.5 * pyrex_z
    pyrex_volume = Volume(
        geometry=Box(field_x, field_y, Float(pyrex_z)),
        material=pyrex_material,
        name="PyrexLightguide",
    )
    pyrex_volume.translate(z=Float(pyrex_local_z))
    cavity_volume.add_child(pyrex_volume)

    # Точечный источник на расстоянии source_distance перед входным торцом коллиматора (z = -source_distance)
    source_node = Source.from_point(
        activity=Float(float(num_particles) * units.Bq),
        energy=nuclide.primary_energy,
        max_polar_angle=Float(max_polar_angle),
        cell_jitter=(preset.pitch_x, preset.pitch_y),
        half_life=Float(np.inf),
    )
    source_node.translate(z=Float(z_enter_face - source_distance))
    world_volume.add_child(source_node)

    # Определяем индексы объемов в FlattenedScene
    flat_scene = FlattenedScene(world_volume)
    crystal_vol_id = -1
    pyrex_vol_id = -1
    collimator_vol_ids: set[int] = set()
    for vol_idx, (vol_obj, _, _) in enumerate(flat_scene.flat_list):
        if vol_obj is crystal_volume:
            crystal_vol_id = vol_idx
        elif vol_obj is pyrex_volume:
            pyrex_vol_id = vol_idx
        elif vol_obj is collimator_node or vol_obj.parent is collimator_node:
            collimator_vol_ids.add(vol_idx)

    if crystal_vol_id < 0 or pyrex_vol_id < 0:
        raise RuntimeError("Не удалось локализовать кристалл NaI(Tl) или световод Pyrex в FlattenedScene.")

    z_crystal_front_global = z_enter_face + float(collimator_z) + detector_gap
    head_handler = NemaDetectorHeadHandler(
        crystal_volume_id=crystal_vol_id,
        pyrex_volume_id=pyrex_vol_id,
        collimator_volume_ids=collimator_vol_ids,
        z_crystal_front=z_crystal_front_global,
        nominal_energy=float(nuclide.primary_energy),
        energy_resolution_ref=float(preset.nema_passport.energy_resolution_fraction),
        intrinsic_fwhm=float(preset.nema_passport.intrinsic_crystal_fwhm),
        window_half_fraction=0.10,
        seed=seed + 777,
    )

    batch_size = min(num_particles, 500_000)
    sim_manager = QuotaBoundedSimulationManager(
        scene=world_volume,
        propagator=_get_head_propagator(),
        stop_time=Float(1.0 * units.s),
        particles_number=batch_size,
        min_energy=Float(5.0 * units.keV),
        buffer_capacity=max(batch_size * 3, 200_000),
        seed=seed,
        consumed_chunk_types=head_handler.consumed_chunk_types,
        max_emitted_particles=num_particles,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_h5 = os.path.join(tmp_dir, "nema_detector_head.hdf5")
        data_manager = DataManager(
            filename=dummy_h5,
            handlers=[head_handler],
            queue=sim_manager.queue,
        )
        data_manager.start()
        start_clock = time.perf_counter()
        sim_manager.start()
        sim_manager.join()
        data_manager.join()
        elapsed_time = float(max(time.perf_counter() - start_clock, 1e-9))

    events = head_handler.events
    crystal_total = int(events.particle_id.size)
    peak_mask = events.photopeak_mask
    peak_count = int(np.sum(peak_mask))
    peak_fraction = float(peak_count) / float(crystal_total) if crystal_total > 0 else 0.0

    res_metrics: ProfileResolutionMetrics = compute_psf_resolution_metrics(
        coord_x=events.relative_x[peak_mask],
        coord_y=events.relative_y[peak_mask],
        spatial_extent=32.0,
        num_bins=121,
    )

    solid_angle_frac = float(source_node.channels[0].angular.solid_angle_fraction)
    decays_per_unit_activity = 1.0e6
    gamma_yield = float(nuclide.gamma_yield_per_decay)
    norm_to_activity = (
        decays_per_unit_activity * gamma_yield * solid_angle_frac / float(max(num_particles, 1))
    )

    sys_sens = float(peak_count) * norm_to_activity
    sys_sens_err = math.sqrt(float(max(peak_count, 1))) * norm_to_activity if peak_count > 0 else 0.0

    mean_doi = float(np.mean(events.interaction_depth[peak_mask])) if peak_count > 0 else 0.0
    coll_scat_in_peak = (
        float(np.mean(events.had_collimator_scatter[peak_mask])) if peak_count > 0 else 0.0
    )
    pyrex_back_in_cryst = (
        float(np.mean(events.had_pyrex_backscatter)) if crystal_total > 0 else 0.0
    )

    metrics = SingleHeadRunMetrics(
        crystal_interacted_count=crystal_total,
        photopeak_count=peak_count,
        photopeak_fraction_in_crystal=peak_fraction,
        system_fwhm=res_metrics.fwhm_mean,
        system_fwtm=res_metrics.fwtm_mean,
        system_activity_sensitivity=sys_sens,
        system_activity_sensitivity_error=sys_sens_err,
        mean_interaction_depth=mean_doi,
        collimator_scatter_fraction_in_peak=coll_scat_in_peak,
        pyrex_backscatter_fraction_in_crystal=pyrex_back_in_cryst,
        elapsed_time=elapsed_time,
    )
    return metrics, events


DEFAULT_NEMA_HEAD_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("LEHR", "Tc99m"),
    ("LEUHR", "Tc99m"),
    ("MEGP", "In111"),
    ("HEGP", "I131"),
)


def run_exp06_nema_detector_head(
    output_dir: Path,
    num_particles: int = 2_500_000,
    scenarios: tuple[tuple[str, str], ...] = DEFAULT_NEMA_HEAD_SCENARIOS,
) -> list[NemaDetectorHeadComparisonResult]:
    """Запускает Эксперимент 6: сквозную верификацию полной детекторной головки GE Discovery по NEMA NU-1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[NemaDetectorHeadComparisonResult] = []

    showcase_energy_centers = np.linspace(0.01, 0.18, 85, dtype=np.float64)
    showcase_energy_hist_param = np.zeros_like(showcase_energy_centers)
    showcase_energy_hist_direct = np.zeros_like(showcase_energy_centers)
    showcase_prof_centers = np.linspace(-30.0, 30.0, 81, dtype=np.float64)
    showcase_prof_hist_param = np.zeros_like(showcase_prof_centers)
    showcase_prof_hist_direct = np.zeros_like(showcase_prof_centers)
    showcase_title = "LEHR (Tc-99m)"
    showcase_window = (0.12645, 0.15455)

    for idx, (collimator_code, nuclide_key) in enumerate(scenarios):
        preset = GE_COLLIMATOR_PRESETS[collimator_code]
        nuclide = RADIONUCLIDE_PRESETS[nuclide_key]
        seed_base = 606000 + idx * 5000

        metrics_param, events_param = run_single_detector_head_simulation(
            preset=preset,
            nuclide=nuclide,
            implementation_kind="parametric",
            num_particles=num_particles,
            seed=seed_base,
        )
        metrics_direct, events_direct = run_single_detector_head_simulation(
            preset=preset,
            nuclide=nuclide,
            implementation_kind="direct",
            num_particles=num_particles,
            seed=seed_base + 1000,
        )

        if idx == 0:
            max_energy_bound = float(nuclide.primary_energy) * 1.30
            energy_edges = np.linspace(0.005, max_energy_bound, 91, dtype=np.float64)
            showcase_energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
            hist_energy_param, _ = np.histogram(events_param.measured_energy, bins=energy_edges)
            hist_energy_direct, _ = np.histogram(events_direct.measured_energy, bins=energy_edges)
            showcase_energy_hist_param = hist_energy_param.astype(np.float64)
            showcase_energy_hist_direct = hist_energy_direct.astype(np.float64)

            prof_edges = np.linspace(-30.0, 30.0, 82, dtype=np.float64)
            showcase_prof_centers = 0.5 * (prof_edges[:-1] + prof_edges[1:])
            hist_x_param, _ = np.histogram(
                events_param.relative_x[events_param.photopeak_mask], bins=prof_edges
            )
            hist_x_direct, _ = np.histogram(
                events_direct.relative_x[events_direct.photopeak_mask], bins=prof_edges
            )
            showcase_prof_hist_param = hist_x_param.astype(np.float64)
            showcase_prof_hist_direct = hist_x_direct.astype(np.float64)
            showcase_title = f"{preset.code} ({nuclide.title})"
            showcase_window = (
                float(nuclide.primary_energy) * 0.90,
                float(nuclide.primary_energy) * 1.10,
            )

        err_comb = math.hypot(
            metrics_param.system_activity_sensitivity_error,
            metrics_direct.system_activity_sensitivity_error,
        )
        sens_z = (
            (metrics_direct.system_activity_sensitivity - metrics_param.system_activity_sensitivity)
            / err_comb
            if err_comb > 0.0
            else 0.0
        )

        passport_fwhm = float(preset.nema_passport.system_fwhm)
        passport_fwtm = float(preset.nema_passport.system_fwtm)
        passport_sens = float(preset.nema_passport.system_activity_sensitivity)

        fwhm_rel_err = (metrics_direct.system_fwhm - passport_fwhm) / max(passport_fwhm, 1e-9)
        sens_rel_err = (metrics_direct.system_activity_sensitivity - passport_sens) / max(passport_sens, 1e-9)

        passed = (
            abs(sens_z) <= 3.5
            and abs(fwhm_rel_err) <= 0.08
            and abs(sens_rel_err) <= 0.12
        )

        results.append(
            NemaDetectorHeadComparisonResult(
                collimator_code=preset.code,
                nuclide_name=nuclide.title,
                photon_energy=float(nuclide.primary_energy),
                num_particles=num_particles,
                passport_system_fwhm=passport_fwhm,
                passport_system_fwtm=passport_fwtm,
                passport_activity_sensitivity=passport_sens,
                parametric=metrics_param,
                direct=metrics_direct,
                sensitivity_z_score_between_models=sens_z,
                fwhm_rel_error_vs_passport_direct=fwhm_rel_err,
                sensitivity_rel_error_vs_passport_direct=sens_rel_err,
                passed=passed,
            )
        )

    plot_nema_detector_head_panel(
        collimator_labels=[f"{item.collimator_code}\n({item.nuclide_name.split()[0]})" for item in results],
        passport_fwhm_list=[item.passport_system_fwhm for item in results],
        measured_fwhm_parametric=[item.parametric.system_fwhm for item in results],
        measured_fwhm_direct=[item.direct.system_fwhm for item in results],
        passport_sens_list=[item.passport_activity_sensitivity for item in results],
        measured_sens_parametric=[item.parametric.system_activity_sensitivity for item in results],
        measured_sens_direct=[item.direct.system_activity_sensitivity for item in results],
        energy_bins_centers=showcase_energy_centers,
        energy_hist_parametric=showcase_energy_hist_param,
        energy_hist_direct=showcase_energy_hist_direct,
        profile_bins_centers=showcase_prof_centers,
        profile_hist_parametric=showcase_prof_hist_param,
        profile_hist_direct=showcase_prof_hist_direct,
        showcase_label=showcase_title,
        photopeak_window_bounds=showcase_window,
        output_path=output_dir / "exp06_nema_detector_head.png",
    )

    json_path = output_dir / "exp06_nema_detector_head.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump([asdict(item) for item in results], json_file, indent=2, ensure_ascii=False)

    md_lines = [
        "# Эксперимент 6: Сквозная (End-to-End) Монте-Карло верификация полной детекторной головки GE Discovery NM/CT 670 по стандарту NEMA NU-1",
        "",
        "## Архитектура полной детекторной головки в симуляции",
        "- **Окружающая среда и полость:** сухой воздух на уровне моря (`Air, Dry (near sea level)`).",
        "- **Защитный кожух (`LeadCasing`):** свинец (`Pb`), габариты $580 \\times 440\\text{ мм}$, толщина стенок $20\\text{ мм}$.",
        "- **Коллиматор (`Parametric` vs `Direct`):** гексагональная решетка каналов GE (`LEHR`, `LEUHR`, `MEGP`, `HEGP`), поле зрения $540 \\times 400\\text{ мм}$.",
        "- **Сцинтилляционный кристалл (`NaICrystal`):** йодид натрия $\\text{NaI(Tl)}$ (`Sodium Iodide`), толщина $3/8'' = 9.5\\text{ мм}$, зазор за коллиматором $2.0\\text{ мм}$.",
        "- **Световод и блок ФЭУ (`PyrexLightguide`):** боросиликатное стекло (`Glass, Borosilicate (Pyrex)`), толщина $76.0\\text{ мм}$ вплотную за кристаллом (формирует физическое обратное комптоновское рассеяние в кристалл).",
        "- **Тракт позиционирования и спектрометрии Ангера:**",
        "  - взвешенный по энерговыделению 3D-центроид каждого каскада взаимодействий в кристалле $\\text{NaI(Tl)}$;",
        "  - энергетическое разрешение ФЭУ $9.8\\%$ FWHM на $140.5\\text{ кэВ}$ ($\\propto 1/\\sqrt{E}$), окно фотопика NEMA $\\pm 10\\%$;",
        "  - собственное пространственное разрешение детектора Ангера $R_{\\text{int}}$ ($3.9\\text{ мм}$ на $140.5\\text{ кэВ}$).",
        "",
        f"Статистика: **{num_particles:,}** историй в конусе ($\\theta_{{\\max}} = 0.18\\text{{ рад}}$) на каждую конфигурацию.",
        "",
        "## 1. Прямое сопоставление сквозной симуляции головки с паспортом GE Healthcare (NEMA NU-1, $Z = 100\\text{ мм}$)",
        "",
        "| Коллиматор (Нуклид) | Паспорт GE $\\text{FWHM}_{\\text{sys}}$, мм | MC $\\text{FWHM}_{\\text{sys}}$ `Param` / `Direct`, мм ($\\Delta\\%$) | Паспорт GE $\\text{FWTM}_{\\text{sys}}$, мм | MC $\\text{FWTM}_{\\text{sys}}$ `Param` / `Direct`, мм | Паспорт GE $S_{\\text{sys}}$, имп./с/МБк | MC $S_{\\text{sys}}$ `Param` / `Direct`, имп./с/МБк ($\\Delta\\%$) | $Z$-score (`Param` vs `Direct`) | Статус |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in results:
        status_str = "PASS" if item.passed else "FAIL"
        param_metrics = item.parametric
        direct_metrics = item.direct
        md_lines.append(
            f"| **{item.collimator_code}** ({item.nuclide_name}) | **{item.passport_system_fwhm:.2f}** | "
            f"{param_metrics.system_fwhm:.2f} / **{direct_metrics.system_fwhm:.2f}** (`{item.fwhm_rel_error_vs_passport_direct * 100.0:+.2f}%`) | "
            f"**{item.passport_system_fwtm:.2f}** | {param_metrics.system_fwtm:.2f} / {direct_metrics.system_fwtm:.2f} | "
            f"**{item.passport_activity_sensitivity:.1f}** | "
            f"{param_metrics.system_activity_sensitivity:.1f} / **{direct_metrics.system_activity_sensitivity:.1f}** (`{item.sensitivity_rel_error_vs_passport_direct * 100.0:+.2f}%`) | "
            f"`{item.sensitivity_z_score_between_models:+.2f}` | **{status_str}** |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Физическая диагностика взаимодействий внутри кристалла $\\text{NaI(Tl)}$ и световода Pyrex",
            "",
            "| Коллиматор | Метод | Взаимодействий в $\\text{NaI(Tl)}$ | Событий в фотопике ($\\pm 10\\%$) | Доля фотопика в кристалле, % | Средняя глубина DOI в кристалле, мм | Вклад рассеяния коллиматора в фотопик, % | Доля обратного рассеяния из Pyrex, % | Время счета, с |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for item in results:
        for method_label, run_metrics in (("Parametric", item.parametric), ("Direct", item.direct)):
            md_lines.append(
                f"| **{item.collimator_code}** | `{method_label}` | {run_metrics.crystal_interacted_count:,} | "
                f"{run_metrics.photopeak_count:,} | {run_metrics.photopeak_fraction_in_crystal * 100.0:.2f}% | "
                f"{run_metrics.mean_interaction_depth:.2f} | "
                f"{run_metrics.collimator_scatter_fraction_in_peak * 100.0:.2f}% | "
                f"{run_metrics.pyrex_backscatter_fraction_in_crystal * 100.0:.2f}% | "
                f"{run_metrics.elapsed_time:.2f} |"
            )

    md_path = output_dir / "exp06_nema_detector_head.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return results


__all__ = [
    "CrystalScintillationEvents",
    "SingleHeadRunMetrics",
    "NemaDetectorHeadComparisonResult",
    "NemaDetectorHeadHandler",
    "DEFAULT_NEMA_HEAD_SCENARIOS",
    "run_single_detector_head_simulation",
    "run_exp06_nema_detector_head",
]
