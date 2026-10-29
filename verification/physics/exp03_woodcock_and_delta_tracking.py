"""Эксперимент 3: Верификация дельта-трекинга и трекинга Вудкока.

Проверяет:
1. Закон Бугера–Ламберта–Бера, моменты распределения длины свободного пробега `_generate_free_path`
   (среднее 1/mu, дисперсия 1/mu^2, согласие кривой выживания с exp(-tau) по критерию хи-квадрат)
   и детерминированные граничные случаи генератора пробега (mu <= 0 -> inf, экстремальные mu, U -> 0 и U -> 1).
2. Аудит физической валидности мажоранты на K-краях поглощения (окно 69.5–88.0 кэВ между K-краями W и Pb,
   где mu_W(80 кэВ) в ~5.5 раз превышает mu_Pb(80 кэВ), демонстрируя необходимость поточечной проверки
   мажоранты mu_maj(E) >= max_m mu_m(E) на всей энергетической сетке).
3. Инвариантность продакшен-ядра дельта-трекинга (`ParticlePropagator` + `WoodcockParametricVolume`)
   к искусственному завышению мажоранты kappa = mu_maj / mu_real для kappa in {1.0, 2.5, 10.0, 50.0}
   в двухслойном барьере (Water -> Lead) со сверкой аналитического среднего числа шагов Вудкока
   на одно реальное столкновение.
4. Тройную эквивалентность в 5-слойном гетерогенном фантоме (Water -> Bone -> Air -> NaI -> Lead)
   между граничным рейкастингом (`Direct Slabs`), 3D-воксельным трекингом Вудкока (`WoodcockVoxelVolume`
   на сетке 16x16x259) и параметрическим трекингом Вудкока (`WoodcockParametricVolume` с `@cfunc`),
   включая:
   - полное ортогональное разбиение исходов истории (поглощение по слоям, прохождение через выходную
     апертуру, обратное отражение / альбедо, боковая утечка);
   - сверку нерассеянного (баллистического) прохождения с законом Бугера–Ламберта–Бера exp(-sum(mu_i * L_i));
   - пособытийный сквозной закон сохранения энергии E_0 = E_dep + E_esc + E_cut с полным транспортом
     вторичных флуоресцентных фотонов через `SecondarySource`;
   - дополнительную 3D-верификацию наклонных лучей в 3D-шахматной воксельной решетке 16x16x16.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import hepunits as units
from llvmlite import ir
from numba import cfunc, njit
from numba.core import types
from numba.extending import intrinsic
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometries import Box
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.woodcock_volumes import WoodcockParametricVolume
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import (
    Float,
    ID,
    Index,
    NumbaFloat,
    NumbaIndex,
    ParticleSpecies,
    ProcessType,
)
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import (
    ChunkType,
    RNGContext,
    SecondaryActivationBuffer,
    SimulationDataBuffer,
)
from core.physics.physics_compiler import PhysicsCompiler
from core.source.secondary_source import SecondarySource
from core.transport.propagator import ParticlePropagator
from core.transport.transport_kernels import _generate_free_path
import settings.database_setting as database_setting
from verification.common.presets import compute_total_linear_attenuation
from verification.physics.plotting import plot_exp03_woodcock_and_delta_tracking


@intrinsic
def _increment_int64_counter(typingctx: Any, addr: Any) -> Any:
    """Последовательно инкрементирует 64-битный целочисленный счетчик по адресу памяти внутри Numba @cfunc."""
    sig = types.void(types.uint64)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        ptr = builder.inttoptr(args[0], ir.IntType(64).as_pointer())
        current_val = builder.load(ptr)
        next_val = builder.add(current_val, ir.Constant(ir.IntType(64), 1))
        builder.store(next_val, ptr)
        return context.get_dummy_value()

    return sig, codegen


@cfunc(types.float64(types.uint64))
def _deterministic_uniform_from_scaled_state_cfunc(scaled_state: int) -> float:
    """Детерминированный CFFI-коллбэк ГСЧ, возвращающий U = scaled_state * 1e-15 для проверки границ."""
    return float(scaled_state) * 1.0e-15


@dataclass(frozen=True)
class FreePathBoundaryAuditMetrics:
    """Метрики детерминированной проверки граничных случаев генератора пробега `_generate_free_path`.

    Attributes:
        zero_lac_returns_inf: Возвращает ли mu = 0.0 (вакуум) бесконечную длину пробега (+inf).
        negative_lac_returns_inf: Возвращает ли mu < 0.0 бесконечную длину пробега (+inf).
        small_lac_value: Тестовое малое значение линейного коэффициента ослабления [1/мм].
        large_lac_value: Тестовое большое значение линейного коэффициента ослабления [1/мм].
        scaling_invariance_max_rel_error: Максимальная относительная погрешность тождества mu * s(mu) = -ln(U).
        zero_uniform_guard_finite: Возвращает ли U = 0.0 конечное положительное число за счет отсечки 1e-300.
        zero_uniform_free_path: Значение свободного пробега при U = 0.0 и mu = 1.0 [мм].
        near_unit_uniform_free_path: Значение свободного пробега при U = 1.0 - 1e-15 и mu = 1.0 [мм].
        unit_uniform_returns_zero: Возвращает ли U = 1.0 нулевой свободный пробег 0.0 [мм].
        passed: Признак успешного выполнения всех граничных инвариантов.
    """

    zero_lac_returns_inf: bool
    negative_lac_returns_inf: bool
    small_lac_value: float
    large_lac_value: float
    scaling_invariance_max_rel_error: float
    zero_uniform_guard_finite: bool
    zero_uniform_free_path: float
    near_unit_uniform_free_path: float
    unit_uniform_returns_zero: bool
    passed: bool


@dataclass(frozen=True)
class MajorantValidityAuditMetrics:
    """Метрики аудита корректности мажоранты на K-краях поглощения (окно между K-краями W и Pb).

    Attributes:
        tungsten_k_edge_energy: Энергия K-края поглощения вольфрама (Z=74) [МэВ].
        lead_k_edge_energy: Энергия K-края поглощения свинца (Z=82) [МэВ].
        probe_energy: Пробная энергия внутри окна [E_K(W), E_K(Pb)] (80 кэВ) [МэВ].
        tungsten_lac_at_probe: Линейный коэффициент ослабления вольфрама при 80 кэВ [1/мм].
        lead_lac_at_probe: Линейный коэффициент ослабления свинца при 80 кэВ [1/мм].
        tungsten_to_lead_lac_ratio: Отношение mu_W(80 кэВ) / mu_Pb(80 кэВ) (~5.5).
        lead_valid_majorant_for_tungsten: Является ли свинец валидной мажорантой для вольфрама во всем диапазоне.
        pointwise_envelope_valid_on_full_grid: Выполняется ли mu_env(E) >= max(mu_W(E), mu_Pb(E)) на всей сетке.
        passed: Признак корректного обнаружения нарушения мажоранты Pb < W в окне K-краев.
    """

    tungsten_k_edge_energy: float
    lead_k_edge_energy: float
    probe_energy: float
    tungsten_lac_at_probe: float
    lead_lac_at_probe: float
    tungsten_to_lead_lac_ratio: float
    lead_valid_majorant_for_tungsten: bool
    pointwise_envelope_valid_on_full_grid: bool
    passed: bool


@dataclass(frozen=True)
class FreePathMomentMetrics:
    """Метрики верификации генератора длины свободного пробега и закона Бугера–Ламберта–Бера.

    Attributes:
        material_name: Название тестовой среды.
        linear_attenuation: Линейный коэффициент ослабления среды [1/мм].
        num_histories: Число разыгранных свободных пробегов.
        mean_free_path_mc: Выборочная средняя длина свободного пробега [мм].
        mean_free_path_theory: Теоретическая средняя длина свободного пробега 1/mu [мм].
        mean_rel_error: Относительная погрешность среднего значения.
        mean_z_score: Асимптотический Z-score отклонения выборочного среднего от 1/mu.
        variance_mc: Выборочная дисперсия длины свободного пробега [мм^2].
        variance_theory: Теоретическая дисперсия 1/mu^2 [мм^2].
        variance_rel_error: Относительная погрешность дисперсии.
        variance_z_score: Асимптотический Z-score отклонения выборочной дисперсии (с учетом куртозиса экспоненты).
        survival_chi2_ndf: Приведенный хи-квадрат Пирсона на полном разбиении [0, inf) (ndf = M - 1).
        passed: Признак выполнения критериев (Z-score моментов < 3.5, chi2/ndf in [0.5, 1.6]).
    """

    material_name: str
    linear_attenuation: float
    num_histories: int
    mean_free_path_mc: float
    mean_free_path_theory: float
    mean_rel_error: float
    mean_z_score: float
    variance_mc: float
    variance_theory: float
    variance_rel_error: float
    variance_z_score: float
    survival_chi2_ndf: float
    passed: bool


@dataclass(frozen=True)
class DeltaMajorantScalingMetrics:
    """Метрики инвариантности продакшен-трекинга Вудкока при масштабировании мажоранты в двухслойном барьере.

    Attributes:
        kappa_factor: Коэффициент искусственного завышения мажоранты kappa = mu_maj / mu_real.
        num_histories: Число запущенных историй в `ParticlePropagator`.
        collided_fraction_mc: Доля частиц, испытавших реальную коллизию внутри барьера.
        collided_fraction_theory: Аналитическая вероятность первой коллизии внутри барьера.
        collided_fraction_z_score: Биномиальный Z-score доли провзаимодействовавших частиц.
        mean_depth_mc: Выборочная средняя глубина первой реальной коллизии [мм].
        mean_depth_theory: Аналитическая средняя глубина первой реальной коллизии [мм].
        mean_depth_z_score: Статистический Z-score отклонения средней глубины.
        profile_chi2_ndf: Приведенный хи-квадрат Пирсона по полному разбиению (включая прошедшие частицы).
        mean_steps_per_collision: Выборочное среднее число шагов Вудкока на одно реальное столкновение.
        mean_steps_per_collision_theory: Аналитическое среднее число шагов Вудкока на одно реальное столкновение.
        mean_steps_rel_error: Относительное отклонение среднего числа шагов Вудкока от теории.
        passed: Признак выполнения критериев (chi2/ndf in [0.5, 1.6], |Z_depth| < 3.0, Z_coll < 3.5).
    """

    kappa_factor: float
    num_histories: int
    collided_fraction_mc: float
    collided_fraction_theory: float
    collided_fraction_z_score: float
    mean_depth_mc: float
    mean_depth_theory: float
    mean_depth_z_score: float
    profile_chi2_ndf: float
    mean_steps_per_collision: float
    mean_steps_per_collision_theory: float
    mean_steps_rel_error: float
    passed: bool


@dataclass(frozen=True)
class Voxel3DObliqueRayMetrics:
    """Метрики верификации 3D-воксельной навигации Вудкока на 3D-шахматной решетке 16x16x16 при косом падении.

    Attributes:
        grid_shape: Размерность 3D-воксельной сетки (Nx, Ny, Nz).
        num_histories: Число запущенных наклонных фотонов.
        beam_direction: Направляющие косинусы наклонного 3D-пучка (dx, dy, dz).
        total_edep_voxel_3d: Полная выделенная энергия в 3D `WoodcockVoxelVolume` [МэВ].
        total_edep_parametric_3d: Полная выделенная энергия в эталонном 3D `WoodcockParametricVolume` [МэВ].
        edep_z_score: Двухвыборочный Z-score разности полных энерговыделений.
        uncollided_fraction_voxel_3d: Доля нерассеянных прошедших фотонов в `WoodcockVoxelVolume`.
        uncollided_fraction_parametric_3d: Доля нерассеянных прошедших фотонов в `WoodcockParametricVolume`.
        uncollided_z_score: Двухвыборочный Z-score доли нерассеянных прошедших фотонов.
        passed: Признак согласия 3D-воксельной и параметрической навигации (|Z| < 3.0).
    """

    grid_shape: list[int]
    num_histories: int
    beam_direction: list[float]
    total_edep_voxel_3d: float
    total_edep_parametric_3d: float
    edep_z_score: float
    uncollided_fraction_voxel_3d: float
    uncollided_fraction_parametric_3d: float
    uncollided_z_score: float
    passed: bool


@dataclass(frozen=True)
class HeterogeneousPhantomParityMetrics:
    """Метрики тройной эквивалентности, ортогонального разбиения и энергобаланса в 5-слойном фантоме.

    Attributes:
        num_histories: Число первичных фотонов в пучке.
        beam_energy: Начальная энергия пучка фотонов [МэВ].
        layer_names: Названия 5 слоев гетерогенного фантома ('Water', 'Bone', 'Air', 'NaI', 'Lead').
        layer_thicknesses: Толщины 5 слоев вдоль оси Z [мм].
        voxel_grid_shape: Форма 3D-воксельной сетки `WoodcockVoxelVolume` (Nx, Ny, Nz).
        edep_direct_slabs: Суммарное энерговыделение по слоям для Direct Slabs [МэВ].
        edep_woodcock_voxel: Суммарное энерговыделение по слоям для WoodcockVoxelVolume [МэВ].
        edep_woodcock_parametric: Суммарное энерговыделение по слоям для WoodcockParametricVolume [МэВ].
        total_edep_direct: Полная поглощенная энергия во всем фантоме для Direct Slabs [МэВ].
        total_edep_voxel: Полная поглощенная энергия во всем фантоме для WoodcockVoxelVolume [МэВ].
        total_edep_parametric: Полная поглощенная энергия во всем фантоме для WoodcockParametricVolume [МэВ].
        max_total_edep_rel_diff: Максимальное попарное относительное расхождение полных энергий.
        total_edep_rel_diff_se: Статистическая стандартная ошибка SE(N) относительной разности полных энергий.
        total_edep_rel_diff_tolerance: Допуск относительного расхождения полных энергий 3.5 * SE(N).
        max_layer_edep_z_score: Максимальный попарный |Z-score| послойного энерговыделения.
        process_counts_direct: Число взаимодействий по процессам для Direct Slabs.
        process_counts_voxel: Число взаимодействий по процессам для WoodcockVoxelVolume.
        process_counts_parametric: Число взаимодействий по процессам для WoodcockParametricVolume.
        max_process_count_z_score: Максимальный попарный |Z-score| по числу процессов.
        transmitted_fraction_direct: Полная доля фотонов, прошедших через выходной торец фантома (Direct Slabs).
        transmitted_fraction_voxel: Полная доля фотонов, прошедших через выходной торец фантома (WoodcockVoxel).
        transmitted_fraction_parametric: Полная доля фотонов, прошедших через выходной торец (WoodcockParametric).
        max_transmitted_z_score: Максимальный попарный |Z-score| доли прошедших фотонов.
        uncollided_transmitted_theory: Аналитическая доля баллистических фотонов exp(-sum(mu_i * L_i)).
        uncollided_transmitted_direct: Доля баллистических фотонов для Direct Slabs.
        uncollided_transmitted_voxel: Доля баллистических фотонов для WoodcockVoxelVolume.
        uncollided_transmitted_parametric: Доля баллистических фотонов для WoodcockParametricVolume.
        max_uncollided_vs_theory_z_score: Максимальный |Z-score| отклонения баллистической доли от закона Бугера.
        absorbed_layer_fractions_direct: Доли первичных историй, поглощенных в каждом из 5 слоев (Direct).
        absorbed_layer_fractions_voxel: Доли первичных историй, поглощенных в каждом из 5 слоев (Voxel).
        absorbed_layer_fractions_parametric: Доли первичных историй, поглощенных в каждом из 5 слоев (Parametric).
        reflected_fraction_direct: Доля обратно отраженных (альбедо, z <= 0) первичных фотонов (Direct).
        reflected_fraction_voxel: Доля обратно отраженных (альбедо, z <= 0) первичных фотонов (Voxel).
        reflected_fraction_parametric: Доля обратно отраженных (альбедо, z <= 0) первичных фотонов (Parametric).
        lateral_escape_fraction_direct: Доля первичных фотонов, вышедших через боковые грани (Direct).
        lateral_escape_fraction_voxel: Доля первичных фотонов, вышедших через боковые грани (Voxel).
        lateral_escape_fraction_parametric: Доля первичных фотонов, вышедших через боковые грани (Parametric).
        outcome_partition_complete: Тождество суммы всех 4 ортогональных категорий исходов единице.
        max_history_energy_imbalance: Максимальный пособытийный дисбаланс |E_0 - (E_dep + E_esc + E_cut)| [МэВ].
        energy_balance_violations: Число историй с нарушением энергобаланса >= 1e-12 МэВ.
        voxel_3d_oblique_metrics: Результаты проверки 3D-шахматной воксельной решетки при косом падении.
        passed: Признак выполнения всех статистических и физических критериев.
    """

    num_histories: int
    beam_energy: float
    layer_names: list[str]
    layer_thicknesses: list[float]
    voxel_grid_shape: list[int]
    edep_direct_slabs: list[float]
    edep_woodcock_voxel: list[float]
    edep_woodcock_parametric: list[float]
    total_edep_direct: float
    total_edep_voxel: float
    total_edep_parametric: float
    max_total_edep_rel_diff: float
    total_edep_rel_diff_se: float
    total_edep_rel_diff_tolerance: float
    max_layer_edep_z_score: float
    process_counts_direct: dict[str, int]
    process_counts_voxel: dict[str, int]
    process_counts_parametric: dict[str, int]
    max_process_count_z_score: float
    transmitted_fraction_direct: float
    transmitted_fraction_voxel: float
    transmitted_fraction_parametric: float
    max_transmitted_z_score: float
    uncollided_transmitted_theory: float
    uncollided_transmitted_direct: float
    uncollided_transmitted_voxel: float
    uncollided_transmitted_parametric: float
    max_uncollided_vs_theory_z_score: float
    absorbed_layer_fractions_direct: list[float]
    absorbed_layer_fractions_voxel: list[float]
    absorbed_layer_fractions_parametric: list[float]
    reflected_fraction_direct: float
    reflected_fraction_voxel: float
    reflected_fraction_parametric: float
    lateral_escape_fraction_direct: float
    lateral_escape_fraction_voxel: float
    lateral_escape_fraction_parametric: float
    outcome_partition_complete: bool
    max_history_energy_imbalance: float
    energy_balance_violations: int
    voxel_3d_oblique_metrics: Voxel3DObliqueRayMetrics
    passed: bool


@dataclass(frozen=True)
class Exp03WoodcockTrackingReport:
    """Корневой отчет верификации дельта-трекинга и трекинга Вудкока (Эксперимент 3).

    Attributes:
        experiment_id: Идентификатор эксперимента ('exp03_woodcock_and_delta_tracking').
        title: Название эксперимента.
        quick_mode: Признак быстрого режима прогона.
        reference_standard_types: Классификация независимости используемых эталонов (A/B/C/D).
        free_path_boundary_audit: Результаты детерминированной проверки граничных случаев `_generate_free_path`.
        majorant_validity_audit: Результаты аудита корректности мажоранты на K-краях W и Pb.
        free_path_metrics: Результаты проверки закона Бугера и моментов `_generate_free_path`.
        delta_scaling_metrics: Результаты проверки инвариантности продакшен-ядра к масштабированию мажоранты.
        phantom_parity_metrics: Результаты тройной сверки на 5-слойном гетерогенном фантоме.
        passed: Итоговый статус верификации.
    """

    experiment_id: str
    title: str
    quick_mode: bool
    reference_standard_types: dict[str, str]
    free_path_boundary_audit: FreePathBoundaryAuditMetrics
    majorant_validity_audit: MajorantValidityAuditMetrics
    free_path_metrics: list[FreePathMomentMetrics]
    delta_scaling_metrics: list[DeltaMajorantScalingMetrics]
    phantom_parity_metrics: HeterogeneousPhantomParityMetrics
    passed: bool


class _TwoLayerBarrierParametricVolume(WoodcockParametricVolume):
    """Двухслойный параметрический объем Вудкока (Water -> Lead) с телеметрией числа вызовов @cfunc."""

    _first_material: Material
    _second_material: Material
    _interface_depth: float
    _counter_addr: int

    def __init__(
        self,
        box_size: tuple[float, float, float],
        majorant_mat: Material,
        first_mat: Material,
        second_mat: Material,
        interface_depth: float,
        counter_addr: int,
        name: str = "TwoLayerWoodcockBarrier",
    ) -> None:
        self._first_material = first_mat
        self._second_material = second_mat
        self._interface_depth = float(interface_depth)
        self._counter_addr = int(counter_addr)
        super().__init__(
            geometry=Box(Float(box_size[0]), Float(box_size[1]), Float(box_size[2])),
            material=majorant_mat,
            name=name,
        )

    @property
    def material_list(self) -> list[Material]:
        """Возвращает список уникальных материалов двухслойного барьера и синтетической мажоранты."""
        combined = [self.material, self._first_material, self._second_material]
        return list({mat.ID: mat for mat in combined}.values())

    def _compile_cfunc(self) -> Any:
        """Компилирует Numba @cfunc определения материала слоев с подсчетом шагов Вудкока."""
        first_id = Index(self._first_material.ID)
        second_id = Index(self._second_material.ID)
        half_thickness = Float(0.5 * float(self.size[2]))
        interface_depth_val = Float(self._interface_depth)
        counter_addr_val = np.uint64(self._counter_addr)

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat))
        def lookup_fn(pos_x: Float, pos_y: Float, pos_z: Float) -> Index:
            _increment_int64_counter(counter_addr_val)
            depth_coord = pos_z + half_thickness
            if depth_coord < interface_depth_val:
                return first_id
            return second_id

        return lookup_fn


class _FiveLayerParametricVolume(WoodcockParametricVolume):
    """Параметрический 5-слойный объем с навигацией Вудкока и @cfunc-коллбэком по координате Z.

    Порядок слоев вдоль оси Z (от 0.0 до 51.8 мм):
    1. Water, Liquid: [0.0, 20.0) мм (20.0 мм)
    2. Bone, Cortical (ICRU-44): [20.0, 30.0) мм (10.0 мм)
    3. Air, Dry (near sea level): [30.0, 45.0) мм (15.0 мм)
    4. Sodium Iodide: [45.0, 51.0) мм (6.0 мм)
    5. Pb: [51.0, 51.8] мм (0.8 мм)
    """

    _layers_materials: list[Material]

    def __init__(
        self,
        box_size: tuple[float, float, float],
        majorant_mat: Material,
        layers_materials: list[Material],
        name: str = "WoodcockParametricPhantom",
    ) -> None:
        self._layers_materials = list(layers_materials)
        super().__init__(
            geometry=Box(Float(box_size[0]), Float(box_size[1]), Float(box_size[2])),
            material=majorant_mat,
            name=name,
        )

    @property
    def material_list(self) -> list[Material]:
        """Возвращает полный список уникальных материалов во всех 5 слоях и мажоранте."""
        combined = [self.material, *self._layers_materials]
        return list({mat.ID: mat for mat in combined}.values())

    def _compile_cfunc(self) -> Any:
        """Компилирует Numba @cfunc, возвращающий ID материала по локальной координате Z."""
        water_id = Index(self._layers_materials[0].ID)
        bone_id = Index(self._layers_materials[1].ID)
        air_id = Index(self._layers_materials[2].ID)
        nai_id = Index(self._layers_materials[3].ID)
        lead_id = Index(self._layers_materials[4].ID)
        half_thickness = Float(0.5 * float(self.size[2]))

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat))
        def lookup_fn(pos_x: Float, pos_y: Float, pos_z: Float) -> Index:
            depth_coord = pos_z + half_thickness
            if depth_coord < 20.0:
                return water_id
            if depth_coord < 30.0:
                return bone_id
            if depth_coord < 45.0:
                return air_id
            if depth_coord < 51.0:
                return nai_id
            return lead_id

        return lookup_fn


class _Checkerboard3DParametricVolume(WoodcockParametricVolume):
    """Трехмерный шахматный параметрический объем Вудкока 16x16x16 для сверки с 3D WoodcockVoxelVolume."""

    _materials_palette: list[Material]
    _voxel_step: float
    _grid_dim: int

    def __init__(
        self,
        box_span: float,
        grid_dim: int,
        majorant_mat: Material,
        materials_palette: list[Material],
        name: str = "Checkerboard3DParametric",
    ) -> None:
        self._materials_palette = list(materials_palette)
        self._grid_dim = int(grid_dim)
        self._voxel_step = float(box_span) / float(grid_dim)
        super().__init__(
            geometry=Box(Float(box_span), Float(box_span), Float(box_span)),
            material=majorant_mat,
            name=name,
        )

    @property
    def material_list(self) -> list[Material]:
        """Возвращает список уникальных материалов 3D-шахматного фантома."""
        combined = [self.material, *self._materials_palette]
        return list({mat.ID: mat for mat in combined}.values())

    def _compile_cfunc(self) -> Any:
        """Компилирует Numba @cfunc 3D-шахматной индексации по всем трем осям (X, Y, Z)."""
        mat_id_0 = Index(self._materials_palette[0].ID)
        mat_id_1 = Index(self._materials_palette[1].ID)
        mat_id_2 = Index(self._materials_palette[2].ID)
        mat_id_3 = Index(self._materials_palette[3].ID)
        half_span = Float(0.5 * float(self.size[0]))
        vox_step = Float(self._voxel_step)
        max_idx = Index(self._grid_dim - 1)

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat))
        def lookup_fn(pos_x: Float, pos_y: Float, pos_z: Float) -> Index:
            ix = Index(np.floor((pos_x + half_span) / vox_step))
            iy = Index(np.floor((pos_y + half_span) / vox_step))
            iz = Index(np.floor((pos_z + half_span) / vox_step))
            ix = max(Index(0), min(ix, max_idx))
            iy = max(Index(0), min(iy, max_idx))
            iz = max(Index(0), min(iz, max_idx))
            selector = (ix + 2 * iy + 3 * iz) % 4
            if selector == 0:
                return mat_id_0
            if selector == 1:
                return mat_id_1
            if selector == 2:
                return mat_id_2
            return mat_id_3

        return lookup_fn


@njit(cache=True)
def _sample_free_paths_kernel(
    linear_attenuation: Float,
    num_samples: int,
    rng_ctx: RNGContext,
) -> NDArray[np.float64]:
    """Генерирует выборку длин свободного пробега ядром `_generate_free_path`."""
    samples = np.empty(num_samples, dtype=np.float64)
    for sample_idx in range(num_samples):
        samples[sample_idx] = float(_generate_free_path(linear_attenuation, rng_ctx))
    return samples


@njit(cache=True)
def _eval_single_free_path_kernel(
    linear_attenuation: Float,
    rng_ctx: RNGContext,
) -> float:
    """Вызывает `_generate_free_path` однократно для детерминированной проверки граничных условий."""
    return float(_generate_free_path(linear_attenuation, rng_ctx))


def _run_free_path_boundary_audit() -> FreePathBoundaryAuditMetrics:
    """Проверяет граничные случаи `_generate_free_path`: mu <= 0, экстремальные mu, U -> 0 и U -> 1."""
    rng_zero = np.random.default_rng(10301)
    rng_ctx_zero = RNGContext.from_numpy_rng(rng_zero)
    path_zero_mu = _eval_single_free_path_kernel(Float(0.0), rng_ctx_zero)
    path_neg_mu = _eval_single_free_path_kernel(Float(-2.5), rng_ctx_zero)
    zero_lac_is_inf = math.isinf(path_zero_mu) and (path_zero_mu > 0.0)
    neg_lac_is_inf = math.isinf(path_neg_mu) and (path_neg_mu > 0.0)

    small_lac = 1.0e-12
    large_lac = 1.0e6
    num_test = 1024
    rng_small = np.random.default_rng(10302)
    rng_large = np.random.default_rng(10302)
    ctx_small = RNGContext.from_numpy_rng(rng_small)
    ctx_large = RNGContext.from_numpy_rng(rng_large)
    samples_small = _sample_free_paths_kernel(Float(small_lac), num_test, ctx_small)
    samples_large = _sample_free_paths_kernel(Float(large_lac), num_test, ctx_large)

    optical_small = samples_small * small_lac
    optical_large = samples_large * large_lac
    scaling_rel_err = float(np.max(np.abs(optical_small - optical_large) / np.maximum(optical_small, 1e-30)))

    ctx_u_zero = RNGContext(state_addr=0, next_double=_deterministic_uniform_from_scaled_state_cfunc.ctypes)
    ctx_u_near_one = RNGContext(
        state_addr=999_999_999_999_999,
        next_double=_deterministic_uniform_from_scaled_state_cfunc.ctypes,
    )
    ctx_u_one = RNGContext(
        state_addr=1_000_000_000_000_000,
        next_double=_deterministic_uniform_from_scaled_state_cfunc.ctypes,
    )

    path_u_zero = _eval_single_free_path_kernel(Float(1.0), ctx_u_zero)
    expected_u_zero = -math.log(1.0e-300)
    zero_guard_ok = (
        math.isfinite(path_u_zero)
        and (path_u_zero > 690.0)
        and (abs(path_u_zero - expected_u_zero) / expected_u_zero < 1e-14)
    )

    path_u_near_one = _eval_single_free_path_kernel(Float(1.0), ctx_u_near_one)
    near_one_ok = math.isfinite(path_u_near_one) and (0.0 < path_u_near_one < 1.0e-13)

    path_u_one = _eval_single_free_path_kernel(Float(1.0), ctx_u_one)
    unit_one_ok = path_u_one == 0.0

    passed = (
        zero_lac_is_inf
        and neg_lac_is_inf
        and bool(np.all(np.isfinite(samples_small)) and np.all(samples_small > 0.0))
        and bool(np.all(np.isfinite(samples_large)) and np.all(samples_large > 0.0))
        and (scaling_rel_err < 1e-14)
        and zero_guard_ok
        and near_one_ok
        and unit_one_ok
    )

    return FreePathBoundaryAuditMetrics(
        zero_lac_returns_inf=zero_lac_is_inf,
        negative_lac_returns_inf=neg_lac_is_inf,
        small_lac_value=small_lac,
        large_lac_value=large_lac,
        scaling_invariance_max_rel_error=scaling_rel_err,
        zero_uniform_guard_finite=zero_guard_ok,
        zero_uniform_free_path=path_u_zero,
        near_unit_uniform_free_path=path_u_near_one,
        unit_uniform_returns_zero=unit_one_ok,
        passed=passed,
    )


def _get_or_register_tungsten_material() -> Material:
    """Возвращает материал элементарного вольфрама (W, Z=74, rho=19.3 г/см^3), регистрируя его при необходимости."""
    mat_db = database_setting.material_database
    att_db = database_setting.attenuation_database
    if "W" in mat_db:
        tungsten_mat = mat_db["W"]
        if tungsten_mat not in att_db:
            att_db.add_material(tungsten_mat)
        return tungsten_mat

    max_id = max((mat.ID for mat in mat_db.values()), default=0)
    w_comp_cls = namedtuple("CompositionW", ["W"])
    tungsten_mat = Material(
        name="W",
        type="Elemental media",
        density=Float(19.3 * units.g / units.cm3),
        composition=w_comp_cls(W=Float(1.0)),
        ZtoA_ratio=Float(74.0 / 183.84),
        ID=max_id + 50,
    )
    mat_db["W"] = tungsten_mat
    att_db.add_material(tungsten_mat)
    return tungsten_mat


def _run_majorant_k_edge_validity_audit() -> MajorantValidityAuditMetrics:
    """Проводит аудит мажоранты в окне между K-краем вольфрама (69.5 кэВ) и K-краем свинца (88.0 кэВ)."""
    att_db = database_setting.attenuation_database
    lead_mat = database_setting.material_database["Pb"]
    tungsten_mat = _get_or_register_tungsten_material()

    w_edges_e, w_edges_lbl = att_db.get_element_edges("W")
    pb_edges_e, pb_edges_lbl = att_db.get_element_edges("Pb")
    w_labels_list = [str(lbl) for lbl in w_edges_lbl]
    pb_labels_list = [str(lbl) for lbl in pb_edges_lbl]
    w_k_edge = float(w_edges_e[w_labels_list.index("K")])
    pb_k_edge = float(pb_edges_e[pb_labels_list.index("K")])

    probe_energy = float(80.0 * units.keV)
    mu_tungsten_80 = float(compute_total_linear_attenuation(tungsten_mat, Float(probe_energy)))
    mu_lead_80 = float(compute_total_linear_attenuation(lead_mat, Float(probe_energy)))
    ratio_w_to_pb = mu_tungsten_80 / mu_lead_80

    # Проверяем на сетке энергий от 10 кэВ до 1 МэВ, что свинец НЕ мажорирует вольфрам в окне [E_K(W), E_K(Pb)]
    energy_grid = np.geomspace(float(10.0 * units.keV), float(1.0 * units.MeV), 256, dtype=np.float64)
    mu_w_grid = np.array(
        [float(compute_total_linear_attenuation(tungsten_mat, Float(ene))) for ene in energy_grid],
        dtype=np.float64,
    )
    mu_pb_grid = np.array(
        [float(compute_total_linear_attenuation(lead_mat, Float(ene))) for ene in energy_grid],
        dtype=np.float64,
    )
    lead_valid_everywhere = bool(np.all(mu_pb_grid >= mu_w_grid))
    envelope_lac = np.maximum(mu_w_grid, mu_pb_grid)
    envelope_valid_everywhere = bool(
        np.all(envelope_lac >= mu_w_grid) and np.all(envelope_lac >= mu_pb_grid)
    )

    passed = (
        (0.069 < w_k_edge < 0.070)
        and (0.0875 < pb_k_edge < 0.0885)
        and (w_k_edge < probe_energy < pb_k_edge)
        and (mu_tungsten_80 > 14.0)
        and (2.5 < mu_lead_80 < 3.0)
        and (ratio_w_to_pb > 5.0)
        and (not lead_valid_everywhere)
        and envelope_valid_everywhere
    )

    return MajorantValidityAuditMetrics(
        tungsten_k_edge_energy=w_k_edge,
        lead_k_edge_energy=pb_k_edge,
        probe_energy=probe_energy,
        tungsten_lac_at_probe=mu_tungsten_80,
        lead_lac_at_probe=mu_lead_80,
        tungsten_to_lead_lac_ratio=ratio_w_to_pb,
        lead_valid_majorant_for_tungsten=lead_valid_everywhere,
        pointwise_envelope_valid_on_full_grid=envelope_valid_everywhere,
        passed=passed,
    )


def _compute_two_layer_exact_moments(
    mu_first: float,
    mu_second: float,
    interface_depth: float,
    total_thickness: float,
) -> tuple[float, float, float, float]:
    """Вычисляет аналитическую вероятность коллизии, среднее, дисперсию глубины и интеграл выживания.

    Интегралы на каждом однородном участке [a, b] с параметром mu и начальной оптической толщиной tau_a:
    I0 = int_a^b mu * exp(-tau(z)) dz = exp(-tau_a) * (1 - exp(-mu * L))
    I1 = int_a^b z * mu * exp(-tau(z)) dz = exp(-tau_a) * ((a + 1/mu) - (b + 1/mu) * exp(-mu * L))
    I2 = int_a^b z^2 * mu * exp(-tau(z)) dz
       = exp(-tau_a) * ((a^2 + 2a/mu + 2/mu^2) - (b^2 + 2b/mu + 2/mu^2) * exp(-mu * L))
    Интеграл выживания:
    S_int = int_0^L exp(-tau(z)) dz = (1 - exp(-mu_1 * L_1)) / mu_1 + exp(-mu_1 * L_1) * (1 - exp(-mu_2 * L_2)) / mu_2.
    """

    def _segment_integrals(
        start_z: float,
        end_z: float,
        mu_val: float,
        tau_start: float,
    ) -> tuple[float, float, float]:
        seg_len = end_z - start_z
        exp_start = math.exp(-tau_start)
        exp_len = math.exp(-mu_val * seg_len)
        inv_mu = 1.0 / mu_val
        inv_mu2 = inv_mu * inv_mu

        int_0 = exp_start * (1.0 - exp_len)
        int_1 = exp_start * ((start_z + inv_mu) - (end_z + inv_mu) * exp_len)
        term_start_2 = start_z * start_z + 2.0 * start_z * inv_mu + 2.0 * inv_mu2
        term_end_2 = end_z * end_z + 2.0 * end_z * inv_mu + 2.0 * inv_mu2
        int_2 = exp_start * (term_start_2 - term_end_2 * exp_len)
        return int_0, int_1, int_2

    i0_first, i1_first, i2_first = _segment_integrals(0.0, interface_depth, mu_first, 0.0)
    tau_interface = mu_first * interface_depth
    i0_second, i1_second, i2_second = _segment_integrals(
        interface_depth, total_thickness, mu_second, tau_interface
    )

    total_prob = i0_first + i0_second
    mean_depth = (i1_first + i1_second) / total_prob
    second_moment = (i2_first + i2_second) / total_prob
    variance_depth = max(second_moment - mean_depth * mean_depth, 1e-16)
    survival_integral = (i0_first / mu_first) + (i0_second / mu_second)
    return total_prob, mean_depth, variance_depth, survival_integral


def _run_beer_lambert_and_free_path_audit(
    quick: bool,
) -> tuple[list[FreePathMomentMetrics], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Выполняет проверку закона Бугера–Ламберта–Бера и моментов `_generate_free_path`."""
    num_histories = 150_000 if quick else 500_000
    reference_energy = Float(140.5 * units.keV)

    test_material_names = [
        "Water, Liquid",
        "Sodium Iodide",
        "Pb",
    ]

    optical_depths = np.linspace(0.2, 5.0, 17, dtype=np.float64)
    metrics_list: list[FreePathMomentMetrics] = []

    plot_depths = np.zeros_like(optical_depths)
    plot_trans_mc = np.zeros_like(optical_depths)
    plot_trans_theory = np.exp(-optical_depths)

    material_seeds = (10330, 10343, 10351)

    for mat_idx, mat_name in enumerate(test_material_names):
        material_obj = database_setting.material_database[mat_name]
        mu_val = float(compute_total_linear_attenuation(material_obj, reference_energy))

        rng = np.random.default_rng(material_seeds[mat_idx])
        rng_ctx = RNGContext.from_numpy_rng(rng)
        samples = _sample_free_paths_kernel(Float(mu_val), num_histories, rng_ctx)

        mean_mc = float(np.mean(samples))
        mean_theory = 1.0 / mu_val
        mean_rel_err = abs(mean_mc - mean_theory) / mean_theory
        mean_std_err = mean_theory / math.sqrt(float(num_histories))
        mean_z_score = abs(mean_mc - mean_theory) / mean_std_err

        var_mc = float(np.var(samples))
        var_theory = 1.0 / (mu_val * mu_val)
        var_rel_err = abs(var_mc - var_theory) / var_theory
        var_std_err = math.sqrt(8.0) * var_theory / math.sqrt(float(num_histories))
        variance_z_score = abs(var_mc - var_theory) / var_std_err

        depths_grid = optical_depths / mu_val
        survival_mc = np.array(
            [float(np.mean(samples > depth_val)) for depth_val in depths_grid],
            dtype=np.float64,
        )

        full_bin_edges = np.concatenate(
            (
                np.array([0.0], dtype=np.float64),
                optical_depths,
                np.array([np.inf], dtype=np.float64),
            )
        )
        full_probs = np.concatenate(
            (
                np.array([1.0 - math.exp(-float(optical_depths[0]))], dtype=np.float64),
                np.exp(-optical_depths[:-1]) - np.exp(-optical_depths[1:]),
                np.array([math.exp(-float(optical_depths[-1]))], dtype=np.float64),
            )
        )
        expected_counts = float(num_histories) * full_probs
        observed_counts, _ = np.histogram(samples * mu_val, bins=full_bin_edges)
        chi2_val = float(
            np.sum((observed_counts.astype(np.float64) - expected_counts) ** 2 / expected_counts)
        )
        chi2_ndf = chi2_val / float(len(full_probs) - 1)

        item_passed = (
            mean_z_score < 3.5
            and variance_z_score < 3.5
            and (0.5 <= chi2_ndf <= 1.6)
        )
        metrics_list.append(
            FreePathMomentMetrics(
                material_name=mat_name,
                linear_attenuation=mu_val,
                num_histories=num_histories,
                mean_free_path_mc=mean_mc,
                mean_free_path_theory=mean_theory,
                mean_rel_error=mean_rel_err,
                mean_z_score=mean_z_score,
                variance_mc=var_mc,
                variance_theory=var_theory,
                variance_rel_error=var_rel_err,
                variance_z_score=variance_z_score,
                survival_chi2_ndf=chi2_ndf,
                passed=item_passed,
            )
        )

        if mat_idx == 0:
            plot_depths = depths_grid
            plot_trans_mc = survival_mc

    return metrics_list, plot_depths, plot_trans_mc, plot_trans_theory


def _run_production_two_layer_barrier_step(
    kappa_factor: float,
    kappa_idx: int,
    num_histories: int,
    probe_energy: Float,
    interface_depth: float,
    total_thickness: float,
    num_bins: int,
    seed: int,
) -> tuple[NDArray[np.int64], int, float, int]:
    """Прогоняет продакшен-транспорт (`ParticlePropagator` + `WoodcockParametricVolume`) через двухслойный барьер.

    Роль мажоранты играет синтетический материал на основе свинца с масштабированной плотностью kappa * rho_Pb.
    Возвращает (hist_counts, collided_count, sum_collision_depth, total_woodcock_steps).
    """
    vacuum_mat = database_setting.material_database["Vacuum"]
    water_mat = database_setting.material_database["Water, Liquid"]
    lead_mat = database_setting.material_database["Pb"]
    att_db = database_setting.attenuation_database

    max_existing_id = max(
        max((mat.ID for mat in database_setting.material_database.values()), default=0),
        max((mat.ID for mat in att_db.keys()), default=0),
    )
    scaled_majorant_mat = Material(
        name=f"SyntheticMajorantPb_kappa_{kappa_idx}",
        type=lead_mat.type,
        density=Float(kappa_factor * float(lead_mat.density)),
        composition=lead_mat.composition,
        ZtoA_ratio=lead_mat.ZtoA_ratio,
        ID=max_existing_id + 10 + kappa_idx,
    )
    att_db.add_material(scaled_majorant_mat)

    step_counter = np.zeros(1, dtype=np.int64)
    counter_addr = int(step_counter.ctypes.data)

    world = Volume(
        geometry=Box(Float(200.0), Float(200.0), Float(200.0)),
        material=vacuum_mat,
        name=f"DeltaBarrierWorld_{kappa_idx}",
    )
    barrier_vol = _TwoLayerBarrierParametricVolume(
        box_size=(100.0, 100.0, total_thickness),
        majorant_mat=scaled_majorant_mat,
        first_mat=water_mat,
        second_mat=lead_mat,
        interface_depth=interface_depth,
        counter_addr=counter_addr,
        name=f"TwoLayerBarrier_kappa_{kappa_idx}",
    )
    barrier_vol.translate(z=Float(0.5 * total_thickness))
    world.add_child(barrier_vol)

    rng = np.random.default_rng(seed)
    rng_ctx = RNGContext.from_numpy_rng(rng)
    propagator = ParticlePropagator(rng=rng)

    geometry_buffer = GeometryCompiler().compile_scene(world)
    physics_buffer = PhysicsCompiler().compile_scene(world, propagator.processes)

    hist_counts = np.zeros(num_bins, dtype=np.int64)
    bin_width = total_thickness / float(num_bins)
    collided_count = 0
    sum_collision_depth = 0.0

    batch_size = min(50_000, num_histories)
    bank = ParticleBank.allocate(batch_size)
    data_buffer = SimulationDataBuffer.allocate_for(
        capacity=batch_size * 2,
        enabled=frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
    )

    processed = 0
    while processed < num_histories:
        cur_batch = min(batch_size, num_histories - processed)
        active_indices = np.arange(cur_batch, dtype=Index)

        bank.state.is_active[:] = False
        bank.state.is_active[:cur_batch] = True

        bank.state.position.x[:cur_batch] = Float(0.0)
        bank.state.position.y[:cur_batch] = Float(0.0)
        bank.state.position.z[:cur_batch] = Float(-1.0)

        bank.state.direction.x[:cur_batch] = Float(0.0)
        bank.state.direction.y[:cur_batch] = Float(0.0)
        bank.state.direction.z[:cur_batch] = Float(1.0)

        bank.state.energy[:cur_batch] = probe_energy
        bank.state.distance_traveled[:cur_batch] = Float(0.0)
        bank.state.species[:cur_batch] = Index(ParticleSpecies.PHOTON)

        bank.navigation_state.current_volume[:cur_batch] = Index(-1)
        bank.navigation_state.boundary_distance[:cur_batch] = Float(0.0)

        particle_ids = np.arange(processed, processed + cur_batch, dtype=ID)
        bank.initial_state.ID[:cur_batch] = particle_ids
        bank.initial_state.parent_ID[:cur_batch] = ID(-1)
        bank.initial_state.emission_time[:cur_batch] = Float(0.0)
        bank.initial_state.emission_energy[:cur_batch] = probe_energy
        bank.initial_state.emission_position.x[:cur_batch] = Float(0.0)
        bank.initial_state.emission_position.y[:cur_batch] = Float(0.0)
        bank.initial_state.emission_position.z[:cur_batch] = Float(-1.0)
        bank.initial_state.emission_direction.x[:cur_batch] = Float(0.0)
        bank.initial_state.emission_direction.y[:cur_batch] = Float(0.0)
        bank.initial_state.emission_direction.z[:cur_batch] = Float(1.0)

        # Шаг 1: доводим пучок из вакуума z = -1.0 мм до входной грани барьера z = +1e-6 мм
        data_buffer.interactions.reset_cursor()
        data_buffer.initial_states.reset_cursor()
        propagator.step(
            bank=bank,
            data_buffer=data_buffer,
            geometry_buffer=geometry_buffer,
            physics_buffer=physics_buffer,
            rng_ctx=rng_ctx,
            active_indices=active_indices,
        )

        # Шаг 2: один вызов propagator.step выполняет дельта-трекинг Вудкока внутри барьера до первой коллизии
        data_buffer.interactions.reset_cursor()
        data_buffer.initial_states.reset_cursor()
        propagator.step(
            bank=bank,
            data_buffer=data_buffer,
            geometry_buffer=geometry_buffer,
            physics_buffer=physics_buffer,
            rng_ctx=rng_ctx,
            active_indices=active_indices,
        )

        num_events = data_buffer.interactions.cursor_value
        if num_events > 0:
            ev_z = data_buffer.interactions.position.z[:num_events]
            collided_count += int(num_events)
            sum_collision_depth += float(np.sum(ev_z))
            bin_indices = np.clip((ev_z / bin_width).astype(np.int64), 0, num_bins - 1)
            np.add.at(hist_counts, bin_indices, 1)

        processed += cur_batch

    return hist_counts, collided_count, sum_collision_depth, int(step_counter[0])


def _run_delta_majorant_invariance_audit(
    quick: bool,
) -> tuple[
    list[DeltaMajorantScalingMetrics],
    NDArray[np.float64],
    list[tuple[str, NDArray[np.float64]]],
    NDArray[np.float64],
]:
    """Проверяет инвариантность продакшен-ядра Вудкока при завышении мажоранты в барьере Water -> Lead."""
    num_histories = 60_000 if quick else 200_000
    probe_energy = Float(1.0 * units.MeV)

    water_mat = database_setting.material_database["Water, Liquid"]
    lead_mat = database_setting.material_database["Pb"]
    mu_water = float(compute_total_linear_attenuation(water_mat, probe_energy))
    mu_lead = float(compute_total_linear_attenuation(lead_mat, probe_energy))

    interface_depth = 25.0
    total_thickness = 50.0
    num_bins = 40
    bin_edges = np.linspace(0.0, total_thickness, num_bins + 1, dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = total_thickness / float(num_bins)

    def _optical_depth_at(coord_z: float) -> float:
        if coord_z <= interface_depth:
            return mu_water * coord_z
        return mu_water * interface_depth + mu_lead * (coord_z - interface_depth)

    bin_probs = np.array(
        [
            math.exp(-_optical_depth_at(float(bin_edges[idx])))
            - math.exp(-_optical_depth_at(float(bin_edges[idx + 1])))
            for idx in range(num_bins)
        ],
        dtype=np.float64,
    )
    theory_pdf = bin_probs / bin_width

    collided_prob_theory, mean_depth_theory, var_depth_theory, survival_integral = (
        _compute_two_layer_exact_moments(
            mu_first=mu_water,
            mu_second=mu_lead,
            interface_depth=interface_depth,
            total_thickness=total_thickness,
        )
    )

    kappa_values = [1.0, 2.5, 10.0, 50.0]
    base_majorant = max(mu_water, mu_lead)
    metrics_list: list[DeltaMajorantScalingMetrics] = []
    delta_profiles: list[tuple[str, NDArray[np.float64]]] = []

    for kappa_idx, kappa_val in enumerate(kappa_values):
        majorant_lac = kappa_val * base_majorant
        hist_counts, collided_count, sum_depth, total_steps = _run_production_two_layer_barrier_step(
            kappa_factor=kappa_val,
            kappa_idx=kappa_idx,
            num_histories=num_histories,
            probe_energy=probe_energy,
            interface_depth=interface_depth,
            total_thickness=total_thickness,
            num_bins=num_bins,
            seed=20315 + kappa_idx * 37,
        )

        pdf_mc = hist_counts.astype(np.float64) / (float(num_histories) * bin_width)
        delta_profiles.append((f"$\\kappa = {kappa_val:.1f}$", pdf_mc))

        expected_counts = float(num_histories) * bin_probs
        trans_prob_theory = 1.0 - collided_prob_theory
        trans_count_mc = num_histories - collided_count
        trans_expected = float(num_histories) * trans_prob_theory

        chi2_val = float(
            np.sum((hist_counts.astype(np.float64) - expected_counts) ** 2 / expected_counts)
            + ((float(trans_count_mc) - trans_expected) ** 2) / trans_expected
        )
        chi2_ndf = chi2_val / float(num_bins)

        mean_depth_mc = sum_depth / float(max(collided_count, 1))
        std_err_mean = math.sqrt(var_depth_theory / float(max(collided_count, 1)))
        z_score = (mean_depth_mc - mean_depth_theory) / std_err_mean

        mean_steps_per_coll_mc = float(total_steps) / float(max(collided_count, 1))
        mean_steps_per_coll_theory = (majorant_lac * survival_integral) / collided_prob_theory
        mean_steps_rel_err = (
            abs(mean_steps_per_coll_mc - mean_steps_per_coll_theory) / mean_steps_per_coll_theory
        )

        collided_frac_mc = float(collided_count) / float(num_histories)
        collided_frac_std_err = math.sqrt(
            collided_prob_theory * (1.0 - collided_prob_theory) / float(num_histories)
        )
        collided_frac_z_score = abs(collided_frac_mc - collided_prob_theory) / collided_frac_std_err
        steps_rel_tol = 4.0 / math.sqrt(float(max(collided_count, 1)))

        item_passed = (
            (0.5 <= chi2_ndf <= 1.6)
            and (abs(z_score) < 3.0)
            and (collided_frac_z_score < 3.5)
            and (mean_steps_rel_err < steps_rel_tol)
        )
        metrics_list.append(
            DeltaMajorantScalingMetrics(
                kappa_factor=kappa_val,
                num_histories=num_histories,
                collided_fraction_mc=collided_frac_mc,
                collided_fraction_theory=collided_prob_theory,
                collided_fraction_z_score=collided_frac_z_score,
                mean_depth_mc=mean_depth_mc,
                mean_depth_theory=mean_depth_theory,
                mean_depth_z_score=z_score,
                profile_chi2_ndf=chi2_ndf,
                mean_steps_per_collision=mean_steps_per_coll_mc,
                mean_steps_per_collision_theory=mean_steps_per_coll_theory,
                mean_steps_rel_error=mean_steps_rel_err,
                passed=item_passed,
            )
        )

    return metrics_list, bin_centers, delta_profiles, theory_pdf


def _build_five_layer_scene(representation_kind: str) -> Volume:
    """Создает сцену с 5-слойным гетерогенным фантомом в одном из трех представлений.

    Порядок слоев вдоль оси Z (от z = 0.0 до z = 51.8 мм, свинец переставлен в конец):
    1. Water, Liquid: [0.0, 20.0] мм (толщина 20.0 мм)
    2. Bone, Cortical (ICRU-44): [20.0, 30.0] мм (толщина 10.0 мм)
    3. Air, Dry (near sea level): [30.0, 45.0] мм (толщина 15.0 мм)
    4. Sodium Iodide: [45.0, 51.0] мм (толщина 6.0 мм)
    5. Pb: [51.0, 51.8] мм (толщина 0.8 мм)

    Суммарная оптическая толщина при 140.5 кэВ составляет ~4.33 MFP (T_uncollided ~ 1.32%),
    обеспечивая высокую статистическую мощность во всех 5 слоях (включая Air и NaI)
    и более 1000 баллистических фотонов на выходе.
    Для `WoodcockVoxelVolume` используется настоящая 3D-воксельная сетка 16 x 16 x 259.
    """
    vacuum_mat = database_setting.material_database["Vacuum"]
    water_mat = database_setting.material_database["Water, Liquid"]
    bone_mat = database_setting.material_database["Bone, Cortical (ICRU-44)"]
    air_mat = database_setting.material_database["Air, Dry (near sea level)"]
    nai_mat = database_setting.material_database["Sodium Iodide"]
    lead_mat = database_setting.material_database["Pb"]

    world = Volume(
        geometry=Box(Float(400.0), Float(400.0), Float(400.0)),
        material=vacuum_mat,
        name=f"World_{representation_kind}",
    )

    lateral_span = 200.0
    total_thickness = 51.8

    if representation_kind == "direct_slabs":
        slab_specs = [
            ("Slab_Water", water_mat, 20.0, 10.0),
            ("Slab_Bone", bone_mat, 10.0, 25.0),
            ("Slab_Air", air_mat, 15.0, 37.5),
            ("Slab_NaI", nai_mat, 6.0, 48.0),
            ("Slab_Lead", lead_mat, 0.8, 51.4),
        ]
        for slab_name, slab_mat, slab_thickness, slab_center_z in slab_specs:
            slab_vol = Volume(
                geometry=Box(Float(lateral_span), Float(lateral_span), Float(slab_thickness)),
                material=slab_mat,
                name=slab_name,
            )
            slab_vol.translate(z=Float(slab_center_z))
            world.add_child(slab_vol)

    elif representation_kind == "woodcock_voxel":
        # Настоящая 3D-воксельная сетка 16 x 16 x 259 (шаг 12.5 x 12.5 x 0.2 мм)
        mat_array = MaterialArray((16, 16, 259))
        mat_array[:, :, 0:100] = water_mat
        mat_array[:, :, 100:150] = bone_mat
        mat_array[:, :, 150:225] = air_mat
        mat_array[:, :, 225:255] = nai_mat
        mat_array[:, :, 255:259] = lead_mat

        voxel_vol = WoodcockVoxelVolume(
            voxel_size=np.array([12.5, 12.5, 0.2], dtype=Float),
            material_distribution=mat_array,
            name="WoodcockVoxelPhantom",
        )
        voxel_vol.translate(z=Float(0.5 * total_thickness))
        world.add_child(voxel_vol)

    elif representation_kind == "woodcock_parametric":
        param_vol = _FiveLayerParametricVolume(
            box_size=(lateral_span, lateral_span, total_thickness),
            majorant_mat=lead_mat,
            layers_materials=[water_mat, bone_mat, air_mat, nai_mat, lead_mat],
            name="WoodcockParametricPhantom",
        )
        param_vol.translate(z=Float(0.5 * total_thickness))
        world.add_child(param_vol)

    else:
        raise ValueError(f"Неизвестный тип представления фантома: {representation_kind}")

    return world


@dataclass(frozen=True)
class _PhantomBeamTransportResult:
    """Внутренняя сводка прогона пучка через 5-слойный гетерогенный фантом."""

    layer_edep_sum: NDArray[np.float64]
    layer_edep_sq_sum: NDArray[np.float64]
    total_edep_sum: float
    total_edep_sq_sum: float
    process_counts: dict[str, int]
    transmitted_count: int
    uncollided_transmitted_count: int
    absorbed_layer_counts: NDArray[np.int64]
    reflected_count: int
    lateral_escape_count: int
    max_history_energy_imbalance: float
    energy_balance_violations: int


def _run_phantom_beam_transport(
    world: Volume,
    num_histories: int,
    beam_energy: Float,
    beam_direction: tuple[float, float, float],
    seed: int,
) -> _PhantomBeamTransportResult:
    """Прогоняет наклонный 3D-пучок фотонов через 5-слойный фантом с SecondarySource и энергобалансом."""
    rng = np.random.default_rng(seed)
    rng_ctx = RNGContext.from_numpy_rng(rng)
    propagator = ParticlePropagator(rng=rng)

    geometry_buffer = GeometryCompiler().compile_scene(world)
    physics_buffer = PhysicsCompiler().compile_scene(world, propagator.processes)

    layer_edges = np.array([0.0, 20.0, 30.0, 45.0, 51.0, 51.8], dtype=np.float64)
    num_layers = 5
    layer_edep_sum = np.zeros(num_layers, dtype=np.float64)
    layer_edep_sq_sum = np.zeros(num_layers, dtype=np.float64)
    total_edep_sum = 0.0
    total_edep_sq_sum = 0.0

    pe_code = int(ProcessType.PHOTOELECTRIC)
    compton_code = int(ProcessType.COMPTON)
    coherent_code = int(ProcessType.COHERENT)
    process_counts: dict[str, int] = {
        "PhotoelectricEffect": 0,
        "ComptonScattering": 0,
        "CoherentScattering": 0,
    }

    transmitted_count = 0
    uncollided_transmitted_count = 0
    absorbed_layer_counts = np.zeros(num_layers, dtype=np.int64)
    reflected_count = 0
    lateral_escape_count = 0
    max_energy_imbalance = 0.0
    energy_balance_violations = 0

    batch_size = min(40_000, num_histories)
    bank = ParticleBank.allocate(batch_size)
    data_buffer = SimulationDataBuffer.allocate_for(
        capacity=batch_size * 4,
        enabled=frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
    )
    secondary_source = SecondarySource(
        activation_buffer=SecondaryActivationBuffer.allocate(
            capacity=batch_size * 3,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        ),
        relaxation_bank=physics_buffer.relaxation_bank,
        stop_time=Float(np.inf),
    )

    min_energy = Float(1.0 * units.keV)
    phantom_exit_z = 51.8
    half_lateral = 100.0
    dir_x, dir_y, dir_z = beam_direction
    beam_energy_val = float(beam_energy)

    # Отдельный детерминированный генератор для начальных поперечных координат (x_0, y_0) пучка
    beam_pos_rng = np.random.default_rng(30399)
    processed_histories = 0

    while processed_histories < num_histories:
        cur_batch = min(batch_size, num_histories - processed_histories)
        start_x = beam_pos_rng.uniform(-40.0, 40.0, size=cur_batch).astype(Float)
        start_y = beam_pos_rng.uniform(-40.0, 40.0, size=cur_batch).astype(Float)

        bank.state.is_active[:] = False
        bank.state.is_active[:cur_batch] = True

        bank.state.position.x[:cur_batch] = start_x
        bank.state.position.y[:cur_batch] = start_y
        bank.state.position.z[:cur_batch] = Float(-0.5)

        bank.state.direction.x[:cur_batch] = Float(dir_x)
        bank.state.direction.y[:cur_batch] = Float(dir_y)
        bank.state.direction.z[:cur_batch] = Float(dir_z)

        bank.state.energy[:cur_batch] = beam_energy
        bank.state.distance_traveled[:cur_batch] = Float(0.0)
        bank.state.species[:cur_batch] = Index(ParticleSpecies.PHOTON)

        bank.navigation_state.current_volume[:cur_batch] = Index(-1)
        bank.navigation_state.boundary_distance[:cur_batch] = Float(0.0)

        local_ids = np.arange(cur_batch, dtype=ID)
        bank.initial_state.ID[:cur_batch] = local_ids
        bank.initial_state.parent_ID[:cur_batch] = ID(-1)
        bank.initial_state.emission_time[:cur_batch] = Float(0.0)
        bank.initial_state.emission_energy[:cur_batch] = beam_energy
        bank.initial_state.emission_position.x[:cur_batch] = start_x
        bank.initial_state.emission_position.y[:cur_batch] = start_y
        bank.initial_state.emission_position.z[:cur_batch] = Float(-0.5)
        bank.initial_state.emission_direction.x[:cur_batch] = Float(dir_x)
        bank.initial_state.emission_direction.y[:cur_batch] = Float(dir_y)
        bank.initial_state.emission_direction.z[:cur_batch] = Float(dir_z)
        bank.count_array[0] = Index(cur_batch)
        secondary_source.activation_buffer.reset_cursor()

        max_possible_ids = cur_batch * 8
        root_history_map = np.full(max_possible_ids, -1, dtype=np.int64)
        root_history_map[:cur_batch] = np.arange(cur_batch, dtype=np.int64)

        history_layer_edep = np.zeros((cur_batch, num_layers), dtype=np.float64)
        history_edep = np.zeros(cur_batch, dtype=np.float64)
        history_escaped_energy = np.zeros(cur_batch, dtype=np.float64)
        history_cutoff_energy = np.zeros(cur_batch, dtype=np.float64)
        primary_interacted = np.zeros(cur_batch, dtype=np.bool_)

        while True:
            if secondary_source.has_pending:
                secondary_source.replenish_bank(bank, rng_ctx, start_slot_cursor=Index(0))
                active_now = bank.active_indices
                if active_now.size > 0:
                    act_ids = bank.initial_state.ID[active_now]
                    act_parents = bank.initial_state.parent_ID[active_now]
                    new_mask = root_history_map[act_ids] < 0
                    if np.any(new_mask):
                        new_ids = act_ids[new_mask]
                        new_parents = act_parents[new_mask]
                        root_history_map[new_ids] = root_history_map[new_parents]

            active_indices = bank.active_indices
            if active_indices.size == 0:
                if secondary_source.has_pending:
                    continue
                break

            data_buffer.interactions.reset_cursor()
            data_buffer.initial_states.reset_cursor()

            propagator.step(
                bank=bank,
                data_buffer=data_buffer,
                geometry_buffer=geometry_buffer,
                physics_buffer=physics_buffer,
                rng_ctx=rng_ctx,
                active_indices=active_indices,
                activation_buffer=secondary_source.activation_buffer,
            )

            num_events = data_buffer.interactions.cursor_value
            if num_events > 0:
                inter_buf = data_buffer.interactions
                ev_pids = inter_buf.particle_ID[:num_events]
                ev_roots = root_history_map[ev_pids]
                ev_z = inter_buf.position.z[:num_events]
                ev_edep = inter_buf.energy_deposit[:num_events]
                ev_proc = inter_buf.process_id[:num_events]

                primary_ev_mask = ev_pids < cur_batch
                if np.any(primary_ev_mask):
                    primary_interacted[ev_pids[primary_ev_mask]] = True

                process_counts["PhotoelectricEffect"] += int(np.sum(ev_proc == pe_code))
                process_counts["ComptonScattering"] += int(np.sum(ev_proc == compton_code))
                process_counts["CoherentScattering"] += int(np.sum(ev_proc == coherent_code))

                layer_indices = np.clip(
                    np.searchsorted(layer_edges, ev_z, side="right") - 1,
                    0,
                    num_layers - 1,
                )
                pos_edep_mask = ev_edep > 0.0
                if np.any(pos_edep_mask):
                    np.add.at(
                        history_layer_edep,
                        (ev_roots[pos_edep_mask], layer_indices[pos_edep_mask]),
                        ev_edep[pos_edep_mask],
                    )
                    np.add.at(history_edep, ev_roots[pos_edep_mask], ev_edep[pos_edep_mask])

            # Классификация завершения треков после шага транспорта
            act_pids = bank.initial_state.ID[active_indices]
            act_roots = root_history_map[act_pids]
            act_is_primary = act_pids < cur_batch

            act_x = bank.state.position.x[active_indices]
            act_y = bank.state.position.y[active_indices]
            act_z = bank.state.position.z[active_indices]
            act_dz = bank.state.direction.z[active_indices]
            act_energy = bank.state.energy[active_indices]
            act_vol = bank.navigation_state.current_volume[active_indices]
            act_bdist = bank.navigation_state.boundary_distance[active_indices]

            below_cutoff = act_energy < min_energy
            alive_energy = ~below_cutoff

            # Ортогональные геометрические условия выхода из фантома [-100, 100]^2 x [0, 51.8]
            exited_lateral = alive_energy & ((np.abs(act_x) > half_lateral) | (np.abs(act_y) > half_lateral))
            inside_aperture = ~exited_lateral
            exited_forward = alive_energy & inside_aperture & (act_z >= phantom_exit_z) & (act_dz > 0.0)
            exited_backward = alive_energy & inside_aperture & (act_z <= 0.0) & (act_dz < 0.0)
            left_world = alive_energy & (~exited_lateral) & (~exited_forward) & (~exited_backward) & (
                (act_vol < 0) & (act_bdist > 0.0)
            )
            escaped_any = exited_lateral | exited_forward | exited_backward | left_world

            if np.any(escaped_any):
                np.add.at(history_escaped_energy, act_roots[escaped_any], act_energy[escaped_any])

            if np.any(below_cutoff):
                residual_pos = below_cutoff & (act_energy > 0.0)
                if np.any(residual_pos):
                    np.add.at(history_cutoff_energy, act_roots[residual_pos], act_energy[residual_pos])

            # Ортогональная классификация исходов для первичных историй
            prim_forward = act_is_primary & exited_forward
            if np.any(prim_forward):
                transmitted_count += int(np.sum(prim_forward))
                prim_forward_pids = act_pids[prim_forward]
                uncollided_transmitted_count += int(np.sum(~primary_interacted[prim_forward_pids]))

            prim_backward = act_is_primary & (exited_backward | left_world)
            if np.any(prim_backward):
                reflected_count += int(np.sum(prim_backward))

            prim_lateral = act_is_primary & exited_lateral
            if np.any(prim_lateral):
                lateral_escape_count += int(np.sum(prim_lateral))

            prim_absorbed = act_is_primary & below_cutoff
            if np.any(prim_absorbed):
                abs_z = act_z[prim_absorbed]
                abs_layers = np.clip(
                    np.searchsorted(layer_edges, abs_z, side="right") - 1,
                    0,
                    num_layers - 1,
                )
                np.add.at(absorbed_layer_counts, abs_layers, 1)

            stop_mask = escaped_any | below_cutoff
            if np.any(stop_mask):
                bank.state.is_active[active_indices[stop_mask]] = False

        # Проверка пособытийного баланса энергии E_0 = E_dep + E_esc + E_cut для всех историй батча
        batch_accounted = history_edep + history_escaped_energy + history_cutoff_energy
        batch_imbalances = np.abs(beam_energy_val - batch_accounted)
        batch_max_imb = float(np.max(batch_imbalances))
        if batch_max_imb > max_energy_imbalance:
            max_energy_imbalance = batch_max_imb
        energy_balance_violations += int(np.sum(batch_imbalances >= 1.0e-12))

        layer_edep_sum += np.sum(history_layer_edep, axis=0)
        layer_edep_sq_sum += np.sum(history_layer_edep * history_layer_edep, axis=0)
        total_edep_sum += float(np.sum(history_edep))
        total_edep_sq_sum += float(np.sum(history_edep * history_edep))
        processed_histories += cur_batch

    return _PhantomBeamTransportResult(
        layer_edep_sum=layer_edep_sum,
        layer_edep_sq_sum=layer_edep_sq_sum,
        total_edep_sum=total_edep_sum,
        total_edep_sq_sum=total_edep_sq_sum,
        process_counts=process_counts,
        transmitted_count=transmitted_count,
        uncollided_transmitted_count=uncollided_transmitted_count,
        absorbed_layer_counts=absorbed_layer_counts,
        reflected_count=reflected_count,
        lateral_escape_count=lateral_escape_count,
        max_history_energy_imbalance=max_energy_imbalance,
        energy_balance_violations=energy_balance_violations,
    )


def _run_voxel_3d_oblique_ray_audit(quick: bool) -> Voxel3DObliqueRayMetrics:
    """Верифицирует 3D-индексацию `WoodcockVoxelVolume` на 3D-шахматной сетке 16x16x16 при косом падении."""
    num_histories = 30_000 if quick else 100_000
    beam_energy = Float(140.5 * units.keV)
    grid_dim = 16
    box_span = 64.0
    vox_step = box_span / float(grid_dim)

    vacuum_mat = database_setting.material_database["Vacuum"]
    water_mat = database_setting.material_database["Water, Liquid"]
    bone_mat = database_setting.material_database["Bone, Cortical (ICRU-44)"]
    air_mat = database_setting.material_database["Air, Dry (near sea level)"]
    nai_mat = database_setting.material_database["Sodium Iodide"]
    palette = [water_mat, bone_mat, air_mat, nai_mat]

    mat_array = MaterialArray((grid_dim, grid_dim, grid_dim))
    for ix in range(grid_dim):
        for iy in range(grid_dim):
            for iz in range(grid_dim):
                mat_array[ix, iy, iz] = palette[(ix + 2 * iy + 3 * iz) % 4]

    world_vox = Volume(
        geometry=Box(Float(200.0), Float(200.0), Float(200.0)),
        material=vacuum_mat,
        name="World3DVoxel",
    )
    vox_vol = WoodcockVoxelVolume(
        voxel_size=np.array([vox_step, vox_step, vox_step], dtype=Float),
        material_distribution=mat_array,
        name="Checkerboard3DVoxel",
    )
    vox_vol.translate(z=Float(0.5 * box_span))
    world_vox.add_child(vox_vol)

    world_param = Volume(
        geometry=Box(Float(200.0), Float(200.0), Float(200.0)),
        material=vacuum_mat,
        name="World3DParametric",
    )
    param_vol = _Checkerboard3DParametricVolume(
        box_span=box_span,
        grid_dim=grid_dim,
        majorant_mat=nai_mat,
        materials_palette=palette,
        name="Checkerboard3DParametric",
    )
    param_vol.translate(z=Float(0.5 * box_span))
    world_param.add_child(param_vol)

    # Сильно наклонный 3D-луч, пересекающий множество вокселей по всем трем осям X, Y, Z
    oblique_dir = (0.28, -0.21, math.sqrt(1.0 - 0.28 * 0.28 - 0.21 * 0.21))

    def _run_3d_scene(scene_world: Volume, seed: int) -> tuple[float, float, int]:
        rng = np.random.default_rng(seed)
        rng_ctx = RNGContext.from_numpy_rng(rng)
        propagator = ParticlePropagator(rng=rng)
        geom_buf = GeometryCompiler().compile_scene(scene_world)
        phys_buf = PhysicsCompiler().compile_scene(scene_world, propagator.processes)

        bank = ParticleBank.allocate(num_histories)
        data_buf = SimulationDataBuffer.allocate_for(
            capacity=num_histories * 4,
            enabled=frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
        )
        pos_rng = np.random.default_rng(40499)
        start_x = pos_rng.uniform(-8.0, 8.0, size=num_histories).astype(Float)
        start_y = pos_rng.uniform(-8.0, 8.0, size=num_histories).astype(Float)

        bank.state.is_active[:] = True
        bank.state.position.x[:] = start_x
        bank.state.position.y[:] = start_y
        bank.state.position.z[:] = Float(-0.5)
        bank.state.direction.x[:] = Float(oblique_dir[0])
        bank.state.direction.y[:] = Float(oblique_dir[1])
        bank.state.direction.z[:] = Float(oblique_dir[2])
        bank.state.energy[:] = beam_energy
        bank.state.distance_traveled[:] = Float(0.0)
        bank.state.species[:] = Index(ParticleSpecies.PHOTON)
        bank.navigation_state.current_volume[:] = Index(-1)
        bank.navigation_state.boundary_distance[:] = Float(0.0)

        pids = np.arange(num_histories, dtype=ID)
        bank.initial_state.ID[:] = pids
        bank.initial_state.parent_ID[:] = ID(-1)
        bank.initial_state.emission_time[:] = Float(0.0)
        bank.initial_state.emission_energy[:] = beam_energy
        bank.initial_state.emission_position.x[:] = start_x
        bank.initial_state.emission_position.y[:] = start_y
        bank.initial_state.emission_position.z[:] = Float(-0.5)
        bank.initial_state.emission_direction.x[:] = Float(oblique_dir[0])
        bank.initial_state.emission_direction.y[:] = Float(oblique_dir[1])
        bank.initial_state.emission_direction.z[:] = Float(oblique_dir[2])

        hist_edep = np.zeros(num_histories, dtype=np.float64)
        interacted = np.zeros(num_histories, dtype=np.bool_)
        uncollided_count = 0
        min_ene = Float(1.0 * units.keV)
        half_span = 0.5 * box_span

        while True:
            act_idx = bank.active_indices
            if act_idx.size == 0:
                break
            data_buf.interactions.reset_cursor()
            data_buf.initial_states.reset_cursor()
            propagator.step(
                bank=bank,
                data_buffer=data_buf,
                geometry_buffer=geom_buf,
                physics_buffer=phys_buf,
                rng_ctx=rng_ctx,
                active_indices=act_idx,
            )
            num_ev = data_buf.interactions.cursor_value
            if num_ev > 0:
                ev_pids = data_buf.interactions.particle_ID[:num_ev]
                ev_edep = data_buf.interactions.energy_deposit[:num_ev]
                interacted[ev_pids] = True
                np.add.at(hist_edep, ev_pids, ev_edep)

            ax = bank.state.position.x[act_idx]
            ay = bank.state.position.y[act_idx]
            az = bank.state.position.z[act_idx]
            adz = bank.state.direction.z[act_idx]
            aene = bank.state.energy[act_idx]
            avol = bank.navigation_state.current_volume[act_idx]
            abdist = bank.navigation_state.boundary_distance[act_idx]
            apids = bank.initial_state.ID[act_idx]

            below = aene < min_ene
            lat_out = (~below) & ((np.abs(ax) > half_span) | (np.abs(ay) > half_span))
            fwd_out = (~below) & (~lat_out) & (az >= box_span) & (adz > 0.0)
            bwd_out = (~below) & (~lat_out) & (az <= 0.0) & (adz < 0.0)
            world_out = (~below) & (avol < 0) & (abdist > 0.0)

            if np.any(fwd_out):
                uncollided_count += int(np.sum(~interacted[apids[fwd_out]]))

            stop = below | lat_out | fwd_out | bwd_out | world_out
            if np.any(stop):
                bank.state.is_active[act_idx[stop]] = False

        return float(np.sum(hist_edep)), float(np.sum(hist_edep * hist_edep)), uncollided_count

    edep_vox, sq_vox, uncoll_vox = _run_3d_scene(world_vox, seed=40401)
    edep_par, sq_par, uncoll_par = _run_3d_scene(world_param, seed=40402)

    edep_z = _compute_edep_pair_z_score(edep_vox, sq_vox, edep_par, sq_par, num_histories)
    uncoll_z = _compute_count_pair_z_score(uncoll_vox, uncoll_par)
    passed = (edep_z < 3.0) and (uncoll_z < 3.0) and (uncoll_vox > 50)

    return Voxel3DObliqueRayMetrics(
        grid_shape=[grid_dim, grid_dim, grid_dim],
        num_histories=num_histories,
        beam_direction=[float(oblique_dir[0]), float(oblique_dir[1]), float(oblique_dir[2])],
        total_edep_voxel_3d=edep_vox,
        total_edep_parametric_3d=edep_par,
        edep_z_score=edep_z,
        uncollided_fraction_voxel_3d=float(uncoll_vox) / float(num_histories),
        uncollided_fraction_parametric_3d=float(uncoll_par) / float(num_histories),
        uncollided_z_score=uncoll_z,
        passed=passed,
    )


def _compute_edep_pair_z_score(
    sum_first: float,
    sq_sum_first: float,
    sum_second: float,
    sq_sum_second: float,
    num_histories: int,
) -> float:
    """Вычисляет двухвыборочный Z-score разности суммарных энерговыделений в слое."""
    mean_first = sum_first / float(num_histories)
    mean_second = sum_second / float(num_histories)
    var_first = max(sq_sum_first / float(num_histories) - mean_first * mean_first, 0.0)
    var_second = max(sq_sum_second / float(num_histories) - mean_second * mean_second, 0.0)
    denom = math.sqrt(float(num_histories) * (var_first + var_second))
    if denom <= 1e-15:
        return 0.0
    return abs(sum_first - sum_second) / denom


def _compute_count_pair_z_score(count_first: int, count_second: int) -> float:
    """Вычисляет двухвыборочный пуассоновский Z-score для пары счетчиков событий."""
    variance_sum = float(max(count_first + count_second, 1))
    return abs(float(count_first - count_second)) / math.sqrt(variance_sum)


def _run_five_layer_triple_equivalence_audit(
    quick: bool,
) -> HeterogeneousPhantomParityMetrics:
    """Выполняет тройную сверку Direct Slabs, 3D WoodcockVoxelVolume и WoodcockParametricVolume."""
    num_histories = 80_000 if quick else 300_000
    beam_energy = Float(140.5 * units.keV)
    layer_names = ["Water", "Bone", "Air", "NaI", "Lead"]
    layer_thicknesses = [20.0, 10.0, 15.0, 6.0, 0.8]
    voxel_grid_shape = [16, 16, 259]

    water_mat = database_setting.material_database["Water, Liquid"]
    bone_mat = database_setting.material_database["Bone, Cortical (ICRU-44)"]
    air_mat = database_setting.material_database["Air, Dry (near sea level)"]
    nai_mat = database_setting.material_database["Sodium Iodide"]
    lead_mat = database_setting.material_database["Pb"]
    layer_materials = [water_mat, bone_mat, air_mat, nai_mat, lead_mat]

    # Косой 3D-луч с направляющими косинусами (dx, dy, dz) для пересечения граней вокселей по X, Y, Z
    dir_x = 0.016
    dir_y = -0.012
    dir_z = math.sqrt(1.0 - dir_x * dir_x - dir_y * dir_y)
    beam_direction = (dir_x, dir_y, dir_z)

    total_optical_depth = sum(
        float(compute_total_linear_attenuation(mat_obj, beam_energy)) * (thickness / dir_z)
        for mat_obj, thickness in zip(layer_materials, layer_thicknesses)
    )
    uncollided_theory = math.exp(-total_optical_depth)

    scene_direct = _build_five_layer_scene("direct_slabs")
    scene_voxel = _build_five_layer_scene("woodcock_voxel")
    scene_parametric = _build_five_layer_scene("woodcock_parametric")

    res_dir = _run_phantom_beam_transport(
        world=scene_direct,
        num_histories=num_histories,
        beam_energy=beam_energy,
        beam_direction=beam_direction,
        seed=30301,
    )
    res_vox = _run_phantom_beam_transport(
        world=scene_voxel,
        num_histories=num_histories,
        beam_energy=beam_energy,
        beam_direction=beam_direction,
        seed=30302,
    )
    res_par = _run_phantom_beam_transport(
        world=scene_parametric,
        num_histories=num_histories,
        beam_energy=beam_energy,
        beam_direction=beam_direction,
        seed=30303,
    )

    total_dir = res_dir.total_edep_sum
    total_vox = res_vox.total_edep_sum
    total_par = res_par.total_edep_sum
    mean_total = (total_dir + total_vox + total_par) / 3.0
    max_total_rel_diff = max(
        abs(total_dir - total_vox) / mean_total,
        abs(total_dir - total_par) / mean_total,
        abs(total_vox - total_par) / mean_total,
    )

    # Статистическая стандартная ошибка SE(N) относительной разности полных выделенных энергий
    var_tot_dir = max(res_dir.total_edep_sq_sum / float(num_histories) - (total_dir / float(num_histories)) ** 2, 0.0)
    var_tot_vox = max(res_vox.total_edep_sq_sum / float(num_histories) - (total_vox / float(num_histories)) ** 2, 0.0)
    var_tot_par = max(res_par.total_edep_sq_sum / float(num_histories) - (total_par / float(num_histories)) ** 2, 0.0)
    max_pair_var_sum = max(
        var_tot_dir + var_tot_vox,
        var_tot_dir + var_tot_par,
        var_tot_vox + var_tot_par,
    )
    total_edep_rel_diff_se = math.sqrt(float(num_histories) * max_pair_var_sum) / mean_total
    total_edep_rel_diff_tolerance = 3.5 * total_edep_rel_diff_se

    max_layer_z = 0.0
    for layer_idx in range(len(layer_names)):
        z_dv = _compute_edep_pair_z_score(
            float(res_dir.layer_edep_sum[layer_idx]),
            float(res_dir.layer_edep_sq_sum[layer_idx]),
            float(res_vox.layer_edep_sum[layer_idx]),
            float(res_vox.layer_edep_sq_sum[layer_idx]),
            num_histories,
        )
        z_dp = _compute_edep_pair_z_score(
            float(res_dir.layer_edep_sum[layer_idx]),
            float(res_dir.layer_edep_sq_sum[layer_idx]),
            float(res_par.layer_edep_sum[layer_idx]),
            float(res_par.layer_edep_sq_sum[layer_idx]),
            num_histories,
        )
        z_vp = _compute_edep_pair_z_score(
            float(res_vox.layer_edep_sum[layer_idx]),
            float(res_vox.layer_edep_sq_sum[layer_idx]),
            float(res_par.layer_edep_sum[layer_idx]),
            float(res_par.layer_edep_sq_sum[layer_idx]),
            num_histories,
        )
        max_layer_z = max(max_layer_z, z_dv, z_dp, z_vp)

    max_proc_z = 0.0
    for proc_name in ("PhotoelectricEffect", "ComptonScattering", "CoherentScattering"):
        z_dv = _compute_count_pair_z_score(
            res_dir.process_counts[proc_name], res_vox.process_counts[proc_name]
        )
        z_dp = _compute_count_pair_z_score(
            res_dir.process_counts[proc_name], res_par.process_counts[proc_name]
        )
        z_vp = _compute_count_pair_z_score(
            res_vox.process_counts[proc_name], res_par.process_counts[proc_name]
        )
        max_proc_z = max(max_proc_z, z_dv, z_dp, z_vp)

    max_trans_z = max(
        _compute_count_pair_z_score(res_dir.transmitted_count, res_vox.transmitted_count),
        _compute_count_pair_z_score(res_dir.transmitted_count, res_par.transmitted_count),
        _compute_count_pair_z_score(res_vox.transmitted_count, res_par.transmitted_count),
    )

    uncoll_dir = float(res_dir.uncollided_transmitted_count) / float(num_histories)
    uncoll_vox = float(res_vox.uncollided_transmitted_count) / float(num_histories)
    uncoll_par = float(res_par.uncollided_transmitted_count) / float(num_histories)
    uncoll_se = math.sqrt(uncollided_theory * (1.0 - uncollided_theory) / float(num_histories))
    max_uncoll_z = max(
        abs(uncoll_dir - uncollided_theory) / uncoll_se,
        abs(uncoll_vox - uncollided_theory) / uncoll_se,
        abs(uncoll_par - uncollided_theory) / uncoll_se,
    )

    # Проверка полноты ортогонального разбиения исходов во всех трех представлениях
    sum_outcomes_dir = (
        int(np.sum(res_dir.absorbed_layer_counts))
        + res_dir.transmitted_count
        + res_dir.reflected_count
        + res_dir.lateral_escape_count
    )
    sum_outcomes_vox = (
        int(np.sum(res_vox.absorbed_layer_counts))
        + res_vox.transmitted_count
        + res_vox.reflected_count
        + res_vox.lateral_escape_count
    )
    sum_outcomes_par = (
        int(np.sum(res_par.absorbed_layer_counts))
        + res_par.transmitted_count
        + res_par.reflected_count
        + res_par.lateral_escape_count
    )
    outcome_partition_complete = (
        (sum_outcomes_dir == num_histories)
        and (sum_outcomes_vox == num_histories)
        and (sum_outcomes_par == num_histories)
    )

    max_history_imbalance = max(
        res_dir.max_history_energy_imbalance,
        res_vox.max_history_energy_imbalance,
        res_par.max_history_energy_imbalance,
    )
    total_balance_violations = (
        res_dir.energy_balance_violations
        + res_vox.energy_balance_violations
        + res_par.energy_balance_violations
    )

    voxel_3d_metrics = _run_voxel_3d_oblique_ray_audit(quick=quick)

    passed = (
        (max_total_rel_diff < total_edep_rel_diff_tolerance)
        and (max_layer_z < 3.5)
        and (max_proc_z < 3.5)
        and (max_trans_z < 3.5)
        and (max_uncoll_z < 3.5)
        and outcome_partition_complete
        and (total_balance_violations == 0)
        and (max_history_imbalance < 1.0e-12)
        and voxel_3d_metrics.passed
    )

    inv_n = 1.0 / float(num_histories)
    return HeterogeneousPhantomParityMetrics(
        num_histories=num_histories,
        beam_energy=float(beam_energy),
        layer_names=layer_names,
        layer_thicknesses=layer_thicknesses,
        voxel_grid_shape=voxel_grid_shape,
        edep_direct_slabs=[float(val) for val in res_dir.layer_edep_sum],
        edep_woodcock_voxel=[float(val) for val in res_vox.layer_edep_sum],
        edep_woodcock_parametric=[float(val) for val in res_par.layer_edep_sum],
        total_edep_direct=total_dir,
        total_edep_voxel=total_vox,
        total_edep_parametric=total_par,
        max_total_edep_rel_diff=max_total_rel_diff,
        total_edep_rel_diff_se=total_edep_rel_diff_se,
        total_edep_rel_diff_tolerance=total_edep_rel_diff_tolerance,
        max_layer_edep_z_score=max_layer_z,
        process_counts_direct=res_dir.process_counts,
        process_counts_voxel=res_vox.process_counts,
        process_counts_parametric=res_par.process_counts,
        max_process_count_z_score=max_proc_z,
        transmitted_fraction_direct=float(res_dir.transmitted_count) * inv_n,
        transmitted_fraction_voxel=float(res_vox.transmitted_count) * inv_n,
        transmitted_fraction_parametric=float(res_par.transmitted_count) * inv_n,
        max_transmitted_z_score=max_trans_z,
        uncollided_transmitted_theory=uncollided_theory,
        uncollided_transmitted_direct=uncoll_dir,
        uncollided_transmitted_voxel=uncoll_vox,
        uncollided_transmitted_parametric=uncoll_par,
        max_uncollided_vs_theory_z_score=max_uncoll_z,
        absorbed_layer_fractions_direct=[
            float(val) * inv_n for val in res_dir.absorbed_layer_counts
        ],
        absorbed_layer_fractions_voxel=[
            float(val) * inv_n for val in res_vox.absorbed_layer_counts
        ],
        absorbed_layer_fractions_parametric=[
            float(val) * inv_n for val in res_par.absorbed_layer_counts
        ],
        reflected_fraction_direct=float(res_dir.reflected_count) * inv_n,
        reflected_fraction_voxel=float(res_vox.reflected_count) * inv_n,
        reflected_fraction_parametric=float(res_par.reflected_count) * inv_n,
        lateral_escape_fraction_direct=float(res_dir.lateral_escape_count) * inv_n,
        lateral_escape_fraction_voxel=float(res_vox.lateral_escape_count) * inv_n,
        lateral_escape_fraction_parametric=float(res_par.lateral_escape_count) * inv_n,
        outcome_partition_complete=outcome_partition_complete,
        max_history_energy_imbalance=max_history_imbalance,
        energy_balance_violations=total_balance_violations,
        voxel_3d_oblique_metrics=voxel_3d_metrics,
        passed=passed,
    )


def run_exp03_woodcock_and_delta_tracking(
    output_dir: Path,
    quick: bool = False,
) -> Exp03WoodcockTrackingReport:
    """Запускает Эксперимент 3 (верификация дельта-трекинга и трекинга Вудкока) и сохраняет отчет.

    Args:
        output_dir: Директория для сохранения JSON-отчета и PNG-графика.
        quick: Если True, использует уменьшенный размер выборки для быстрой проверки.

    Returns:
        Структурированный отчет Exp03WoodcockTrackingReport.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary_audit = _run_free_path_boundary_audit()
    majorant_audit = _run_majorant_k_edge_validity_audit()
    free_path_metrics, depths_att, trans_mc, trans_theory = _run_beer_lambert_and_free_path_audit(
        quick=quick
    )
    delta_metrics, depth_centers, delta_profiles, delta_theory_pdf = (
        _run_delta_majorant_invariance_audit(quick=quick)
    )
    phantom_metrics = _run_five_layer_triple_equivalence_audit(quick=quick)

    png_path = output_dir / "exp03_woodcock_tracking.png"
    plot_exp03_woodcock_and_delta_tracking(
        depths_attenuation=depths_att,
        transmission_mc=trans_mc,
        transmission_theory=trans_theory,
        depth_centers_delta=depth_centers,
        delta_profiles=delta_profiles,
        delta_theory_pdf=delta_theory_pdf,
        layer_labels=phantom_metrics.layer_names,
        edep_direct_slabs=phantom_metrics.edep_direct_slabs,
        edep_woodcock_voxel=phantom_metrics.edep_woodcock_voxel,
        edep_woodcock_parametric=phantom_metrics.edep_woodcock_parametric,
        output_path=png_path,
    )

    overall_passed = (
        boundary_audit.passed
        and majorant_audit.passed
        and all(item.passed for item in free_path_metrics)
        and all(item.passed for item in delta_metrics)
        and phantom_metrics.passed
    )

    reference_standard_types = {
        "free_path_moments_and_boundary": "Type B (аналитическое распределение Бугера–Ламберта exp(-mu*s) + детерминированные границы)",
        "majorant_k_edge_validity": "Type B (анализ K-краев поглощения W и Pb из базы сечений NIST XCOM)",
        "delta_majorant_scaling_production": "Type B (аналитическая кусочно-экспоненциальная плотность и теорема Кэмпбелла для шагов Вудкока на продакшен-ядре)",
        "phantom_uncollided_transmission": "Type B (аналитический интеграл Бугера–Ламберта–Бера exp(-sum(mu_i * L_i)))",
        "phantom_history_energy_balance": "Type C (точное алгебраическое тождество закона сохранения энергии E_0 = E_dep + E_esc + E_cut)",
        "phantom_triple_equivalence_and_3d_voxel": "Type D (кросс-верификация Direct Slabs vs 3D WoodcockVoxelVolume vs WoodcockParametricVolume)",
    }

    report = Exp03WoodcockTrackingReport(
        experiment_id="exp03_woodcock_and_delta_tracking",
        title="Эксперимент 3: Верификация дельта-трекинга и трекинга Вудкока",
        quick_mode=quick,
        reference_standard_types=reference_standard_types,
        free_path_boundary_audit=boundary_audit,
        majorant_validity_audit=majorant_audit,
        free_path_metrics=free_path_metrics,
        delta_scaling_metrics=delta_metrics,
        phantom_parity_metrics=phantom_metrics,
        passed=overall_passed,
    )

    json_path = output_dir / "exp03_woodcock_tracking.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(report), json_file, indent=2, ensure_ascii=False)

    return report


__all__ = [
    "FreePathBoundaryAuditMetrics",
    "MajorantValidityAuditMetrics",
    "FreePathMomentMetrics",
    "DeltaMajorantScalingMetrics",
    "Voxel3DObliqueRayMetrics",
    "HeterogeneousPhantomParityMetrics",
    "Exp03WoodcockTrackingReport",
    "run_exp03_woodcock_and_delta_tracking",
]
