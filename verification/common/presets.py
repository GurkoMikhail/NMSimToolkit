"""
Модуль паспортных пресетов оборудования (серия коллиматоров GE Healthcare: LEHR, LEUHR, MEGP, HEGP),
спектров радионуклидов и аналитических эталонов теории коллиматоров Ангера (Anger Theory).

Все физические величины выражены строго в базовых единицах hepunits:
- длина: мм (units.mm == 1.0)
- энергия: МэВ (units.MeV == 1.0)
- время: нс (units.ns == 1.0)
- плоский угол: рад (units.rad == 1.0)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import hepunits as units
import numpy as np

from core.geometry.direct_collimators import CollimatorHoleShape
from core.materials.materials import Material
from core.other.typing_definitions import Float
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PhotoelectricEffect,
    Process,
)
import settings.database_setting as database_setting


@dataclass(frozen=True)
class NemaPassportSpecification:
    """
    Официальная паспортная спецификация детекторной системы гамма-камеры GE Discovery NM/CT 670 / Infinia
    по стандарту NEMA NU-1 на расстоянии Z = 100 мм от входного торца коллиматора (кристалл NaI(Tl) 3/8 дюйма = 9.5 мм).

    Attributes:
        reference_nuclide_code: Код эталонного радионуклида паспортного теста ('Tc99m', 'In111', 'I131').
        system_fwhm: Паспортное системное пространственное разрешение FWHM на 100 мм [мм].
        system_fwtm: Паспортное системное пространственное разрешение FWTM на 100 мм [мм].
        system_activity_sensitivity: Паспортная системная чувствительность NEMA NU-1 [имп./с/МБк].
        intrinsic_crystal_fwhm: Собственное пространственное разрешение кристалла и ФЭУ R_int (FWHM) [мм].
        crystal_thickness: Толщина сцинтилляционного кристалла NaI(Tl) [мм].
        detector_gap: Технологический зазор между выходным торцом коллиматора и кристаллом [мм].
        effective_doi_offset: Эффективное смещение глубины взаимодействия в кристалле (зазор + средняя глубина DOI) [мм].
        energy_resolution_fraction: Энергетическое разрешение тракта ФЭУ (FWHM / E) на линии 140.5 кэВ.
        nominal_photopeak_efficiency: Эффективная вероятность регистрации в окне фотопика +-10% кристалла 9.5 мм.
    """

    reference_nuclide_code: str
    system_fwhm: Float
    system_fwtm: Float
    system_activity_sensitivity: Float
    intrinsic_crystal_fwhm: Float
    crystal_thickness: Float = Float(9.5 * units.mm)
    detector_gap: Float = Float(2.0 * units.mm)
    effective_doi_offset: Float = Float(3.5 * units.mm)
    energy_resolution_fraction: Float = Float(0.098)
    nominal_photopeak_efficiency: Float = Float(0.865)


@dataclass(frozen=True)
class CollimatorPreset:
    """
    Паспортная спецификация параллельного коллиматора гамма-камеры.

    Все линейные размеры задаются в базовых единицах проекта (мм).
    """

    code: str
    title: str
    manufacturer: str
    size: Tuple[Float, Float, Float]
    hole_diameter: Float
    septa: Float
    hole_shape: CollimatorHoleShape
    nema_passport: NemaPassportSpecification
    material_name: str = "Pb"
    hole_material_name: str = "Vacuum"

    @property
    def thickness(self) -> Float:
        """Длина каналов (толщина коллиматора вдоль оси Z)."""
        return self.size[2]

    @property
    def pitch_x(self) -> Float:
        """Шаг периодической решетки каналов вдоль оси X (d + septa)."""
        return Float(self.hole_diameter + self.septa)

    @property
    def pitch_y(self) -> Float:
        """Шаг центрированной гексагональной решетки вдоль оси Y (sqrt(3) * (d + septa))."""
        return Float(np.sqrt(3.0) * self.pitch_x)


@dataclass(frozen=True)
class RadionuclidePreset:
    """
    Паспортная спецификация радионуклида для верификационных экспериментов.

    Энергии задаются в базовых единицах (МэВ), период полураспада — в нс.
    Поле gamma_yield_per_decay задает абсолютный квантовый выход основной линии на один распад.
    Поле spectrum содержит пары (энергия, относительный выход).
    """

    code: str
    title: str
    primary_energy: Float
    half_life: Float
    gamma_yield_per_decay: Float
    spectrum: Tuple[Tuple[Float, Float], ...]


# Стандартные габариты поля зрения детектора GE Discovery NM/CT 670 / Infinia (540 x 400 мм)
GE_FIELD_SIZE_X: Float = Float(540.0 * units.mm)
GE_FIELD_SIZE_Y: Float = Float(400.0 * units.mm)


GE_COLLIMATOR_PRESETS: Dict[str, CollimatorPreset] = {
    "LEHR": CollimatorPreset(
        code="LEHR",
        title="GE Low Energy High Resolution (LEHR)",
        manufacturer="GE Healthcare",
        size=(GE_FIELD_SIZE_X, GE_FIELD_SIZE_Y, Float(35.0 * units.mm)),
        hole_diameter=Float(1.50 * units.mm),
        septa=Float(0.20 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
        nema_passport=NemaPassportSpecification(
            reference_nuclide_code="Tc99m",
            system_fwhm=Float(7.4 * units.mm),
            system_fwtm=Float(13.4 * units.mm),
            system_activity_sensitivity=Float(74.0),
            intrinsic_crystal_fwhm=Float(3.9 * units.mm),
            crystal_thickness=Float(9.5 * units.mm),
            detector_gap=Float(2.0 * units.mm),
            effective_doi_offset=Float(3.5 * units.mm),
            energy_resolution_fraction=Float(0.098),
            nominal_photopeak_efficiency=Float(0.865),
        ),
    ),
    "LEUHR": CollimatorPreset(
        code="LEUHR",
        title="GE Low Energy Ultra High Resolution (LEUHR)",
        manufacturer="GE Healthcare",
        size=(GE_FIELD_SIZE_X, GE_FIELD_SIZE_Y, Float(40.0 * units.mm)),
        hole_diameter=Float(1.22 * units.mm),
        septa=Float(0.15 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
        nema_passport=NemaPassportSpecification(
            reference_nuclide_code="Tc99m",
            system_fwhm=Float(6.0 * units.mm),
            system_fwtm=Float(11.0 * units.mm),
            system_activity_sensitivity=Float(37.0),
            intrinsic_crystal_fwhm=Float(3.9 * units.mm),
            crystal_thickness=Float(9.5 * units.mm),
            detector_gap=Float(2.0 * units.mm),
            effective_doi_offset=Float(3.5 * units.mm),
            energy_resolution_fraction=Float(0.098),
            nominal_photopeak_efficiency=Float(0.865),
        ),
    ),
    "MEGP": CollimatorPreset(
        code="MEGP",
        title="GE Medium Energy General Purpose (MEGP)",
        manufacturer="GE Healthcare",
        size=(GE_FIELD_SIZE_X, GE_FIELD_SIZE_Y, Float(58.0 * units.mm)),
        hole_diameter=Float(3.00 * units.mm),
        septa=Float(1.05 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
        nema_passport=NemaPassportSpecification(
            reference_nuclide_code="In111",
            system_fwhm=Float(9.4 * units.mm),
            system_fwtm=Float(17.2 * units.mm),
            system_activity_sensitivity=Float(46.0),
            intrinsic_crystal_fwhm=Float(3.4 * units.mm),
            crystal_thickness=Float(9.5 * units.mm),
            detector_gap=Float(2.0 * units.mm),
            effective_doi_offset=Float(4.5 * units.mm),
            energy_resolution_fraction=Float(0.098),
            nominal_photopeak_efficiency=Float(0.425),
        ),
    ),
    "HEGP": CollimatorPreset(
        code="HEGP",
        title="GE High Energy General Purpose (HEGP)",
        manufacturer="GE Healthcare",
        size=(GE_FIELD_SIZE_X, GE_FIELD_SIZE_Y, Float(66.0 * units.mm)),
        hole_diameter=Float(4.00 * units.mm),
        septa=Float(1.80 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
        nema_passport=NemaPassportSpecification(
            reference_nuclide_code="I131",
            system_fwhm=Float(12.0 * units.mm),
            system_fwtm=Float(22.5 * units.mm),
            system_activity_sensitivity=Float(28.0),
            intrinsic_crystal_fwhm=Float(3.2 * units.mm),
            crystal_thickness=Float(9.5 * units.mm),
            detector_gap=Float(2.0 * units.mm),
            effective_doi_offset=Float(5.0 * units.mm),
            energy_resolution_fraction=Float(0.098),
            nominal_photopeak_efficiency=Float(0.185),
        ),
    ),
}


RADIONUCLIDE_PRESETS: Dict[str, RadionuclidePreset] = {
    "Tc99m": RadionuclidePreset(
        code="Tc99m",
        title="99mTc (140.5 кэВ)",
        primary_energy=Float(140.5 * units.keV),
        half_life=Float(6.0067 * units.hour),
        gamma_yield_per_decay=Float(0.8906),
        spectrum=((Float(140.5 * units.keV), Float(1.0)),),
    ),
    "I123": RadionuclidePreset(
        code="I123",
        title="123I (159.0 кэВ)",
        primary_energy=Float(159.0 * units.keV),
        half_life=Float(13.223 * units.hour),
        gamma_yield_per_decay=Float(0.8330),
        spectrum=((Float(159.0 * units.keV), Float(1.0)),),
    ),
    "Lu177": RadionuclidePreset(
        code="Lu177",
        title="177Lu (208.0 кэВ)",
        primary_energy=Float(208.0 * units.keV),
        half_life=Float(6.647 * units.day),
        gamma_yield_per_decay=Float(0.1038),
        spectrum=(
            (Float(208.0 * units.keV), Float(0.1038)),
            (Float(112.95 * units.keV), Float(0.0620)),
        ),
    ),
    "In111": RadionuclidePreset(
        code="In111",
        title="111In (245.4 кэВ)",
        primary_energy=Float(245.4 * units.keV),
        half_life=Float(2.8047 * units.day),
        gamma_yield_per_decay=Float(0.9410),
        spectrum=(
            (Float(171.3 * units.keV), Float(0.906)),
            (Float(245.4 * units.keV), Float(0.941)),
        ),
    ),
    "I131": RadionuclidePreset(
        code="I131",
        title="131I (364.5 кэВ)",
        primary_energy=Float(364.5 * units.keV),
        half_life=Float(8.0252 * units.day),
        gamma_yield_per_decay=Float(0.8150),
        spectrum=((Float(364.5 * units.keV), Float(1.0)),),
    ),
    "Tb161": RadionuclidePreset(
        code="Tb161",
        title="161Tb (Спектр Juget et al. 2021/2023)",
        primary_energy=Float(74.57 * units.keV),
        half_life=Float(6.953 * units.day),
        gamma_yield_per_decay=Float(0.1028),
        spectrum=(
            (Float(25.65 * units.keV), Float(23.154)),
            (Float(45.21 * units.keV), Float(7.747)),
            (Float(46.00 * units.keV), Float(12.779)),
            (Float(48.92 * units.keV), Float(17.73)),
            (Float(52.19 * units.keV), Float(3.741)),
            (Float(53.64 * units.keV), Float(1.0185)),
            (Float(57.19 * units.keV), Float(2.066)),
            (Float(59.24 * units.keV), Float(0.2002)),
            (Float(74.57 * units.keV), Float(10.28)),
            (Float(77.42 * units.keV), Float(0.0459)),
            (Float(87.94 * units.keV), Float(0.1996)),
            (Float(103.07 * units.keV), Float(0.1077)),
            (Float(106.11 * units.keV), Float(0.0816)),
            (Float(286.48 * units.keV), Float(0.01574)),
            (Float(292.40 * units.keV), Float(0.07102)),
            (Float(319.66 * units.keV), Float(0.00367)),
            (Float(341.40 * units.keV), Float(0.00375)),
            (Float(343.63 * units.keV), Float(0.01473)),
            (Float(418.47 * units.keV), Float(0.00925)),
            (Float(475.66 * units.keV), Float(0.01660)),
            (Float(550.25 * units.keV), Float(0.04258)),
        ),
    ),
}


def compute_total_linear_attenuation(
    material: Material,
    energy: Float | float,
    processes: Sequence[Process] | None = None,
) -> Float:
    """
    Вычисляет полный линейный коэффициент ослабления mu(E) материала по базе данных сечений ядра.
    """
    if Float(energy) <= 0.0:
        raise ValueError(f"Энергия должна быть строго положительной (> 0), получено {energy}")

    active_processes: List[Process]
    if processes is None:
        rng = np.random.default_rng(0)
        active_processes = [
            PhotoelectricEffect(database_setting.attenuation_database, rng),
            ComptonScattering(database_setting.attenuation_database, rng),
            CoherentScattering(database_setting.attenuation_database, rng),
        ]
    else:
        active_processes = list(processes)

    total_lac = Float(0.0)
    query_energy = Float(energy)
    for proc in active_processes:
        if material in proc.attenuation_function:
            lac_val = proc.attenuation_function(material, query_energy)
            total_lac = Float(total_lac + float(lac_val))
    return total_lac


def compute_anger_effective_length(
    preset: CollimatorPreset,
    energy: Float | float,
    linear_attenuation: Float | float | None = None,
) -> Float:
    """
    Вычисляет эффективную длину канала коллиматора по формуле Ангера:
    L_eff = L - 2 / mu(E).
    """
    if linear_attenuation is None:
        lead_mat = database_setting.material_database[preset.material_name]
        mu_val = compute_total_linear_attenuation(lead_mat, energy)
    else:
        mu_val = Float(linear_attenuation)

    if mu_val <= 0.0:
        return preset.thickness

    effective_len = Float(preset.thickness - 2.0 / mu_val)
    return max(effective_len, Float(1e-3 * units.mm))


def compute_anger_geometric_fwhm(
    preset: CollimatorPreset,
    source_distance: Float | float,
    energy: Float | float,
    detector_gap: Float | float = Float(0.0),
    use_effective_length: bool = True,
) -> Float:
    """
    Вычисляет теоретическое геометрическое пространственное разрешение R_g (FWHM)
    на расстоянии source_distance от входного торца коллиматора по формуле Ангера:
    R_g(Z) = d * (L_eff + Z + gap) / L_eff.
    """
    if Float(source_distance) < 0.0:
        raise ValueError(f"Расстояние от источника до коллиматора не может быть отрицательным: {source_distance}")

    channel_length = (
        compute_anger_effective_length(preset, energy) if use_effective_length else preset.thickness
    )
    fwhm_geometric = Float(
        preset.hole_diameter * (channel_length + Float(source_distance) + Float(detector_gap)) / channel_length
    )
    return fwhm_geometric


def compute_anger_geometric_sensitivity(
    preset: CollimatorPreset,
    energy: Float | float,
    use_effective_length: bool = True,
) -> Float:
    """
    Вычисляет теоретическую геометрическую эффективность (чувствительность g, доля фотонов на распад в 4*pi)
    для гексагонального коллиматора с параллельными каналами по формуле Ангера:
    g = [ K * d^2 / (L_eff * (d + septa)) ]^2, где K_hex = sqrt(sqrt(3) / (8 * pi)) для flat-to-flat d,
    то есть g = (sqrt(3) / (8 * pi)) * [ d^2 / (L_eff * (d + septa)) ]^2 = A_hole^2 / (4 * pi * L_eff^2 * A_cell).

    Примечание: Площадь правильного шестиугольника с расстоянием между противоположными гранями d
    равна A_hole = (sqrt(3) / 2) * d^2. Площадь элементарной ячейки периодичности на один канал
    равна A_cell = (sqrt(3) / 2) * (d + septa)^2.
    Телесный угол через апертуру на большом расстоянии дает точную формулу:
    g = A_hole^2 / (4 * pi * L_eff^2 * A_cell) = (sqrt(3) / (8 * pi)) * (d^2 / (L_eff * (d + septa)))^2.
    """
    channel_length = (
        compute_anger_effective_length(preset, energy) if use_effective_length else preset.thickness
    )
    pitch = Float(preset.hole_diameter + preset.septa)
    ratio = Float((preset.hole_diameter * preset.hole_diameter) / (channel_length * pitch))
    factor = Float(np.sqrt(3.0) / (8.0 * np.pi))
    return Float(factor * ratio * ratio)


def compute_nema_system_fwhm(
    collimator_fwhm: Float | float,
    preset: CollimatorPreset,
    source_distance: Float | float,
    energy: Float | float,
) -> Float:
    """
    Пересчитывает пространственное разрешение на выходном торце коллиматора в системное разрешение
    гамма-камеры по стандарту NEMA NU-1 с учетом эффективной глубины взаимодействия в кристалле NaI(Tl)
    и собственного разрешения детектора Ангера R_int:
    R_coll_det = collimator_fwhm * (L_eff + Z + delta_Z_doi) / (L_eff + Z),
    FWHM_sys = sqrt(R_coll_det^2 + R_int^2).
    """
    effective_len = compute_anger_effective_length(preset, energy)
    base_span = Float(effective_len + Float(source_distance))
    projected_span = Float(base_span + preset.nema_passport.effective_doi_offset)
    projected_collimator_fwhm = Float(Float(collimator_fwhm) * (projected_span / max(base_span, Float(1e-6))))
    intrinsic_fwhm = preset.nema_passport.intrinsic_crystal_fwhm
    return Float(np.hypot(projected_collimator_fwhm, intrinsic_fwhm))


def compute_nema_activity_sensitivity(
    collimator_sensitivity: Float | float,
    preset: CollimatorPreset,
    nuclide: RadionuclidePreset,
) -> Float:
    """
    Пересчитывает безразмерную чувствительность коллиматора (долю фотонов в 4*pi на 1 испущенный гамма-квант)
    в паспортную системную чувствительность гамма-камеры по стандарту NEMA NU-1 [имп./с/МБк]
    с учетом пропускания входного герметизирующего алюминиевого окна кристалла (Al) и фотопиковой эффективности NaI(Tl):
    S_sys = 10^6 * S_coll * I_gamma * exp(-mu_Al(E) * d_window) * epsilon_photopeak.
    """
    decays_per_unit_activity = Float(1.0e6)
    aluminum_mat = database_setting.material_database["Al"]
    mu_aluminum = compute_total_linear_attenuation(aluminum_mat, nuclide.primary_energy)
    window_transmission = Float(np.exp(-float(mu_aluminum) * float(preset.nema_passport.detector_gap)))
    photopeak_efficiency = preset.nema_passport.nominal_photopeak_efficiency
    return Float(
        decays_per_unit_activity
        * Float(collimator_sensitivity)
        * nuclide.gamma_yield_per_decay
        * window_transmission
        * photopeak_efficiency
    )


__all__ = [
    "NemaPassportSpecification",
    "CollimatorPreset",
    "RadionuclidePreset",
    "GE_FIELD_SIZE_X",
    "GE_FIELD_SIZE_Y",
    "GE_COLLIMATOR_PRESETS",
    "RADIONUCLIDE_PRESETS",
    "compute_total_linear_attenuation",
    "compute_anger_effective_length",
    "compute_anger_geometric_fwhm",
    "compute_anger_geometric_sensitivity",
    "compute_nema_system_fwhm",
    "compute_nema_activity_sensitivity",
]
