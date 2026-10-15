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
    Поле spectrum содержит пары (энергия, относительный выход).
    """

    code: str
    title: str
    primary_energy: Float
    half_life: Float
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
    ),
    "LEUHR": CollimatorPreset(
        code="LEUHR",
        title="GE Low Energy Ultra High Resolution (LEUHR)",
        manufacturer="GE Healthcare",
        size=(GE_FIELD_SIZE_X, GE_FIELD_SIZE_Y, Float(40.0 * units.mm)),
        hole_diameter=Float(1.22 * units.mm),
        septa=Float(0.15 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
    ),
    "MEGP": CollimatorPreset(
        code="MEGP",
        title="GE Medium Energy General Purpose (MEGP)",
        manufacturer="GE Healthcare",
        size=(GE_FIELD_SIZE_X, GE_FIELD_SIZE_Y, Float(58.0 * units.mm)),
        hole_diameter=Float(3.00 * units.mm),
        septa=Float(1.05 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
    ),
    "HEGP": CollimatorPreset(
        code="HEGP",
        title="GE High Energy General Purpose (HEGP)",
        manufacturer="GE Healthcare",
        size=(GE_FIELD_SIZE_X, GE_FIELD_SIZE_Y, Float(66.0 * units.mm)),
        hole_diameter=Float(4.00 * units.mm),
        septa=Float(1.80 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
    ),
}


RADIONUCLIDE_PRESETS: Dict[str, RadionuclidePreset] = {
    "Tc99m": RadionuclidePreset(
        code="Tc99m",
        title="99mTc (140.5 кэВ)",
        primary_energy=Float(140.5 * units.keV),
        half_life=Float(6.0067 * units.hour),
        spectrum=((Float(140.5 * units.keV), Float(1.0)),),
    ),
    "I123": RadionuclidePreset(
        code="I123",
        title="123I (159.0 кэВ)",
        primary_energy=Float(159.0 * units.keV),
        half_life=Float(13.223 * units.hour),
        spectrum=((Float(159.0 * units.keV), Float(1.0)),),
    ),
    "Lu177": RadionuclidePreset(
        code="Lu177",
        title="177Lu (208.0 кэВ)",
        primary_energy=Float(208.0 * units.keV),
        half_life=Float(6.647 * units.day),
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
        spectrum=((Float(364.5 * units.keV), Float(1.0)),),
    ),
    "Tb161": RadionuclidePreset(
        code="Tb161",
        title="161Tb (Полный спектр ICRP 107)",
        primary_energy=Float(74.57 * units.keV),
        half_life=Float(6.89 * units.day),
        spectrum=(
            (Float(74.57 * units.keV), Float(10.3)),
            (Float(48.9 * units.keV), Float(17.0)),
            (Float(46.0 * units.keV), Float(11.2)),
            (Float(25.65 * units.keV), Float(23.2)),
            (Float(52.0 * units.keV), Float(1.4)),
            (Float(57.4 * units.keV), Float(0.4)),
            (Float(106.1 * units.keV), Float(0.05)),
            (Float(131.8 * units.keV), Float(0.03)),
            (Float(160.0 * units.keV), Float(0.004)),
            (Float(292.0 * units.keV), Float(0.002)),
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


__all__ = [
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
]
