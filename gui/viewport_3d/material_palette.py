"""
Семантическая палитра цветов материалов и расчет физической рентгеновской непрозрачности (Opacity).

Модуль предоставляет:
1. Каталог визуальных цветов для всех 144 материалов базы данных NIST (tables/NIST Materials.h5)
   с разделением по физическим группам (биологические ткани, полимеры, сцинтилляторы, кристаллы, металлы, газы).
2. Физический расчет линейного коэффициента ослабления фотонов (mu в единицах hepunits) для материалов на основе NIST XCOM.
3. Расчет оптической непрозрачности (Opacity) по закону Бугера-Ламберта-Бера в зависимости от энергии квантов.
4. Режим отображения в псевдорентгене (Pseudo-X-ray Mode).
5. Визуальные акценты для детекторов (золотистый янтарь) и подсветки ребер выделенных узлов.
"""

import hashlib
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import hepunits as units

import settings.database_setting as database_setting

_logger = logging.getLogger(__name__)

# Визуальные акценты для детекторов и выделения узлов сцены
DETECTOR_ACCENT_COLOR: Tuple[float, float, float] = (1.0, 0.78, 0.10)
DETECTOR_ACCENT_OPACITY: float = 0.75

SELECTED_EDGE_HIGHLIGHT_COLOR: Tuple[float, float, float] = (1.0, 0.55, 0.0)
SELECTED_EDGE_HIGHLIGHT_WIDTH: float = 2.5

# Базовая семантическая палитра материалов (RGB в диапазоне [0.0, 1.0])
MATERIAL_COLOR_PALETTE: Dict[str, Tuple[float, float, float]] = {
    # Вакуум и фоновые среды
    "Vacuum": (0.45, 0.48, 0.55),
    "Air, Dry (near sea level)": (0.75, 0.88, 0.95),
    "Water, Liquid": (0.30, 0.65, 0.92),

    # Анатомические и биологические ткани (ICRU-44)
    "Adipose Tissue (ICRU-44)": (0.92, 0.86, 0.58),
    "Blood, Whole (ICRU-44)": (0.78, 0.18, 0.18),
    "Bone, Cortical (ICRU-44)": (0.92, 0.90, 0.78),
    "Brain, Grey\\White Matter (ICRU-44)": (0.85, 0.72, 0.75),
    "Breast Tissue (ICRU-44)": (0.92, 0.78, 0.72),
    "Eye Lens (ICRU-44)": (0.70, 0.88, 0.95),
    "Lung": (0.68, 0.55, 0.62),
    "Lung Tissue (ICRU-44)": (0.70, 0.58, 0.65),
    "Muscle, Skeletal (ICRU-44)": (0.82, 0.32, 0.32),
    "Ovary (ICRU-44)": (0.80, 0.55, 0.58),
    "Testis (ICRU-44)": (0.82, 0.62, 0.60),
    "Tissue, Soft (ICRU Four-Component)": (0.88, 0.62, 0.62),
    "Tissue, Soft (ICRU-44)": (0.86, 0.60, 0.60),

    # Тканеэквивалентные пластики и газы
    "A-150 Tissue-Equivalent Plastic": (0.80, 0.72, 0.65),
    "B-100 Bone-Equivalent Plastic": (0.88, 0.85, 0.72),
    "C-552 Air-equivalent Plastic": (0.70, 0.72, 0.75),
    "Tissue-Equivalent Gas, Methane Based": (0.78, 0.84, 0.92),
    "Tissue-Equivalent Gas, Propane Based": (0.76, 0.82, 0.90),

    # Полимеры, пластмассы и смолы
    "Bakelite": (0.48, 0.28, 0.20),
    "Polyethylene": (0.88, 0.90, 0.92),
    "Polyethylene Terephthalate, (Mylar)": (0.75, 0.80, 0.90),
    "Polymethyl Methacrylate": (0.65, 0.85, 0.92),
    "Polystyrene": (0.82, 0.86, 0.90),
    "Polytetrafluoroethylene, (Teflon)": (0.94, 0.94, 0.96),
    "Polyvinyl Chloride": (0.78, 0.72, 0.52),

    # Сцинтилляционные кристаллы и полупроводниковые детекторы
    "Sodium Iodide": (0.72, 0.68, 0.92),
    "Cesium Iodide": (0.68, 0.72, 0.92),
    "Cadmium Telluride": (0.32, 0.35, 0.38),
    "Cadmium Zinc Telluride": (0.35, 0.40, 0.48),
    "Gadolinium Oxysulfide": (0.55, 0.72, 0.58),
    "Gallium Arsenide": (0.42, 0.38, 0.32),
    "Mercuric Iodide": (0.85, 0.25, 0.20),
    "Calcium Fluoride": (0.85, 0.92, 0.88),
    "Calcium Sulfate": (0.90, 0.89, 0.85),
    "Lithium Fluride": (0.92, 0.92, 0.95),
    "Lithium Tetraborate": (0.88, 0.90, 0.92),
    "Magnesium Tetroborate": (0.86, 0.88, 0.90),
    "Plastic Scintillator, Vinyltoluene": (0.30, 0.88, 0.65),

    # Стекла и строительные материалы радиационной защиты
    "Concrete, Ordinary": (0.65, 0.65, 0.62),
    "Concrete, Barite (TYPE BA)": (0.55, 0.52, 0.48),
    "Glass, Borosilicate (Pyrex)": (0.65, 0.82, 0.85),
    "Glass, Lead": (0.72, 0.72, 0.58),

    # Дозиметрические растворы и пленки
    "15 mmol L-1 Ceric Ammonium Sulfate Solution": (0.85, 0.75, 0.40),
    "Alanine": (0.90, 0.88, 0.82),
    "Ferrous Sulfate Standard Fricke": (0.75, 0.82, 0.60),
    "Gafchromic Sensor": (0.62, 0.25, 0.65),
    "Radiochromic Dye Film, Nylon Base": (0.70, 0.28, 0.60),
    "Photographic Emulsion (Kodak Type AA)": (0.72, 0.62, 0.48),
    "Photographic Emulsion (Standard Nuclear)": (0.68, 0.60, 0.52),

    # Тяжелые металлы радиационной защиты и коллиматоров
    "Pb": (0.30, 0.32, 0.38),
    "W": (0.35, 0.36, 0.40),
    "Ta": (0.40, 0.42, 0.45),
    "Mo": (0.45, 0.46, 0.50),
    "Au": (0.95, 0.80, 0.20),
    "Ag": (0.88, 0.90, 0.92),
    "Pt": (0.82, 0.84, 0.88),
    "Cu": (0.85, 0.45, 0.25),
    "Fe": (0.50, 0.52, 0.55),
    "Al": (0.78, 0.80, 0.82),
    "Ti": (0.55, 0.58, 0.62),
    "Ni": (0.60, 0.62, 0.65),
    "Cr": (0.65, 0.68, 0.72),
    "Zn": (0.72, 0.75, 0.82),
    "Sn": (0.75, 0.78, 0.80),
    "Cd": (0.68, 0.70, 0.72),
    "Bi": (0.78, 0.72, 0.76),
    "U": (0.32, 0.38, 0.32),
    "Th": (0.36, 0.38, 0.35),

    # Переходные металлы и тугоплавкие элементы
    "Sc": (0.75, 0.78, 0.80),
    "V": (0.60, 0.62, 0.65),
    "Mn": (0.60, 0.48, 0.65),
    "Co": (0.45, 0.55, 0.75),
    "Y": (0.65, 0.75, 0.75),
    "Zr": (0.60, 0.68, 0.72),
    "Nb": (0.52, 0.58, 0.65),
    "Tc": (0.42, 0.65, 0.60),
    "Ru": (0.55, 0.60, 0.65),
    "Rh": (0.65, 0.70, 0.75),
    "Pd": (0.75, 0.78, 0.82),
    "Hf": (0.48, 0.52, 0.56),
    "Re": (0.45, 0.48, 0.52),
    "Os": (0.35, 0.40, 0.48),
    "Ir": (0.45, 0.50, 0.55),

    # Лантаноиды и актиноиды
    "La": (0.68, 0.82, 0.88),
    "Ce": (1.00, 1.00, 0.78),
    "Pr": (0.70, 0.85, 0.75),
    "Nd": (0.62, 0.78, 0.85),
    "Pm": (0.58, 0.75, 0.80),
    "Sm": (0.60, 0.75, 0.70),
    "Eu": (0.62, 0.88, 0.75),
    "Gd": (0.50, 0.70, 0.65),
    "Tb": (0.55, 0.80, 0.65),
    "Dy": (0.55, 0.85, 0.75),
    "Ho": (0.60, 0.85, 0.70),
    "Er": (0.65, 0.85, 0.70),
    "Tm": (0.50, 0.75, 0.65),
    "Yb": (0.55, 0.72, 0.68),
    "Lu": (0.45, 0.65, 0.70),
    "Ac": (0.44, 0.67, 0.75),
    "Pa": (0.40, 0.55, 0.65),

    # Постпереходные металлы и полупроводники
    "Ga": (0.76, 0.78, 0.82),
    "Ge": (0.55, 0.65, 0.65),
    "As": (0.74, 0.50, 0.89),
    "Se": (0.80, 0.50, 0.30),
    "In": (0.65, 0.72, 0.75),
    "Sb": (0.65, 0.58, 0.72),
    "Te": (0.70, 0.60, 0.40),
    "Tl": (0.60, 0.62, 0.65),
    "Po": (0.65, 0.55, 0.45),
    "At": (0.46, 0.33, 0.27),

    # Щелочные и щелочноземельные металлы
    "Li": (0.80, 0.50, 1.00),
    "Be": (0.76, 0.85, 0.70),
    "Na": (0.65, 0.55, 0.80),
    "Mg": (0.75, 0.78, 0.75),
    "K": (0.60, 0.50, 0.75),
    "Ca": (0.80, 0.80, 0.75),
    "Rb": (0.65, 0.40, 0.75),
    "Sr": (0.60, 0.75, 0.60),
    "Cs": (0.58, 0.48, 0.70),
    "Ba": (0.65, 0.70, 0.68),
    "Fr": (0.50, 0.40, 0.65),
    "Ra": (0.70, 0.75, 0.70),

    # Неметаллы, газы и легкие элементы
    "H": (0.85, 0.90, 1.00),
    "He": (0.80, 0.92, 0.98),
    "B": (1.00, 0.71, 0.71),
    "C": (0.25, 0.25, 0.25),
    "N": (0.35, 0.60, 0.90),
    "O": (0.90, 0.25, 0.25),
    "F": (0.75, 0.90, 0.60),
    "Ne": (0.95, 0.55, 0.40),
    "Si": (0.50, 0.55, 0.65),
    "P": (0.90, 0.55, 0.20),
    "S": (0.92, 0.85, 0.20),
    "Cl": (0.65, 0.85, 0.35),
    "Ar": (0.75, 0.80, 0.90),
    "Br": (0.65, 0.22, 0.15),
    "Kr": (0.70, 0.78, 0.85),
    "I": (0.55, 0.30, 0.65),
    "Xe": (0.60, 0.70, 0.82),
    "Hg": (0.72, 0.74, 0.78),
    "Rn": (0.65, 0.70, 0.75),
}

def generate_deterministic_color(material_name: str) -> Tuple[float, float, float]:
    """
    Генерирует детерминированный, согласованный по контрасту RGB цвет
    для любого материала на основе хэш-функции его названия.
    """
    hasher = hashlib.md5(material_name.encode('utf-8'))
    hash_bytes = hasher.digest()
    red_val = float(hash_bytes[0]) / 255.0
    green_val = float(hash_bytes[1]) / 255.0
    blue_val = float(hash_bytes[2]) / 255.0

    # Коррекция диапазона яркости: избегаем слишком темных или перенасыщенных цветов
    min_component = 0.35
    max_component = 0.90
    red_clamped = min_component + (max_component - min_component) * red_val
    green_clamped = min_component + (max_component - min_component) * green_val
    blue_clamped = min_component + (max_component - min_component) * blue_val
    return (float(red_clamped), float(green_clamped), float(blue_clamped))


def get_material_color(material_name: str) -> Tuple[float, float, float]:
    """
    Возвращает RGB цвет материала [0.0, 1.0].
    """
    if material_name in MATERIAL_COLOR_PALETTE:
        return MATERIAL_COLOR_PALETTE[material_name]
    return generate_deterministic_color(material_name)


def compute_material_linear_attenuation(
    material_name: str,
    energy: float = 140.0 * units.keV,
) -> float:
    """
    Рассчитывает физический линейный коэффициент ослабления mu (в единицах hepunits)
    для материала при заданной энергии фотонов на основе NIST XCOM.
    """
    if energy <= 0.0:
        raise ValueError(f"Энергия фотонов должна быть строго положительной (> 0), получено: {energy}")

    if material_name == "Vacuum":
        return 0.0

    materials_db = database_setting.material_database
    if material_name not in materials_db:
        raise KeyError(f"Материал '{material_name}' отсутствует в базе данных NIST.")

    target_material = materials_db[material_name]
    attenuations_db = database_setting.attenuation_database
    if target_material not in attenuations_db:
        raise KeyError(
            f"Материал '{material_name}' отсутствует в базе данных ослабления NIST."
        )

    energy_table_scale = float(energy / units.MeV)
    mac_table = attenuations_db[target_material]
    energy_array = mac_table['Energy']
    coefficients_table = mac_table['Coefficient']

    total_mass_attenuation_array = sum(
        coefficients_table[process_name]
        for process_name in coefficients_table.dtype.names
    )

    interpolated_mass_coeff = float(np.interp(energy_table_scale, energy_array, total_mass_attenuation_array))
    linear_attenuation_internal = interpolated_mass_coeff * target_material.density
    linear_attenuation = float(linear_attenuation_internal / (1.0 / units.mm))
    return max(0.0, linear_attenuation)


def compute_xray_opacity(
    linear_attenuation: float,
    characteristic_length: float = 25.0 * units.mm,
    min_opacity: float = 0.12,
    max_opacity: float = 0.95,
) -> float:
    """
    Преобразует линейный коэффициент ослабления mu в оптическую непрозрачность (Opacity)
    по физическому экспоненциальному закону Бугера-Ламберта-Бера: Opacity = 1 - exp(-mu * L).
    """
    if characteristic_length <= 0.0:
        raise ValueError(
            f"Характерный размер объема должен быть строго положительным (> 0), получено: {characteristic_length}"
        )
    if linear_attenuation < 0.0:
        raise ValueError(
            f"Линейный коэффициент ослабления должен быть неотрицательным (>= 0), получено: {linear_attenuation}"
        )
    if not (0.0 <= min_opacity <= max_opacity <= 1.0):
        raise ValueError(
            f"Недопустимый диапазон непрозрачности: min_opacity={min_opacity}, max_opacity={max_opacity}"
        )
    if linear_attenuation <= 1e-7:
        return 0.06

    optical_depth = float(linear_attenuation * characteristic_length)
    absorbed_ratio = float(1.0 - np.exp(-optical_depth))
    visual_opacity = min_opacity + (max_opacity - min_opacity) * (absorbed_ratio ** 0.8)
    return float(np.clip(visual_opacity, min_opacity, max_opacity))


def get_material_opacity(
    material_name: str,
    energy: float = 140.0 * units.keV,
    characteristic_length: float = 25.0 * units.mm,
) -> float:
    """
    Возвращает рассчитанную оптическую непрозрачность (Opacity) материала
    при заданной энергии фотонов и характерном размере в единицах hepunits.
    """
    try:
        linear_attenuation = compute_material_linear_attenuation(material_name, energy=energy)
    except KeyError:
        # Для нестандартных/пользовательских материалов, отсутствующих в базе NIST,
        # используем стандартное ослабление мягких тканей/воды (0.015 / units.mm)
        linear_attenuation = 0.015 * (1.0 / units.mm)
    return compute_xray_opacity(linear_attenuation, characteristic_length=characteristic_length)


def get_material_rgba(
    material_name: str,
    energy: float = 140.0 * units.keV,
    characteristic_length: float = 25.0 * units.mm,
) -> Tuple[float, float, float, float]:
    """
    Возвращает полный кортеж (Red, Green, Blue, Alpha) для материала,
    где цвет определяется семантической палитрой, а прозрачность — рентгеновской плотностью.
    """
    red_color, green_color, blue_color = get_material_color(material_name)
    opacity_value = get_material_opacity(
        material_name,
        energy=energy,
        characteristic_length=characteristic_length,
    )
    return (red_color, green_color, blue_color, opacity_value)


def get_pseudo_xray_rgba(
    material_name: str,
    energy: float = 140.0 * units.keV,
    characteristic_length: float = 25.0 * units.mm,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Возвращает RGB цвет и Opacity материала для режима визуализации 'Псевдорентген'.
    Рентгеноплотные материалы (свинец, вольфрам, кость) получают высокую яркость и непрозрачность,
    тогда как мягкие ткани и воздух становятся слабовидимыми и полупрозрачными.
    """
    try:
        linear_attenuation = compute_material_linear_attenuation(material_name, energy=energy)
    except KeyError:
        linear_attenuation = 0.015 * (1.0 / units.mm)

    opacity_value = compute_xray_opacity(linear_attenuation, characteristic_length=characteristic_length)

    optical_depth = float(linear_attenuation * characteristic_length)
    radiographic_brightness = float(np.clip(0.20 + 0.80 * (1.0 - np.exp(-optical_depth)), 0.20, 1.00))

    # Стилизация холодного рентгеновского свечения
    rgb_radiographic = (
        float(np.clip(radiographic_brightness * 0.92, 0.0, 1.0)),
        float(np.clip(radiographic_brightness * 0.96, 0.0, 1.0)),
        float(np.clip(radiographic_brightness * 1.00, 0.0, 1.0)),
    )
    return (rgb_radiographic, opacity_value)
