from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import vtk

# Предопределенные цветовые шкалы ядерной медицины (x, r, g, b в диапазоне [0, 1])
DICOM_COLORMAP_DEFINITIONS: Dict[str, List[Tuple[float, float, float, float]]] = {
    'Hot Iron': [
        (0.00, 0.00, 0.00, 0.00),
        (0.25, 0.50, 0.00, 0.00),
        (0.50, 0.90, 0.20, 0.00),
        (0.75, 1.00, 0.80, 0.00),
        (1.00, 1.00, 1.00, 1.00),
    ],
    'Rainbow': [
        (0.00, 0.00, 0.00, 0.60),
        (0.20, 0.00, 0.40, 1.00),
        (0.40, 0.00, 0.90, 0.50),
        (0.60, 0.80, 0.90, 0.00),
        (0.80, 1.00, 0.50, 0.00),
        (1.00, 1.00, 0.00, 0.00),
    ],
    'GE Color': [
        (0.00, 0.00, 0.00, 0.00),
        (0.15, 0.10, 0.10, 0.50),
        (0.35, 0.00, 0.60, 0.80),
        (0.55, 0.00, 0.80, 0.20),
        (0.75, 0.90, 0.80, 0.00),
        (1.00, 1.00, 0.10, 0.10),
    ],
    'NIH': [
        (0.00, 0.00, 0.00, 0.00),
        (0.30, 0.60, 0.00, 0.60),
        (0.60, 0.90, 0.40, 0.00),
        (0.85, 1.00, 0.90, 0.20),
        (1.00, 1.00, 1.00, 1.00),
    ],
    'PET 20 Step': [
        (i / 19.0,
         min(1.0, max(0.0, 1.5 - abs(3.0 * i / 19.0 - 2.0))),
         min(1.0, max(0.0, 1.5 - abs(3.0 * i / 19.0 - 1.0))),
         min(1.0, max(0.0, 1.5 - abs(3.0 * i / 19.0 - 0.0))))
        for i in range(20)
    ]
}


def get_available_colormaps() -> List[str]:
    """
    Возвращает список доступных стандартных палитр ядерной медицины.
    """
    return list(DICOM_COLORMAP_DEFINITIONS.keys())


def get_colormap_lut(name: str, num_colors: int = 256) -> np.ndarray:
    """
    Генерирует матрицу цветов RGBA формы (num_colors, 4) на основе интерполяции опорных точек.
    """
    if name not in DICOM_COLORMAP_DEFINITIONS:
        name = 'Hot Iron'

    control_points = DICOM_COLORMAP_DEFINITIONS[name]
    x_pts = [pt[0] for pt in control_points]
    r_pts = [pt[1] for pt in control_points]
    g_pts = [pt[2] for pt in control_points]
    b_pts = [pt[3] for pt in control_points]

    x_new = np.linspace(0.0, 1.0, num_colors)
    r_new = np.interp(x_new, x_pts, r_pts)
    g_new = np.interp(x_new, x_pts, g_pts)
    b_new = np.interp(x_new, x_pts, b_pts)
    a_new = np.ones(num_colors, dtype=float)

    lut = np.column_stack((r_new, g_new, b_new, a_new))
    return lut


def import_lut_file(filepath: Union[str, Path]) -> np.ndarray:
    """
    Импортирует цветовую таблицу из текстового файла формата LUT / CSV (значения R G B в строках).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"LUT-файл не найден: {filepath}")

    data = np.loadtxt(path, delimiter=None)
    if data.shape[1] < 3:
        raise ValueError("LUT-файл должен содержать как минимум 3 столбца (R, G, B)")

    # Если значения в диапазоне [0, 255], нормализуем к [0, 1]
    if np.max(data[:, :3]) > 1.0:
        data[:, :3] /= 255.0

    if data.shape[1] == 3:
        alpha = np.ones((data.shape[0], 1), dtype=float)
        data = np.hstack((data, alpha))

    return data[:, :4]


def to_vtk_color_transfer_function(name: str, scalar_range: Tuple[float, float] = (0.0, 1.0)) -> Any:
    """
    Создает и настраивает экземпляр vtkColorTransferFunction для объемного рендеринга.
    """
    func = vtk.vtkColorTransferFunction()
    lut = get_colormap_lut(name, num_colors=256)
    s_min, s_max = scalar_range
    step = (s_max - s_min) / max(1, len(lut) - 1)

    for i, row in enumerate(lut):
        val = s_min + i * step
        func.AddRGBPoint(val, float(row[0]), float(row[1]), float(row[2]))

    return func


def to_vtk_piecewise_function(
    scalar_range: Tuple[float, float] = (0.0, 1.0),
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    ramp_type: str = 'linear',
    threshold: Optional[float] = None,
    preset: str = 'air_cutoff'
) -> Any:
    """
    Создает и настраивает vtkPiecewiseFunction для задания прозрачности объема.
    Поддерживает пресеты: 'air_cutoff', 'linear', 'soft_tissue', 'xray_translucent', 'step'.
    """
    opacity = vtk.vtkPiecewiseFunction()
    s_min, s_max = scalar_range
    if s_max <= s_min:
        s_max = s_min + 1.0

    max_alpha = float(np.clip(max_alpha, 0.0, 1.0))
    min_alpha = float(np.clip(min_alpha, 0.0, max_alpha))
    thresh_val = 0.05 if threshold is None else float(np.clip(threshold, 0.0, 1.0))
    mid = s_min + thresh_val * (s_max - s_min)

    if preset == 'step' or ramp_type == 'step':
        opacity.AddPoint(s_min, min_alpha)
        opacity.AddPoint(mid, min_alpha)
        opacity.AddPoint(min(mid + 1e-4, s_max), max_alpha)
        opacity.AddPoint(s_max, max_alpha)
    elif preset == 'soft_tissue':
        # Отсечение фона, плавная видимость мягких тканей, акцент на плотных структурах
        p1 = s_min + max(thresh_val, 0.05) * (s_max - s_min)
        p2 = s_min + 0.4 * (s_max - s_min)
        opacity.AddPoint(s_min, 0.0)
        opacity.AddPoint(p1, 0.0)
        opacity.AddPoint(p2, max_alpha * 0.35)
        opacity.AddPoint(s_max, max_alpha)
    elif preset == 'xray_translucent':
        # Полупрозрачный режим для удобного обзора внутренних источников
        opacity.AddPoint(s_min, 0.0)
        if thresh_val > 0.001:
            opacity.AddPoint(mid, 0.0)
        opacity.AddPoint(s_max, max_alpha * 0.45)
    else:  # 'air_cutoff' или 'linear'
        if thresh_val > 0.001:
            opacity.AddPoint(s_min, min_alpha)
            opacity.AddPoint(mid, min_alpha)
            opacity.AddPoint(s_max, max_alpha)
        else:
            opacity.AddPoint(s_min, min_alpha)
            opacity.AddPoint(s_max, max_alpha)

    return opacity

