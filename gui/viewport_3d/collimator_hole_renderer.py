"""
Модуль аппаратной 3D-визуализации каналов параллельного коллиматора на базе GPU-инстансинга VTK.
Обеспечивает высокопроизводительный рендеринг десятков тысяч сквозных каналов коллиматора
через vtkGlyph3DMapper и полые шестиугольные призмы без торцевых крышек по оси Z.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import logging
import numpy as np
import vtk

from core.geometry.geometries import PeriodicHexPrism
from core.geometry.direct_collimators import CollimatorHoleShape
from gui.viewport_3d.material_palette import (
    COLLIMATOR_HOLE_ACCENT_COLOR,
    COLLIMATOR_HOLE_OPACITY,
    get_collimator_visual_properties,
)

_logger = logging.getLogger(__name__)


def create_hollow_hex_prism_prototype(hole_diameter: float, height_z: float) -> vtk.vtkPolyData:
    """
    Создает полигональный меш (vtkPolyData) полой правильной шестиугольной призмы без торцевых крышек по оси Z.
    Сквозная геометрия позволяет визуально наблюдать структуру каналов коллиматора на просвет.

    Ориентация:
    - Плоские противоположные грани расположены на x = +/- hole_diameter / 2.
    - Внешний радиус вершин: outer_radius = hole_diameter / sqrt(3).
    - Углы 6 вершин в плоскости XY: (1 + 2*k) * pi / 6 для k in [0, 5].
    """
    if hole_diameter <= 0.0:
        raise ValueError(f"Диаметр канала должен быть строго положительным (> 0), получено: {hole_diameter}")
    if height_z <= 0.0:
        raise ValueError(f"Высота призмы по Z должна быть строго положительной (> 0), получено: {height_z}")

    outer_radius = float(hole_diameter / np.sqrt(3.0))
    half_height_z = float(height_z / 2.0)

    # 1. Формирование 12 вершин: 6 нижних (z = -half_height_z) и 6 верхних (z = +half_height_z)
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(12)

    for vertex_index in range(6):
        vertex_angle = float((1.0 + 2.0 * float(vertex_index)) * np.pi / 6.0)
        coord_x = float(outer_radius * np.cos(vertex_angle))
        coord_y = float(outer_radius * np.sin(vertex_angle))

        # Нижняя вершина (индекс 0..5)
        points.SetPoint(vertex_index, coord_x, coord_y, -half_height_z)
        # Верхняя вершина (индекс 6..11)
        points.SetPoint(vertex_index + 6, coord_x, coord_y, half_height_z)

    # 2. Формирование 6 боковых граней (четырехугольники vtkQuad)
    # Порядок обхода ориентирован против часовой стрелки для формирования нормалей наружу
    cell_array = vtk.vtkCellArray()
    for side_index in range(6):
        next_side_index = (side_index + 1) % 6
        quad_cell = vtk.vtkQuad()
        quad_cell.GetPointIds().SetId(0, side_index)
        quad_cell.GetPointIds().SetId(1, next_side_index)
        quad_cell.GetPointIds().SetId(2, next_side_index + 6)
        quad_cell.GetPointIds().SetId(3, side_index + 6)
        cell_array.InsertNextCell(quad_cell)

    raw_poly_data = vtk.vtkPolyData()
    raw_poly_data.SetPoints(points)
    raw_poly_data.SetPolys(cell_array)

    # 3. Расчет нормалей граней для корректного шейдинга
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(raw_poly_data)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.SplittingOff()
    normals_filter.Update()

    return normals_filter.GetOutput()


def create_hollow_square_prism_prototype(hole_width: float, height_z: float) -> vtk.vtkPolyData:
    """
    Создает полигональный меш (vtkPolyData) полой правильной квадратной призмы без торцевых крышек по оси Z.
    Сквозная геометрия позволяет визуально наблюдать структуру каналов коллиматора на просвет.

    Поперечное сечение в плоскости XY — квадрат с размером ребра hole_width
    (от -half_width до +half_width по X и Y).
    Высота призмы по Z — от -half_height_z до +half_height_z.
    """
    if hole_width <= 0.0:
        raise ValueError(f"Ширина канала должна быть строго положительной (> 0), получено: {hole_width}")
    if height_z <= 0.0:
        raise ValueError(f"Высота призмы по Z должна быть строго положительной (> 0), получено: {height_z}")

    half_width = float(0.5 * hole_width)
    half_height_z = float(0.5 * height_z)

    # 1. Формирование 8 вершин: 4 нижние (z = -half_height_z) и 4 верхние (z = +half_height_z)
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(8)

    # 4 вершины квадрата в плоскости XY с обходом против часовой стрелки (CCW)
    square_xy_vertices = (
        (-half_width, -half_width),
        (half_width, -half_width),
        (half_width, half_width),
        (-half_width, half_width),
    )

    for vertex_index, (coord_x, coord_y) in enumerate(square_xy_vertices):
        # Нижняя вершина (индекс 0..3)
        points.SetPoint(vertex_index, coord_x, coord_y, -half_height_z)
        # Верхняя вершина (индекс 4..7)
        points.SetPoint(vertex_index + 4, coord_x, coord_y, half_height_z)

    # 2. Формирование 4 боковых граней (четырехугольники vtkQuad)
    # Порядок обхода ориентирован против часовой стрелки для формирования нормалей наружу
    cell_array = vtk.vtkCellArray()
    for side_index in range(4):
        next_side_index = (side_index + 1) % 4
        quad_cell = vtk.vtkQuad()
        quad_cell.GetPointIds().SetId(0, side_index)
        quad_cell.GetPointIds().SetId(1, next_side_index)
        quad_cell.GetPointIds().SetId(2, next_side_index + 4)
        quad_cell.GetPointIds().SetId(3, side_index + 4)
        cell_array.InsertNextCell(quad_cell)

    raw_poly_data = vtk.vtkPolyData()
    raw_poly_data.SetPoints(points)
    raw_poly_data.SetPolys(cell_array)

    # 3. Расчет нормалей граней для корректного шейдинга
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(raw_poly_data)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.SplittingOff()
    normals_filter.Update()

    return normals_filter.GetOutput()


def create_hole_prototype(
    shape: CollimatorHoleShape,
    hole_diameter: float,
    height_z: float,
) -> vtk.vtkPolyData:
    """
    Фабрика прототипов геометрии сквозного канала коллиматора по заданной форме.
    Масштабируемый контракт для различных форм геометрии (гексагональная, квадратная).
    """
    if shape == CollimatorHoleShape.HEXAGONAL:
        return create_hollow_hex_prism_prototype(hole_diameter=hole_diameter, height_z=height_z)
    if shape == CollimatorHoleShape.SQUARE:
        return create_hollow_square_prism_prototype(hole_width=hole_diameter, height_z=height_z)
    raise NotImplementedError(
        f"Прототип геометрии канала для формы '{shape.value}' еще не реализован в графической подсистеме."
    )


def generate_hex_hole_centers(
    collimator_size: Sequence[float],
    hole_diameter: float,
    septa: float,
) -> np.ndarray:
    """
    Генерирует центры отверстий периодической гексагональной решетки коллиматора в плоскости XY.
    Строго соответствует расположению ячеек в расчетном ядре (_hex_prism_intersect):
    - Подрешетка 1: (i * x_period, j * y_period, 0.0)
    - Подрешетка 2: ((i + 0.5) * x_period, (j + 0.5) * y_period, 0.0)
    где x_period = hole_diameter + septa, y_period = sqrt(3) * x_period.

    Возвращает:
    np.ndarray размерности (N, 3) с координатами центров каналов в локальной СК коллиматора.
    """
    collimator_size_array = np.asarray(collimator_size, dtype=float)
    if len(collimator_size_array) < 3 or any(dim_val <= 0.0 for dim_val in collimator_size_array[:3]):
        raise ValueError(f"Размеры коллиматора должны быть строго положительными, получено: {collimator_size}")
    if hole_diameter <= 0.0:
        raise ValueError(f"Диаметр отверстия должен быть строго положительным, получено: {hole_diameter}")
    if septa <= 0.0:
        raise ValueError(f"Толщина септы должна быть строго положительной, получено: {septa}")

    size_x, size_y = float(collimator_size_array[0]), float(collimator_size_array[1])
    half_size_x = 0.5 * size_x
    half_size_y = 0.5 * size_y

    x_period = float(hole_diameter + septa)
    y_period = float(np.sqrt(3.0) * x_period)

    # Ограничения для отсечения каналов, выходящих за габариты коллиматора
    half_width_x = 0.5 * hole_diameter
    outer_radius_y = float(hole_diameter / np.sqrt(3.0))

    max_index_x = int(np.ceil(half_size_x / x_period)) + 1
    max_index_y = int(np.ceil(half_size_y / y_period)) + 1

    centers_list = []

    # Подрешетка 1: узлы (i * x_period, j * y_period)
    for index_i in range(-max_index_x, max_index_x + 1):
        center_x_first = float(index_i * x_period)
        if abs(center_x_first) + half_width_x > half_size_x:
            continue
        for index_j in range(-max_index_y, max_index_y + 1):
            center_y_first = float(index_j * y_period)
            if abs(center_y_first) + outer_radius_y <= half_size_y:
                centers_list.append((center_x_first, center_y_first, 0.0))

    # Подрешетка 2: узлы ((i + 0.5) * x_period, (j + 0.5) * y_period)
    shift_x = 0.5 * x_period
    shift_y = 0.5 * y_period
    for index_i in range(-max_index_x - 1, max_index_x + 1):
        center_x_second = float(index_i * x_period + shift_x)
        if abs(center_x_second) + half_width_x > half_size_x:
            continue
        for index_j in range(-max_index_y - 1, max_index_y + 1):
            center_y_second = float(index_j * y_period + shift_y)
            if abs(center_y_second) + outer_radius_y <= half_size_y:
                centers_list.append((center_x_second, center_y_second, 0.0))

    if not centers_list:
        return np.zeros((0, 3), dtype=float)

    return np.asarray(centers_list, dtype=float)


def generate_square_hole_centers(
    collimator_size: Sequence[float],
    hole_width: float,
    septa: float,
) -> np.ndarray:
    """
    Генерирует центры отверстий периодической квадратной решетки коллиматора в плоскости XY.
    Строго соответствует расположению ячеек в расчетном ядре ParametricParallelCollimator:
    - Период решетки по осям X и Y: period = hole_width + septa
    - Координаты узлов: (i * period, j * period, 0.0)

    Возвращает:
    np.ndarray размерности (N, 3) с координатами центров каналов в локальной СК коллиматора.
    """
    collimator_size_array = np.asarray(collimator_size, dtype=float)
    if len(collimator_size_array) < 3 or any(dim_val <= 0.0 for dim_val in collimator_size_array[:3]):
        raise ValueError(f"Размеры коллиматора должны быть строго положительными, получено: {collimator_size}")
    if hole_width <= 0.0:
        raise ValueError(f"Ширина отверстия должна быть строго положительной, получено: {hole_width}")
    if septa <= 0.0:
        raise ValueError(f"Толщина септы должна быть строго положительной, получено: {septa}")

    size_x, size_y = float(collimator_size_array[0]), float(collimator_size_array[1])
    half_size_x = 0.5 * size_x
    half_size_y = 0.5 * size_y

    period = float(hole_width + septa)
    half_hole = 0.5 * float(hole_width)

    max_index_x = int(np.ceil(half_size_x / period)) + 1
    max_index_y = int(np.ceil(half_size_y / period)) + 1

    centers_list = []
    for index_i in range(-max_index_x, max_index_x + 1):
        center_x = float(index_i * period)
        if abs(center_x) + half_hole > half_size_x:
            continue
        for index_j in range(-max_index_y, max_index_y + 1):
            center_y = float(index_j * period)
            if abs(center_y) + half_hole <= half_size_y:
                centers_list.append((center_x, center_y, 0.0))

    if not centers_list:
        return np.zeros((0, 3), dtype=float)

    return np.asarray(centers_list, dtype=float)


class CollimatorHoleRenderer:
    """
    Рендерер каналов коллиматора на основе аппаратного GPU-инстансинга VTK (vtkGlyph3DMapper).
    Обеспечивает высокую частоту кадров при отрисовке десятков тысяч отверстий.
    """

    def __init__(self, viewport: Any) -> None:
        self.viewport = viewport
        self._actors: Dict[str, vtk.vtkActor] = {}
        self._last_geometries: Dict[str, Tuple[float, float, float, float, float, str]] = {}

    @property
    def actors(self) -> Dict[str, vtk.vtkActor]:
        """Словарь активных акторов каналов коллиматора."""
        return self._actors

    def render_holes(
        self,
        actor_name: str,
        geometry_or_size: Optional[Union[PeriodicHexPrism, Sequence[float]]] = None,
        global_matrix: Optional[np.ndarray] = None,
        hole_diameter: Optional[float] = None,
        septa: Optional[float] = None,
        hole_shape: CollimatorHoleShape = CollimatorHoleShape.HEXAGONAL,
        hole_color: Tuple[float, float, float] = COLLIMATOR_HOLE_ACCENT_COLOR,
        hole_opacity: float = COLLIMATOR_HOLE_OPACITY,
        geometry: Optional[PeriodicHexPrism] = None,
    ) -> Optional[vtk.vtkActor]:
        """
        Создает или обновляет инстансированный GPU-меш каналов коллиматора.
        Поддерживает как передачу объекта геометрии PeriodicHexPrism (geometry/geometry_or_size),
        так и прямую передачу габаритов (collimator_size), диаметра отверстия и септы.
        """
        actual_matrix = global_matrix if global_matrix is not None else np.eye(4)
        target_geometry = geometry if geometry is not None else geometry_or_size
        if target_geometry is None:
            raise ValueError("Необходимо передать либо geometry, либо geometry_or_size.")

        if isinstance(target_geometry, PeriodicHexPrism):
            hole_diameter_val = float(target_geometry.hole_diameter)
            septa_val = float(target_geometry.septa)
            collimator_size = target_geometry.size
        else:
            collimator_size = target_geometry
            if hole_diameter is None or septa is None:
                raise ValueError("При передаче размеров коллиматора hole_diameter и septa обязательны.")
            hole_diameter_val = float(hole_diameter)
            septa_val = float(septa)

        height_z = float(collimator_size[2])

        geometry_key = (
            float(collimator_size[0]),
            float(collimator_size[1]),
            height_z,
            hole_diameter_val,
            septa_val,
            hole_shape.value,
        )

        existing_actor = self._actors.get(actor_name)
        cached_key = self._last_geometries.get(actor_name)

        if existing_actor is not None and cached_key == geometry_key:
            # Геометрия не изменилась: обновляем цвет, непрозрачность и матрицу
            actor_property = existing_actor.GetProperty()
            actor_property.SetColor(float(hole_color[0]), float(hole_color[1]), float(hole_color[2]))
            actor_property.SetOpacity(float(hole_opacity))
            self.viewport.update_actor_transform(actor_name, actual_matrix)
            return existing_actor

        # 1. Создание геометрического прототипа полого канала
        prototype_polydata = create_hole_prototype(
            shape=hole_shape,
            hole_diameter=hole_diameter_val,
            height_z=height_z,
        )

        # 2. Генерация центров каналов в зависимости от формы
        if hole_shape == CollimatorHoleShape.SQUARE:
            centers_array = generate_square_hole_centers(
                collimator_size=collimator_size,
                hole_width=hole_diameter_val,
                septa=septa_val,
            )
        else:
            centers_array = generate_hex_hole_centers(
                collimator_size=collimator_size,
                hole_diameter=hole_diameter_val,
                septa=septa_val,
            )

        if len(centers_array) == 0:
            self.remove_actor(actor_name)
            return None

        # 3. Формирование point cloud в vtkPolyData
        points_vtk = vtk.vtkPoints()
        points_vtk.SetNumberOfPoints(len(centers_array))
        for point_index, center_point in enumerate(centers_array):
            points_vtk.SetPoint(
                point_index,
                float(center_point[0]),
                float(center_point[1]),
                float(center_point[2]),
            )

        points_polydata = vtk.vtkPolyData()
        points_polydata.SetPoints(points_vtk)

        # 4. Настройка GPU-инстансинга через vtkGlyph3DMapper
        glyph_mapper = vtk.vtkGlyph3DMapper()
        glyph_mapper.SetInputData(points_polydata)
        glyph_mapper.SetSourceData(prototype_polydata)
        glyph_mapper.SetScaleModeToNoDataScaling()
        glyph_mapper.SetScaling(False)
        glyph_mapper.SetOrientationModeToDirection()

        # 5. Создание и настройка VTK-актора
        hole_actor = vtk.vtkActor()
        hole_actor.SetMapper(glyph_mapper)

        actor_property = hole_actor.GetProperty()
        actor_property.SetColor(float(hole_color[0]), float(hole_color[1]), float(hole_color[2]))
        actor_property.SetOpacity(float(hole_opacity))
        actor_property.BackfaceCullingOff()
        actor_property.SetAmbient(0.2)
        actor_property.SetDiffuse(0.8)

        # 6. Регистрация актора во вьюпорте
        self.viewport.add_actor(actor_name, hole_actor)
        self.viewport.update_actor_transform(actor_name, global_matrix)

        self._actors[actor_name] = hole_actor
        self._last_geometries[actor_name] = geometry_key

        return hole_actor

    def remove_actor(self, actor_name: str) -> None:
        """Удаляет актор каналов коллиматора из вьюпорта и внутреннего кэша."""
        if actor_name in self._actors:
            self.viewport.remove_actor(actor_name)
            self._actors.pop(actor_name, None)
            self._last_geometries.pop(actor_name, None)

    def clear(self) -> None:
        """Очищает все акторы каналов коллиматора."""
        for actor_name in list(self._actors.keys()):
            self.remove_actor(actor_name)


__all__ = [
    "create_hollow_hex_prism_prototype",
    "create_hollow_square_prism_prototype",
    "create_hole_prototype",
    "generate_hex_hole_centers",
    "generate_square_hole_centers",
    "CollimatorHoleRenderer",
]
