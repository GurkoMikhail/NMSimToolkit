import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pyvista as pv
import vtk
from PySide6.QtCore import QObject, Signal

from core.geometry.volumes import Volume

_logger = logging.getLogger(__name__)


@runtime_checkable
class IViewport(Protocol):
    """Контракт интерактивного 3D-вьюпорта для манипулятора трансформаций."""

    @property
    def interactor(self) -> Optional[Any]: ...

    @property
    def plotter(self) -> Optional[Any]: ...

    def add_mesh_actor(
        self,
        name: str,
        mesh: Any,
        color: Optional[str] = 'white',
        opacity: float = 1.0,
        style: str = 'surface',
        wireframe: bool = False,
        rgb: bool = False,
        **kwargs: Any
    ) -> Optional[Any]: ...

    def remove_actor(self, name: str) -> None: ...

    def render(self) -> None: ...


from gui.viewport_3d.gizmo_types import GizmoAxis, GizmoMode, GizmoSpace



class TransformGizmo(QObject):
    """
    Интерактивный 3D-манипулятор трансформации объектов сцены в стиле Unreal Engine (Transform Gizmo).
    Поддерживает:
    - 3 режима манипуляции: Перемещение (W), Вращение (E), Масштабирование (R).
    - Системы координат: World Space (Мировая) и Local Space (Локальная объекта).
    - UE-Style ступенчатый снаппинг (координатная сетка, углы поворота, шаг масштаба)
      с возможностью временного отключения снаппинга через клавишу Shift.
    - Двухосевые плоскости перемещения (XY, XZ, YZ) и центральный равномерный масштаб (XYZ).
    - Прямое обновление матриц узлов NodeViewModel с инвалидацией геометрического кэша.
    """

    mode_changed = Signal(object)        # GizmoMode
    space_changed = Signal(object)       # GizmoSpace
    transform_started = Signal()
    transform_changed = Signal(object)   # np.ndarray (новая локальная матрица)
    transform_ended = Signal()
    status_message_requested = Signal(str)

    def __init__(
        self,
        viewport: Any,
        target_node: Optional[NodeViewModel] = None,
        grid_snap_step: float = 10.0,
        angle_snap_step: float = 5.0,
        scale_snap_step: float = 0.1,
        gizmo_size: float = 80.0,
        min_gizmo_size: float = 40.0,
        max_gizmo_size: float = 350.0,
        adaptive_size: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.viewport = viewport
        self._target_node: Optional[NodeViewModel] = target_node
        self._constraint: Optional[Any] = None
        self._mode: GizmoMode = GizmoMode.TRANSLATE
        self._space: GizmoSpace = GizmoSpace.WORLD

        self._grid_snap_step: float = float(grid_snap_step)
        self._angle_snap_step: float = float(angle_snap_step)
        self._scale_snap_step: float = float(scale_snap_step)
        self._gizmo_size: float = float(gizmo_size)
        self._min_gizmo_size: float = float(min_gizmo_size)
        self._max_gizmo_size: float = float(max_gizmo_size)
        self._adaptive_size: bool = bool(adaptive_size)

        self._active_axis: GizmoAxis = GizmoAxis.NONE
        self._is_dragging: bool = False
        self._initial_drag_matrix: Optional[np.ndarray] = None
        self._initial_drag_position: Optional[np.ndarray] = None
        self._accumulated_world_delta: np.ndarray = np.zeros(3, dtype=np.float64)
        self._accumulated_angle_deg: float = 0.0
        self._accumulated_scale_factor: float = 1.0
        self._last_changed_data: Optional[Dict[str, Any]] = None
        self._last_built_size: float = float(gizmo_size)
        self._last_mouse_pos: Optional[Tuple[int, int]] = None
        self._hovered_actor_name: Optional[str] = None
        self._observer_tags: Dict[str, int] = {}
        self._mesh_actors: Dict[str, Any] = {}

        self._actor_names: List[str] = []
        self._colors: Dict[GizmoAxis, str] = {
            GizmoAxis.X: "#ff3b30",    # Красный (Red)
            GizmoAxis.Y: "#34c759",    # Зеленый (Green)
            GizmoAxis.Z: "#007aff",    # Синий (Blue)
            GizmoAxis.XY: "#ffd60a",   # Желтый (Yellow)
            GizmoAxis.XZ: "#32ade6",   # Голубой (Cyan)
            GizmoAxis.YZ: "#ff2d55",   # Маджента (Magenta)
            GizmoAxis.XYZ: "#f2f2f7",  # Белый (White)
        }

        self._actor_axis_map: Dict[str, GizmoAxis] = {
            "gizmo_translate_x": GizmoAxis.X,
            "gizmo_translate_y": GizmoAxis.Y,
            "gizmo_translate_z": GizmoAxis.Z,
            "gizmo_translate_xy": GizmoAxis.XY,
            "gizmo_translate_xz": GizmoAxis.XZ,
            "gizmo_translate_yz": GizmoAxis.YZ,
            "gizmo_rotate_x": GizmoAxis.X,
            "gizmo_rotate_y": GizmoAxis.Y,
            "gizmo_rotate_z": GizmoAxis.Z,
            "gizmo_scale_x": GizmoAxis.X,
            "gizmo_scale_y": GizmoAxis.Y,
            "gizmo_scale_z": GizmoAxis.Z,
            "gizmo_scale_xyz": GizmoAxis.XYZ,
        }

        self._setup_interactor()

        if self._target_node is not None:
            self.update_visuals()

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства конфигурации и состояния
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def target_node(self) -> Optional[NodeViewModel]:
        return self._target_node

    @property
    def constraint(self) -> Optional[Any]:
        """Кинематическое ограничение манипулятора."""
        return self._constraint

    @constraint.setter
    def constraint(self, new_constraint: Optional[Any]) -> None:
        self.set_constraint(new_constraint)

    def set_constraint(self, constraint: Optional[Any]) -> None:
        """
        Устанавливает кинематическое ограничение степеней свободы манипулятора.
        При наличии принудительной системы координат принудительно переключает space.
        При запрете масштабирования сбрасывает текущий режим в TRANSLATE.
        """
        if self._is_dragging:
            self._is_dragging = False
            self._active_axis = GizmoAxis.NONE
            self._last_mouse_pos = None
            self._initial_drag_matrix = None
            self._accumulated_world_delta = np.zeros(3, dtype=np.float64)
            self._accumulated_angle_deg = 0.0
            self._accumulated_scale_factor = 1.0
            self._last_changed_data = None
            self.transform_ended.emit()
            self._reset_highlight()

        self._constraint = constraint
        if self._constraint is not None:
            forced_space = self._constraint.get_forced_space()
            if forced_space is not None and self._space != forced_space:
                self._space = forced_space
                self.space_changed.emit(self._space)
            if not self._constraint.is_scale_allowed() and self._mode == GizmoMode.SCALE:
                self._mode = GizmoMode.TRANSLATE
                self.mode_changed.emit(self._mode)
        self.update_visuals(render=True)

    @property
    def mode(self) -> GizmoMode:
        return self._mode

    @mode.setter
    def mode(self, new_mode: GizmoMode) -> None:
        if self._constraint is not None and not self._constraint.is_scale_allowed() and new_mode == GizmoMode.SCALE:
            return
        if self._mode != new_mode:
            self._mode = new_mode
            self.mode_changed.emit(self._mode)
            self.update_visuals(render=True)

    @property
    def space(self) -> GizmoSpace:
        return self._space

    @space.setter
    def space(self, new_space: GizmoSpace) -> None:
        if self._constraint is not None:
            forced_space = self._constraint.get_forced_space()
            if forced_space is not None and new_space != forced_space:
                return
        if self._space != new_space:
            self._space = new_space
            self.space_changed.emit(self._space)
            self.update_visuals(render=True)

    @property
    def grid_snap_step(self) -> float:
        return self._grid_snap_step

    @grid_snap_step.setter
    def grid_snap_step(self, step: float) -> None:
        self._grid_snap_step = max(0.0, float(step))

    @property
    def angle_snap_step(self) -> float:
        return self._angle_snap_step

    @angle_snap_step.setter
    def angle_snap_step(self, step: float) -> None:
        self._angle_snap_step = max(0.0, float(step))

    @property
    def scale_snap_step(self) -> float:
        return self._scale_snap_step

    @scale_snap_step.setter
    def scale_snap_step(self, step: float) -> None:
        self._scale_snap_step = max(0.0, float(step))

    @property
    def gizmo_size(self) -> float:
        return self._gizmo_size

    @gizmo_size.setter
    def gizmo_size(self, size: float) -> None:
        self._gizmo_size = max(1.0, float(size))
        if self._gizmo_size > self._max_gizmo_size:
            self._max_gizmo_size = self._gizmo_size * 2.0
        self.update_visuals(render=True)

    @property
    def min_gizmo_size(self) -> float:
        """Минимально допустимый размер манипулятора (мм)."""
        return self._min_gizmo_size

    @min_gizmo_size.setter
    def min_gizmo_size(self, size: float) -> None:
        self._min_gizmo_size = max(1.0, float(size))
        self.update_visuals(render=True)

    @property
    def max_gizmo_size(self) -> float:
        """Максимально допустимый размер манипулятора (мм)."""
        return self._max_gizmo_size

    @max_gizmo_size.setter
    def max_gizmo_size(self, size: float) -> None:
        self._max_gizmo_size = max(self._min_gizmo_size, float(size))
        self.update_visuals(render=True)

    @property
    def adaptive_size(self) -> bool:
        """Включен ли режим адаптивного вычисления размера манипулятора."""
        return self._adaptive_size

    @adaptive_size.setter
    def adaptive_size(self, enabled: bool) -> None:
        self._adaptive_size = bool(enabled)
        self.update_visuals(render=True)

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    @property
    def active_axis(self) -> GizmoAxis:
        return self._active_axis

    # ------------------------------------------------------------------------------------------------------------------
    # Привязка целевого узла
    # ------------------------------------------------------------------------------------------------------------------

    def set_target_node(self, node_vm: Optional[NodeViewModel]) -> None:
        """
        Устанавливает целевой узел для интерактивной трансформации.
        При передаче None манипулятор отсоединяется и скрывается.
        """
        if self._target_node is node_vm:
            return

        self._target_node = node_vm
        if self._target_node is not None:
            self._setup_interactor()
            self.update_visuals(render=True)
        else:
            self._constraint = None
            self.remove_visuals()
            if self.viewport is not None:
                self.viewport.render()

    def detach(self) -> None:
        """Отсоединяет целевой узел и удаляет визуальные акторы манипулятора."""
        if self._is_dragging:
            self._is_dragging = False
            self._active_axis = GizmoAxis.NONE
            self.transform_ended.emit()
        self._constraint = None
        self.set_target_node(None)

    def close(self) -> None:
        """Освобождение всех ресурсов и отключение обработчиков VTK."""
        self._remove_interactor_observers()
        self.detach()

    # ------------------------------------------------------------------------------------------------------------------
    # Алгоритмы ступенчатого снаппинга (UE-Style Snapping)
    # ------------------------------------------------------------------------------------------------------------------

    def snap_coordinate(self, coordinate_value: float, shift_modifier: bool = False) -> float:
        """Снаппинг линейной координаты к узлам координатной сетки."""
        if shift_modifier or self._grid_snap_step <= 0.0:
            return float(coordinate_value)
        snapped_coord = round(coordinate_value / self._grid_snap_step) * self._grid_snap_step
        return float(snapped_coord)

    def snap_angle(self, angle_deg_value: float, shift_modifier: bool = False) -> float:
        """Снаппинг угла поворота в градусах к угловой сетке."""
        if shift_modifier or self._angle_snap_step <= 0.0:
            return float(angle_deg_value)
        snapped_angle = round(angle_deg_value / self._angle_snap_step) * self._angle_snap_step
        return float(snapped_angle)

    def snap_scale_factor(self, scale_factor_value: float, shift_modifier: bool = False) -> float:
        """Снаппинг коэффициента масштаба к ступенчатому множителю."""
        if shift_modifier or self._scale_snap_step <= 0.0:
            return float(scale_factor_value)
        snapped_scale = round(scale_factor_value / self._scale_snap_step) * self._scale_snap_step
        return float(max(0.001, snapped_scale))

    def snap_translation_vector(
        self,
        position_vector: np.ndarray,
        shift_modifier: bool = False
    ) -> np.ndarray:
        """Снаппинг 3D-вектора абсолютной позиции к координатной сетке."""
        if shift_modifier or self._grid_snap_step <= 0.0:
            return position_vector.copy()
        snapped_vec = np.round(position_vector / self._grid_snap_step) * self._grid_snap_step
        return snapped_vec.astype(position_vector.dtype)

    def snap_rotation_vector(
        self,
        angles_deg_vector: np.ndarray,
        shift_modifier: bool = False
    ) -> np.ndarray:
        """Снаппинг углов Эйлера (градусы) к угловой сетке."""
        if shift_modifier or self._angle_snap_step <= 0.0:
            return angles_deg_vector.copy()
        snapped_angles = np.round(angles_deg_vector / self._angle_snap_step) * self._angle_snap_step
        return snapped_angles.astype(angles_deg_vector.dtype)

    def snap_scale_vector(
        self,
        scale_vector: np.ndarray,
        shift_modifier: bool = False
    ) -> np.ndarray:
        """Снаппинг 3D-вектора масштабирования."""
        if shift_modifier or self._scale_snap_step <= 0.0:
            return scale_vector.copy()
        snapped_scale = np.round(scale_vector / self._scale_snap_step) * self._scale_snap_step
        return np.maximum(0.001, snapped_scale).astype(scale_vector.dtype)

    # ------------------------------------------------------------------------------------------------------------------
    # Математические трансформации и вычисление дельт
    # ------------------------------------------------------------------------------------------------------------------

    def apply_translation(
        self,
        delta_vector: np.ndarray,
        shift_modifier: bool = False
    ) -> Optional[np.ndarray]:
        """
        Применение смещения к целевому объекту с учетом снаппинга и выбранной системы координат.
        Возвращает обновленную локальную матрицу трансформации 4x4.
        """
        if self._target_node is None:
            return None

        if self._constraint is not None:
            initial_matrix = self._target_node.local_matrix.copy()
            if self._space == GizmoSpace.LOCAL:
                dir_x, dir_y, dir_z = self._extract_orthonormal_basis(self._target_node.global_matrix)
                rotation_basis = np.column_stack([dir_x, dir_y, dir_z])
                world_delta = rotation_basis @ delta_vector
            else:
                world_delta = delta_vector

            filtered_delta, changed_data = self._constraint.filter_translation(
                self._target_node,
                world_delta,
                initial_matrix,
            )
            if 'matrix' in changed_data:
                new_matrix = changed_data['matrix']
            else:
                new_matrix = initial_matrix.copy()
                new_matrix[0:3, 3] = initial_matrix[0:3, 3] + filtered_delta
            self._target_node.local_matrix = new_matrix
            self.transform_changed.emit(new_matrix)
            self._constraint.on_transform_changed(self._target_node, changed_data)
            self._constraint.on_transform_committed(self._target_node, changed_data)
            if 'status_message' in changed_data:
                self.status_message_requested.emit(str(changed_data['status_message']))
            self.update_visuals()
            return new_matrix

        current_matrix = self._target_node.local_matrix.copy()
        current_pos = current_matrix[0:3, 3].copy()

        if self._space == GizmoSpace.LOCAL:
            dir_x, dir_y, dir_z = self._extract_orthonormal_basis(self._target_node.global_matrix)
            rotation_basis = np.column_stack([dir_x, dir_y, dir_z])
            world_delta = rotation_basis @ delta_vector
        else:
            world_delta = delta_vector

        # Преобразование мирового смещения в систему координат родительского узла
        if self._target_node.parent_vm is not None:
            parent_dir_x, parent_dir_y, parent_dir_z = self._extract_orthonormal_basis(
                self._target_node.parent_vm.global_matrix
            )
            parent_basis = np.column_stack([parent_dir_x, parent_dir_y, parent_dir_z])
            effective_delta = parent_basis.T @ world_delta
        else:
            effective_delta = world_delta

        target_pos = current_pos + effective_delta
        snapped_target_pos = self.snap_translation_vector(target_pos, shift_modifier=shift_modifier)

        current_matrix[0:3, 3] = snapped_target_pos
        self._target_node.local_matrix = current_matrix
        self.transform_changed.emit(current_matrix)
        self.update_visuals()
        return current_matrix

    def apply_rotation(
        self,
        axis_name: str,
        angle_deg: float,
        shift_modifier: bool = False
    ) -> Optional[np.ndarray]:
        """
        Применение поворота вокруг заданной оси ('x', 'y', 'z') с учетом снаппинга.
        """
        if self._target_node is None:
            return None

        if self._constraint is not None:
            initial_matrix = self._target_node.local_matrix.copy()
            dir_x, dir_y, dir_z = self._extract_orthonormal_basis(self._target_node.global_matrix)
            axis_map = {'x': dir_x, 'y': dir_y, 'z': dir_z}
            axis_vec = axis_map.get(axis_name.lower(), dir_z)
            filtered_axis, filtered_angle, changed_data = self._constraint.filter_rotation(
                self._target_node,
                axis_vec,
                angle_deg,
                initial_matrix,
            )
            if 'matrix' in changed_data:
                new_matrix = changed_data['matrix']
            else:
                new_matrix = initial_matrix.copy()
            self._target_node.local_matrix = new_matrix
            self.transform_changed.emit(new_matrix)
            self._constraint.on_transform_changed(self._target_node, changed_data)
            self._constraint.on_transform_committed(self._target_node, changed_data)
            if 'status_message' in changed_data:
                self.status_message_requested.emit(str(changed_data['status_message']))
            self.update_visuals()
            return new_matrix

        snapped_angle_deg = self.snap_angle(angle_deg, shift_modifier=shift_modifier)
        angle_rad = math.radians(snapped_angle_deg)

        rotation_mat_3x3 = np.eye(3, dtype=np.float64)
        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        if axis_name.lower() == 'x':
            rotation_mat_3x3 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, cos_val, -sin_val],
                [0.0, sin_val, cos_val]
            ], dtype=np.float64)
        elif axis_name.lower() == 'y':
            rotation_mat_3x3 = np.array([
                [cos_val, 0.0, sin_val],
                [0.0, 1.0, 0.0],
                [-sin_val, 0.0, cos_val]
            ], dtype=np.float64)
        elif axis_name.lower() == 'z':
            rotation_mat_3x3 = np.array([
                [cos_val, -sin_val, 0.0],
                [sin_val, cos_val, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
        else:
            raise ValueError(f"Неизвестная ось вращения: {axis_name}")

        current_matrix = self._target_node.local_matrix.copy()
        current_center = current_matrix[0:3, 3].copy()

        if self._space == GizmoSpace.LOCAL:
            # Вращение в локальной системе координат
            current_matrix[0:3, 0:3] = current_matrix[0:3, 0:3] @ rotation_mat_3x3
        else:
            # Вращение в глобальной (мировой) системе вокруг центра объекта
            if self._target_node.parent_vm is not None:
                parent_dir_x, parent_dir_y, parent_dir_z = self._extract_orthonormal_basis(
                    self._target_node.parent_vm.global_matrix
                )
                parent_basis = np.column_stack([parent_dir_x, parent_dir_y, parent_dir_z])
                local_rot = parent_basis.T @ rotation_mat_3x3 @ parent_basis
                current_matrix[0:3, 0:3] = local_rot @ current_matrix[0:3, 0:3]
            else:
                current_matrix[0:3, 0:3] = rotation_mat_3x3 @ current_matrix[0:3, 0:3]

        current_matrix[0:3, 3] = current_center
        self._target_node.local_matrix = current_matrix
        self.transform_changed.emit(current_matrix)
        self.update_visuals()
        return current_matrix

    def apply_scale(
        self,
        scale_multipliers: np.ndarray,
        shift_modifier: bool = False
    ) -> Optional[np.ndarray]:
        """
        Применение масштабирования к объекту вдоль его локальных осей.
        """
        if self._target_node is None:
            return None

        if self._constraint is not None and not self._constraint.is_scale_allowed():
            return None

        snapped_multipliers = self.snap_scale_vector(scale_multipliers, shift_modifier=shift_modifier)
        current_matrix = self._target_node.local_matrix.copy()

        # Масштабирование применяется к базисным векторам столбцов локальной матрицы
        for col_idx in range(3):
            current_matrix[0:3, col_idx] *= snapped_multipliers[col_idx]

        self._target_node.local_matrix = current_matrix
        self.transform_changed.emit(current_matrix)
        self.update_visuals()
        return current_matrix

    # ------------------------------------------------------------------------------------------------------------------
    # Управление горячими клавишами (UE-Style Shortcuts)
    # ------------------------------------------------------------------------------------------------------------------

    def handle_key_action(self, key_character: str) -> bool:
        """
        Обработка горячих клавиш переключения режимов и систем координат.
        W / Ц — Перемещение (Translate)
        E / У — Вращение (Rotate)
        R / К — Масштабирование (Scale)
        Q / Й — Переключение World / Local Space
        """
        key_upper = key_character.strip().upper()
        if key_upper in ('W', 'Ц'):
            self.mode = GizmoMode.TRANSLATE
            return True
        elif key_upper in ('E', 'У'):
            self.mode = GizmoMode.ROTATE
            return True
        elif key_upper in ('R', 'К'):
            if self._constraint is not None and not self._constraint.is_scale_allowed():
                return False
            self.mode = GizmoMode.SCALE
            return True
        elif key_upper in ('Q', 'Й'):
            if self._constraint is not None and self._constraint.get_forced_space() is not None:
                return False
            self.space = GizmoSpace.WORLD if self._space == GizmoSpace.LOCAL else GizmoSpace.LOCAL
            return True
        return False

    # ------------------------------------------------------------------------------------------------------------------
    # Вспомогательные алгоритмы ориентации и масштабирования
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _extract_orthonormal_basis(
        matrix_4x4: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Извлекает строго ортонормированный 3D-базис (direction_x, direction_y, direction_z) из матрицы трансформации.
        Гарантирует, что направления осей имеют строго единичную длину (1.0), взаимно ортогональны
        и сохраняют истинные направления локальных осей объекта даже при отрицательных масштабах (зеркалировании),
        деформациях или скосах матрицы целевого узла.
        """
        sub_matrix_3x3 = np.asarray(matrix_4x4[0:3, 0:3], dtype=np.float64)

        column_x = sub_matrix_3x3[:, 0].copy()
        column_y = sub_matrix_3x3[:, 1].copy()
        column_z = sub_matrix_3x3[:, 2].copy()

        norm_x = float(np.linalg.norm(column_x))
        norm_y = float(np.linalg.norm(column_y))
        norm_z = float(np.linalg.norm(column_z))

        direction_x = column_x / norm_x if norm_x > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        direction_y = column_y / norm_y if norm_y > 1e-9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        direction_z = column_z / norm_z if norm_z > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Модифицированный процесс Грама-Шмидта для устранения скосов и деформаций
        direction_y = direction_y - float(np.dot(direction_y, direction_x)) * direction_x
        norm_y_ortho = float(np.linalg.norm(direction_y))
        if norm_y_ortho > 1e-9:
            direction_y = direction_y / norm_y_ortho
        else:
            candidate_vector = np.array([0.0, 1.0, 0.0], dtype=np.float64) if abs(direction_x[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
            direction_y = candidate_vector - float(np.dot(candidate_vector, direction_x)) * direction_x
            direction_y /= float(np.linalg.norm(direction_y))

        direction_z = direction_z - float(np.dot(direction_z, direction_x)) * direction_x - float(np.dot(direction_z, direction_y)) * direction_y
        norm_z_ortho = float(np.linalg.norm(direction_z))
        if norm_z_ortho > 1e-9:
            direction_z = direction_z / norm_z_ortho
        else:
            determinant_value = float(np.linalg.det(sub_matrix_3x3))
            cross_vector = np.cross(direction_x, direction_y)
            cross_norm = float(np.linalg.norm(cross_vector))
            if cross_norm > 1e-9:
                direction_z = (cross_vector / cross_norm) * (-1.0 if determinant_value < -1e-9 else 1.0)
            else:
                direction_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        return (
            np.ascontiguousarray(direction_x, dtype=np.float64),
            np.ascontiguousarray(direction_y, dtype=np.float64),
            np.ascontiguousarray(direction_z, dtype=np.float64),
        )

    def _compute_gizmo_size(self, gizmo_center: np.ndarray) -> float:
        """
        Вычисляет эргономичный и стабильный размер манипулятора (Gizmo size).
        Размер изолирован от внутреннего масштаба (scale) матрицы целевого узла
        и ограничивается разумными пределами [min_gizmo_size, max_gizmo_size],
        либо рассчитывается относительно камеры/сцены.
        """
        base_size = self._gizmo_size
        if not self._adaptive_size:
            return float(max(self._min_gizmo_size, min(base_size, self._max_gizmo_size)))

        # 1. Попытка расчета относительно камеры для сохранения комфортного экранного размера
        camera_size: Optional[float] = None
        if self.viewport is not None and self.viewport.plotter is not None:
            renderer = self.viewport.plotter.renderer
            if renderer is not None:
                try:
                    camera = renderer.GetActiveCamera()
                    if camera is not None:
                        if bool(camera.GetParallelProjection()):
                            parallel_scale = float(camera.GetParallelScale())
                            if parallel_scale > 0.0:
                                camera_size = parallel_scale * 0.28
                        else:
                            camera_position = np.array(camera.GetPosition(), dtype=np.float64)
                            distance_to_camera = float(np.linalg.norm(camera_position - gizmo_center))
                            view_angle_deg = float(camera.GetViewAngle())
                            if distance_to_camera > 1.0 and view_angle_deg > 0.0:
                                visible_height = 2.0 * distance_to_camera * math.tan(
                                    math.radians(view_angle_deg / 2.0)
                                )
                                camera_size = visible_height * 0.15
                except Exception as camera_err:
                    _logger.debug(f"Ошибка вычисления размера манипулятора по камере: {camera_err}")
                    camera_size = None

        target_volume: Optional[Volume] = None
        if isinstance(self._target_node, Volume):
            target_volume = self._target_node
        elif self._target_node is not None and isinstance(self._target_node.core_node, Volume):
            target_volume = self._target_node.core_node

        if camera_size is not None and camera_size > 0.0:
            target_size = camera_size
        elif target_volume is not None:
            # 2. Адаптивный режим по геометрическим габаритам объема.
            # Внимание: берется чистый local_bound геометрии (без домножения на scale матрицы трансформации объекта).
            local_dimensions = target_volume.local_bound
            max_dimension = float(np.max(local_dimensions))
            if max_dimension > 0.0:
                half_extent = max_dimension * 0.5
                target_size = max(base_size, half_extent * 1.35)
            else:
                target_size = base_size
        else:
            target_size = base_size

        # Ограничение разумными пределами [min_gizmo_size, max_gizmo_size]
        clamped_size = max(self._min_gizmo_size, min(target_size, self._max_gizmo_size))
        return float(clamped_size)

    # ------------------------------------------------------------------------------------------------------------------
    # Визуализация манипулятора в PyVista / VTK
    # ------------------------------------------------------------------------------------------------------------------

    def update_visuals(self, render: bool = True) -> None:
        """
        Построение или обновление полигональных мешей манипулятора во вьюпорте.
        """
        self.remove_visuals()

        if self._target_node is None or self.viewport is None:
            return

        if self.viewport.plotter is None:
            return

        global_matrix = self._target_node.global_matrix
        gizmo_center = global_matrix[0:3, 3]

        if self._space == GizmoSpace.LOCAL:
            dir_x, dir_y, dir_z = self._extract_orthonormal_basis(global_matrix)
        else:
            dir_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            dir_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            dir_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        size = self._compute_gizmo_size(gizmo_center)
        self._last_built_size = size

        try:
            if self._mode == GizmoMode.TRANSLATE:
                self._build_translate_visuals(gizmo_center, dir_x, dir_y, dir_z, size)
            elif self._mode == GizmoMode.ROTATE:
                self._build_rotate_visuals(gizmo_center, dir_x, dir_y, dir_z, size)
            elif self._mode == GizmoMode.SCALE:
                self._build_scale_visuals(gizmo_center, dir_x, dir_y, dir_z, size)

            if render:
                self.viewport.render()
        except Exception as err:
            _logger.error(f"Ошибка построения мешей манипулятора Gizmo: {err}")

    def _build_translate_visuals(
        self,
        center: np.ndarray,
        dir_x: np.ndarray,
        dir_y: np.ndarray,
        dir_z: np.ndarray,
        size: float
    ) -> None:
        """Построение 3 осевых стрелок и 3 координатных плоскостей перемещения."""
        allowed_axes: Set[GizmoAxis] = (
            self._constraint.get_allowed_axes(GizmoMode.TRANSLATE)
            if self._constraint is not None
            else {GizmoAxis.X, GizmoAxis.Y, GizmoAxis.Z, GizmoAxis.XY, GizmoAxis.XZ, GizmoAxis.YZ}
        )

        arrow_shaft_radius = size * 0.03
        arrow_tip_length = size * 0.25
        arrow_tip_radius = size * 0.08

        # Стрелка X
        if GizmoAxis.X in allowed_axes:
            arrow_x = pv.Arrow(
                start=center,
                direction=dir_x,
                tip_length=arrow_tip_length / size,
                tip_radius=arrow_tip_radius / size,
                shaft_radius=arrow_shaft_radius / size,
                scale=size
            )
            self._add_gizmo_mesh("gizmo_translate_x", arrow_x, self._colors[GizmoAxis.X])

        # Стрелка Y
        if GizmoAxis.Y in allowed_axes:
            arrow_y = pv.Arrow(
                start=center,
                direction=dir_y,
                tip_length=arrow_tip_length / size,
                tip_radius=arrow_tip_radius / size,
                shaft_radius=arrow_shaft_radius / size,
                scale=size
            )
            self._add_gizmo_mesh("gizmo_translate_y", arrow_y, self._colors[GizmoAxis.Y])

        # Стрелка Z
        if GizmoAxis.Z in allowed_axes:
            arrow_z = pv.Arrow(
                start=center,
                direction=dir_z,
                tip_length=arrow_tip_length / size,
                tip_radius=arrow_tip_radius / size,
                shaft_radius=arrow_shaft_radius / size,
                scale=size
            )
            self._add_gizmo_mesh("gizmo_translate_z", arrow_z, self._colors[GizmoAxis.Z])

        # Плоскости XY, XZ, YZ
        plane_size = size * 0.25
        plane_offset = size * 0.2

        if GizmoAxis.XY in allowed_axes:
            center_xy = center + dir_x * plane_offset + dir_y * plane_offset
            plane_xy = pv.Plane(center=center_xy, direction=dir_z, i_size=plane_size, j_size=plane_size)
            self._add_gizmo_mesh("gizmo_translate_xy", plane_xy, self._colors[GizmoAxis.XY], opacity=0.45)

        if GizmoAxis.XZ in allowed_axes:
            center_xz = center + dir_x * plane_offset + dir_z * plane_offset
            plane_xz = pv.Plane(center=center_xz, direction=dir_y, i_size=plane_size, j_size=plane_size)
            self._add_gizmo_mesh("gizmo_translate_xz", plane_xz, self._colors[GizmoAxis.XZ], opacity=0.45)

        if GizmoAxis.YZ in allowed_axes:
            center_yz = center + dir_y * plane_offset + dir_z * plane_offset
            plane_yz = pv.Plane(center=center_yz, direction=dir_x, i_size=plane_size, j_size=plane_size)
            self._add_gizmo_mesh("gizmo_translate_yz", plane_yz, self._colors[GizmoAxis.YZ], opacity=0.45)

    def _build_rotate_visuals(
        self,
        center: np.ndarray,
        dir_x: np.ndarray,
        dir_y: np.ndarray,
        dir_z: np.ndarray,
        size: float
    ) -> None:
        """Построение 3 круговых колец вращения вокруг осей X, Y, Z."""
        allowed_axes: Set[GizmoAxis] = (
            self._constraint.get_allowed_axes(GizmoMode.ROTATE)
            if self._constraint is not None
            else {GizmoAxis.X, GizmoAxis.Y, GizmoAxis.Z}
        )

        ring_radius = size * 0.9
        inner_r = ring_radius * 0.96
        outer_r = ring_radius * 1.04

        # Кольцо вокруг X
        if GizmoAxis.X in allowed_axes:
            ring_x = pv.Disc(center=center, inner=inner_r, outer=outer_r, normal=dir_x, r_res=1, c_res=48)
            self._add_gizmo_mesh("gizmo_rotate_x", ring_x, self._colors[GizmoAxis.X], opacity=0.85)

        # Кольцо вокруг Y
        if GizmoAxis.Y in allowed_axes:
            ring_y = pv.Disc(center=center, inner=inner_r, outer=outer_r, normal=dir_y, r_res=1, c_res=48)
            self._add_gizmo_mesh("gizmo_rotate_y", ring_y, self._colors[GizmoAxis.Y], opacity=0.85)

        # Кольцо вокруг Z
        if GizmoAxis.Z in allowed_axes:
            ring_z = pv.Disc(center=center, inner=inner_r, outer=outer_r, normal=dir_z, r_res=1, c_res=48)
            self._add_gizmo_mesh("gizmo_rotate_z", ring_z, self._colors[GizmoAxis.Z], opacity=0.85)

    def _build_scale_visuals(
        self,
        center: np.ndarray,
        dir_x: np.ndarray,
        dir_y: np.ndarray,
        dir_z: np.ndarray,
        size: float
    ) -> None:
        """Построение осевых кубиков масштабирования и центрального кубика (XYZ)."""
        if self._constraint is not None and not self._constraint.is_scale_allowed():
            return
        box_edge = size * 0.12
        axis_len = size * 0.85
        shaft_rad = size * 0.02

        # Осевой брусок + кубик X
        cyl_x = pv.Cylinder(center=center + dir_x * (axis_len * 0.5), direction=dir_x, radius=shaft_rad, height=axis_len)
        box_x = pv.Cube(center=center + dir_x * axis_len, x_length=box_edge, y_length=box_edge, z_length=box_edge)
        mesh_x = cyl_x.merge(box_x)
        self._add_gizmo_mesh("gizmo_scale_x", mesh_x, self._colors[GizmoAxis.X])

        # Осевой брусок + кубик Y
        cyl_y = pv.Cylinder(center=center + dir_y * (axis_len * 0.5), direction=dir_y, radius=shaft_rad, height=axis_len)
        box_y = pv.Cube(center=center + dir_y * axis_len, x_length=box_edge, y_length=box_edge, z_length=box_edge)
        mesh_y = cyl_y.merge(box_y)
        self._add_gizmo_mesh("gizmo_scale_y", mesh_y, self._colors[GizmoAxis.Y])

        # Осевой брусок + кубик Z
        cyl_z = pv.Cylinder(center=center + dir_z * (axis_len * 0.5), direction=dir_z, radius=shaft_rad, height=axis_len)
        box_z = pv.Cube(center=center + dir_z * axis_len, x_length=box_edge, y_length=box_edge, z_length=box_edge)
        mesh_z = cyl_z.merge(box_z)
        self._add_gizmo_mesh("gizmo_scale_z", mesh_z, self._colors[GizmoAxis.Z])

        # Центральный кубик для равномерного 3-осевого масштабирования (XYZ)
        center_box = pv.Cube(center=center, x_length=box_edge * 1.2, y_length=box_edge * 1.2, z_length=box_edge * 1.2)
        self._add_gizmo_mesh("gizmo_scale_xyz", center_box, self._colors[GizmoAxis.XYZ], opacity=0.9)

    def _add_gizmo_mesh(
        self,
        actor_name: str,
        mesh_polydata: Any,
        color_hex: str,
        opacity: float = 1.0
    ) -> None:
        """Регистрация и добавление меша манипулятора в сцену вьюпорта."""
        if self.viewport is None:
            return
        mesh_actor = self.viewport.add_mesh_actor(
            actor_name,
            mesh_polydata,
            color=color_hex,
            opacity=opacity,
            reset_camera=False
        )
        self._actor_names.append(actor_name)
        if mesh_actor is not None:
            try:
                mesh_actor.SetPickable(1)
                actor_property = mesh_actor.GetProperty()
                if actor_property is not None:
                    actor_property.BackfaceCullingOff()
            except (AttributeError, TypeError):
                pass
            self._mesh_actors[actor_name] = mesh_actor

    def remove_visuals(self) -> None:
        """Удаление всех активных акторов манипулятора из сцены вьюпорта."""
        self._reset_highlight()
        if self.viewport is not None:
            for name in self._actor_names:
                self.viewport.remove_actor(name)
        self._actor_names.clear()
        self._mesh_actors.clear()

    # ------------------------------------------------------------------------------------------------------------------
    # Интерактивное взаимодействие через VTK Interactor (Task 3.3)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_interactor(self) -> Optional[Any]:
        """Получение низкоуровневого vtkRenderWindowInteractor из вьюпорта."""
        if self.viewport is None:
            return None
        if isinstance(self.viewport, IViewport):
            return self.viewport.interactor
        return None

    def _setup_interactor(self) -> None:
        """Подключение обработчиков событий мыши и клавиатуры к VTK Interactor."""
        if self._observer_tags:
            return
        interactor = self._get_interactor()
        if interactor is None:
            return

        try:
            # Приоритет 10.0 для первоочередного перехвата клика по манипулятору до камеры
            self._observer_tags['LeftButtonPressEvent'] = interactor.AddObserver(
                'LeftButtonPressEvent', self._on_left_button_press, 10.0
            )
            self._observer_tags['MouseMoveEvent'] = interactor.AddObserver(
                'MouseMoveEvent', self._on_mouse_move, 10.0
            )
            self._observer_tags['LeftButtonReleaseEvent'] = interactor.AddObserver(
                'LeftButtonReleaseEvent', self._on_left_button_release, 10.0
            )
            self._observer_tags['KeyPressEvent'] = interactor.AddObserver(
                'KeyPressEvent', self._on_key_press, 10.0
            )
            self._observer_tags['EndInteractionEvent'] = interactor.AddObserver(
                'EndInteractionEvent', self._on_camera_view_changed, 0.0
            )
            self._observer_tags['MouseWheelForwardEvent'] = interactor.AddObserver(
                'MouseWheelForwardEvent', self._on_camera_view_changed, 0.0
            )
            self._observer_tags['MouseWheelBackwardEvent'] = interactor.AddObserver(
                'MouseWheelBackwardEvent', self._on_camera_view_changed, 0.0
            )
        except (AttributeError, TypeError) as setup_err:
            _logger.debug(f"Не удалось подключить обработчики VTK к interactor: {setup_err}")

    def _remove_interactor_observers(self) -> None:
        """Отключение обработчиков событий VTK."""
        if not self._observer_tags:
            return
        interactor = self._get_interactor()
        if interactor is not None:
            for tag in self._observer_tags.values():
                try:
                    interactor.RemoveObserver(tag)
                except (AttributeError, TypeError):
                    pass
        self._observer_tags.clear()

    def _abort_event(self, caller: Any, event_name: str) -> None:
        """
        Отмена дальнейшей обработки события наблюдателями VTK с более низким приоритетом
        (включая камеру vtkInteractorStyleTrackballCamera).
        """
        # 1. Прямая поддержка mock-объектов или команд с методом SetAbortFlag
        try:
            caller.SetAbortFlag(1)
            return
        except (AttributeError, TypeError):
            pass

        # 2. Нативный механизм VTK: получение vtkCommand по зарегистрированному тегу
        tag = self._observer_tags.get(event_name)
        if tag is not None:
            try:
                cmd = caller.GetCommand(tag)
                if cmd is not None:
                    cmd.SetAbortFlag(1)
            except Exception as err:
                _logger.debug(f"Не удалось выставить AbortFlag для события {event_name}: {err}")

    def _pick_gizmo_axis(self, event_x: int, event_y: int) -> Tuple[GizmoAxis, Optional[str]]:
        """
        Определение активной оси/плоскости манипулятора по экранным координатам курсора с помощью vtkPropPicker
        и fallback на vtkCellPicker с допуском для надежного захвата стрелок и колец.
        """
        if self.viewport is None or self.viewport.plotter is None or not self._mesh_actors:
            return GizmoAxis.NONE, None

        renderer = self.viewport.plotter.renderer
        if renderer is None:
            return GizmoAxis.NONE, None

        # 1. Быстрый точный пикинг через vtkPropPicker
        picker = vtk.vtkPropPicker()
        picker.PickFromListOn()
        for mesh_actor in self._mesh_actors.values():
            picker.AddPickList(mesh_actor)

        picker.Pick(float(event_x), float(event_y), 0.0, renderer)
        picked_actor = picker.GetActor()
        if picked_actor is not None:
            for actor_name, mesh_actor in self._mesh_actors.items():
                if mesh_actor is picked_actor or mesh_actor == picked_actor:
                    return self._actor_axis_map.get(actor_name, GizmoAxis.NONE), actor_name

        # 2. Fallback на vtkCellPicker с допуском (tolerance ~1.5% размера окна)
        try:
            cell_picker = vtk.vtkCellPicker()
            cell_picker.SetTolerance(0.015)
            cell_picker.PickFromListOn()
            for mesh_actor in self._mesh_actors.values():
                cell_picker.AddPickList(mesh_actor)

            cell_picker.Pick(float(event_x), float(event_y), 0.0, renderer)
            picked_actor = cell_picker.GetActor()
            if picked_actor is not None:
                for actor_name, mesh_actor in self._mesh_actors.items():
                    if mesh_actor is picked_actor or mesh_actor == picked_actor:
                        return self._actor_axis_map.get(actor_name, GizmoAxis.NONE), actor_name
        except Exception:
            pass

        return GizmoAxis.NONE, None

    def _on_left_button_press(self, interactor_obj: Any, event_name: str) -> None:
        """Обработка нажатия левой кнопки мыши во вьюпорте."""
        if self._target_node is None or self.viewport is None or self.viewport.plotter is None:
            return
        interactor = self._get_interactor()
        if interactor is None:
            return

        event_x, event_y = interactor.GetEventPosition()
        picked_axis, picked_name = self._pick_gizmo_axis(event_x, event_y)
        if picked_axis != GizmoAxis.NONE:
            self._active_axis = picked_axis
            self._is_dragging = True
            self._last_mouse_pos = (int(event_x), int(event_y))
            self._initial_drag_matrix = self._target_node.local_matrix.copy()
            self._accumulated_world_delta = np.zeros(3, dtype=np.float64)
            self._accumulated_angle_deg = 0.0
            self._accumulated_scale_factor = 1.0
            self._last_changed_data = None
            self.transform_started.emit()
            self._abort_event(interactor_obj, 'LeftButtonPressEvent')

    def _on_mouse_move(self, interactor_obj: Any, event_name: str) -> None:
        """Обработка перемещения мыши: трансформация при перетаскивании или ховер-подсветка."""
        if self.viewport is None or self.viewport.plotter is None:
            return
        interactor = self._get_interactor()
        if interactor is None:
            return

        event_x, event_y = interactor.GetEventPosition()

        if self._is_dragging and self._active_axis != GizmoAxis.NONE and self._last_mouse_pos is not None:
            self._abort_event(interactor_obj, 'MouseMoveEvent')
            last_x, last_y = self._last_mouse_pos
            shift_modifier = bool(interactor.GetShiftKey())

            self._process_drag(last_x, last_y, int(event_x), int(event_y), shift_modifier)
            self._last_mouse_pos = (int(event_x), int(event_y))
            self.viewport.render()
        else:
            self._process_hover(int(event_x), int(event_y))

    def _on_left_button_release(self, interactor_obj: Any, event_name: str) -> None:
        """Обработка отпускания левой кнопки мыши."""
        if self._is_dragging:
            if self._constraint is not None and self._last_changed_data is not None and self._target_node is not None:
                self._constraint.on_transform_committed(self._target_node, self._last_changed_data)
                self._last_changed_data = None
            self._is_dragging = False
            self._active_axis = GizmoAxis.NONE
            self._last_mouse_pos = None
            self._initial_drag_matrix = None
            self._accumulated_world_delta = np.zeros(3, dtype=np.float64)
            self._accumulated_angle_deg = 0.0
            self._accumulated_scale_factor = 1.0
            self.transform_ended.emit()
            self._abort_event(interactor_obj, 'LeftButtonReleaseEvent')
            self._reset_highlight()
            if self.viewport is not None:
                self.viewport.render()

    def _on_key_press(self, interactor_obj: Any, event_name: str) -> None:
        """Обработка нажатия клавиш переключения режимов (W/E/R/Q) во вьюпорте."""
        interactor = self._get_interactor()
        if interactor is None:
            return
        key_symbol = interactor.GetKeySym()
        if key_symbol is not None and self.handle_key_action(key_symbol):
            self._abort_event(interactor_obj, 'KeyPressEvent')
            self.update_visuals(render=True)

    def _on_camera_view_changed(self, interactor_obj: Any, event_name: str) -> None:
        """Обновление размера манипулятора при зуме или завершении движения камеры во вьюпорте."""
        if self._target_node is None or self._is_dragging or not self._adaptive_size:
            return
        gizmo_center = self._target_node.global_matrix[0:3, 3]
        new_size = self._compute_gizmo_size(gizmo_center)
        if self._last_built_size > 0.0:
            relative_difference = abs(new_size - self._last_built_size) / self._last_built_size
            if relative_difference < 0.05:
                return
        self.update_visuals(render=True)

    def _process_drag(
        self,
        last_x: int,
        last_y: int,
        cur_x: int,
        cur_y: int,
        shift_modifier: bool
    ) -> None:
        """Вычисление и применение пространственной дельты манипуляции."""
        if self._target_node is None or self.viewport is None or self.viewport.plotter is None:
            return
        if self._initial_drag_matrix is None:
            return

        if self._mode == GizmoMode.TRANSLATE:
            step_world_delta = self._compute_screen_to_world_delta(last_x, last_y, cur_x, cur_y)
            self._accumulated_world_delta += step_world_delta
            self._apply_drag_translation(self._accumulated_world_delta, shift_modifier)
        elif self._mode == GizmoMode.ROTATE:
            step_angle = self._compute_screen_rotation_angle(last_x, last_y, cur_x, cur_y)
            self._accumulated_angle_deg += step_angle
            self._apply_drag_rotation(self._accumulated_angle_deg, shift_modifier)
        elif self._mode == GizmoMode.SCALE:
            pixel_delta = float((cur_x - last_x) + (cur_y - last_y))
            multiplier = 1.0 + pixel_delta * 0.01
            self._accumulated_scale_factor = max(0.01, self._accumulated_scale_factor * multiplier)
            self._apply_drag_scale(self._accumulated_scale_factor, shift_modifier)

    def _compute_screen_rotation_angle(
        self,
        last_x: int,
        last_y: int,
        cur_x: int,
        cur_y: int
    ) -> float:
        """
        Вычисляет угловое приращение (в градусах) при круговом или касательном движении мыши
        вокруг проекции центра манипулятора на плоскость экрана.
        """
        if self._target_node is None or self.viewport is None or self.viewport.plotter is None:
            return float(cur_x - last_x) * 0.5

        renderer = self.viewport.plotter.renderer
        if renderer is None:
            return float(cur_x - last_x) * 0.5

        global_matrix = self._target_node.global_matrix
        gizmo_center = global_matrix[0:3, 3]

        renderer.SetWorldPoint(float(gizmo_center[0]), float(gizmo_center[1]), float(gizmo_center[2]), 1.0)
        renderer.WorldToDisplay()
        center_disp = renderer.GetDisplayPoint()
        cx, cy = center_disp[0], center_disp[1]

        v_prev_x = float(last_x) - cx
        v_prev_y = float(last_y) - cy
        v_cur_x = float(cur_x) - cx
        v_cur_y = float(cur_y) - cy

        norm_prev = math.hypot(v_prev_x, v_prev_y)
        norm_cur = math.hypot(v_cur_x, v_cur_y)

        if norm_prev < 5.0 or norm_cur < 5.0:
            return float((cur_x - last_x) - (cur_y - last_y)) * 0.5

        angle_prev = math.atan2(v_prev_y, v_prev_x)
        angle_cur = math.atan2(v_cur_y, v_cur_x)
        delta_rad = angle_cur - angle_prev

        while delta_rad > math.pi:
            delta_rad -= 2.0 * math.pi
        while delta_rad < -math.pi:
            delta_rad += 2.0 * math.pi

        delta_deg = math.degrees(delta_rad)

        camera = renderer.GetActiveCamera()
        if camera is not None:
            view_dir = np.array(camera.GetDirectionOfProjection(), dtype=np.float64)
            dir_x, dir_y, dir_z = self._extract_orthonormal_basis(global_matrix)
            if self._active_axis == GizmoAxis.X:
                axis_vec = dir_x if self._space == GizmoSpace.LOCAL else np.array([1.0, 0.0, 0.0], dtype=np.float64)
            elif self._active_axis == GizmoAxis.Y:
                axis_vec = dir_y if self._space == GizmoSpace.LOCAL else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                axis_vec = dir_z if self._space == GizmoSpace.LOCAL else np.array([0.0, 0.0, 1.0], dtype=np.float64)

            norm_axis = np.linalg.norm(axis_vec)
            if norm_axis > 0:
                axis_vec = axis_vec / norm_axis
                if np.dot(axis_vec, view_dir) > 0.0:
                    delta_deg = -delta_deg

        return float(delta_deg)

    def _compute_screen_to_world_delta(
        self,
        last_x: int,
        last_y: int,
        cur_x: int,
        cur_y: int
    ) -> np.ndarray:
        """Пересчет экранного смещения мыши в 3D мировое смещение через обратную проекцию луча камеры."""
        renderer = self.viewport.plotter.renderer
        global_matrix = self._target_node.global_matrix
        gizmo_center = global_matrix[0:3, 3]

        renderer.SetWorldPoint(float(gizmo_center[0]), float(gizmo_center[1]), float(gizmo_center[2]), 1.0)
        renderer.WorldToDisplay()
        center_display = renderer.GetDisplayPoint()
        display_depth = float(center_display[2])

        def _screen_point_to_world(screen_x: float, screen_y: float) -> np.ndarray:
            renderer.SetDisplayPoint(screen_x, screen_y, display_depth)
            renderer.DisplayToWorld()
            world_pt = renderer.GetWorldPoint()
            if world_pt[3] != 0.0:
                return np.array([world_pt[0], world_pt[1], world_pt[2]], dtype=np.float64) / world_pt[3]
            return np.array([world_pt[0], world_pt[1], world_pt[2]], dtype=np.float64)

        world_prev = _screen_point_to_world(float(last_x), float(last_y))
        world_cur = _screen_point_to_world(float(cur_x), float(cur_y))
        return world_cur - world_prev

    def _apply_drag_translation(self, total_world_delta: np.ndarray, shift_modifier: bool) -> None:
        """Проекция общего мирового смещения на активную ось/плоскость относительно исходной точки перетаскивания."""
        if self._initial_drag_matrix is None or self._target_node is None:
            return

        if self._constraint is not None:
            filtered_delta, changed_data = self._constraint.filter_translation(
                self._target_node,
                total_world_delta,
                self._initial_drag_matrix,
                active_axis=self._active_axis,
            )
            self._last_changed_data = changed_data
            if 'matrix' in changed_data:
                new_matrix = changed_data['matrix']
            else:
                new_matrix = self._initial_drag_matrix.copy()
                new_matrix[0:3, 3] = self._initial_drag_matrix[0:3, 3] + filtered_delta

            self._target_node.local_matrix = new_matrix
            self.transform_changed.emit(new_matrix)
            self._constraint.on_transform_changed(self._target_node, changed_data)
            if 'status_message' in changed_data:
                self.status_message_requested.emit(str(changed_data['status_message']))
            self.update_visuals()
            return

        initial_matrix = self._initial_drag_matrix.copy()
        start_pos = initial_matrix[0:3, 3].copy()

        if self._space == GizmoSpace.LOCAL:
            dir_x, dir_y, dir_z = self._extract_orthonormal_basis(self._target_node.global_matrix)
            axis_x = dir_x
            axis_y = dir_y
            axis_z = dir_z
        else:
            axis_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            axis_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        if self._active_axis == GizmoAxis.X:
            projected_delta = float(np.dot(total_world_delta, axis_x)) * axis_x
        elif self._active_axis == GizmoAxis.Y:
            projected_delta = float(np.dot(total_world_delta, axis_y)) * axis_y
        elif self._active_axis == GizmoAxis.Z:
            projected_delta = float(np.dot(total_world_delta, axis_z)) * axis_z
        elif self._active_axis == GizmoAxis.XY:
            projected_delta = total_world_delta - float(np.dot(total_world_delta, axis_z)) * axis_z
        elif self._active_axis == GizmoAxis.XZ:
            projected_delta = total_world_delta - float(np.dot(total_world_delta, axis_y)) * axis_y
        elif self._active_axis == GizmoAxis.YZ:
            projected_delta = total_world_delta - float(np.dot(total_world_delta, axis_x)) * axis_x
        else:
            projected_delta = total_world_delta

        # Преобразование мирового смещения в систему координат родительского узла
        if self._target_node.parent_vm is not None:
            parent_dir_x, parent_dir_y, parent_dir_z = self._extract_orthonormal_basis(
                self._target_node.parent_vm.global_matrix
            )
            parent_basis = np.column_stack([parent_dir_x, parent_dir_y, parent_dir_z])
            effective_delta = parent_basis.T @ projected_delta
        else:
            effective_delta = projected_delta

        target_pos = start_pos + effective_delta
        snapped_target_pos = self.snap_translation_vector(target_pos, shift_modifier=shift_modifier)

        new_matrix = initial_matrix.copy()
        new_matrix[0:3, 3] = snapped_target_pos
        self._target_node.local_matrix = new_matrix
        self.transform_changed.emit(new_matrix)
        self.update_visuals()

    def _apply_drag_rotation(self, total_angle_deg: float, shift_modifier: bool) -> None:
        """Применение суммарного поворота вокруг выбранной оси относительно начальной матрицы."""
        if self._initial_drag_matrix is None or self._target_node is None:
            return

        if self._constraint is not None:
            dir_x, dir_y, dir_z = self._extract_orthonormal_basis(self._target_node.global_matrix)
            if self._active_axis == GizmoAxis.X:
                axis_vec = dir_x if self._space == GizmoSpace.LOCAL else np.array([1.0, 0.0, 0.0], dtype=np.float64)
            elif self._active_axis == GizmoAxis.Y:
                axis_vec = dir_y if self._space == GizmoSpace.LOCAL else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                axis_vec = dir_z if self._space == GizmoSpace.LOCAL else np.array([0.0, 0.0, 1.0], dtype=np.float64)

            filtered_axis, filtered_angle, changed_data = self._constraint.filter_rotation(
                self._target_node,
                axis_vec,
                total_angle_deg,
                self._initial_drag_matrix,
            )
            self._last_changed_data = changed_data
            if 'matrix' in changed_data:
                new_matrix = changed_data['matrix']
            else:
                new_matrix = self._initial_drag_matrix.copy()

            self._target_node.local_matrix = new_matrix
            self.transform_changed.emit(new_matrix)
            self._constraint.on_transform_changed(self._target_node, changed_data)
            if 'status_message' in changed_data:
                self.status_message_requested.emit(str(changed_data['status_message']))
            self.update_visuals()
            return

        snapped_angle_deg = self.snap_angle(total_angle_deg, shift_modifier=shift_modifier)
        angle_rad = math.radians(snapped_angle_deg)

        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        if self._active_axis == GizmoAxis.X:
            rotation_mat_3x3 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, cos_val, -sin_val],
                [0.0, sin_val, cos_val]
            ], dtype=np.float64)
        elif self._active_axis == GizmoAxis.Y:
            rotation_mat_3x3 = np.array([
                [cos_val, 0.0, sin_val],
                [0.0, 1.0, 0.0],
                [-sin_val, 0.0, cos_val]
            ], dtype=np.float64)
        else:
            rotation_mat_3x3 = np.array([
                [cos_val, -sin_val, 0.0],
                [sin_val, cos_val, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

        initial_matrix = self._initial_drag_matrix.copy()
        current_center = initial_matrix[0:3, 3].copy()
        new_matrix = initial_matrix.copy()

        if self._space == GizmoSpace.LOCAL:
            new_matrix[0:3, 0:3] = initial_matrix[0:3, 0:3] @ rotation_mat_3x3
        else:
            if self._target_node.parent_vm is not None:
                parent_dir_x, parent_dir_y, parent_dir_z = self._extract_orthonormal_basis(
                    self._target_node.parent_vm.global_matrix
                )
                parent_basis = np.column_stack([parent_dir_x, parent_dir_y, parent_dir_z])
                local_rot = parent_basis.T @ rotation_mat_3x3 @ parent_basis
                new_matrix[0:3, 0:3] = local_rot @ initial_matrix[0:3, 0:3]
            else:
                new_matrix[0:3, 0:3] = rotation_mat_3x3 @ initial_matrix[0:3, 0:3]

        new_matrix[0:3, 3] = current_center
        self._target_node.local_matrix = new_matrix
        self.transform_changed.emit(new_matrix)
        self.update_visuals()

    def _apply_drag_scale(self, total_scale_factor: float, shift_modifier: bool) -> None:
        """Применение суммарного масштабирования относительно начальной матрицы."""
        if self._initial_drag_matrix is None or self._target_node is None:
            return

        snapped_scale = self.snap_scale_factor(total_scale_factor, shift_modifier=shift_modifier)

        if self._active_axis == GizmoAxis.XYZ:
            multipliers = np.array([snapped_scale, snapped_scale, snapped_scale], dtype=np.float64)
        elif self._active_axis == GizmoAxis.X:
            multipliers = np.array([snapped_scale, 1.0, 1.0], dtype=np.float64)
        elif self._active_axis == GizmoAxis.Y:
            multipliers = np.array([1.0, snapped_scale, 1.0], dtype=np.float64)
        elif self._active_axis == GizmoAxis.Z:
            multipliers = np.array([1.0, 1.0, snapped_scale], dtype=np.float64)
        else:
            multipliers = np.array([snapped_scale, snapped_scale, snapped_scale], dtype=np.float64)

        initial_matrix = self._initial_drag_matrix.copy()
        new_matrix = initial_matrix.copy()

        for col_idx in range(3):
            new_matrix[0:3, col_idx] = initial_matrix[0:3, col_idx] * multipliers[col_idx]

        self._target_node.local_matrix = new_matrix
        self.transform_changed.emit(new_matrix)
        self.update_visuals()

    def _process_hover(self, event_x: int, event_y: int) -> None:
        """Подсветка активной оси манипулятора при наведении курсора мыши."""
        picked_axis, picked_name = self._pick_gizmo_axis(event_x, event_y)
        if picked_name != self._hovered_actor_name:
            self._reset_highlight()
            if picked_name is not None and picked_name in self._mesh_actors:
                self._hovered_actor_name = picked_name
                actor = self._mesh_actors[picked_name]
                actor.GetProperty().SetColor(1.0, 1.0, 0.0)
                if self.viewport is not None:
                    self.viewport.render()

    def _reset_highlight(self) -> None:
        """Сброс подсветки манипулятора к исходным цветам осей."""
        if self._hovered_actor_name is not None and self._hovered_actor_name in self._mesh_actors:
            original_axis = self._actor_axis_map.get(self._hovered_actor_name, GizmoAxis.NONE)
            if original_axis in self._colors:
                color_hex = self._colors[original_axis]
                rgb = pv.Color(color_hex).float_rgb
                self._mesh_actors[self._hovered_actor_name].GetProperty().SetColor(rgb[0], rgb[1], rgb[2])
            self._hovered_actor_name = None




__all__ = [
    'IViewport',
    'GizmoMode',
    'GizmoSpace',
    'GizmoAxis',
    'TransformGizmo',
]

