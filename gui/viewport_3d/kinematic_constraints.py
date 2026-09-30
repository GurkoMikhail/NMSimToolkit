import logging
import math
import weakref
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

import numpy as np

from core.scene.gamma_camera_node import GammaCameraNode
from core.geometry.spect_kinematics import compute_orbit_matrix
from core.scene.gantry_node import GantryNode
from gui.viewport_3d.gizmo_types import GizmoAxis, GizmoMode, GizmoSpace

_logger = logging.getLogger(__name__)



@runtime_checkable
class IRotatableProcedure(Protocol):
    """
    Протокол процедуры исследования с поддержкой начального угла поворота ротора.
    """
    start_angle: float


@runtime_checkable
class IKinematicConstraint(Protocol):
    """
    Протокол кинематических ограничений для 3D-манипулятора TransformGizmo.
    """

    def filter_translation(
        self,
        target_node: Any,
        proposed_world_delta: np.ndarray,
        initial_matrix: np.ndarray,
        active_axis: Optional[GizmoAxis] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Проецирует произвольное 3D-смещение на разрешенные процедурой степени свободы.
        Возвращает:
          - отфильтрованное мировое смещение (np.ndarray shape (3,))
          - словарь параметров для строки состояния и процедуры (например, {'radius': 280.0, 'angle': 45.0})
        """
        ...

    def filter_rotation(
        self,
        target_node: Any,
        axis: np.ndarray,
        proposed_angle_deg: float,
        initial_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Ограничивает вращение только разрешенными осями.
        """
        ...

    def is_scale_allowed(self) -> bool:
        """Разрешено ли масштабирование целевого узла."""
        ...

    def get_forced_space(self) -> Optional[GizmoSpace]:
        """
        Принудительная система координат (например, GizmoSpace.LOCAL для ОФЭКТ)
        или None, если свободное переключение World/Local разрешено.
        """
        ...

    def get_allowed_axes(self, mode: GizmoMode) -> Set[GizmoAxis]:
        """Возвращает набор визуально активных осей и колец для заданного режима."""
        ...

    def on_transform_changed(self, target_node: Any, changed_data: Dict[str, Any]) -> None:
        """
        Непрерывное инкрементальное обновление на каждый тик мыши при перетаскивании:
        динамическое масштабирование визуалов орбиты и синхронный предпросмотр положений головок.
        """
        ...

    def on_transform_committed(self, target_node: Any, commit_data: Dict[str, Any]) -> None:
        """Фиксация параметров в процедуре при отпускании кнопки мыши (LMB release)."""
        ...



class FixedSubcomponentKinematicConstraint:
    """
    Кинематическое ограничение для жестко зафиксированных внутренних компонентов (дочерних узлов).
    Блокирует любые перемещения, вращения и масштабирование, делая компонент полностью неподвижным.
    """

    def filter_translation(
        self,
        target_node: Any,
        proposed_world_delta: np.ndarray,
        initial_matrix: np.ndarray,
        active_axis: Optional[GizmoAxis] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return (np.zeros(3, dtype=np.float64), {'status_message': 'Геометрия компонента зафиксирована'})

    def filter_rotation(
        self,
        target_node: Any,
        axis: np.ndarray,
        proposed_angle_deg: float,
        initial_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        return (axis, 0.0, {'status_message': 'Геометрия компонента зафиксирована'})

    def is_translation_allowed(self) -> bool:
        return False

    def is_rotation_allowed(self) -> bool:
        return False

    def is_scale_allowed(self) -> bool:
        return False

    def get_forced_space(self) -> Optional[GizmoSpace]:
        return None

    def get_allowed_axes(self, mode: GizmoMode) -> Set[GizmoAxis]:
        return set()

    def on_transform_changed(self, target_node: Any, changed_data: Dict[str, Any]) -> None:
        pass

    def on_transform_committed(self, target_node: Any, commit_data: Dict[str, Any]) -> None:
        pass



class SpectOrbitKinematicConstraint:
    """
    Кинематическое ограничение для гамма-камер в протоколе ОФЭКТ (круговая орбита гантри).
    Принудительно фиксирует локальную систему координат детектора (GizmoSpace.LOCAL),
    блокирует масштабирование и паразитные наклоны/крены (Pitch/Tilt).
    """

    def __init__(
        self,
        procedure_vm: Optional[Any] = None,
        camera_vm: Optional[Any] = None,
        spect_manipulator: Optional[Any] = None,
    ) -> None:
        self._procedure_ref = weakref.ref(procedure_vm) if procedure_vm is not None else None
        self._camera_vm: Optional[Any] = camera_vm
        self.spect_manipulator: Optional[Any] = spect_manipulator
        self._min_radius: float = 50.0
        self._max_radius: float = 600.0

    @property
    def procedure_vm(self) -> Optional[Any]:
        """Возвращает процедуру исследования или None."""
        return self._procedure_ref() if self._procedure_ref is not None else None

    @procedure_vm.setter
    def procedure_vm(self, procedure: Optional[Any]) -> None:
        """Устанавливает процедуру исследования."""
        self._procedure_ref = weakref.ref(procedure) if procedure is not None else None

    @property
    def camera_vm(self) -> Optional[Any]:
        """Возвращает связанный узел гамма-камеры."""
        return self._camera_vm

    @camera_vm.setter
    def camera_vm(self, camera: Optional[Any]) -> None:
        """Устанавливает связанный узел гамма-камеры."""
        self._camera_vm = camera

    def is_scale_allowed(self) -> bool:
        """Масштабирование гамма-камер физически невозможно и заблокировано."""
        return False

    def get_forced_space(self) -> Optional[GizmoSpace]:
        """Принудительная локальная система координат (LOCAL). Переключение Q блокируется."""
        return GizmoSpace.LOCAL

    def get_allowed_axes(self, mode: GizmoMode) -> Set[GizmoAxis]:
        """
        Возвращает допустимые оси для режима:
        - TRANSLATE: 3 осевые стрелки (X: орбитальный угол, Y: осевая Z, Z: радиус орбиты).
          Плоскости XY/XZ/YZ скрыты.
        - ROTATE: только кольцо Z (вращение детектора в собственной плоскости: альбом/портрет).
        - SCALE: пустой набор.
        """
        if mode == GizmoMode.TRANSLATE:
            return {GizmoAxis.X, GizmoAxis.Y, GizmoAxis.Z}
        elif mode == GizmoMode.ROTATE:
            return {GizmoAxis.Z}
        return set()

    def filter_translation(
        self,
        target_node: Any,
        proposed_world_delta: np.ndarray,
        initial_matrix: np.ndarray,
        active_axis: Optional[GizmoAxis] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Проецирует перемещение на кинематические параметры орбиты ОФЭКТ:
        - Стрелка Z (перпендикуляр к детектору) -> изменение радиуса R.
        - Стрелка X (боковая, вдоль детектора) -> поворот гантри на угол theta.
        - Стрелка Y (вдоль стола) -> смещение по высоте среза Z.
        """
        camera_half_thickness = (
            target_node.half_thickness
            if isinstance(target_node.core_node, GammaCameraNode)
            else (self._camera_vm.half_thickness if self._camera_vm is not None else 0.0)
        )


        pos_x = float(initial_matrix[0, 3])
        pos_y = float(initial_matrix[1, 3])
        pos_z = float(initial_matrix[2, 3])
        center_radius = float(np.hypot(pos_x, pos_y))
        initial_radius = max(self._min_radius, center_radius - camera_half_thickness)
        initial_angle = float(np.degrees(np.arctan2(pos_y, pos_x)) % 360.0)
        initial_z = pos_z

        rad = math.radians(initial_angle)
        radial_unit = np.array([math.cos(rad), math.sin(rad), 0.0], dtype=np.float64)
        tangent_unit = np.array([-math.sin(rad), math.cos(rad), 0.0], dtype=np.float64)
        axial_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        action: str
        if active_axis == GizmoAxis.Z:
            action = 'radial'
        elif active_axis == GizmoAxis.X:
            axis_x_dir = initial_matrix[0:3, 0]
            if abs(float(np.dot(axis_x_dir, tangent_unit))) >= abs(float(np.dot(axis_x_dir, axial_unit))):
                action = 'tangential'
            else:
                action = 'axial'
        elif active_axis == GizmoAxis.Y:
            axis_y_dir = initial_matrix[0:3, 1]
            if abs(float(np.dot(axis_y_dir, axial_unit))) >= abs(float(np.dot(axis_y_dir, tangent_unit))):
                action = 'axial'
            else:
                action = 'tangential'
        else:
            dot_z = abs(float(np.dot(proposed_world_delta, radial_unit)))
            dot_t = abs(float(np.dot(proposed_world_delta, tangent_unit)))
            dot_y = abs(float(np.dot(proposed_world_delta, axial_unit)))
            if dot_z >= dot_t and dot_z >= dot_y:
                action = 'radial'
            elif dot_t >= dot_z and dot_t >= dot_y:
                action = 'tangential'
            else:
                action = 'axial'

        if action == 'radial':
            delta_r = float(np.dot(proposed_world_delta, radial_unit))
            new_radius_raw = initial_radius + delta_r
            snapped_radius = round(new_radius_raw / 10.0) * 10.0
            new_radius = float(np.clip(snapped_radius, self._min_radius, self._max_radius))
            new_angle = initial_angle
            new_z = initial_z
            status_message = f"ОФЭКТ: Радиус орбиты R = {new_radius:.1f} мм (Сетка: 10 мм)"
        elif action == 'tangential':
            delta_tangent = float(np.dot(proposed_world_delta, tangent_unit))
            effective_radius = initial_radius + camera_half_thickness
            delta_angle_deg = math.degrees(delta_tangent / effective_radius) if effective_radius > 1e-3 else 0.0
            new_angle_raw = initial_angle + delta_angle_deg
            snapped_angle = round(new_angle_raw / 5.0) * 5.0
            new_angle = float(snapped_angle % 360.0)
            new_radius = initial_radius
            new_z = initial_z
            status_message = f"ОФЭКТ: Угол штатива θ = {new_angle:.1f}° (Угол: 5°)"
        else:  # axial
            delta_z = float(np.dot(proposed_world_delta, axial_unit))
            new_z_raw = initial_z + delta_z
            snapped_z = round(new_z_raw / 10.0) * 10.0
            new_z = float(snapped_z)
            new_radius = initial_radius
            new_angle = initial_angle
            status_message = f"ОФЭКТ: Смещение Z = {new_z:.1f} мм (Сетка: 10 мм)"

        new_matrix = compute_orbit_matrix(
            radius=new_radius,
            angle_deg=new_angle,
            z=new_z,
            half_thickness=camera_half_thickness,
        )

        initial_dir_z = initial_matrix[0:3, 2]
        initial_dir_y = initial_matrix[0:3, 1]
        cos_roll = float(np.clip(np.dot(initial_dir_y, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        sin_roll = float(np.dot(np.cross(np.array([0.0, 0.0, 1.0]), initial_dir_y), initial_dir_z))
        roll_angle_deg = math.degrees(math.atan2(sin_roll, cos_roll))
        if abs(roll_angle_deg) > 1e-3:
            roll_rad = math.radians(roll_angle_deg)
            roll_mat = np.array([
                [math.cos(roll_rad), -math.sin(roll_rad), 0.0],
                [math.sin(roll_rad), math.cos(roll_rad), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            new_matrix[0:3, 0:3] = new_matrix[0:3, 0:3] @ roll_mat

        filtered_world_delta = new_matrix[0:3, 3] - initial_matrix[0:3, 3]
        changed_data = {
            'radius': new_radius,
            'angle': new_angle,
            'z': new_z,
            'matrix': new_matrix,
            'status_message': status_message,
            'axis': active_axis,
        }
        return (filtered_world_delta, changed_data)

    def filter_rotation(
        self,
        target_node: Any,
        axis: np.ndarray,
        proposed_angle_deg: float,
        initial_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Ограничивает вращение только осью, перпендикулярной детектору (+Z).
        Наклоны и крены (Pitch/Tilt) блокируются.
        Снаппинг с шагом 90 градусов (Альбомная / Книжная ориентация).
        """
        dir_z_global = (
            target_node.global_matrix[0:3, 2]
            if target_node is not None and target_node.global_matrix is not None
            else initial_matrix[0:3, 2]
        )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            return (dir_z_global, 0.0, {})

        unit_axis = axis / axis_norm
        dot_product = float(np.dot(unit_axis, dir_z_global))

        if abs(dot_product) < 0.7:
            return (dir_z_global, 0.0, {})

        sign = 1.0 if dot_product > 0.0 else -1.0
        effective_angle = proposed_angle_deg * sign
        snapped_angle_deg = float(round(effective_angle / 90.0) * 90.0)

        angle_rad = math.radians(snapped_angle_deg)
        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        rotation_mat_3x3 = np.array([
            [cos_val, -sin_val, 0.0],
            [sin_val, cos_val, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        new_matrix = initial_matrix.copy()
        new_matrix[0:3, 0:3] = initial_matrix[0:3, 0:3] @ rotation_mat_3x3

        is_portrait = int(round(snapped_angle_deg / 90.0)) % 2 != 0
        orientation_text = "Книжная (Portrait)" if is_portrait else "Альбомная (Landscape)"
        status_message = f"ОФЭКТ: Ориентация детектора = {orientation_text} ({snapped_angle_deg:.0f}°)"

        changed_data = {
            'matrix': new_matrix,
            'angle_deg': snapped_angle_deg,
            'orientation': orientation_text,
            'status_message': status_message,
        }
        return (dir_z_global, snapped_angle_deg, changed_data)

    def on_transform_changed(self, target_node: Any, changed_data: Dict[str, Any]) -> None:
        """
        Непрерывное обновление при перетаскивании:
        - Динамическое масштабирование направляющей SPECTManipulator
        - Синхронное обновление положений спаренных головок детекторов в сцене
        """
        new_radius = changed_data.get('radius')
        new_angle = changed_data.get('angle')
        new_z = changed_data.get('z')

        if self.spect_manipulator is not None and new_radius is not None:
            camera_node = target_node if target_node is not None else self._camera_vm
            default_angle = 0.0
            default_z = 0.0
            if camera_node is not None and camera_node.local_matrix is not None:
                matrix_pos = camera_node.local_matrix[0:3, 3]
                default_angle = float(np.degrees(np.arctan2(matrix_pos[1], matrix_pos[0])) % 360.0)
                default_z = float(matrix_pos[2])

            self.spect_manipulator.set_orbit_parameters(
                radius=float(new_radius),
                angle_deg=float(new_angle if new_angle is not None else default_angle),
                z=float(new_z if new_z is not None else default_z),
                render=False,
                emit_signal=False,
            )

        if target_node.parent_vm is not None:
            sibling_cameras = [
                child for child in target_node.parent_vm.children
                if isinstance(child.core_node, GammaCameraNode)
            ]
            if len(sibling_cameras) > 1 and target_node in sibling_cameras:
                procedure = self._procedure_ref() if self._procedure_ref is not None else None
                node_idx = sibling_cameras.index(target_node)

                if new_radius is not None:
                    for sibling in sibling_cameras:
                        if sibling is not target_node:
                            sibling_position = sibling.local_matrix[0:3, 3]
                            sibling_angle = float(np.degrees(np.arctan2(sibling_position[1], sibling_position[0])) % 360.0)
                            sibling_dir_z = sibling.local_matrix[0:3, 2]
                            sibling_dir_y = sibling.local_matrix[0:3, 1]
                            cos_roll = float(np.clip(np.dot(sibling_dir_y, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
                            sin_roll = float(np.dot(np.cross(np.array([0.0, 0.0, 1.0]), sibling_dir_y), sibling_dir_z))
                            sibling_roll_deg = float(np.degrees(np.arctan2(sin_roll, cos_roll)))
                            sibling.local_matrix = compute_orbit_matrix(
                                radius=float(new_radius),
                                angle_deg=sibling_angle,
                                z=float(sibling_position[2]),
                                half_thickness=sibling.half_thickness,
                                roll_deg=sibling_roll_deg,
                            )

                if new_angle is not None and procedure is not None:
                    head_angles = procedure.head_angles
                    offset = head_angles[node_idx] if node_idx < len(head_angles) else (360.0 / len(sibling_cameras)) * node_idx
                    base_start_angle = (float(new_angle) - offset) % 360.0
                    for sibling_idx, sibling in enumerate(sibling_cameras):
                        if sibling is not target_node:
                            sib_offset = head_angles[sibling_idx] if sibling_idx < len(head_angles) else (360.0 / len(sibling_cameras)) * sibling_idx
                            sibling_position = sibling.local_matrix[0:3, 3]
                            sibling_radius = max(self._min_radius, float(np.hypot(sibling_position[0], sibling_position[1])) - sibling.half_thickness)
                            sibling_dir_z = sibling.local_matrix[0:3, 2]
                            sibling_dir_y = sibling.local_matrix[0:3, 1]
                            cos_roll = float(np.clip(np.dot(sibling_dir_y, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
                            sin_roll = float(np.dot(np.cross(np.array([0.0, 0.0, 1.0]), sibling_dir_y), sibling_dir_z))
                            sibling_roll_deg = float(np.degrees(np.arctan2(sin_roll, cos_roll)))
                            sibling.local_matrix = compute_orbit_matrix(
                                radius=sibling_radius,
                                angle_deg=(base_start_angle + sib_offset) % 360.0,
                                z=float(sibling_position[2]),
                                half_thickness=sibling.half_thickness,
                                roll_deg=sibling_roll_deg,
                            )

    def on_transform_committed(self, target_node: Any, commit_data: Dict[str, Any]) -> None:
        """
        Фиксация параметров в процедуре при отпускании кнопки мыши:
        - Запись нового радиуса в procedure_vm.radius
        - Запись нового угла в procedure_vm.start_angle
        - Финализация всех головок через procedure_vm.sync_cameras
        """
        procedure = self._procedure_ref() if self._procedure_ref is not None else None
        if procedure is None:
            return

        camera_nodes = [
            child for child in target_node.parent_vm.children
            if isinstance(child.core_node, GammaCameraNode)
        ] if target_node.parent_vm is not None else [target_node]


        if 'radius' in commit_data:
            procedure.radius = float(commit_data['radius'])

        if 'angle' in commit_data and target_node in camera_nodes:
            node_idx = camera_nodes.index(target_node)
            head_angles = procedure.head_angles
            offset = head_angles[node_idx] if node_idx < len(head_angles) else (360.0 / max(1, len(camera_nodes))) * node_idx
            base_start_angle = (float(commit_data['angle']) - offset) % 360.0
            procedure.start_angle = base_start_angle

        if 'radius' in commit_data or 'angle' in commit_data:
            procedure.sync_cameras(camera_nodes)


class CameraMountKinematicConstraint(SpectOrbitKinematicConstraint):
    """
    Кинематическое ограничение для каретки гамма-камеры на направляющих ротора гантри.
    Степени свободы:
    - Радиальный вылет R (стрелка нормали детектора Z);
    - Угол монтажа phi на направляющих гантри;
    - Поворот в собственной плоскости (Landscape / Portrait).
    Тангенциальный перехват: при попытке оператора потянуть камеру по касательной к орбите
    констрейнт транслирует перемещение во вращение родительского узла GantryViewModel
    ("потянуть за ручку аппарата"), сохраняя взаимную ориентацию детекторов на роторе.
    """

    def __init__(
        self,
        procedure_vm: Optional[Any] = None,
        camera_vm: Optional[Any] = None,
        spect_manipulator: Optional[Any] = None,
        gantry_vm: Optional[Any] = None,
    ) -> None:
        super().__init__(
            procedure_vm=procedure_vm,
            camera_vm=camera_vm,
            spect_manipulator=spect_manipulator,
        )
        self._gantry_vm: Optional[Any] = gantry_vm

    @property
    def gantry_vm(self) -> Optional[Any]:
        """Возвращает связанный узел станины томографа."""
        return self._gantry_vm

    @gantry_vm.setter
    def gantry_vm(self, gantry: Optional[Any]) -> None:
        """Устанавливает связанный узел станины томографа."""
        self._gantry_vm = gantry

    def filter_translation(
        self,
        target_node: Any,
        proposed_world_delta: np.ndarray,
        initial_matrix: np.ndarray,
        active_axis: Optional[GizmoAxis] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node.parent_vm
        is_mounted_on_gantry = (
            gantry_node_vm is not None
            and isinstance(gantry_node_vm.core_node, GantryNode)
        )
        if not is_mounted_on_gantry:
            return super().filter_translation(target_node, proposed_world_delta, initial_matrix, active_axis)

        camera_half_thickness = (
            target_node.half_thickness
            if isinstance(target_node.core_node, GammaCameraNode)
            else (self._camera_vm.half_thickness if self._camera_vm is not None else 0.0)
        )

        # Преобразование мирового вектора перемещения в систему координат станины
        gantry_rotation = gantry_node_vm.global_matrix[0:3, 0:3]
        delta_gantry = gantry_rotation.T @ proposed_world_delta

        pos_x = float(initial_matrix[0, 3])
        pos_y = float(initial_matrix[1, 3])
        pos_z = float(initial_matrix[2, 3])
        center_radius = float(np.hypot(pos_x, pos_y))
        initial_radius = max(self._min_radius, center_radius - camera_half_thickness)
        initial_angle = float(np.degrees(np.arctan2(pos_y, pos_x)) % 360.0)
        initial_z = pos_z

        rad = math.radians(initial_angle)
        radial_unit = np.array([math.cos(rad), math.sin(rad), 0.0], dtype=np.float64)
        tangent_unit = np.array([-math.sin(rad), math.cos(rad), 0.0], dtype=np.float64)
        axial_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        action: str
        if active_axis == GizmoAxis.Z:
            action = 'radial'
        elif active_axis == GizmoAxis.X:
            axis_x_dir = initial_matrix[0:3, 0]
            if abs(float(np.dot(axis_x_dir, tangent_unit))) >= abs(float(np.dot(axis_x_dir, axial_unit))):
                action = 'tangential'
            else:
                action = 'axial'
        elif active_axis == GizmoAxis.Y:
            axis_y_dir = initial_matrix[0:3, 1]
            if abs(float(np.dot(axis_y_dir, axial_unit))) >= abs(float(np.dot(axis_y_dir, tangent_unit))):
                action = 'axial'
            else:
                action = 'tangential'
        else:
            dot_z = abs(float(np.dot(delta_gantry, radial_unit)))
            dot_t = abs(float(np.dot(delta_gantry, tangent_unit)))
            dot_y = abs(float(np.dot(delta_gantry, axial_unit)))
            if dot_z >= dot_t and dot_z >= dot_y:
                action = 'radial'
            elif dot_t >= dot_z and dot_t >= dot_y:
                action = 'tangential'
            else:
                action = 'axial'

        if action == 'radial':
            delta_radius = float(np.dot(delta_gantry, radial_unit))
            new_radius_raw = initial_radius + delta_radius
            snapped_radius = round(new_radius_raw / 10.0) * 10.0
            new_radius = float(np.clip(snapped_radius, self._min_radius, self._max_radius))
            new_angle = initial_angle
            new_z = initial_z
            status_message = f"ОФЭКТ (Гантри): Радиус орбиты R = {new_radius:.1f} мм (Сетка: 10 мм)"
            gantry_angle = gantry_node_vm.gantry_angle_deg
        elif action == 'tangential':
            delta_tangent = float(np.dot(delta_gantry, tangent_unit))
            effective_radius = initial_radius + camera_half_thickness
            delta_angle_deg = math.degrees(delta_tangent / effective_radius) if effective_radius > 1e-3 else 0.0
            current_gantry_angle = gantry_node_vm.gantry_angle_deg
            new_gantry_raw = current_gantry_angle + delta_angle_deg
            snapped_gantry = round(new_gantry_raw / 5.0) * 5.0
            gantry_angle = float(snapped_gantry % 360.0)
            new_radius = initial_radius
            new_angle = initial_angle
            new_z = initial_z
            status_message = f"ОФЭКТ (Гантри): Угол ротора θ = {gantry_angle:.1f}° (Сетка: 5°)"
        else:  # axial
            delta_z = float(np.dot(delta_gantry, axial_unit))
            new_z_raw = initial_z + delta_z
            snapped_z = round(new_z_raw / 10.0) * 10.0
            new_z = float(snapped_z)
            new_radius = initial_radius
            new_angle = initial_angle
            gantry_angle = gantry_node_vm.gantry_angle_deg
            status_message = f"ОФЭКТ (Гантри): Смещение Z = {new_z:.1f} мм (Сетка: 10 мм)"

        new_matrix = compute_orbit_matrix(
            radius=new_radius,
            angle_deg=new_angle,
            z=new_z,
            half_thickness=camera_half_thickness,
        )

        initial_dir_z = initial_matrix[0:3, 2]
        initial_dir_y = initial_matrix[0:3, 1]
        cos_roll = float(np.clip(np.dot(initial_dir_y, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        sin_roll = float(np.dot(np.cross(np.array([0.0, 0.0, 1.0]), initial_dir_y), initial_dir_z))
        roll_angle_deg = math.degrees(math.atan2(sin_roll, cos_roll))
        if abs(roll_angle_deg) > 1e-3:
            roll_rad = math.radians(roll_angle_deg)
            roll_mat = np.array([
                [math.cos(roll_rad), -math.sin(roll_rad), 0.0],
                [math.sin(roll_rad), math.cos(roll_rad), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            new_matrix[0:3, 0:3] = new_matrix[0:3, 0:3] @ roll_mat

        filtered_local_delta = new_matrix[0:3, 3] - initial_matrix[0:3, 3]
        filtered_world_delta = gantry_rotation @ filtered_local_delta
        changed_data = {
            'radius': new_radius,
            'angle': gantry_angle,
            'gantry_angle': gantry_angle,
            'z': new_z,
            'matrix': new_matrix,
            'status_message': status_message,
            'axis': active_axis,
            'action': action,
        }
        return (filtered_world_delta, changed_data)

    def on_transform_changed(self, target_node: Any, changed_data: Dict[str, Any]) -> None:
        gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node.parent_vm
        if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
            procedure = self._procedure_ref() if self._procedure_ref is not None else None

            # 1. Поворот станины при тангенциальном перемещении
            if changed_data.get('action') == 'tangential':
                new_gantry_angle = changed_data.get('gantry_angle', changed_data.get('angle'))
                if new_gantry_angle is not None:
                    gantry_node_vm.gantry_angle_deg = float(new_gantry_angle)
                    if isinstance(procedure, IRotatableProcedure):
                        procedure.start_angle = float(new_gantry_angle)

            # 2. Синхронизация радиуса со всеми спаренными головками
            if changed_data.get('action') == 'radial' and 'radius' in changed_data:
                new_radius = changed_data.get('radius')
                if new_radius is not None:
                    for sibling in gantry_node_vm.children:
                        if isinstance(sibling.core_node, GammaCameraNode) and sibling is not target_node:
                            sibling_pos = sibling.local_matrix[0:3, 3]
                            sibling_angle = float(np.degrees(np.arctan2(sibling_pos[1], sibling_pos[0])) % 360.0)
                            sibling.local_matrix = compute_orbit_matrix(
                                radius=float(new_radius),
                                angle_deg=sibling_angle,
                                z=float(sibling_pos[2]),
                                half_thickness=sibling.half_thickness,
                            )
                    if procedure is not None:
                        procedure.radius = float(new_radius)

            if self.spect_manipulator is not None:
                current_radius = float(changed_data['radius']) if 'radius' in changed_data else (
                    float(np.hypot(target_node.local_matrix[0, 3], target_node.local_matrix[1, 3])) - target_node.half_thickness
                )
                current_gantry_angle = gantry_node_vm.gantry_angle_deg
                current_z = float(changed_data.get('z', target_node.local_matrix[2, 3]))
                self.spect_manipulator.set_orbit_parameters(
                    radius=current_radius,
                    angle_deg=current_gantry_angle,
                    z=current_z,
                    render=False,
                    emit_signal=False,
                )
        else:
            super().on_transform_changed(target_node, changed_data)

    def on_transform_committed(self, target_node: Any, commit_data: Dict[str, Any]) -> None:
        gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node.parent_vm
        if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
            procedure = self._procedure_ref() if self._procedure_ref is not None else None
            if procedure is not None:
                if commit_data.get('action') == 'radial' and 'radius' in commit_data:
                    procedure.radius = float(commit_data['radius'])
                if commit_data.get('action') == 'tangential':
                    new_gantry_angle = commit_data.get('gantry_angle', commit_data.get('angle'))
                    if new_gantry_angle is not None and isinstance(procedure, IRotatableProcedure):
                        procedure.start_angle = float(new_gantry_angle)
                if commit_data.get('action') in ('radial', 'tangential'):
                    camera_nodes = [
                        child for child in gantry_node_vm.children
                        if isinstance(child.core_node, GammaCameraNode)
                    ]
                    procedure.sync_cameras(camera_nodes)
        else:
            super().on_transform_committed(target_node, commit_data)



class GantryKinematicConstraint:
    """
    Кинематическое ограничение для станины томографа GantryViewModel (1-DOF вращение вокруг оси Z).
    Блокирует линейные перемещения и масштабирование, разрешая только вращение ротора вокруг оси Z в изоцентре.
    """

    def __init__(
        self,
        procedure_vm: Optional[Any] = None,
        gantry_vm: Optional[Any] = None,
    ) -> None:
        self._procedure_ref = weakref.ref(procedure_vm) if procedure_vm is not None else None
        self._gantry_vm = gantry_vm

    @property
    def procedure_vm(self) -> Optional[Any]:
        """Возвращает процедуру исследования или None."""
        return self._procedure_ref() if self._procedure_ref is not None else None

    @procedure_vm.setter
    def procedure_vm(self, procedure: Optional[Any]) -> None:
        """Устанавливает процедуру исследования."""
        self._procedure_ref = weakref.ref(procedure) if procedure is not None else None

    @property
    def gantry_vm(self) -> Optional[Any]:
        """Возвращает связанный узел станины томографа."""
        return self._gantry_vm

    @gantry_vm.setter
    def gantry_vm(self, gantry: Optional[Any]) -> None:
        """Устанавливает связанный узел станины томографа."""
        self._gantry_vm = gantry

    def is_scale_allowed(self) -> bool:
        """Масштабирование станины заблокировано."""
        return False

    def get_forced_space(self) -> Optional[GizmoSpace]:
        """Вращение ротора выполняется вокруг фиксированной мировой оси Z."""
        return GizmoSpace.WORLD

    def get_allowed_axes(self, mode: GizmoMode) -> Set[GizmoAxis]:
        """
        - TRANSLATE: пусто (перемещения станины заблокированы)
        - ROTATE: строго кольцо Z
        - SCALE: пусто
        """
        if mode == GizmoMode.ROTATE:
            return {GizmoAxis.Z}
        return set()

    def filter_translation(
        self,
        target_node: Any,
        proposed_world_delta: np.ndarray,
        initial_matrix: np.ndarray,
        active_axis: Optional[GizmoAxis] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Линейные перемещения станины заблокированы."""
        return (np.zeros(3, dtype=np.float64), {'status_message': 'Перемещение станины заблокировано'})

    def filter_rotation(
        self,
        target_node: Any,
        axis: np.ndarray,
        proposed_angle_deg: float,
        initial_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Разрешает вращение только вокруг оси Z.
        Снаппинг угла с шагом 5 градусов.
        """
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        snapped_angle_deg = float(round(proposed_angle_deg / 5.0) * 5.0)
        angle_rad = math.radians(snapped_angle_deg)
        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        rot_z = np.array([
            [cos_val, -sin_val, 0.0, 0.0],
            [sin_val,  cos_val, 0.0, 0.0],
            [0.0,      0.0,     1.0, 0.0],
            [0.0,      0.0,     0.0, 1.0],
        ], dtype=np.float64)

        new_matrix = rot_z @ initial_matrix
        new_gantry_angle = float(math.degrees(math.atan2(new_matrix[1, 0], new_matrix[0, 0])) % 360.0)

        changed_data = {
            'matrix': new_matrix,
            'angle_deg': new_gantry_angle,
            'delta_angle_deg': snapped_angle_deg,
            'status_message': f"Гантри: Угол ротора θ = {new_gantry_angle:.1f}° (Сетка: 5°)",
        }
        return (z_axis, snapped_angle_deg, changed_data)

    def on_transform_changed(self, target_node: Any, changed_data: Dict[str, Any]) -> None:
        """Обновление при непрерывном вращении ротора."""
        angle_deg = changed_data.get('angle_deg')
        if angle_deg is not None:
            gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node
            if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
                if abs(gantry_node_vm.gantry_angle_deg - float(angle_deg)) > 1e-4:
                    gantry_node_vm.gantry_angle_deg = float(angle_deg)
            procedure = self._procedure_ref() if self._procedure_ref is not None else None
            if isinstance(procedure, IRotatableProcedure):
                procedure.start_angle = float(angle_deg)

    def on_transform_committed(self, target_node: Any, commit_data: Dict[str, Any]) -> None:
        """Фиксация угла ротора в процедуре."""
        angle_deg = commit_data.get('angle_deg')
        if angle_deg is not None:
            gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node
            if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
                if abs(gantry_node_vm.gantry_angle_deg - float(angle_deg)) > 1e-4:
                    gantry_node_vm.gantry_angle_deg = float(angle_deg)
            procedure = self._procedure_ref() if self._procedure_ref is not None else None
            if isinstance(procedure, IRotatableProcedure):
                procedure.start_angle = float(angle_deg)



__all__ = [
    'IRotatableProcedure',
    'IKinematicConstraint',
    'FixedSubcomponentKinematicConstraint',
    'SpectOrbitKinematicConstraint',
    'CameraMountKinematicConstraint',
    'GantryKinematicConstraint',
]
