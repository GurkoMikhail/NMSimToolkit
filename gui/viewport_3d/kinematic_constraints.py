import logging
import math
import weakref
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

import numpy as np

from core.scene.gamma_camera_node import GammaCameraNode
from core.geometry.spect_kinematics import compute_orbit_matrix, extract_orbit_parameters
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
        proposed_angle: float,
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

    _translation_blocked_message: str = 'Геометрия компонента зафиксирована'
    _rotation_blocked_message: str = 'Геометрия компонента зафиксирована'

    def filter_translation(
        self,
        target_node: Any,
        proposed_world_delta: np.ndarray,
        initial_matrix: np.ndarray,
        active_axis: Optional[GizmoAxis] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return (
            np.zeros(3, dtype=np.float64),
            {'status_message': self._translation_blocked_message, 'changed': False},
        )

    def filter_rotation(
        self,
        target_node: Any,
        axis: np.ndarray,
        proposed_angle: float,
        initial_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        return (
            axis,
            0.0,
            {'status_message': self._rotation_blocked_message, 'changed': False},
        )

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



class RootVolumeKinematicConstraint(FixedSubcomponentKinematicConstraint):
    """
    Кинематическое ограничение для корневого объема сцены (World Volume).
    Блокирует любые перемещения и вращения в пространстве через манипулятор Gizmo,
    гарантируя неподвижность мировой системы отсчета относительно центра координат.
    """

    _translation_blocked_message: str = 'Перемещение рутового объема заблокировано'
    _rotation_blocked_message: str = 'Вращение рутового объема заблокировано'



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

    def _resolve_camera_half_thickness(self, target_node: Any) -> float:
        """Определяет полутолщину гамма-камеры вдоль локальной оси Z."""
        if target_node is not None and isinstance(target_node.core_node, GammaCameraNode):
            return float(target_node.half_thickness)
        if self._camera_vm is not None:
            return float(self._camera_vm.half_thickness)
        return 0.0

    @staticmethod
    def _resolve_orbit_action(
        initial_matrix: np.ndarray,
        delta_vec: np.ndarray,
        active_axis: Optional[GizmoAxis],
        radial_unit: np.ndarray,
        tangent_unit: np.ndarray,
        axial_unit: np.ndarray,
    ) -> str:
        """Классифицирует направление перемещения на орбите: radial, tangential или axial."""
        if active_axis == GizmoAxis.Z:
            return 'radial'
        if active_axis == GizmoAxis.X:
            axis_x_dir = initial_matrix[0:3, 0]
            if abs(float(np.dot(axis_x_dir, tangent_unit))) >= abs(float(np.dot(axis_x_dir, axial_unit))):
                return 'tangential'
            return 'axial'
        if active_axis == GizmoAxis.Y:
            axis_y_dir = initial_matrix[0:3, 1]
            if abs(float(np.dot(axis_y_dir, axial_unit))) >= abs(float(np.dot(axis_y_dir, tangent_unit))):
                return 'axial'
            return 'tangential'

        dot_z = abs(float(np.dot(delta_vec, radial_unit)))
        dot_t = abs(float(np.dot(delta_vec, tangent_unit)))
        dot_y = abs(float(np.dot(delta_vec, axial_unit)))
        if dot_z >= dot_t and dot_z >= dot_y:
            return 'radial'
        if dot_t >= dot_z and dot_t >= dot_y:
            return 'tangential'
        return 'axial'

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
        camera_half_thickness = self._resolve_camera_half_thickness(target_node)
        raw_radius, initial_angle, initial_z, roll_angle = extract_orbit_parameters(
            initial_matrix, half_thickness=camera_half_thickness
        )
        initial_radius = max(self._min_radius, raw_radius)

        orbit_angle = math.radians(initial_angle)
        radial_unit = np.array([math.cos(orbit_angle), math.sin(orbit_angle), 0.0], dtype=np.float64)
        tangent_unit = np.array([-math.sin(orbit_angle), math.cos(orbit_angle), 0.0], dtype=np.float64)
        axial_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        action = self._resolve_orbit_action(
            initial_matrix, proposed_world_delta, active_axis, radial_unit, tangent_unit, axial_unit
        )

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
            delta_angle = math.degrees(delta_tangent / effective_radius) if effective_radius > 1e-3 else 0.0
            new_angle_raw = initial_angle + delta_angle
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

        is_changed = True
        if target_node is not None and target_node.local_matrix is not None:
            curr_r, curr_a, curr_z, _ = extract_orbit_parameters(
                target_node.local_matrix, half_thickness=camera_half_thickness
            )
            if action == 'radial':
                is_changed = abs(new_radius - curr_r) > 1e-4
            elif action == 'tangential':
                is_changed = abs((new_angle - curr_a + 180.0) % 360.0 - 180.0) > 1e-4
            else:
                is_changed = abs(new_z - curr_z) > 1e-4

        if not is_changed and target_node is not None and target_node.local_matrix is not None:
            new_matrix = target_node.local_matrix
        else:
            new_matrix = compute_orbit_matrix(
                radius=new_radius,
                angle=new_angle,
                z=new_z,
                half_thickness=camera_half_thickness,
                roll=roll_angle,
            )

        filtered_world_delta = new_matrix[0:3, 3] - initial_matrix[0:3, 3]
        changed_data = {
            'radius': new_radius,
            'angle': new_angle,
            'z': new_z,
            'matrix': new_matrix,
            'status_message': status_message,
            'axis': active_axis,
            'action': action,
            'changed': is_changed,
        }
        return (filtered_world_delta, changed_data)

    def filter_rotation(
        self,
        target_node: Any,
        axis: np.ndarray,
        proposed_angle: float,
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
        effective_angle = proposed_angle * sign
        snapped_angle = float(round(effective_angle / 90.0) * 90.0)

        rotation_angle = math.radians(snapped_angle)
        cos_val = math.cos(rotation_angle)
        sin_val = math.sin(rotation_angle)

        rotation_mat_3x3 = np.array([
            [cos_val, -sin_val, 0.0],
            [sin_val, cos_val, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        new_matrix = initial_matrix.copy()
        new_matrix[0:3, 0:3] = initial_matrix[0:3, 0:3] @ rotation_mat_3x3

        is_changed = True
        if target_node is not None and target_node.local_matrix is not None:
            is_changed = not np.allclose(new_matrix, target_node.local_matrix, atol=1e-4)

        result_matrix = (
            target_node.local_matrix
            if (not is_changed and target_node is not None and target_node.local_matrix is not None)
            else new_matrix
        )

        is_portrait = int(round(snapped_angle / 90.0)) % 2 != 0
        orientation_text = "Книжная (Portrait)" if is_portrait else "Альбомная (Landscape)"
        status_message = f"ОФЭКТ: Ориентация детектора = {orientation_text} ({snapped_angle:.0f}°)"

        changed_data = {
            'matrix': result_matrix,
            'angle': snapped_angle,
            'orientation': orientation_text,
            'status_message': status_message,
            'changed': is_changed,
        }
        return (dir_z_global, snapped_angle, changed_data)

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
                _, default_angle, default_z, _ = extract_orbit_parameters(
                    camera_node.local_matrix,
                    half_thickness=self._resolve_camera_half_thickness(camera_node),
                )

            self.spect_manipulator.set_orbit_parameters(
                radius=float(new_radius),
                angle=float(new_angle if new_angle is not None else default_angle),
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

                shift_pressed = bool(
                    changed_data.get('shift_pressed', changed_data.get('shift_modifier', False))
                )
                if new_radius is not None and not shift_pressed:
                    target_r = float(new_radius)
                    for sibling in sibling_cameras:
                        if sibling is not target_node:
                            sib_r, sib_angle, sib_z, sib_roll = extract_orbit_parameters(
                                sibling.local_matrix, half_thickness=sibling.half_thickness
                            )
                            if abs(sib_r - target_r) > 1e-4:
                                sibling.local_matrix = compute_orbit_matrix(
                                    radius=target_r,
                                    angle=sib_angle,
                                    z=sib_z,
                                    half_thickness=sibling.half_thickness,
                                    roll=sib_roll,
                                )

                if new_angle is not None and procedure is not None:
                    head_angles = procedure.head_angles
                    offset = head_angles[node_idx] if node_idx < len(head_angles) else (360.0 / len(sibling_cameras)) * node_idx
                    base_start_angle = (float(new_angle) - offset) % 360.0
                    for sibling_idx, sibling in enumerate(sibling_cameras):
                        if sibling is not target_node:
                            sib_offset = head_angles[sibling_idx] if sibling_idx < len(head_angles) else (360.0 / len(sibling_cameras)) * sibling_idx
                            target_sib_angle = (base_start_angle + sib_offset) % 360.0
                            sib_r, sib_angle, sib_z, sib_roll = extract_orbit_parameters(
                                sibling.local_matrix, half_thickness=sibling.half_thickness
                            )
                            sibling_radius = max(self._min_radius, sib_r)
                            if abs((sib_angle - target_sib_angle + 180.0) % 360.0 - 180.0) > 1e-4:
                                sibling.local_matrix = compute_orbit_matrix(
                                    radius=sibling_radius,
                                    angle=target_sib_angle,
                                    z=sib_z,
                                    half_thickness=sibling.half_thickness,
                                    roll=sib_roll,
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

        shift_pressed = bool(
            commit_data.get('shift_pressed', commit_data.get('shift_modifier', False))
        )

        if 'radius' in commit_data and not shift_pressed:
            procedure.radius = float(commit_data['radius'])

        if 'angle' in commit_data and target_node in camera_nodes:
            node_idx = camera_nodes.index(target_node)
            head_angles = procedure.head_angles
            offset = head_angles[node_idx] if node_idx < len(head_angles) else (360.0 / max(1, len(camera_nodes))) * node_idx
            base_start_angle = (float(commit_data['angle']) - offset) % 360.0
            procedure.start_angle = base_start_angle

        if ('radius' in commit_data or 'angle' in commit_data) and not shift_pressed:
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
        self._drag_initial_matrix_id: Optional[int] = None
        self._drag_initial_gantry_angle: Optional[float] = None
        self._drag_initial_gantry_rotation: Optional[np.ndarray] = None
        self._last_applied_gantry_angle: Optional[float] = None

    def _reset_drag_gantry_state(self) -> None:
        """Сбрасывает кэшированное начальное состояние станины для текущей сессии перетаскивания."""
        self._drag_initial_matrix_id = None
        self._drag_initial_gantry_angle = None
        self._drag_initial_gantry_rotation = None
        self._last_applied_gantry_angle = None

    @property
    def gantry_vm(self) -> Optional[Any]:
        """Возвращает связанный узел станины томографа."""
        return self._gantry_vm

    @gantry_vm.setter
    def gantry_vm(self, gantry: Optional[Any]) -> None:
        """Устанавливает связанный узел станины томографа."""
        self._gantry_vm = gantry
        self._reset_drag_gantry_state()

    @property
    def gantry_node_vm(self) -> Optional[Any]:
        """Псевдоним связанного узла станины томографа."""
        return self._gantry_vm

    @gantry_node_vm.setter
    def gantry_node_vm(self, gantry: Optional[Any]) -> None:
        """Устанавливает связанный узел станины томографа через псевдоним."""
        self._gantry_vm = gantry
        self._reset_drag_gantry_state()

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

        camera_half_thickness = self._resolve_camera_half_thickness(target_node)
        current_gantry_angle = float(gantry_node_vm.gantry_angle)

        matrix_id = id(initial_matrix)
        if (
            self._drag_initial_matrix_id != matrix_id
            or self._drag_initial_gantry_angle is None
            or self._drag_initial_gantry_rotation is None
            or (
                self._last_applied_gantry_angle is not None
                and abs((current_gantry_angle - self._last_applied_gantry_angle + 180.0) % 360.0 - 180.0) > 1e-4
            )
        ):
            self._drag_initial_matrix_id = matrix_id
            self._drag_initial_gantry_angle = current_gantry_angle
            self._drag_initial_gantry_rotation = np.asarray(
                gantry_node_vm.global_matrix[0:3, 0:3], dtype=np.float64
            ).copy()
            self._last_applied_gantry_angle = current_gantry_angle

        # Преобразование мирового вектора перемещения в систему координат станины на начало перетаскивания
        gantry_rotation = self._drag_initial_gantry_rotation
        base_gantry_angle = self._drag_initial_gantry_angle
        delta_gantry = gantry_rotation.T @ proposed_world_delta

        raw_radius, initial_angle, initial_z, roll_angle = extract_orbit_parameters(
            initial_matrix, half_thickness=camera_half_thickness
        )
        initial_radius = max(self._min_radius, raw_radius)

        orbit_angle = math.radians(initial_angle)
        radial_unit = np.array([math.cos(orbit_angle), math.sin(orbit_angle), 0.0], dtype=np.float64)
        tangent_unit = np.array([-math.sin(orbit_angle), math.cos(orbit_angle), 0.0], dtype=np.float64)
        axial_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        action = self._resolve_orbit_action(
            initial_matrix, delta_gantry, active_axis, radial_unit, tangent_unit, axial_unit
        )

        if action == 'radial':
            delta_radius = float(np.dot(delta_gantry, radial_unit))
            new_radius_raw = initial_radius + delta_radius
            snapped_radius = round(new_radius_raw / 10.0) * 10.0
            new_radius = float(np.clip(snapped_radius, self._min_radius, self._max_radius))
            new_angle = initial_angle
            new_z = initial_z
            status_message = f"ОФЭКТ (Гантри): Радиус орбиты R = {new_radius:.1f} мм (Сетка: 10 мм)"
            gantry_angle = current_gantry_angle
        elif action == 'tangential':
            delta_tangent = float(np.dot(delta_gantry, tangent_unit))
            effective_radius = initial_radius + camera_half_thickness
            delta_angle = math.degrees(delta_tangent / effective_radius) if effective_radius > 1e-3 else 0.0
            new_gantry_raw = base_gantry_angle + delta_angle
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
            gantry_angle = current_gantry_angle
            status_message = f"ОФЭКТ (Гантри): Смещение Z = {new_z:.1f} мм (Сетка: 10 мм)"

        is_changed = True
        if action == 'tangential':
            is_changed = abs((gantry_angle - current_gantry_angle + 180.0) % 360.0 - 180.0) > 1e-4
        elif target_node is not None and target_node.local_matrix is not None:
            curr_r, _, curr_z, _ = extract_orbit_parameters(
                target_node.local_matrix, half_thickness=camera_half_thickness
            )
            if action == 'radial':
                is_changed = abs(new_radius - curr_r) > 1e-4
            else:
                is_changed = abs(new_z - curr_z) > 1e-4

        if not is_changed and target_node is not None and target_node.local_matrix is not None:
            new_matrix = target_node.local_matrix
        else:
            new_matrix = compute_orbit_matrix(
                radius=new_radius,
                angle=new_angle,
                z=new_z,
                half_thickness=camera_half_thickness,
                roll=roll_angle,
            )

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
            'changed': is_changed,
        }
        return (filtered_world_delta, changed_data)

    def on_transform_changed(self, target_node: Any, changed_data: Dict[str, Any]) -> None:
        gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node.parent_vm
        if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
            procedure = self._procedure_ref() if self._procedure_ref is not None else None
            shift_pressed = bool(
                changed_data.get('shift_pressed', changed_data.get('shift_modifier', False))
            )

            # 1. Поворот станины при тангенциальном перемещении
            if changed_data.get('action') == 'tangential':
                new_gantry_angle = changed_data.get('gantry_angle', changed_data.get('angle'))
                if new_gantry_angle is not None:
                    target_angle = float(new_gantry_angle)
                    self._last_applied_gantry_angle = target_angle
                    if abs((gantry_node_vm.gantry_angle - target_angle + 180.0) % 360.0 - 180.0) > 1e-4:
                        gantry_node_vm.gantry_angle = target_angle
                    if isinstance(procedure, IRotatableProcedure):
                        if abs((procedure.start_angle - target_angle + 180.0) % 360.0 - 180.0) > 1e-4:
                            procedure.start_angle = target_angle

            # 2. Синхронизация радиуса со всеми спаренными головками (отключается при зажатом Shift)
            if changed_data.get('action') == 'radial' and 'radius' in changed_data and not shift_pressed:
                new_radius = changed_data.get('radius')
                if new_radius is not None:
                    target_r = float(new_radius)
                    already_synced_by_procedure = (
                        procedure is not None and abs(float(procedure.radius) - target_r) <= 1e-4
                    )
                    if not already_synced_by_procedure:
                        for sibling in gantry_node_vm.children:
                            if isinstance(sibling.core_node, GammaCameraNode) and sibling is not target_node:
                                sib_r, sib_angle, sib_z, sib_roll = extract_orbit_parameters(
                                    sibling.local_matrix, half_thickness=sibling.half_thickness
                                )
                                if abs(sib_r - target_r) > 1e-4:
                                    sibling.local_matrix = compute_orbit_matrix(
                                        radius=target_r,
                                        angle=sib_angle,
                                        z=sib_z,
                                        half_thickness=sibling.half_thickness,
                                        roll=sib_roll,
                                    )
                        if procedure is not None:
                            procedure.radius = target_r

            if self.spect_manipulator is not None:
                if 'radius' in changed_data:
                    current_radius = float(changed_data['radius'])
                else:
                    current_radius, _, _, _ = extract_orbit_parameters(
                        target_node.local_matrix, half_thickness=target_node.half_thickness
                    )
                current_gantry_angle = gantry_node_vm.gantry_angle
                current_z = float(changed_data.get('z', target_node.local_matrix[2, 3]))
                self.spect_manipulator.set_orbit_parameters(
                    radius=current_radius,
                    angle=current_gantry_angle,
                    z=current_z,
                    render=False,
                    emit_signal=False,
                )
        else:
            super().on_transform_changed(target_node, changed_data)

    def on_transform_committed(self, target_node: Any, commit_data: Dict[str, Any]) -> None:
        self._reset_drag_gantry_state()
        gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node.parent_vm
        if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
            procedure = self._procedure_ref() if self._procedure_ref is not None else None
            shift_pressed = bool(
                commit_data.get('shift_pressed', commit_data.get('shift_modifier', False))
            )
            if procedure is not None:
                if commit_data.get('action') == 'radial' and 'radius' in commit_data and not shift_pressed:
                    procedure.radius = float(commit_data['radius'])
                if commit_data.get('action') == 'tangential':
                    new_gantry_angle = commit_data.get('gantry_angle', commit_data.get('angle'))
                    if new_gantry_angle is not None and isinstance(procedure, IRotatableProcedure):
                        procedure.start_angle = float(new_gantry_angle)
                if commit_data.get('action') in ('radial', 'tangential') and not shift_pressed:
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
        return (
            np.zeros(3, dtype=np.float64),
            {'status_message': 'Перемещение станины заблокировано', 'changed': False},
        )

    def filter_rotation(
        self,
        target_node: Any,
        axis: np.ndarray,
        proposed_angle: float,
        initial_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Разрешает вращение только вокруг оси Z.
        Снаппинг угла с шагом 5 градусов.
        """
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        snapped_angle = float(round(proposed_angle / 5.0) * 5.0)
        rotation_angle = math.radians(snapped_angle)
        cos_val = math.cos(rotation_angle)
        sin_val = math.sin(rotation_angle)

        rot_z = np.array([
            [cos_val, -sin_val, 0.0, 0.0],
            [sin_val,  cos_val, 0.0, 0.0],
            [0.0,      0.0,     1.0, 0.0],
            [0.0,      0.0,     0.0, 1.0],
        ], dtype=np.float64)

        new_matrix = rot_z @ initial_matrix
        new_gantry_angle = float(math.degrees(math.atan2(new_matrix[1, 0], new_matrix[0, 0])) % 360.0)

        gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node
        if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
            is_changed = abs((new_gantry_angle - gantry_node_vm.gantry_angle + 180.0) % 360.0 - 180.0) > 1e-4
        elif target_node is not None and target_node.local_matrix is not None:
            is_changed = not np.allclose(new_matrix, target_node.local_matrix, atol=1e-4)
        else:
            is_changed = True

        result_matrix = (
            target_node.local_matrix
            if (not is_changed and target_node is not None and target_node.local_matrix is not None)
            else new_matrix
        )

        changed_data = {
            'matrix': result_matrix,
            'angle': new_gantry_angle,
            'delta_angle': snapped_angle,
            'status_message': f"Гантри: Угол ротора θ = {new_gantry_angle:.1f}° (Сетка: 5°)",
            'changed': is_changed,
        }
        return (z_axis, snapped_angle, changed_data)

    def on_transform_changed(self, target_node: Any, changed_data: Dict[str, Any]) -> None:
        """Обновление при непрерывном вращении ротора."""
        angle = changed_data.get('angle')
        if angle is not None:
            gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node
            if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
                if abs((gantry_node_vm.gantry_angle - float(angle) + 180.0) % 360.0 - 180.0) > 1e-4:
                    gantry_node_vm.gantry_angle = float(angle)
            procedure = self._procedure_ref() if self._procedure_ref is not None else None
            if isinstance(procedure, IRotatableProcedure):
                procedure.start_angle = float(angle)

    def on_transform_committed(self, target_node: Any, commit_data: Dict[str, Any]) -> None:
        """Фиксация угла ротора в процедуре."""
        angle = commit_data.get('angle')
        if angle is not None:
            gantry_node_vm = self._gantry_vm if self._gantry_vm is not None else target_node
            if gantry_node_vm is not None and isinstance(gantry_node_vm.core_node, GantryNode):
                if abs((gantry_node_vm.gantry_angle - float(angle) + 180.0) % 360.0 - 180.0) > 1e-4:
                    gantry_node_vm.gantry_angle = float(angle)
            procedure = self._procedure_ref() if self._procedure_ref is not None else None
            if isinstance(procedure, IRotatableProcedure):
                procedure.start_angle = float(angle)



__all__ = [
    'IRotatableProcedure',
    'IKinematicConstraint',
    'FixedSubcomponentKinematicConstraint',
    'RootVolumeKinematicConstraint',
    'SpectOrbitKinematicConstraint',
    'CameraMountKinematicConstraint',
    'GantryKinematicConstraint',
]

