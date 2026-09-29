import logging
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union, runtime_checkable
import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject

from gui.viewport_3d.vtk_viewport import VTKViewport
from gui.viewport_3d.track_renderer import TrackRenderer
from gui.viewport_3d.spect_manipulator import SPECTManipulator
from gui.viewport_3d.pet_manipulator import PETManipulator
from gui.viewport_3d.voxel_volume_renderer import VoxelVolumeRenderer
from gui.viewport_3d.dose_volume_renderer import DoseVolumeRenderer
from gui.viewport_3d.transform_gizmo import TransformGizmo
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.nodes.pet_scanner_vm import PetScannerViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.procedure_viewmodel import BaseProcedureViewModel, SpectProcedureViewModel
from gui.viewport_3d.kinematic_constraints import (
    GantryKinematicConstraint,
    IKinematicConstraint,
    SpectOrbitKinematicConstraint,
)

from core.geometry import compute_spect_poses

_logger = logging.getLogger(__name__)


@runtime_checkable
class IDoseGeometryProvider(Protocol):
    """
    Контракт источника геометрических параметров карты накопленной дозы.
    Исключает утиную типизацию и использование hasattr/getattr.
    """
    @property
    def dose_origin(self) -> Optional[Tuple[float, float, float]]: ...

    @property
    def dose_voxel_size(self) -> Optional[float]: ...

    @property
    def dose_transform_matrix(self) -> Optional[np.ndarray]: ...


class SceneViewportController(QObject):
    """
    Выделенный контроллер синхронизации сцены и 3D-вьюпорта (ViewportBridge / Controller).
    Управляет жизненным циклом VTK-акторов, инкрементальным обновлением трансформаций,
    кинематикой ОФЭКТ/ПЭТ манипуляторов и маршрутизацией данных объемной дозы.
    """

    def __init__(
        self,
        viewport: VTKViewport,
        scene_vm: Optional[SceneViewModel] = None,
        procedure_vm: Optional[BaseProcedureViewModel] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.viewport = viewport
        self.scene_vm = scene_vm
        self.procedure_vm = procedure_vm

        # Рендереры и манипуляторы вьюпорта
        self.track_renderer = TrackRenderer(self.viewport, render_as_lines=True)
        self.spect_manipulator = SPECTManipulator(self.viewport)
        self.spect_manipulator.orbit_changed.connect(self.on_spect_manipulator_changed)
        self.pet_manipulator = PETManipulator(self.viewport)
        self.voxel_renderer = VoxelVolumeRenderer(self.viewport)
        self.dose_renderer = DoseVolumeRenderer(self.viewport)
        self.transform_gizmo = TransformGizmo(self.viewport)

        # Кэш параметров активной воксельной сетки дозы
        self._active_dose_voxel_size: float = 5.0
        self._active_dose_origin: Optional[Tuple[float, float, float]] = None
        self._active_dose_transform_matrix: Optional[np.ndarray] = None

        # Словарь подписок на события узлов: node_id -> (node_vm, [connections])
        self._node_connections: Dict[int, Tuple[NodeViewModel, List[Any]]] = {}

        if self.scene_vm is not None:
            self.set_scene_viewmodel(self.scene_vm)

    @property
    def node_connections(self) -> Dict[int, Tuple[NodeViewModel, List[Any]]]:
        return self._node_connections

    @property
    def active_dose_origin(self) -> Optional[Tuple[float, float, float]]:
        """Точка начала координат активной сетки дозы (X, Y, Z) в мм."""
        return self._active_dose_origin

    @active_dose_origin.setter
    def active_dose_origin(self, origin: Optional[Tuple[float, float, float]]) -> None:
        self._active_dose_origin = origin

    @property
    def active_dose_voxel_size(self) -> float:
        """Размер вокселя активной сетки дозы (мм)."""
        return self._active_dose_voxel_size

    @active_dose_voxel_size.setter
    def active_dose_voxel_size(self, voxel_size: float) -> None:
        self._active_dose_voxel_size = float(voxel_size)

    @property
    def active_dose_transform_matrix(self) -> Optional[np.ndarray]:
        """Матрица пространственной трансформации активной сетки дозы (4x4)."""
        return self._active_dose_transform_matrix

    @active_dose_transform_matrix.setter
    def active_dose_transform_matrix(self, matrix: Optional[np.ndarray]) -> None:
        self._active_dose_transform_matrix = matrix

    def set_scene_viewmodel(self, scene_vm: Optional[SceneViewModel]) -> None:
        """Привязка новой модели представления сцены с обновлением подписок."""
        self.disconnect_all_nodes()
        self.scene_vm = scene_vm
        if self.scene_vm is not None:
            self.sync_viewport_scene()

    def sync_viewport_scene(self) -> None:
        """
        Полная синхронизация визуальных 3D-мешей в VTKViewport с графом SceneViewModel.
        """
        if self.viewport is None or self.scene_vm is None or self.scene_vm.root_vm is None:
            return

        all_nodes = self.scene_vm.all_nodes()
        current_actor_names = set()
        has_spect = False
        has_pet = False

        for node_vm in all_nodes:
            actor_name = f"mesh_{id(node_vm)}"
            current_actor_names.add(actor_name)
            self.add_or_update_node_actor(node_vm)
            if isinstance(node_vm, GammaCameraViewModel):
                has_spect = True
            if isinstance(node_vm, PetScannerViewModel):
                has_pet = True

        # Исключение паразитной отрисовки манипуляторов при отсутствии узлов
        if not has_spect:
            self.spect_manipulator.remove_visuals()
        if not has_pet:
            self.pet_manipulator.remove_visuals()

        # Удаляем акторы и отключаем подписки узлов, которых больше нет в сцене
        current_node_ids = {id(n) for n in all_nodes}
        for node_id in list(self._node_connections.keys()):
            if node_id not in current_node_ids:
                self.disconnect_node(node_id)

        for existing in list(self.viewport._actors.keys()):
            if existing.startswith("mesh_") and existing not in current_actor_names:
                self.viewport.remove_actor(existing)

        self.viewport.render()

    def disconnect_node(self, node_id: int) -> None:
        """Отключение Qt-сигналов и удаление ссылки на ViewModel узла."""
        if node_id in self._node_connections:
            node_vm, conns = self._node_connections.pop(node_id)
            for conn in conns:
                try:
                    QObject.disconnect(conn)
                except (RuntimeError, TypeError):
                    pass

    def disconnect_all_nodes(self) -> None:
        """Полное отключение подписок на все узлы сцены."""
        for node_id in list(self._node_connections.keys()):
            self.disconnect_node(node_id)

    def add_or_update_node_actor(self, node_vm: NodeViewModel) -> None:
        """
        Добавление или обновление геометрического актора узла в 3D вьюпорте.
        """
        actor_name = f"mesh_{id(node_vm)}"

        if isinstance(node_vm, VolumeViewModel):
            volume_size = node_vm.size
            box = pv.Box(bounds=(-volume_size[0]/2, volume_size[0]/2, -volume_size[1]/2, volume_size[1]/2, -volume_size[2]/2, volume_size[2]/2))
            color = node_vm.color
            rgb_color = color[:3] if isinstance(color, tuple) and len(color) >= 3 else (0.2, 0.6, 1.0)
            self.viewport.add_mesh_actor(actor_name, box, color=rgb_color, opacity=0.45)
            self.viewport.update_actor_transform(actor_name, node_vm.global_matrix)

        elif isinstance(node_vm, VoxelVolumeViewModel):
            dist = node_vm.core_node.material_distribution
            if dist is not None:
                data = np.asarray(dist.ID, dtype=np.float32)
                if float(np.max(data)) == 0.0:
                    data = np.asarray(dist.view(np.ndarray), dtype=np.float32)
                if float(np.max(data)) == 0.0:
                    data = np.asarray(dist.density, dtype=np.float32)
                self.voxel_renderer.set_volume_data(
                    data,
                    voxel_size=node_vm.voxel_size,
                    origin=node_vm.origin
                )
                self.voxel_renderer.set_colormap(node_vm.colormap_name)
                self.voxel_renderer.set_opacity_parameters(
                    max_opacity=float(node_vm.max_opacity),
                    threshold=float(node_vm.opacity_threshold),
                    preset=node_vm.opacity_preset
                )
                self.voxel_renderer.set_lod_factor(float(node_vm.lod_factor))
                self.viewport.update_actor_transform(self.voxel_renderer.actor_name, node_vm.global_matrix)

        elif isinstance(node_vm, SourceViewModel):
            if node_vm.is_point_source:
                sphere = pv.Sphere(radius=8.0)
                self.viewport.add_mesh_actor(actor_name, sphere, color=(1.0, 0.2, 0.2), opacity=0.85)
            else:
                sz = node_vm.size
                if any(s <= 0 for s in sz):
                    sz = (50.0, 50.0, 50.0)
                box = pv.Box(bounds=(-sz[0]/2, sz[0]/2, -sz[1]/2, sz[1]/2, -sz[2]/2, sz[2]/2))
                self.viewport.add_mesh_actor(actor_name, box, color=(1.0, 0.8, 0.1), opacity=0.35, style='wireframe')
            self.viewport.update_actor_transform(actor_name, node_vm.global_matrix)

        elif isinstance(node_vm, DoseGridViewModel):
            sz = node_vm.size
            if any(s <= 0 for s in sz):
                sz = (100.0, 100.0, 100.0)
            box = pv.Box(bounds=(-sz[0]/2, sz[0]/2, -sz[1]/2, sz[1]/2, -sz[2]/2, sz[2]/2))
            color = (0.2, 0.9, 0.3)
            opacity = 0.85 if node_vm.is_active else 0.3
            self.viewport.add_mesh_actor(
                actor_name,
                box,
                color=color,
                opacity=opacity,
                style='wireframe',
                line_width=2.0
            )
            self.viewport.update_actor_transform(actor_name, node_vm.global_matrix)

        elif isinstance(node_vm, PetScannerViewModel):
            self.pet_manipulator.set_parameters(
                diameter=float(node_vm.diameter),
                axial_length=float(node_vm.axial_length),
                num_sectors=int(node_vm.num_sectors),
            )
            self.viewport.update_actor_transform(self.pet_manipulator.actor_name, node_vm.global_matrix)

        # Подписка на изменение матрицы и свойств узла для инкрементального обновления
        node_id = id(node_vm)
        if node_id not in self._node_connections:
            conn1 = node_vm.transform_changed.connect(
                lambda observed_node_vm=node_vm: self.on_node_transform_changed(observed_node_vm)
            )
            conn2 = node_vm.property_changed.connect(
                lambda property_name, property_value, observed_node_vm=node_vm: self.on_node_property_changed(observed_node_vm, property_name, property_value)
            )
            self._node_connections[node_id] = (node_vm, [conn1, conn2])

    def disconnect_node(self, node_id: int) -> None:
        """
        Отключение сигналов отслеживаемого узла и удаление из кэша соединений.
        """
        if node_id in self._node_connections:
            node_vm, connections = self._node_connections.pop(node_id)
            for connection in connections:
                try:
                    QObject.disconnect(connection)
                except (RuntimeError, TypeError):
                    pass

    def disconnect_all_nodes(self) -> None:
        """
        Отключение сигналов всех отслеживаемых узлов сцены и деактивация манипуляторов.
        """
        for node_id in list(self._node_connections.keys()):
            self.disconnect_node(node_id)
        self._node_connections.clear()
        if self.transform_gizmo is not None:
            self.transform_gizmo.detach()

    def on_node_transform_changed(self, node_vm: NodeViewModel) -> None:
        """
        Инкрементальное обновление матрицы трансформации актора без пересоздания меша.
        """
        actor_name = f"mesh_{id(node_vm)}"
        self.viewport.update_actor_transform(actor_name, node_vm.global_matrix)
        if isinstance(node_vm, VoxelVolumeViewModel) and self.voxel_renderer is not None:
            self.viewport.update_actor_transform(self.voxel_renderer.actor_name, node_vm.global_matrix)
        elif isinstance(node_vm, PetScannerViewModel) and self.pet_manipulator is not None:
            self.viewport.update_actor_transform(self.pet_manipulator.actor_name, node_vm.global_matrix)
        if self.transform_gizmo is not None and self.transform_gizmo.target_node is node_vm:
            self.transform_gizmo.update_visuals()

    def on_node_property_changed(self, node_vm: NodeViewModel, prop_name: str, value: Any) -> None:
        """
        Инкрементальное обновление параметров актора при смене геометрии, цвета или физических свойств.
        """
        if prop_name in ('size', 'color', 'voxel_size', 'is_point_source', 'file_path', 'dose_voxel_size', 'is_active'):
            self.add_or_update_node_actor(node_vm)
            self.viewport.render()
        elif prop_name in ('diameter', 'axial_length', 'num_sectors') and isinstance(node_vm, PetScannerViewModel):
            self.pet_manipulator.set_parameters(
                diameter=float(node_vm.diameter),
                axial_length=float(node_vm.axial_length),
                num_sectors=int(node_vm.num_sectors),
            )
            self.viewport.render()
        elif prop_name == 'colormap_name' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_colormap(str(value))
                self.viewport.render()
        elif prop_name == 'opacity_threshold' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_opacity_threshold(float(value))
                self.viewport.render()
        elif prop_name == 'max_opacity' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_max_opacity(float(value))
                self.viewport.render()
        elif prop_name == 'opacity_preset' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_opacity_preset(str(value))
                self.viewport.render()
        elif prop_name == 'lod_factor' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_lod_factor(float(value))
                self.viewport.render()

    def on_node_added(self, node_vm: NodeViewModel) -> None:
        """Точечное добавление нового актора в сцену (рекурсивно для дочерних узлов)."""
        def _add_recursive(vm: NodeViewModel) -> None:
            self.add_or_update_node_actor(vm)
            for child in vm.children:
                _add_recursive(child)

        _add_recursive(node_vm)
        self.viewport.render()

    def on_node_removed(self, node_vm: NodeViewModel) -> None:
        """Точечное удаление актора из сцены (рекурсивно для дочерних узлов)."""
        def _remove_recursive(vm: NodeViewModel) -> None:
            actor_name = f"mesh_{id(vm)}"
            self.viewport.remove_actor(actor_name)
            if isinstance(vm, VoxelVolumeViewModel) and self.voxel_renderer is not None:
                self.viewport.remove_actor(self.voxel_renderer.actor_name)
            self.disconnect_node(id(vm))
            for child in vm.children:
                _remove_recursive(child)

        _remove_recursive(node_vm)
        self.viewport.render()

    def on_procedure_changed(self, new_procedure_vm: BaseProcedureViewModel) -> None:
        """Смена активной процедуры: динамическое обновление кинематических ограничений."""
        self.procedure_vm = new_procedure_vm
        if self.scene_vm is not None:
            for scene_node_vm in self.scene_vm.all_nodes():
                if isinstance(scene_node_vm, GantryViewModel):
                    scene_node_vm.procedure_vm = self.procedure_vm

        if self.transform_gizmo is not None and self.transform_gizmo.target_node is not None:
            current_target = self.transform_gizmo.target_node
            new_constraint = current_target.get_effective_kinematic_constraint()
            if new_constraint is None and self.procedure_vm is not None:
                new_constraint = self.procedure_vm.get_kinematic_constraint_for_node(current_target)

            if isinstance(new_constraint, SpectOrbitKinematicConstraint):
                new_constraint.spect_manipulator = self.spect_manipulator
            if isinstance(new_constraint, (SpectOrbitKinematicConstraint, GantryKinematicConstraint)):
                if new_constraint.procedure_vm is None and self.procedure_vm is not None:
                    new_constraint.procedure_vm = self.procedure_vm

            self.transform_gizmo.set_constraint(new_constraint)
            if new_constraint is not None:
                forced_space = new_constraint.get_forced_space()
                if forced_space is not None:
                    self.transform_gizmo.space = forced_space
            self.transform_gizmo.update_visuals()

            # Обновление направляющих манипуляторов ОФЭКТ/ПЭТ
            if isinstance(current_target, GammaCameraViewModel) and isinstance(self.procedure_vm, SpectProcedureViewModel):
                self.spect_manipulator.half_thickness = current_target.half_thickness
                self.spect_manipulator.set_orbit_parameters(
                    current_target.orbit_radius,
                    current_target.orbit_angle,
                    z=current_target.orbit_z,
                    render=False,
                    emit_signal=False,
                )
                self.pet_manipulator.remove_visuals()
            elif isinstance(current_target, PetScannerViewModel):
                self.spect_manipulator.remove_visuals()
                self.pet_manipulator.set_parameters(
                    diameter=float(current_target.diameter),
                    axial_length=float(current_target.axial_length),
                    num_sectors=int(current_target.num_sectors),
                )
            else:
                self.spect_manipulator.remove_visuals()
                self.pet_manipulator.remove_visuals()

            self.viewport.render()

    def on_node_selected(self, selected_node_vm: Optional[NodeViewModel]) -> None:
        """Синхронизация ОФЭКТ/ПЭТ-манипулятора и Transform Gizmo при выборе узла в сцене."""
        if selected_node_vm is None:
            if self.transform_gizmo is not None:
                self.transform_gizmo.detach()
            self.spect_manipulator.remove_visuals()
            self.pet_manipulator.remove_visuals()
            self.viewport.render()
            return

        # 1. Запрашиваем кинематическое ограничение:
        # Первичный источник: граф сцены (узел сам или через родителя знает свои ограничения)
        constraint: Optional[IKinematicConstraint] = selected_node_vm.get_effective_kinematic_constraint()
        if constraint is None and self.procedure_vm is not None:
            # Fallback для обратной совместимости с внешними процедурами
            constraint = self.procedure_vm.get_kinematic_constraint_for_node(selected_node_vm)

        if isinstance(constraint, SpectOrbitKinematicConstraint):
            constraint.spect_manipulator = self.spect_manipulator

        if isinstance(constraint, (SpectOrbitKinematicConstraint, GantryKinematicConstraint)):
            if constraint.procedure_vm is None and self.procedure_vm is not None:
                constraint.procedure_vm = self.procedure_vm

        # Если выбран GantryViewModel или его дочерний узел, гарантируем привязку procedure_vm
        if isinstance(selected_node_vm, GantryViewModel) and selected_node_vm.procedure_vm is None:
            selected_node_vm.procedure_vm = self.procedure_vm
        elif isinstance(selected_node_vm.parent_vm, GantryViewModel) and selected_node_vm.parent_vm.procedure_vm is None:
            selected_node_vm.parent_vm.procedure_vm = self.procedure_vm

        # 2. Настраиваем манипулятор TransformGizmo:
        if self.transform_gizmo is not None:
            self.transform_gizmo.set_constraint(constraint)
            if constraint is not None:
                forced_space = constraint.get_forced_space()
                if forced_space is not None:
                    self.transform_gizmo.space = forced_space
            self.transform_gizmo.set_target_node(selected_node_vm)

        # 3. Визуальные направляющие ОФЭКТ (круговая орбита):
        if isinstance(selected_node_vm, GammaCameraViewModel) and isinstance(self.procedure_vm, SpectProcedureViewModel):
            self.spect_manipulator.half_thickness = selected_node_vm.half_thickness
            self.spect_manipulator.set_orbit_parameters(
                selected_node_vm.orbit_radius,
                selected_node_vm.orbit_angle,
                z=selected_node_vm.orbit_z,
                render=False,
                emit_signal=False,
            )
            self.pet_manipulator.remove_visuals()
        elif isinstance(selected_node_vm, PetScannerViewModel):
            self.spect_manipulator.remove_visuals()
            self.pet_manipulator.set_parameters(
                diameter=float(selected_node_vm.diameter),
                axial_length=float(selected_node_vm.axial_length),
                num_sectors=int(selected_node_vm.num_sectors),
            )
        else:
            self.spect_manipulator.remove_visuals()
            self.pet_manipulator.remove_visuals()

        self.viewport.render()


    def on_spect_manipulator_changed(self, radius: float, angle_deg: float, z_pos: float) -> None:
        """Обработка перемещения ОФЭКТ-манипулятора в 3D-пространстве."""
        if self.scene_vm is not None and isinstance(self.scene_vm.selected_node, GammaCameraViewModel):
            self.scene_vm.selected_node.set_orbit_position(radius, angle_deg, z=z_pos)

    def apply_job_angles_to_viewport(
        self,
        context: Dict[str, Any],
        procedure_vm: Optional[BaseProcedureViewModel] = None
    ) -> None:
        """Применяет углы из контекста задачи к гамма-камерам во вьюпорте."""
        if self.scene_vm is None:
            return

        camera_vms = [node for node in self.scene_vm.all_nodes() if isinstance(node, GammaCameraViewModel)]
        if not camera_vms:
            return

        radius = 250.0
        if isinstance(procedure_vm, SpectProcedureViewModel):
            radius = float(procedure_vm.radius)

        for camera_index, camera_view_model in enumerate(camera_vms):
            angle_val = context.get(f"head_{camera_index}_angle")
            if angle_val is None:
                angle_val = context.get("current_angle")
            if angle_val is not None:
                camera_view_model.set_orbit_position(radius, float(angle_val), camera_view_model.orbit_z)

        if isinstance(self.scene_vm.selected_node, GammaCameraViewModel):
            selected_camera = self.scene_vm.selected_node
            self.spect_manipulator.set_orbit_parameters(
                selected_camera.orbit_radius,
                selected_camera.orbit_angle,
                z=selected_camera.orbit_z,
                render=False,
                emit_signal=False
            )

        self.viewport.render()

    def preview_view(
        self,
        view_number_1based: int,
        procedure_vm: Optional[BaseProcedureViewModel] = None
    ) -> float:
        """
        Предварительный кинематический поворот всех детекторных головок в 3D-сцене на выбранный ракурс ОФЭКТ.
        Возвращает базовый угол поворота в градусах.
        """
        if self.scene_vm is None:
            return 0.0

        view_index = max(0, view_number_1based - 1)
        camera_vms = [node for node in self.scene_vm.all_nodes() if isinstance(node, GammaCameraViewModel)]
        if not camera_vms:
            return 0.0

        base_angle = 0.0
        active_procedure = procedure_vm if procedure_vm is not None else self.procedure_vm
        if isinstance(active_procedure, SpectProcedureViewModel):
            previous_sync_state = active_procedure._is_syncing_with_scene
            active_procedure._is_syncing_with_scene = True
            try:
                radius = float(active_procedure.radius)
                calculated_poses = compute_spect_poses(
                    views_or_protocol=active_procedure.to_config(),
                    gamma_cameras=active_procedure.gamma_cameras,
                    start_angle_deg=active_procedure.start_angle,
                    end_angle_deg=active_procedure.end_angle,
                    head_angle_offsets=active_procedure.head_angles if active_procedure.head_angles else None,
                    endpoint=active_procedure.endpoint,
                )
                pose_index = min(view_index, len(calculated_poses) - 1) if calculated_poses else 0
                angles = calculated_poses[pose_index] if calculated_poses else [0.0] * len(camera_vms)
                base_angle = angles[0] if angles else 0.0

                gantry_vms = [node for node in self.scene_vm.all_nodes() if isinstance(node, GantryViewModel)]
                if gantry_vms and any(cam_vm.parent_vm in gantry_vms for cam_vm in camera_vms):
                    gantry_vm = gantry_vms[0]
                    head_offset_0 = active_procedure.head_angles[0] if active_procedure.head_angles else 0.0
                    gantry_rot_angle = (angles[0] - head_offset_0) % 360.0
                    gantry_vm.gantry_angle_deg = gantry_rot_angle
                    for camera_index, cam_vm in enumerate(camera_vms):
                        head_offset = (
                            active_procedure.head_angles[camera_index]
                            if camera_index < len(active_procedure.head_angles)
                            else (360.0 / max(1, len(camera_vms))) * camera_index
                        )
                        cam_vm.set_orbit_position(radius, float(head_offset), cam_vm.orbit_z)
                else:
                    for camera_index, cam_vm in enumerate(camera_vms):
                        angle_val = angles[camera_index] if camera_index < len(angles) else angles[0]
                        cam_vm.set_orbit_position(radius, float(angle_val), cam_vm.orbit_z)

                if isinstance(self.scene_vm.selected_node, GammaCameraViewModel):
                    selected_camera = self.scene_vm.selected_node
                    self.spect_manipulator.set_orbit_parameters(
                        selected_camera.orbit_radius,
                        selected_camera.orbit_angle,
                        z=selected_camera.orbit_z,
                        render=False,
                        emit_signal=False
                    )

                self.viewport.render()
            finally:
                active_procedure._is_syncing_with_scene = previous_sync_state
        return base_angle

    def on_dose_volume_received(
        self,
        dose_data: np.ndarray,
        session: Optional[Union[IDoseGeometryProvider, Any]] = None
    ) -> None:
        """
        Прием и отображение очередного снимка 3D-карты дозы.
        Использует геометрические параметры активной сессии или последние сохраненные
        параметры сетки, гарантируя неизменность origin и матриц при остановке симуляции.
        """
        if self.dose_renderer is None:
            return

        if session is not None and isinstance(session, IDoseGeometryProvider):
            if session.dose_origin is not None:
                self._active_dose_origin = session.dose_origin
            if session.dose_voxel_size is not None:
                self._active_dose_voxel_size = session.dose_voxel_size
            if session.dose_transform_matrix is not None:
                self._active_dose_transform_matrix = session.dose_transform_matrix

        self.dose_renderer.update_dose_data(
            dose_data,
            voxel_size=self._active_dose_voxel_size,
            origin=self._active_dose_origin,
            transform_matrix=self._active_dose_transform_matrix
        )

    def clear_dose_volume(self) -> None:
        """Очистка отображения карты дозы и сброс параметров сетки."""
        if self.dose_renderer is not None:
            self.dose_renderer.clear()
        self._active_dose_origin = None
        self._active_dose_transform_matrix = None

    def clear_tracks(self) -> None:
        """Очистка отображения треков частиц."""
        if self.track_renderer is not None:
            self.track_renderer.clear()

    def close(self) -> None:
        """Освобождение всех ресурсов и отключение подписок."""
        self.disconnect_all_nodes()
        if self.transform_gizmo is not None:
            self.transform_gizmo.detach()
        self.spect_manipulator.remove_visuals()
        self.pet_manipulator.remove_visuals()
        self.clear_tracks()
        self.clear_dose_volume()
