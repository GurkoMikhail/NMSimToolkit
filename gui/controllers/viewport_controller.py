import logging
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union, runtime_checkable
import numpy as np
import pyvista as pv
import hepunits as units
from PySide6.QtCore import QObject

from gui.viewport_3d.vtk_viewport import VTKViewport, ISceneViewport
from gui.viewport_3d.material_palette import (
    get_material_rgba,
    get_material_opacity,
    get_material_color,
    get_pseudo_xray_rgba,
    DETECTOR_ACCENT_COLOR,
    DETECTOR_ACCENT_OPACITY,
    SELECTED_EDGE_HIGHLIGHT_COLOR,
    SELECTED_EDGE_HIGHLIGHT_WIDTH,
    COLLIMATOR_BODY_OPACITY,
    COLLIMATOR_HOLE_ACCENT_COLOR,
    COLLIMATOR_HOLE_OPACITY,
    get_collimator_visual_properties,
)
from gui.viewport_3d.track_renderer import TrackRenderer
from gui.viewport_3d.spect_manipulator import SPECTManipulator
from gui.viewport_3d.pet_manipulator import PETManipulator
from gui.viewport_3d.voxel_volume_renderer import VoxelVolumeRenderer
from gui.viewport_3d.dose_volume_renderer import DoseVolumeRenderer
from gui.viewport_3d.collimator_hole_renderer import CollimatorHoleRenderer
from gui.viewport_3d.transform_gizmo import TransformGizmo, GizmoMode
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.nodes.pet_scanner_vm import PetScannerViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    StepAndShootProcedureViewModel,
    SpectProcedureViewModel,
)
from gui.viewmodels.data_handler_viewmodel import DataManagerViewModel
from gui.viewport_3d.kinematic_constraints import (
    GantryKinematicConstraint,
    IKinematicConstraint,
    RootVolumeKinematicConstraint,
    SpectOrbitKinematicConstraint,
)

from core.geometry import compute_spect_poses, compute_orbit_matrix, extract_orbit_parameters
from core.geometry.geometries import PeriodicHexPrism
from core.geometry.direct_collimators import DirectParallelCollimator
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.scene.nodes import CompositeNode

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
        data_manager_vm: Optional[DataManagerViewModel] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.viewport = viewport
        self.scene_vm = scene_vm
        self.procedure_vm = procedure_vm
        self._data_manager_vm: Optional[DataManagerViewModel] = None

        # Рендереры и манипуляторы вьюпорта
        self.track_renderer = TrackRenderer(self.viewport, render_as_lines=True)
        self.spect_manipulator = SPECTManipulator(self.viewport)
        self.spect_manipulator.orbit_changed.connect(self.on_spect_manipulator_changed)
        self.pet_manipulator = PETManipulator(self.viewport)
        self.voxel_renderer = VoxelVolumeRenderer(self.viewport)
        self.dose_renderer = DoseVolumeRenderer(self.viewport)
        self.collimator_hole_renderer = CollimatorHoleRenderer(self.viewport)
        self.transform_gizmo = TransformGizmo(self.viewport)

        # Кэш параметров активной воксельной сетки дозы
        self._active_dose_voxel_size: float = 5.0
        self._active_dose_origin: Optional[Tuple[float, float, float]] = None
        self._active_dose_transform_matrix: Optional[np.ndarray] = None

        # Состояние выделения узлов и параметров рентгеновской визуализации
        self._selected_node_vm: Optional[NodeViewModel] = None
        self._active_selection_box_actor: Optional[str] = None
        self._xray_mode: bool = False
        self._xray_energy: float = 140.0 * units.keV

        # Словарь подписок на события узлов: node_id -> (node_vm, [connections])
        self._node_connections: Dict[int, Tuple[NodeViewModel, List[Any]]] = {}
        self._sensitive_conn: Optional[Any] = None

        if self.scene_vm is not None:
            self.set_scene_viewmodel(self.scene_vm)
        if data_manager_vm is not None:
            self.set_data_manager_viewmodel(data_manager_vm)

    @property
    def selected_node_vm(self) -> Optional[NodeViewModel]:
        """Текущий выделенный узел сцены."""
        return self._selected_node_vm

    @property
    def xray_mode(self) -> bool:
        """Флаг активного режима отображения 'Псевдорентген'."""
        return self._xray_mode

    @xray_mode.setter
    def xray_mode(self, enabled: bool) -> None:
        self.set_xray_parameters(self._xray_energy, enabled)

    @property
    def xray_energy(self) -> float:
        """Энергия фотонов в единицах hepunits для расчета физической рентгеновской непрозрачности."""
        return self._xray_energy

    @xray_energy.setter
    def xray_energy(self, energy: float) -> None:
        self.set_xray_parameters(energy, self._xray_mode)

    def set_xray_parameters(self, energy: float, pseudo_xray_mode: bool) -> None:
        """
        Устанавливает параметры рентгеновской визуализации и обновляет акторы объемов.
        """
        if energy <= 0.0:
            raise ValueError(f"Энергия фотонов должна быть строго положительной (> 0), получено: {energy}")
        self._xray_energy = float(energy)
        self._xray_mode = bool(pseudo_xray_mode)
        if self.scene_vm is not None:
            for node_vm in self.scene_vm.all_nodes():
                if isinstance(node_vm, (VolumeViewModel, CollimatorViewModel)):
                    self.add_or_update_node_actor(node_vm)
                elif isinstance(node_vm, VoxelVolumeViewModel):
                    if self.voxel_renderer is not None and self.voxel_renderer.volume_actor is not None:
                        if node_vm.colormap_name == 'Physical Materials' or self._xray_mode:
                            core_voxel = node_vm.core_node
                            distribution_instance = (
                                core_voxel.material_distribution
                                if isinstance(core_voxel, WoodcockVoxelVolume)
                                else None
                            )
                            if distribution_instance is not None:
                                voxel_characteristic_length = float(np.mean(node_vm.voxel_size))
                                self.voxel_renderer.apply_material_transfer_functions(
                                    element_list=distribution_instance.element_list,
                                    pseudo_xray_mode=self._xray_mode,
                                    energy=self._xray_energy,
                                    characteristic_length=voxel_characteristic_length,
                                )
                        else:
                            self.voxel_renderer.set_colormap(node_vm.colormap_name)
                    else:
                        self.add_or_update_node_actor(node_vm)
            self.viewport.render()

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

    @property
    def data_manager_vm(self) -> Optional[DataManagerViewModel]:
        """Текущая модель представления диспетчера данных."""
        return self._data_manager_vm

    @data_manager_vm.setter
    def data_manager_vm(self, vm: Optional[DataManagerViewModel]) -> None:
        self.set_data_manager_viewmodel(vm)

    def set_data_manager_viewmodel(self, data_manager_vm: Optional[DataManagerViewModel]) -> None:
        """Привязка модели диспетчера данных и обновление подписок на чувствительные объемы."""
        if self._sensitive_conn is not None:
            try:
                QObject.disconnect(self._sensitive_conn)
            except (RuntimeError, TypeError):
                pass
            self._sensitive_conn = None
        self._data_manager_vm = data_manager_vm
        if self._data_manager_vm is not None:
            self._sensitive_conn = self._data_manager_vm.changed.connect(self.sync_viewport_scene)
            self.sync_viewport_scene()

    def set_scene_viewmodel(self, scene_vm: Optional[SceneViewModel]) -> None:
        """Привязка новой модели представления сцены с обновлением подписок."""
        if self._selected_node_vm is not None:
            self._set_node_edge_highlight(self._selected_node_vm, False)
            self._selected_node_vm = None
        self.disconnect_all_nodes()
        self.scene_vm = scene_vm
        if self.scene_vm is not None:
            self.sync_viewport_scene()
            if self.scene_vm.selected_node is not None:
                self.on_node_selected(self.scene_vm.selected_node)

    def sync_viewport_scene(self) -> None:
        """
        Полная синхронизация визуальных 3D-мешей в VTKViewport с графом SceneViewModel.
        """
        if self.viewport is None or self.scene_vm is None or self.scene_vm.root_vm is None:
            return

        all_nodes = self.scene_vm.all_nodes()
        current_actor_names = set()
        current_hole_actor_names = set()
        has_spect = False
        has_pet = False

        for node_vm in all_nodes:
            actor_name = f"mesh_{id(node_vm)}"
            current_actor_names.add(actor_name)
            if (
                isinstance(node_vm, VolumeViewModel)
                and isinstance(node_vm.core_node, Volume)
                and isinstance(node_vm.core_node.geometry, PeriodicHexPrism)
            ):
                current_hole_actor_names.add(f"holes_{id(node_vm)}")
            elif isinstance(node_vm, CollimatorViewModel) and isinstance(node_vm.core_node, Volume):
                current_hole_actor_names.add(f"holes_{id(node_vm)}")
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
        current_node_ids = {id(existing_node_vm) for existing_node_vm in all_nodes}
        for node_id in list(self._node_connections.keys()):
            if node_id not in current_node_ids:
                self.disconnect_node(node_id)

        for existing in list(self.viewport._actors.keys()):
            if existing.startswith("mesh_") and existing not in current_actor_names:
                self.viewport.remove_actor(existing)
            elif existing.startswith("holes_") and existing not in current_hole_actor_names:
                self.collimator_hole_renderer.remove_actor(existing)

        self.viewport.render()

    def add_or_update_node_actor(self, node_vm: NodeViewModel) -> None:
        """
        Добавление или обновление геометрического актора узла в 3D вьюпорте.
        """
        node_id = id(node_vm)
        parent_id = id(node_vm.parent_vm) if node_vm.parent_vm is not None else None
        self.viewport.set_node_local_transform(node_id, node_vm.local_matrix, parent_id=parent_id)

        actor_name = f"mesh_{node_id}"
        holes_actor_name = f"holes_{node_id}"

        if (
            isinstance(node_vm, VolumeViewModel)
            and isinstance(node_vm.core_node, Volume)
            and isinstance(node_vm.core_node.geometry, PeriodicHexPrism)
        ):
            # Каналы коллиматора: удаляем сплошной меш Box и визуализируем через CollimatorHoleRenderer
            self.viewport.remove_actor(actor_name)
            prism_geometry: PeriodicHexPrism = node_vm.core_node.geometry

            parent_material_name = "Pb"
            if node_vm.parent_vm is not None and isinstance(node_vm.parent_vm, (VolumeViewModel, CollimatorViewModel)):
                parent_material_name = node_vm.parent_vm.material_name
            elif isinstance(node_vm.core_node.parent, Volume):
                parent_material_name = node_vm.core_node.parent.material.name
            elif isinstance(node_vm.core_node.parent, DirectParallelCollimator):
                parent_material_name = node_vm.core_node.parent.material.name

            ((_, _), (hole_rgb, hole_opacity)) = get_collimator_visual_properties(
                body_material_name=parent_material_name,
                hole_material_name=node_vm.material_name,
                energy=self._xray_energy,
                pseudo_xray_mode=self._xray_mode,
            )

            self.collimator_hole_renderer.render_holes(
                actor_name=holes_actor_name,
                geometry=prism_geometry,
                global_matrix=None,
                hole_color=hole_rgb,
                hole_opacity=hole_opacity,
            )
            self.viewport.attach_actor_to_node(holes_actor_name, node_id)

        elif isinstance(node_vm, CollimatorViewModel) and isinstance(node_vm.core_node, DirectParallelCollimator):
            # Составной узел прямого коллиматора: рекурсивно обновляем акторы его дочерних узлов (lead_body и channels)
            for child_vm in node_vm.children:
                self.add_or_update_node_actor(child_vm)

        elif isinstance(node_vm, VolumeViewModel) or (isinstance(node_vm, CollimatorViewModel) and isinstance(node_vm.core_node, Volume)):
            volume_size = node_vm.size
            box = pv.Box(bounds=(-volume_size[0]/2, volume_size[0]/2, -volume_size[1]/2, volume_size[1]/2, -volume_size[2]/2, volume_size[2]/2))

            # Проверяем, является ли объем свинцовым корпусом коллиматора
            is_collimator_body = False
            if isinstance(node_vm, CollimatorViewModel):
                is_collimator_body = True
            elif isinstance(node_vm.parent_vm, CollimatorViewModel):
                is_collimator_body = True
            elif isinstance(node_vm.core_node.parent, DirectParallelCollimator):
                is_collimator_body = True
            elif isinstance(node_vm.core_node, CompositeNode) and any(
                isinstance(child_core.geometry, PeriodicHexPrism)
                for child_core in node_vm.core_node.childs
                if isinstance(child_core, Volume)
            ):
                is_collimator_body = True

            if self._xray_mode:
                rgb_color, calculated_opacity = get_pseudo_xray_rgba(
                    node_vm.material_name,
                    energy=self._xray_energy,
                )
            elif self.data_manager_vm is not None and self.data_manager_vm.is_sensitive_volume(node_vm):
                rgb_color = DETECTOR_ACCENT_COLOR
                calculated_opacity = DETECTOR_ACCENT_OPACITY
            elif is_collimator_body:
                rgb_color = get_material_color(node_vm.material_name)
                calculated_opacity = COLLIMATOR_BODY_OPACITY
            else:
                volume_color = node_vm.color
                if isinstance(volume_color, (tuple, list)) and len(volume_color) >= 3:
                    rgb_color = (float(volume_color[0]), float(volume_color[1]), float(volume_color[2]))
                else:
                    rgb_color = get_material_color(node_vm.material_name)
                calculated_opacity = get_material_opacity(
                    node_vm.material_name,
                    energy=self._xray_energy,
                )

            self.viewport.add_mesh_actor(actor_name, box, color=rgb_color, opacity=calculated_opacity)
            self.viewport.attach_actor_to_node(actor_name, node_id)

            if self._is_node_selected_or_child(node_vm):
                if isinstance(self.viewport, ISceneViewport):
                    self.viewport.set_actor_edge_highlight(
                        actor_name,
                        True,
                        color=SELECTED_EDGE_HIGHLIGHT_COLOR,
                        line_width=SELECTED_EDGE_HIGHLIGHT_WIDTH,
                    )

            if isinstance(node_vm, CollimatorViewModel) and isinstance(node_vm.core_node, Volume):
                ((_, _), (hole_rgb, hole_opacity)) = get_collimator_visual_properties(
                    body_material_name=node_vm.material_name,
                    hole_material_name=node_vm.hole_material_name or "Vacuum",
                    energy=self._xray_energy,
                    pseudo_xray_mode=self._xray_mode,
                )
                self.collimator_hole_renderer.render_holes(
                    actor_name=holes_actor_name,
                    geometry=node_vm.size,
                    global_matrix=None,
                    hole_diameter=node_vm.hole_diameter,
                    septa=node_vm.septa,
                    hole_shape=node_vm.hole_shape,
                    hole_color=hole_rgb,
                    hole_opacity=hole_opacity,
                )
                self.viewport.attach_actor_to_node(holes_actor_name, node_id)

        elif isinstance(node_vm, VoxelVolumeViewModel):
            core_voxel_node = node_vm.core_node
            dist = (
                core_voxel_node.material_distribution
                if isinstance(core_voxel_node, WoodcockVoxelVolume)
                else None
            )
            if dist is not None:
                data = np.asarray(dist.view(np.ndarray), dtype=np.float32)
                voxel_spacing_tuple: Tuple[float, float, float] = (
                    float(node_vm.voxel_size[0]),
                    float(node_vm.voxel_size[1]),
                    float(node_vm.voxel_size[2]),
                )
                self.voxel_renderer.set_volume_data(
                    data,
                    voxel_size=voxel_spacing_tuple,
                    origin=node_vm.origin,
                    element_list=dist.element_list,
                )
                if node_vm.colormap_name == 'Physical Materials' or self._xray_mode:
                    voxel_characteristic_length = float(np.mean(node_vm.voxel_size))
                    self.voxel_renderer.apply_material_transfer_functions(
                        element_list=dist.element_list,
                        pseudo_xray_mode=self._xray_mode,
                        energy=self._xray_energy,
                        characteristic_length=voxel_characteristic_length,
                    )
                else:
                    self.voxel_renderer.set_colormap(node_vm.colormap_name)
                    self.voxel_renderer.set_opacity_parameters(
                        max_opacity=float(node_vm.max_opacity),
                        threshold=float(node_vm.opacity_threshold),
                        preset=node_vm.opacity_preset
                    )
                self.voxel_renderer.set_lod_factor(float(node_vm.lod_factor))
                self.viewport.attach_actor_to_node(self.voxel_renderer.actor_name, node_id)

                if self._is_node_selected_or_child(node_vm):
                    self._create_or_update_voxel_selection_box(node_vm)
                else:
                    self._remove_voxel_selection_box(node_vm)

        elif isinstance(node_vm, SourceViewModel):
            if node_vm.is_point_source:
                sphere = pv.Sphere(radius=8.0)
                self.viewport.add_mesh_actor(actor_name, sphere, color=(1.0, 0.2, 0.2), opacity=0.85)
            else:
                raw_source_size = node_vm.size
                if any(dimension_val <= 0 for dimension_val in raw_source_size):
                    source_size_tuple: Tuple[float, float, float] = (50.0, 50.0, 50.0)
                else:
                    source_size_tuple = (
                        float(raw_source_size[0]),
                        float(raw_source_size[1]),
                        float(raw_source_size[2]),
                    )
                box = pv.Box(
                    bounds=(
                        -source_size_tuple[0] / 2.0,
                        source_size_tuple[0] / 2.0,
                        -source_size_tuple[1] / 2.0,
                        source_size_tuple[1] / 2.0,
                        -source_size_tuple[2] / 2.0,
                        source_size_tuple[2] / 2.0,
                    )
                )
                self.viewport.add_mesh_actor(actor_name, box, color=(1.0, 0.8, 0.1), opacity=0.35, style='wireframe')
            self.viewport.attach_actor_to_node(actor_name, node_id)

        elif isinstance(node_vm, DoseGridViewModel):
            raw_grid_size = node_vm.size
            if any(dimension_val <= 0 for dimension_val in raw_grid_size):
                grid_size_tuple: Tuple[float, float, float] = (100.0, 100.0, 100.0)
            else:
                grid_size_tuple = (
                    float(raw_grid_size[0]),
                    float(raw_grid_size[1]),
                    float(raw_grid_size[2]),
                )
            box = pv.Box(
                bounds=(
                    -grid_size_tuple[0] / 2.0,
                    grid_size_tuple[0] / 2.0,
                    -grid_size_tuple[1] / 2.0,
                    grid_size_tuple[1] / 2.0,
                    -grid_size_tuple[2] / 2.0,
                    grid_size_tuple[2] / 2.0,
                )
            )
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
            self.viewport.attach_actor_to_node(actor_name, node_id)

        elif isinstance(node_vm, PetScannerViewModel):
            self.pet_manipulator.set_parameters(
                diameter=float(node_vm.diameter),
                axial_length=float(node_vm.axial_length),
                num_sectors=int(node_vm.num_sectors),
            )
            self.viewport.update_actor_transform(self.pet_manipulator.actor_name, node_vm.global_matrix)

        # Подписка на изменение локальной матрицы и свойств узла для инкрементального обновления
        if node_id not in self._node_connections:
            conn1 = node_vm.local_transform_changed.connect(
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
        if self.collimator_hole_renderer is not None:
            self.collimator_hole_renderer.remove_actor(f"holes_{node_id}")
        self.viewport.remove_node_transform(node_id)

    def disconnect_all_nodes(self) -> None:
        """
        Отключение сигналов всех отслеживаемых узлов сцены и деактивация манипуляторов.
        """
        for node_id in list(self._node_connections.keys()):
            self.disconnect_node(node_id)
        self._node_connections.clear()
        if self.transform_gizmo is not None:
            self.transform_gizmo.detach()

    @staticmethod
    def _is_ancestor_or_self(ancestor_vm: NodeViewModel, target_vm: Optional[NodeViewModel]) -> bool:
        """Проверяет, является ли ancestor_vm самим узлом target_vm или его предком в иерархии."""
        current: Optional[NodeViewModel] = target_vm
        while current is not None:
            if current is ancestor_vm:
                return True
            current = current.parent_vm
        return False

    def _sync_non_hierarchical_descendants(self, node_vm: NodeViewModel) -> None:
        """
        Синхронизирует сервисные рендереры (PetScannerViewModel, DoseGridViewModel),
        не использующие VTK SetUserTransform, если они сами или их родитель изменили трансформацию.
        """
        if isinstance(node_vm, PetScannerViewModel) and self.pet_manipulator is not None:
            self.viewport.update_actor_transform(self.pet_manipulator.actor_name, node_vm.global_matrix)
        elif isinstance(node_vm, DoseGridViewModel) and self.dose_renderer is not None:
            self.dose_renderer.update_grid_transform(node_vm.name, node_vm.global_matrix)
            if "default" in self.dose_renderer.entries and len(self.dose_renderer.entries) == 1:
                self.dose_renderer.update_grid_transform("default", node_vm.global_matrix)
        for child_vm in node_vm.children:
            self._sync_non_hierarchical_descendants(child_vm)

    def on_node_transform_changed(self, node_vm: NodeViewModel) -> None:
        """
        Инкрементальное обновление локальной матрицы трансформации узла в C++ иерархии vtkTransform.
        Все дочерние акторы обновляют свои мировые координаты автоматически на стороне VTK.
        """
        node_id = id(node_vm)
        parent_id = id(node_vm.parent_vm) if node_vm.parent_vm is not None else None
        self.viewport.set_node_local_transform(node_id, node_vm.local_matrix, parent_id=parent_id)

        self._sync_non_hierarchical_descendants(node_vm)

        if (
            self.transform_gizmo is not None
            and not self.transform_gizmo.is_dragging
            and self._is_ancestor_or_self(node_vm, self.transform_gizmo.target_node)
        ):
            self.transform_gizmo.update_visuals(render=False)

    def on_node_property_changed(self, node_vm: NodeViewModel, prop_name: str, value: Any) -> None:
        """
        Инкрементальное обновление параметров актора при смене геометрии, цвета или физических свойств.
        """
        if isinstance(node_vm, DoseGridViewModel) and prop_name in ('is_active', 'dose_visible'):
            if self.dose_renderer is not None:
                is_grid_visible = bool(node_vm.is_active and node_vm.dose_visible)
                self.dose_renderer.set_grid_visible(node_vm.name, is_grid_visible)
                if "default" in self.dose_renderer.entries and len(self.dose_renderer.entries) == 1:
                    self.dose_renderer.set_grid_visible("default", is_grid_visible)

        if prop_name in ('size', 'color', 'voxel_size', 'is_point_source', 'file_path', 'dose_voxel_size', 'is_active', 'material_name', 'is_sensitive_detector', 'hole_diameter', 'hole_width', 'septa', 'hole_shape', 'hole_material_name'):
            self.add_or_update_node_actor(node_vm)
            self.viewport.request_render()
        elif prop_name == 'half_thickness' and isinstance(node_vm, GammaCameraViewModel):
            for sub_vm in (
                node_vm.detector_box_vm,
                node_vm.collimator_vm,
                node_vm.crystal_vm,
                node_vm.glass_backend_vm,
            ):
                if sub_vm is not None:
                    sub_parent_id = id(sub_vm.parent_vm) if sub_vm.parent_vm is not None else None
                    self.viewport.set_node_local_transform(id(sub_vm), sub_vm.local_matrix, parent_id=sub_parent_id)
        elif prop_name in ('diameter', 'axial_length', 'num_sectors') and isinstance(node_vm, PetScannerViewModel):
            self.pet_manipulator.set_parameters(
                diameter=float(node_vm.diameter),
                axial_length=float(node_vm.axial_length),
                num_sectors=int(node_vm.num_sectors),
            )
        elif prop_name == 'material_distribution' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                core_voxel_node = node_vm.core_node
                dist = (
                    core_voxel_node.material_distribution
                    if isinstance(core_voxel_node, WoodcockVoxelVolume)
                    else None
                )
                if dist is not None:
                    voxel_characteristic_length = float(np.mean(node_vm.voxel_size))
                    self.voxel_renderer.apply_material_transfer_functions(
                        element_list=dist.element_list,
                        pseudo_xray_mode=self._xray_mode,
                        energy=self._xray_energy,
                        characteristic_length=voxel_characteristic_length,
                    )
                    self.viewport.request_render()
        elif prop_name == 'colormap_name' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                if str(value) == 'Physical Materials' or self._xray_mode:
                    core_voxel_node = node_vm.core_node
                    dist = (
                        core_voxel_node.material_distribution
                        if isinstance(core_voxel_node, WoodcockVoxelVolume)
                        else None
                    )
                    if dist is not None:
                        voxel_characteristic_length = float(np.mean(node_vm.voxel_size))
                        self.voxel_renderer.apply_material_transfer_functions(
                            element_list=dist.element_list,
                            pseudo_xray_mode=self._xray_mode,
                            energy=self._xray_energy,
                            characteristic_length=voxel_characteristic_length,
                        )
                else:
                    self.voxel_renderer.set_colormap(str(value))
                    self.voxel_renderer.set_opacity_parameters(
                        max_opacity=float(node_vm.max_opacity),
                        threshold=float(node_vm.opacity_threshold),
                        preset=node_vm.opacity_preset
                    )
                self.viewport.request_render()
        elif prop_name == 'opacity_threshold' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_opacity_threshold(float(value))
                self.viewport.request_render()
        elif prop_name == 'max_opacity' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_max_opacity(float(value))
                self.viewport.request_render()
        elif prop_name == 'opacity_preset' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_opacity_preset(str(value))
                self.viewport.request_render()
        elif prop_name == 'lod_factor' and isinstance(node_vm, VoxelVolumeViewModel):
            if self.voxel_renderer is not None:
                self.voxel_renderer.set_lod_factor(float(value))
                self.viewport.request_render()

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
        if self._selected_node_vm is not None:
            current_ancestor: Optional[NodeViewModel] = self._selected_node_vm
            while current_ancestor is not None:
                if current_ancestor is node_vm:
                    self._selected_node_vm = None
                    self._remove_voxel_selection_box()
                    break
                current_ancestor = current_ancestor.parent_vm

        def _remove_recursive(vm: NodeViewModel) -> None:
            actor_name = f"mesh_{id(vm)}"
            self.viewport.remove_actor(actor_name)
            if isinstance(vm, VoxelVolumeViewModel):
                box_actor_name = f"selection_box_{id(vm)}"
                self.viewport.remove_actor(box_actor_name)
                if self.voxel_renderer is not None:
                    self.viewport.remove_actor(self.voxel_renderer.actor_name)
            elif isinstance(vm, DoseGridViewModel) and self.dose_renderer is not None:
                self.dose_renderer.remove_grid(vm.name, render=False)
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
            if isinstance(current_target, GammaCameraViewModel) and isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
                raw_r, camera_angle, pos_z, _ = extract_orbit_parameters(
                    current_target.local_matrix, half_thickness=current_target.half_thickness
                )
                camera_radius = max(0.0, raw_r)
                self.spect_manipulator.half_thickness = current_target.half_thickness
                self.spect_manipulator.set_orbit_parameters(
                    camera_radius,
                    camera_angle,
                    z=pos_z,
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

    def _is_node_selected_or_child(self, node_vm: NodeViewModel) -> bool:
        """Проверяет, является ли узел выделенным или потомком выделенного узла любой глубины вложенности."""
        if self._selected_node_vm is None:
            return False
        current_node_vm: Optional[NodeViewModel] = node_vm
        while current_node_vm is not None:
            if current_node_vm is self._selected_node_vm:
                return True
            current_node_vm = current_node_vm.parent_vm
        return False

    def _set_node_edge_highlight(
        self,
        node_vm: Optional[NodeViewModel],
        visible: bool,
        color: Tuple[float, float, float] = SELECTED_EDGE_HIGHLIGHT_COLOR,
        line_width: float = SELECTED_EDGE_HIGHLIGHT_WIDTH,
    ) -> None:
        """
        Рекурсивно включает или выключает подсветку контура (ребер) для узла
        и всех его потомков в графе сцены на любой глубине вложенности.
        """
        if node_vm is None or not isinstance(self.viewport, ISceneViewport):
            return

        def _apply_highlight_recursive(target_node_vm: NodeViewModel) -> None:
            actor_name = f"mesh_{id(target_node_vm)}"
            self.viewport.set_actor_edge_highlight(
                actor_name,
                visible,
                color=color,
                line_width=line_width,
            )
            for child_vm in target_node_vm.children:
                _apply_highlight_recursive(child_vm)

        _apply_highlight_recursive(node_vm)

    def _create_or_update_voxel_selection_box(self, node_vm: VoxelVolumeViewModel) -> None:
        """
        Создает или обновляет контурную янтарную рамку выделения вокруг воксельного фантома.
        """
        box_actor_name = f"selection_box_{id(node_vm)}"
        self._active_selection_box_actor = box_actor_name

        if self.voxel_renderer is not None and self.voxel_renderer.grid is not None:
            raw_bounds = self.voxel_renderer.grid.bounds
            box_bounds: Tuple[float, float, float, float, float, float] = (
                float(raw_bounds[0]),
                float(raw_bounds[1]),
                float(raw_bounds[2]),
                float(raw_bounds[3]),
                float(raw_bounds[4]),
                float(raw_bounds[5]),
            )
        else:
            voxel_dimensions = node_vm.dimensions
            if any(dim_val <= 0 for dim_val in voxel_dimensions):
                return
            voxel_origin = node_vm.origin
            voxel_spacing = node_vm.voxel_size
            box_bounds = (
                float(voxel_origin[0]),
                float(voxel_origin[0] + max(0, voxel_dimensions[0] - 1) * voxel_spacing[0]),
                float(voxel_origin[1]),
                float(voxel_origin[1] + max(0, voxel_dimensions[1] - 1) * voxel_spacing[1]),
                float(voxel_origin[2]),
                float(voxel_origin[2] + max(0, voxel_dimensions[2] - 1) * voxel_spacing[2]),
            )

        bounding_box = pv.Box(bounds=box_bounds)
        self.viewport.add_mesh_actor(
            box_actor_name,
            bounding_box,
            color=SELECTED_EDGE_HIGHLIGHT_COLOR,
            style='wireframe',
            line_width=SELECTED_EDGE_HIGHLIGHT_WIDTH,
        )
        node_id = id(node_vm)
        parent_id = id(node_vm.parent_vm) if node_vm.parent_vm is not None else None
        self.viewport.set_node_local_transform(node_id, node_vm.local_matrix, parent_id=parent_id)
        self.viewport.attach_actor_to_node(box_actor_name, node_id)

    def _remove_voxel_selection_box(self, node_vm: Optional[NodeViewModel] = None) -> None:
        """
        Удаляет актор контурной рамки выделения воксельного фантома.
        """
        if node_vm is not None:
            box_actor_name = f"selection_box_{id(node_vm)}"
            self.viewport.remove_actor(box_actor_name)
        if self._active_selection_box_actor is not None:
            self.viewport.remove_actor(self._active_selection_box_actor)
            self._active_selection_box_actor = None

    def _apply_voxel_selection_recursive(self, target_node_vm: NodeViewModel, is_selected: bool) -> None:
        """
        Рекурсивно включает или выключает контурную рамку выделения для воксельных фантомов.
        """
        if isinstance(target_node_vm, VoxelVolumeViewModel):
            if is_selected:
                self._create_or_update_voxel_selection_box(target_node_vm)
            else:
                self._remove_voxel_selection_box(target_node_vm)
        for child_node_vm in target_node_vm.children:
            self._apply_voxel_selection_recursive(child_node_vm, is_selected)

    def _is_root_volume(self, node_view_model: Optional[NodeViewModel]) -> bool:
        """Проверяет, является ли узел рутовым Volume сцены."""
        if node_view_model is None or not isinstance(node_view_model, VolumeViewModel):
            return False
        if self.scene_vm is not None:
            return self.scene_vm.is_root_volume(node_view_model)
        return node_view_model.is_root_volume

    def on_node_selected(self, selected_node_vm: Optional[NodeViewModel]) -> None:
        """Синхронизация подсветки ребер, ОФЭКТ/ПЭТ-манипулятора и Transform Gizmo при выборе узла в сцене."""
        if self._selected_node_vm is not None:
            self._set_node_edge_highlight(self._selected_node_vm, False)
            self._apply_voxel_selection_recursive(self._selected_node_vm, False)
        self._selected_node_vm = selected_node_vm
        if selected_node_vm is not None:
            self._set_node_edge_highlight(selected_node_vm, True)
            self._apply_voxel_selection_recursive(selected_node_vm, True)

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
        if constraint is None and self._is_root_volume(selected_node_vm):
            constraint = RootVolumeKinematicConstraint()
            selected_node_vm.set_self_kinematic_constraint(constraint)

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
            if constraint is not None:
                has_trans = len(constraint.get_allowed_axes(GizmoMode.TRANSLATE)) > 0
                has_rot = len(constraint.get_allowed_axes(GizmoMode.ROTATE)) > 0
                has_scale = constraint.is_scale_allowed()
                if not has_trans and not has_rot and not has_scale:
                    self.transform_gizmo.detach()
                else:
                    self.transform_gizmo.set_constraint(constraint)
                    forced_space = constraint.get_forced_space()
                    if forced_space is not None:
                        self.transform_gizmo.space = forced_space
                    self.transform_gizmo.set_target_node(selected_node_vm)
            else:
                self.transform_gizmo.set_constraint(None)
                self.transform_gizmo.set_target_node(selected_node_vm)

        # 3. Визуальные направляющие ОФЭКТ (круговая орбита):
        if isinstance(selected_node_vm, GammaCameraViewModel) and isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
            self.spect_manipulator.half_thickness = selected_node_vm.half_thickness
            raw_r, cam_angle, cam_z, _ = extract_orbit_parameters(
                selected_node_vm.local_matrix, half_thickness=selected_node_vm.half_thickness
            )
            cam_radius = max(10.0, raw_r)
            self.spect_manipulator.set_orbit_parameters(
                cam_radius,
                cam_angle,
                z=cam_z,
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


    @property
    def dose_volume_renderer(self) -> DoseVolumeRenderer:
        """Псевдоним рендерера объемной карты дозы."""
        return self.dose_renderer

    @staticmethod
    def _extract_camera_roll(camera_vm: GammaCameraViewModel) -> float:
        """Извлекает угол крена детектора (roll, в градусах) в собственной плоскости из local_matrix."""
        matrix = camera_vm.local_matrix
        if matrix is None:
            return 0.0
        _, _, _, roll = extract_orbit_parameters(matrix, half_thickness=camera_vm.half_thickness)
        return roll

    def on_spect_manipulator_changed(self, radius: float, angle: float, z_pos: float) -> None:
        """Обработка перемещения ОФЭКТ-манипулятора в 3D-пространстве."""
        if self.scene_vm is not None and isinstance(self.scene_vm.selected_node, GammaCameraViewModel):
            camera_vm = self.scene_vm.selected_node
            _, local_angle, _, roll_angle = extract_orbit_parameters(
                camera_vm.local_matrix, half_thickness=camera_vm.half_thickness
            )
            if isinstance(camera_vm.parent_vm, GantryViewModel):
                camera_vm.parent_vm.gantry_angle = angle
                camera_vm.local_matrix = compute_orbit_matrix(
                    radius=radius,
                    angle=local_angle,
                    z=z_pos,
                    half_thickness=camera_vm.half_thickness,
                    roll=roll_angle,
                )
            else:
                camera_vm.local_matrix = compute_orbit_matrix(
                    radius=radius,
                    angle=angle,
                    z=z_pos,
                    half_thickness=camera_vm.half_thickness,
                    roll=roll_angle,
                )

    def apply_job_angles_to_viewport(
        self,
        context: Dict[str, Any],
        procedure_vm: Optional[BaseProcedureViewModel] = None
    ) -> None:
        """Применяет углы из контекста задачи к станине гантри и гамма-камерам во вьюпорте."""
        if self.scene_vm is None:
            return

        active_procedure = procedure_vm if procedure_vm is not None else self.procedure_vm

        # Если передан индекс ракурса и активна процедура StepAndShoot без переопределения радиуса/угла, делегируем в preview_view
        if (
            "view_index" in context
            and isinstance(active_procedure, StepAndShootProcedureViewModel)
            and "radius" not in context
            and "gantry_angle" not in context
            and "current_angle" not in context
        ):
            self.preview_view(int(context["view_index"]) + 1, active_procedure)
            return

        all_nodes = self.scene_vm.all_nodes()
        camera_vms = [node for node in all_nodes if isinstance(node, GammaCameraViewModel)]
        gantry_vms = [node for node in all_nodes if isinstance(node, GantryViewModel)]
        if not camera_vms and not gantry_vms:
            return

        radius = 250.0
        has_explicit_radius = "radius" in context and context["radius"] is not None
        if has_explicit_radius:
            radius = float(context["radius"])
        elif isinstance(active_procedure, StepAndShootProcedureViewModel):
            radius = float(active_procedure.radius)

        is_radians_context = ("step_index" in context or "gantry_angle" in context)

        def _to_degrees(raw_val: float) -> float:
            if is_radians_context:
                return float(np.degrees(raw_val))
            return float(raw_val)

        previous_sync_state = active_procedure._is_syncing_with_scene if active_procedure is not None else False
        if active_procedure is not None:
            active_procedure._is_syncing_with_scene = True
        self.viewport.begin_update()
        try:
            raw_gantry_angle = context.get("gantry_angle", context.get("current_angle", context.get("head_0_angle")))
            target_gantry_angle: Optional[float] = (
                _to_degrees(float(raw_gantry_angle)) % 360.0
                if raw_gantry_angle is not None
                else None
            )

            for gantry_vm in gantry_vms:
                if target_gantry_angle is not None:
                    gantry_vm.gantry_angle = target_gantry_angle

            for camera_index, camera_view_model in enumerate(camera_vms):
                z_pos = 0.0
                roll_angle = 0.0
                local_cam_angle = 0.0
                if camera_view_model.local_matrix is not None:
                    _, local_cam_angle, z_pos, roll_angle = extract_orbit_parameters(
                        camera_view_model.local_matrix, half_thickness=camera_view_model.half_thickness
                    )

                if isinstance(active_procedure, StepAndShootProcedureViewModel) and camera_index < len(active_procedure.head_angles):
                    head_offset = float(active_procedure.head_angles[camera_index])
                elif len(camera_vms) > 1:
                    head_offset = (360.0 / len(camera_vms)) * camera_index
                else:
                    head_offset = local_cam_angle

                if isinstance(camera_view_model.parent_vm, GantryViewModel):
                    # Камера смонтирована на станине: поворот станины уже обновил global_matrix.
                    # Локальную матрицу обновляем только при явном изменении радиуса или монтажного угла.
                    explicit_head_angle = context.get(f"head_{camera_index}_angle")
                    if has_explicit_radius or explicit_head_angle is not None:
                        local_mount_angle = (
                            _to_degrees(float(explicit_head_angle)) % 360.0
                            if explicit_head_angle is not None
                            else head_offset
                        )
                        camera_view_model.local_matrix = compute_orbit_matrix(
                            radius=radius,
                            angle=local_mount_angle,
                            z=z_pos,
                            half_thickness=camera_view_model.half_thickness,
                            roll=roll_angle,
                        )
                else:
                    # Автономная камера без станины гантри
                    explicit_head_angle = context.get(f"head_{camera_index}_angle")
                    if explicit_head_angle is not None:
                        head_angle = _to_degrees(float(explicit_head_angle)) % 360.0
                    elif target_gantry_angle is not None:
                        head_angle = (target_gantry_angle + head_offset) % 360.0
                    else:
                        head_angle = head_offset

                    camera_view_model.local_matrix = compute_orbit_matrix(
                        radius=radius,
                        angle=head_angle,
                        z=z_pos,
                        half_thickness=camera_view_model.half_thickness,
                        roll=roll_angle,
                    )

            if isinstance(self.scene_vm.selected_node, GammaCameraViewModel):
                selected_camera = self.scene_vm.selected_node
                raw_r, local_angle, cam_z, _ = extract_orbit_parameters(
                    selected_camera.local_matrix, half_thickness=selected_camera.half_thickness
                )
                cam_radius = max(10.0, raw_r)
                cam_angle = (
                    selected_camera.parent_vm.gantry_angle
                    if isinstance(selected_camera.parent_vm, GantryViewModel)
                    else local_angle
                )
                self.spect_manipulator.set_orbit_parameters(
                    cam_radius,
                    cam_angle,
                    z=cam_z,
                    render=False,
                    emit_signal=False
                )

            self.viewport.request_render()
        finally:
            self.viewport.end_update()
            if active_procedure is not None:
                active_procedure._is_syncing_with_scene = previous_sync_state

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
        all_nodes = self.scene_vm.all_nodes()
        camera_vms = [node for node in all_nodes if isinstance(node, GammaCameraViewModel)]
        gantry_vms = [node for node in all_nodes if isinstance(node, GantryViewModel)]
        if not camera_vms and not gantry_vms:
            return 0.0

        base_angle = 0.0
        active_procedure = procedure_vm if procedure_vm is not None else self.procedure_vm
        if isinstance(active_procedure, StepAndShootProcedureViewModel):
            previous_sync_state = active_procedure._is_syncing_with_scene
            active_procedure._is_syncing_with_scene = True
            self.viewport.begin_update()
            try:
                radius = float(active_procedure.radius)
                calculated_poses = compute_spect_poses(
                    views_or_protocol=active_procedure.to_config(),
                    gamma_cameras=active_procedure.gamma_cameras,
                    start_angle=active_procedure.start_angle,
                    end_angle=active_procedure.end_angle,
                    head_angle_offsets=active_procedure.head_angles if active_procedure.head_angles else None,
                    endpoint=active_procedure.endpoint,
                )
                pose_index = min(view_index, len(calculated_poses) - 1) if calculated_poses else 0
                angles = calculated_poses[pose_index] if calculated_poses else [0.0] * max(1, len(camera_vms))
                base_angle = float(angles[0]) if angles else 0.0
                head_offset_0 = float(active_procedure.head_angles[0]) if active_procedure.head_angles else 0.0
                gantry_rot_angle = (base_angle - head_offset_0) % 360.0

                for gantry_vm in gantry_vms:
                    gantry_vm.gantry_angle = gantry_rot_angle

                for camera_index, cam_vm in enumerate(camera_vms):
                    head_offset = (
                        float(active_procedure.head_angles[camera_index])
                        if camera_index < len(active_procedure.head_angles)
                        else (360.0 / max(1, len(camera_vms))) * camera_index
                    )
                    z_pos = 0.0
                    roll_angle = 0.0
                    if cam_vm.local_matrix is not None:
                        _, _, z_pos, roll_angle = extract_orbit_parameters(
                            cam_vm.local_matrix, half_thickness=cam_vm.half_thickness
                        )

                    if isinstance(cam_vm.parent_vm, GantryViewModel):
                        cam_vm.local_matrix = compute_orbit_matrix(
                            radius=radius,
                            angle=head_offset,
                            z=z_pos,
                            half_thickness=cam_vm.half_thickness,
                            roll=roll_angle,
                        )
                    else:
                        angle_val = float(angles[camera_index]) if camera_index < len(angles) else (gantry_rot_angle + head_offset) % 360.0
                        cam_vm.local_matrix = compute_orbit_matrix(
                            radius=radius,
                            angle=angle_val,
                            z=z_pos,
                            half_thickness=cam_vm.half_thickness,
                            roll=roll_angle,
                        )

                if isinstance(self.scene_vm.selected_node, GammaCameraViewModel):
                    selected_camera = self.scene_vm.selected_node
                    raw_r, local_angle, cam_z, _ = extract_orbit_parameters(
                        selected_camera.local_matrix, half_thickness=selected_camera.half_thickness
                    )
                    cam_radius = max(10.0, raw_r)
                    cam_angle = (
                        selected_camera.parent_vm.gantry_angle
                        if isinstance(selected_camera.parent_vm, GantryViewModel)
                        else local_angle
                    )
                    self.spect_manipulator.set_orbit_parameters(
                        cam_radius,
                        cam_angle,
                        z=cam_z,
                        render=False,
                        emit_signal=False
                    )

                self.viewport.request_render()
            finally:
                self.viewport.end_update()
                active_procedure._is_syncing_with_scene = previous_sync_state
        return base_angle

    def on_dose_volume_received(
        self,
        dose_data: Union[np.ndarray, Dict[str, np.ndarray]],
        session: Optional[Union[IDoseGeometryProvider, float, Any]] = None,
        simulation_time: float = 0.0,
    ) -> None:
        """
        Прием и отображение очередного снимка 3D-карты дозы (одиночного массива или словаря сеток).
        Поддерживает одновременную визуализацию всех активных и видимых сеток DoseGridViewModel.
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

        if isinstance(dose_data, dict):
            if not dose_data:
                return

            dose_nodes_by_name: Dict[str, DoseGridViewModel] = {}
            if self.scene_vm is not None:
                for node in self.scene_vm.all_nodes():
                    if isinstance(node, DoseGridViewModel):
                        dose_nodes_by_name[node.name] = node

            if "default" in self.dose_renderer.entries and "default" not in dose_data:
                self.dose_renderer.remove_grid("default", render=False)

            for existing_grid_name in list(self.dose_renderer.entries.keys()):
                if existing_grid_name in dose_nodes_by_name:
                    existing_vm = dose_nodes_by_name[existing_grid_name]
                    if not (existing_vm.is_active and existing_vm.dose_visible):
                        self.dose_renderer.set_grid_visible(existing_grid_name, False, render=False)

            has_updated_grids = False
            for grid_name, raw_arr in dose_data.items():
                grid_vm = dose_nodes_by_name.get(grid_name)
                if grid_vm is not None:
                    if not (grid_vm.is_active and grid_vm.dose_visible):
                        self.dose_renderer.set_grid_visible(grid_name, False, render=False)
                        continue
                    self.dose_renderer.set_grid_visible(grid_name, True, render=False)
                    grid_voxel_size: Any = float(grid_vm.dose_voxel_size)
                    grid_origin: Optional[Tuple[float, float, float]] = grid_vm.origin
                    grid_transform: Optional[np.ndarray] = grid_vm.global_matrix
                    self._active_dose_voxel_size = float(grid_vm.dose_voxel_size)
                    self._active_dose_origin = grid_origin
                    self._active_dose_transform_matrix = grid_transform
                else:
                    grid_voxel_size = self._active_dose_voxel_size
                    grid_origin = self._active_dose_origin
                    grid_transform = self._active_dose_transform_matrix

                self.dose_renderer.update_dose_data(
                    np.asarray(raw_arr),
                    voxel_size=grid_voxel_size,
                    origin=grid_origin,
                    transform_matrix=grid_transform,
                    grid_name=grid_name,
                    render=False,
                )
                has_updated_grids = True

            selected_vm = (
                self._selected_node_vm
                if self._selected_node_vm is not None
                else (self.scene_vm.selected_node if self.scene_vm is not None else None)
            )
            if isinstance(selected_vm, DoseGridViewModel) and selected_vm.name in self.dose_renderer.entries:
                self.dose_renderer._active_grid_name = selected_vm.name
                self._active_dose_voxel_size = float(selected_vm.dose_voxel_size)
                self._active_dose_origin = selected_vm.origin
                self._active_dose_transform_matrix = selected_vm.global_matrix

            if has_updated_grids:
                self.dose_renderer.request_render()
            return

        dose_array = np.asarray(dose_data)
        if self._active_dose_origin is None and self.scene_vm is not None:
            selected_vm = (
                self._selected_node_vm
                if self._selected_node_vm is not None
                else self.scene_vm.selected_node
            )
            if isinstance(selected_vm, DoseGridViewModel):
                self._active_dose_voxel_size = float(selected_vm.dose_voxel_size)
                self._active_dose_origin = selected_vm.origin
                self._active_dose_transform_matrix = selected_vm.global_matrix
            else:
                for node_vm in self.scene_vm.all_nodes():
                    if isinstance(node_vm, DoseGridViewModel) and node_vm.is_active:
                        self._active_dose_voxel_size = float(node_vm.dose_voxel_size)
                        self._active_dose_origin = node_vm.origin
                        self._active_dose_transform_matrix = node_vm.global_matrix
                        break

        self.dose_renderer.update_dose_data(
            dose_array,
            voxel_size=self._active_dose_voxel_size,
            origin=self._active_dose_origin,
            transform_matrix=self._active_dose_transform_matrix,
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


__all__ = [
    "IDoseGeometryProvider",
    "SceneViewportController",
]

