import logging
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Sequence, Set, Tuple, Union
import weakref
import numpy as np
import hepunits as units
from PySide6.QtCore import QObject, Signal

from core.scene.nodes import SpatialNode, CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from core.geometry.volumes import Volume
from core.materials.materials import Material, MaterialArray
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.parametric_collimators import ParametricParallelCollimator, ParametricParallelSquareCollimator
from core.source.sources import Source, PointSource
import settings.database_setting as database_setting
from core.other.typing_definitions import Float
from gui.viewmodels.decorators import core_field, gui_field

_logger = logging.getLogger(__name__)


class NodeViewModel(QObject):
    """
    Базовая модель представления для узла графа сцены (паттерн MVVM).
    Инкапсулирует SpatialNode или CompositeNode ядра, предоставляя
    Qt-сигналы для синхронизации с 3D-вьюпортом и инспектором свойств.
    """

    property_changed = Signal(str, object)
    child_added = Signal(object)
    child_removed = Signal(object)
    transform_changed = Signal()

    name = core_field('name', default='Node')
    visible = gui_field(default=True)

    def __init__(self, core_node: SpatialNode, parent_vm: Optional['NodeViewModel'] = None) -> None:
        super().__init__()
        self.core_node = core_node
        self.parent_vm = parent_vm
        self.children: List['NodeViewModel'] = []

        # Инициализация дочерних узлов, если core_node является CompositeNode
        if isinstance(core_node, CompositeNode):
            for child_core in core_node.childs:
                if child_core.parent is not core_node:
                    child_core.parent = core_node
                child_vm = create_node_viewmodel(child_core, parent_vm=self)
                self.children.append(child_vm)

    def _notify_transform_changed(self) -> None:
        """
        Испускает сигнал transform_changed для текущего узла и рекурсивно
        уведомляет всех потомков, так как их эффективная global_matrix изменилась.
        """
        self.transform_changed.emit()
        for child in self.children:
            child._notify_transform_changed()

    @property
    def node_type(self) -> str:
        """
        Человекочитаемый тип узла сцены.
        """
        return self.core_node.__class__.__name__

    @property
    def local_matrix(self) -> np.ndarray:
        return self.core_node.local_matrix

    @local_matrix.setter
    def local_matrix(self, matrix: np.ndarray) -> None:
        self.core_node.local_matrix = np.asarray(matrix, dtype=self.core_node.local_matrix.dtype)
        self.core_node.invalidate_matrix_cache()
        self._notify_transform_changed()
        self.property_changed.emit('local_matrix', self.core_node.local_matrix)

    @property
    def global_matrix(self) -> np.ndarray:
        return self.core_node.global_matrix

    def translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, in_local: bool = False) -> None:
        """
        Перемещение узла с уведомлением подписчиков.
        """
        self.core_node.translate(x=x, y=y, z=z, in_local=in_local)
        self._notify_transform_changed()
        self.property_changed.emit('local_matrix', self.core_node.local_matrix)

    def rotate(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        rotation_center: Sequence[float] = (0.0, 0.0, 0.0),
        in_local: bool = False
    ) -> None:
        """
        Вращение узла с уведомлением подписчиков.
        """
        self.core_node.rotate(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rotation_center=rotation_center,
            in_local=in_local
        )
        self._notify_transform_changed()
        self.property_changed.emit('local_matrix', self.core_node.local_matrix)

    def add_child(self, child_vm: 'NodeViewModel') -> None:
        """
        Добавление дочернего ViewModel и соответствующего узла в ядро.
        """
        if not isinstance(self.core_node, CompositeNode):
            raise TypeError("Cannot add child to a non-composite node")
        if child_vm is self:
            raise ValueError("Cannot add node as a child of itself")

        # Проверка на циклические зависимости
        curr: Optional['NodeViewModel'] = self
        while curr is not None:
            if curr is child_vm:
                raise ValueError("Cannot add an ancestor as a child (cycle detected)")
            curr = curr.parent_vm

        # Если узел уже является дочерним для self, повторно не добавляем
        if child_vm.parent_vm is self and child_vm in self.children and child_vm.core_node in self.core_node.childs:
            return

        # Если узел уже имел другого родителя в дереве ViewModel, отсоединяем
        if child_vm.parent_vm is not None and child_vm.parent_vm is not self:
            child_vm.parent_vm.remove_child(child_vm)
        elif child_vm.core_node.parent is not None and child_vm.core_node.parent is not self.core_node:
            if isinstance(child_vm.core_node.parent, CompositeNode):
                child_vm.core_node.parent.remove_child(child_vm.core_node)

        self.core_node.add_child(child_vm.core_node)

        child_vm.parent_vm = self
        if child_vm not in self.children:
            self.children.append(child_vm)
        child_vm._notify_transform_changed()
        self.child_added.emit(child_vm)

    def remove_child(self, child_vm: 'NodeViewModel') -> None:
        """
        Удаление дочернего ViewModel и отсоединение узла из ядра.
        """
        if not isinstance(self.core_node, CompositeNode):
            raise TypeError("Cannot remove child from a non-composite node")
        if child_vm in self.children:
            self.core_node.remove_child(child_vm.core_node)
            child_vm.parent_vm = None
            self.children.remove(child_vm)
            child_vm._notify_transform_changed()
            self.child_removed.emit(child_vm)

    def sync_children_from_core(self) -> None:
        """
        Синхронизация списка children ViewModel со списком core_node.childs
        при изменениях графа со стороны ядра.
        """
        if not isinstance(self.core_node, CompositeNode):
            for child in list(self.children):
                child.parent_vm = None
                if child.core_node.parent is self.core_node:
                    child.core_node.parent = None
                    child.core_node.invalidate_matrix_cache()
                child._notify_transform_changed()
                self.child_removed.emit(child)
            self.children.clear()
            return

        core_child_map = {id(c): c for c in self.core_node.childs}
        current_vms = {id(vm.core_node): vm for vm in list(self.children)}

        # Удаление узлов, которых больше нет в core_node.childs
        for core_id, vm in list(current_vms.items()):
            if core_id not in core_child_map:
                self.children.remove(vm)
                vm.parent_vm = None
                if vm.core_node.parent is self.core_node:
                    vm.core_node.parent = None
                    vm.core_node.invalidate_matrix_cache()
                vm._notify_transform_changed()
                self.child_removed.emit(vm)

        # Добавление новых узлов или актуализация существующих
        for child_core in self.core_node.childs:
            if child_core.parent is not self.core_node:
                child_core.parent = self.core_node

            if id(child_core) not in current_vms:
                child_vm = create_node_viewmodel(child_core, parent_vm=self)
                self.children.append(child_vm)
                child_vm._notify_transform_changed()
                self.child_added.emit(child_vm)
            else:
                existing_vm = current_vms[id(child_core)]
                if existing_vm.parent_vm is not self:
                    existing_vm.parent_vm = self
                existing_vm.sync_children_from_core()

        # Синхронизация порядка children с core_node.childs
        core_order = {id(c): idx for idx, c in enumerate(self.core_node.childs)}
        self.children.sort(key=lambda vm: core_order.get(id(vm.core_node), 0))


class VolumeViewModel(NodeViewModel):
    """
    ViewModel для геометрического объема Volume.
    """
    color = gui_field(default=(0.5, 0.7, 1.0, 0.4))
    _sensitive_core_volumes: ClassVar[weakref.WeakSet[Volume]] = weakref.WeakSet()

    def __init__(self, core_node: Volume, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        self._is_sensitive_detector: bool = False

    @classmethod
    def get_sensitive_volumes(cls) -> Set[Volume]:
        """Возвращает набор узлов Volume расчетного ядра, помеченных в GUI как чувствительные детекторы."""
        return set(cls._sensitive_core_volumes)

    @classmethod
    def clear_sensitive_volumes(cls) -> None:
        """Очищает глобальный реестр чувствительных объемов ядра."""
        cls._sensitive_core_volumes.clear()

    @property
    def local_bound(self) -> np.ndarray:
        """Локальные габариты геометрии объема [Lx, Ly, Lz]."""
        return np.asarray(self.core_node.local_bound, dtype=float)

    @property
    def is_sensitive_detector(self) -> bool:
        return self._is_sensitive_detector

    @is_sensitive_detector.setter
    def is_sensitive_detector(self, val: bool) -> None:
        val_bool = bool(val)
        if self._is_sensitive_detector == val_bool:
            return
        self._is_sensitive_detector = val_bool
        if val_bool and isinstance(self.core_node, Volume):
            VolumeViewModel._sensitive_core_volumes.add(self.core_node)
        elif isinstance(self.core_node, Volume):
            VolumeViewModel._sensitive_core_volumes.discard(self.core_node)
        self.property_changed.emit('is_sensitive_detector', val_bool)

    @property
    def material_name(self) -> str:
        mat = self.core_node.material if isinstance(self.core_node, Volume) else None
        return mat.name if mat is not None else "Vacuum"

    @material_name.setter
    def material_name(self, name: str) -> None:
        if name == "Vacuum":
            self.core_node.material = Material(name="Vacuum")
        elif name in database_setting.material_database:
            self.core_node.material = database_setting.material_database[name]
        else:
            raise KeyError(f"Material '{name}' is not found in the material database.")
        self.core_node.invalidate_geometry()
        self.property_changed.emit('material_name', name)

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.core_node.size, dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        new_size_arr = np.asarray(new_size, dtype=float)
        self.core_node.size = new_size_arr
        self.property_changed.emit('size', new_size_arr)


class CollimatorViewModel(VolumeViewModel):
    """
    Базовая модель представления для коллиматоров гамма-камер.
    """
    def __init__(self, core_node: Volume, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        self.color = (0.35, 0.35, 0.35, 0.7)

    @property
    def collimator_type(self) -> str:
        """Человекочитаемый тип геометрии каналов коллиматора."""
        if isinstance(self.core_node, ParametricParallelSquareCollimator):
            return "Квадратный (CZT)"
        elif isinstance(self.core_node, ParametricParallelCollimator):
            return "Гексагональный (LEHR/LEGP)"
        return "Параллельный"

    @property
    def septa_thickness(self) -> float:
        """Толщина септ коллиматора в мм."""
        if isinstance(self.core_node, (ParametricParallelCollimator, ParametricParallelSquareCollimator)):
            return float(self.core_node.septa)
        return 0.2

    @septa_thickness.setter
    def septa_thickness(self, val: float) -> None:
        v = float(val)
        if isinstance(self.core_node, (ParametricParallelCollimator, ParametricParallelSquareCollimator)):
            self.core_node.septa = Float(v)
            self.property_changed.emit('septa_thickness', v)


class ParametricParallelCollimatorViewModel(CollimatorViewModel):
    """
    Модель представления для коллиматора с круглыми/гексагональными каналами (LEHR, LEGP).
    """
    def __init__(self, core_node: ParametricParallelCollimator, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)

    @property
    def hole_diameter(self) -> float:
        """Диаметр отверстий коллиматора в мм."""
        if isinstance(self.core_node, ParametricParallelCollimator):
            return float(self.core_node.hole_diameter)
        return 1.5

    @hole_diameter.setter
    def hole_diameter(self, val: float) -> None:
        if isinstance(self.core_node, ParametricParallelCollimator):
            v = float(val)
            self.core_node.hole_diameter = Float(v)
            self.property_changed.emit('hole_diameter', v)


class ParametricParallelSquareCollimatorViewModel(CollimatorViewModel):
    """
    Модель представления для коллиматора с квадратными каналами (CZT).
    """
    def __init__(self, core_node: ParametricParallelSquareCollimator, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)

    @property
    def hole_width(self) -> float:
        """Ширина квадратного отверстия коллиматора в мм."""
        if isinstance(self.core_node, ParametricParallelSquareCollimator):
            return float(self.core_node.hole_width)
        return 1.5

    @hole_width.setter
    def hole_width(self, val: float) -> None:
        if isinstance(self.core_node, ParametricParallelSquareCollimator):
            v = float(val)
            self.core_node.hole_width = Float(v)
            self.property_changed.emit('hole_width', v)


class VoxelVolumeViewModel(NodeViewModel):
    """
    ViewModel для воксельного фантома WoodcockVoxelVolume.
    """
    colormap_name = gui_field(default='Hot Iron')
    lod_factor = gui_field(default=1.0)
    opacity_threshold = gui_field(default=0.05)
    max_opacity = gui_field(default=0.4)
    opacity_preset = gui_field(default='air_cutoff')
    file_path = gui_field(default='')

    def __init__(self, core_node: WoodcockVoxelVolume, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        if core_node.distribution_path:
            self.file_path = str(core_node.distribution_path)

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.core_node.size, dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        new_size_arr = np.asarray(new_size, dtype=float)
        self.core_node.size = new_size_arr
        dims = self.dimensions
        if all(d > 0 for d in dims):
            new_voxel_size = new_size_arr / np.asarray(dims, dtype=float)
            self.core_node.voxel_size = new_voxel_size
        self.core_node.invalidate_geometry()
        self.property_changed.emit('size', new_size_arr)

    @property
    def voxel_size(self) -> np.ndarray:
        return np.asarray(self.core_node.voxel_size)

    @voxel_size.setter
    def voxel_size(self, value: Union[float, Sequence[float]]) -> None:
        val_arr = np.asarray(value, dtype=float)
        dist = self.core_node.material_distribution
        if dist is not None:
            self.core_node.geometry.size = np.asarray(dist.shape, dtype=float) * val_arr
        self.core_node.voxel_size = val_arr
        self.core_node.invalidate_geometry()
        self.property_changed.emit('voxel_size', val_arr)
        self.property_changed.emit('size', self.size)

    @property
    def dimensions(self) -> Tuple[int, ...]:
        dist = self.core_node.material_distribution
        if dist is not None:
            return tuple(dist.shape)
        return (0, 0, 0)

    @property
    def origin(self) -> Tuple[float, float, float]:
        """
        Возвращает смещение начала координат сетки фантома для центрирования в локальной СК.
        """
        sp = self.voxel_size
        dims = self.dimensions
        return tuple(-0.5 * d * s for d, s in zip(dims, sp))

    def reload_distribution(self, path: str, shape: Optional[Tuple[int, ...]] = None, order: str = 'F') -> bool:
        """
        Перезагрузка матрицы фантома из файла (.npy, .dat, .raw).
        """
        p = Path(path)
        if not p.exists():
            return False
        try:
            if p.suffix.lower() == '.npy':
                data = np.load(p, allow_pickle=True)
            else:
                s = shape or self.dimensions
                try:
                    data = np.loadtxt(p).reshape(s, order=order)
                except (ValueError, OSError):
                    data = np.fromfile(p, dtype=np.float32).reshape(s, order=order)

            if isinstance(data, MaterialArray):
                mat_arr = data
            else:
                mat_arr = MaterialArray(data.shape)
                existing_list: List[Material] = []
                if self.core_node.material_distribution is not None:
                    existing_list = list(self.core_node.material_distribution.element_list)

                mdb = database_setting.material_database
                if not existing_list or len(existing_list) <= 1:
                    existing_list = [
                        Material(name='Vacuum', ID=0),
                        mdb.get('Water, Liquid', Material(name='Water', ID=1)),
                        mdb.get('Tissue, Soft (ICRU-44)', Material(name='Tissue', ID=2)),
                        mdb.get('Bone, Cortical (ICRU-44)', Material(name='Bone', ID=3)),
                        mdb.get('Lung (ICRP)', Material(name='Lung', ID=4)),
                        mdb.get('Adipose Tissue (ICRU-44)', Material(name='Adipose', ID=5)),
                    ]

                int_data = data.astype(int)
                max_val = int(np.nanmax(int_data)) if int_data.size > 0 else 0
                all_mats = list(mdb.values())
                mat_idx = 0
                while len(existing_list) <= max_val:
                    if mat_idx < len(all_mats):
                        cand = all_mats[mat_idx]
                        if cand not in existing_list:
                            existing_list.append(cand)
                        mat_idx += 1
                    else:
                        new_id = len(existing_list)
                        existing_list.append(Material(name=f"Material_{new_id}", ID=new_id))

                mat_arr.element_list = existing_list
                mat_arr.view(np.ndarray)[:] = int_data

            self.core_node.material_distribution = mat_arr
            self.core_node.distribution_path = str(p)
            new_size = np.asarray(mat_arr.shape, dtype=float) * np.asarray(self.core_node.voxel_size, dtype=float)
            self.core_node.size = new_size
            self.core_node.invalidate_geometry()

            self.file_path = str(p)
            self.property_changed.emit('file_path', self.file_path)
            self.property_changed.emit('size', new_size)
            self.property_changed.emit('voxel_size', self.voxel_size)
            return True
        except (OSError, ValueError, TypeError, KeyError) as e:
            _logger.warning(f"Ошибка загрузки файла фантома: {e}")
            return False


class GammaCameraViewModel(VolumeViewModel):
    """
    ViewModel для ОФЭКТ гамма-камеры с поддержкой параметров орбиты.
    """
    orbit_radius = gui_field(default=250.0)
    orbit_angle = gui_field(default=0.0)
    orbit_z = gui_field(default=0.0)

    def __init__(self, core_node: GammaCamera, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        self._sync_orbit_params_from_matrix()
        det_vm = self.detector_vm
        if det_vm is not None:
            det_vm.is_sensitive_detector = True

    @property
    def detector_vm(self) -> Optional[VolumeViewModel]:
        """
        ViewModel чувствительного объема детектора гамма-камеры.
        """
        if isinstance(self.core_node, GammaCamera):
            det_core = self.core_node.detector
            stack = list(self.children)
            while stack:
                curr = stack.pop()
                if curr.core_node is det_core and isinstance(curr, VolumeViewModel):
                    return curr
                stack.extend(curr.children)
        return None

    @property
    def collimator_vm(self) -> Optional[VolumeViewModel]:
        """
        ViewModel коллиматора гамма-камеры.
        """
        if isinstance(self.core_node, GammaCamera):
            col_core = self.core_node.collimator
            stack = list(self.children)
            while stack:
                curr = stack.pop()
                if curr.core_node is col_core and isinstance(curr, VolumeViewModel):
                    return curr
                stack.extend(curr.children)
        return None

    @property
    def gap(self) -> float:
        """Внутренний зазор между коллиматором и детектором (мм)."""
        if isinstance(self.core_node, GammaCamera):
            return float(self.core_node.gap)
        return 1.0

    @gap.setter
    def gap(self, val: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            v = float(val)
            self.core_node.gap = Float(v)
            self.property_changed.emit('gap', v)
            self.property_changed.emit('size', self.size)

    @property
    def shielding_thickness(self) -> float:
        """Толщина свинцовой защиты корпуса гамма-камеры (мм)."""
        if isinstance(self.core_node, GammaCamera):
            return float(self.core_node.shielding_thickness)
        return 20.0

    @shielding_thickness.setter
    def shielding_thickness(self, val: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            v = float(val)
            self.core_node.shielding_thickness = Float(v)
            self.property_changed.emit('shielding_thickness', v)
            self.property_changed.emit('size', self.size)

    @property
    def glass_backend_thickness(self) -> float:
        """Толщина подложки оптического стекла (мм)."""
        if isinstance(self.core_node, GammaCamera):
            return float(self.core_node.glass_backend_thickness)
        return 50.0

    @glass_backend_thickness.setter
    def glass_backend_thickness(self, val: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            v = float(val)
            self.core_node.glass_backend_thickness = Float(v)
            self.property_changed.emit('glass_backend_thickness', v)
            self.property_changed.emit('size', self.size)

    def _sync_orbit_params_from_matrix(self) -> None:
        """
        Синхронизирует параметры орбиты (orbit_radius, orbit_angle, orbit_z)
        из текущей матрицы local_matrix ядра.
        Радиус орбиты отсчитывается до лицевой поверхности гамма-камеры.
        """
        if self.core_node.local_matrix is not None:
            x = float(self.core_node.local_matrix[0, 3])
            y = float(self.core_node.local_matrix[1, 3])
            z = float(self.core_node.local_matrix[2, 3])
            r = float(np.hypot(x, y))
            self.orbit_z = z
            if r > 1e-4:
                self.orbit_radius = max(0.0, r - self.half_thickness)
                ang = float(np.degrees(np.arctan2(y, x)) % 360.0)
                if np.isclose(ang, 360.0) or np.isclose(ang, 0.0):
                    ang = 0.0
                elif np.isclose(ang, round(ang), atol=1e-5):
                    ang = float(round(ang, 5))
                self.orbit_angle = ang

    @NodeViewModel.local_matrix.setter
    def local_matrix(self, matrix: np.ndarray) -> None:
        NodeViewModel.local_matrix.fset(self, matrix)
        self._sync_orbit_params_from_matrix()

    def translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, in_local: bool = False) -> None:
        super().translate(x=x, y=y, z=z, in_local=in_local)
        self._sync_orbit_params_from_matrix()

    def rotate(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        rotation_center: Sequence[float] = (0.0, 0.0, 0.0),
        in_local: bool = False
    ) -> None:
        super().rotate(alpha=alpha, beta=beta, gamma=gamma, rotation_center=rotation_center, in_local=in_local)
        self._sync_orbit_params_from_matrix()

    @property
    def half_thickness(self) -> float:
        """
        Половина толщины гамма-камеры вдоль оси Z (мм).
        Лицевая поверхность коллиматора/камеры находится на расстоянии half_thickness
        от геометрического центра камеры в направлении нормали (+Z, к центру орбиты).
        """
        sz = self.size
        return float(sz[2]) / 2.0 if len(sz) >= 3 and sz[2] > 0 else 0.0

    @staticmethod
    def compute_orbit_matrix(radius: float, angle_deg: float, z: float = 0.0, half_thickness: float = 0.0) -> np.ndarray:
        """
        Вычисляет кинематическую матрицу трансформации 4x4 для круговой орбиты ОФЭКТ,
        ориентирующую гамма-камеру к центру орбиты (0, 0, z).
        Делегирует физико-математический расчет классу ядра GammaCamera.
        """
        return GammaCamera.compute_orbit_matrix(radius=radius, angle_deg=angle_deg, z=z, half_thickness=half_thickness)

    def set_orbit_position(self, radius: float, angle_deg: float, z: float = 0.0) -> None:
        """
        Установка положения гамма-камеры на круговой орбите (ОФЭКТ манипулятор).
        Радиус орбиты задается до лицевой поверхности гамма-камеры.
        """
        self.orbit_radius = float(radius)
        self.orbit_angle = float(angle_deg % 360.0)
        self.orbit_z = float(z)
        self.local_matrix = self.compute_orbit_matrix(radius, angle_deg, z, half_thickness=self.half_thickness)


class PetScannerViewModel(VolumeViewModel):
    """
    ViewModel для ПЭТ-сканера с кольцевой геометрией детекторов.
    """
    diameter = gui_field(default=600.0)
    axial_length = gui_field(default=200.0)
    num_sectors = gui_field(default=32)

    def __init__(self, core_node: Any, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)


def _on_source_activity_change(instance: Any, value: float) -> None:
    if isinstance(instance.core_node, Source):
        instance.core_node.initial_activity = Float(float(value) * 1e6 * units.Bq)


def _on_source_energy_change(instance: Any, value: float) -> None:
    if isinstance(instance.core_node, Source):
        en_val = float(value) * units.keV
        instance.core_node.energy = np.zeros(1, dtype=[("energy", Float), ("probability", Float)])
        instance.core_node.energy["energy"] = Float(en_val)
        instance.core_node.energy["probability"] = Float(1.0)


def _on_source_radiation_type_change(instance: Any, value: str) -> None:
    if isinstance(instance.core_node, Source):
        instance.core_node.radiation_type = str(value)


def _on_source_half_life_change(instance: Any, value: float) -> None:
    if isinstance(instance.core_node, Source):
        val = float(value)
        instance.core_node.half_life = Float(val * 3600.0 * units.second) if val > 0 else Float(np.inf)


class SourceViewModel(NodeViewModel):
    """
    ViewModel для источника излучения (Source, PointSource).
    """
    activity = gui_field(default=100.0, on_change=_on_source_activity_change)      # МБк
    energy = gui_field(default=140.5, on_change=_on_source_energy_change)          # кэВ
    radiation_type = gui_field(default='Gamma', on_change=_on_source_radiation_type_change)
    half_life = gui_field(default=6.0, on_change=_on_source_half_life_change)     # часы
    file_path = gui_field(default='')

    def __init__(self, core_node: Any, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        self._sync_from_core()

    def _sync_from_core(self) -> None:
        if isinstance(self.core_node, Source):
            try:
                act = float(np.sum(self.core_node.initial_activity))
                if act < 1e3:
                    act_bq = act / units.Bq
                else:
                    act_bq = act
                self.activity = float(act_bq / 1e6)
            except Exception:
                pass

            en = self.core_node.energy
            try:
                if isinstance(en, np.ndarray) and en.dtype.names and 'energy' in en.dtype.names:
                    raw_en = float(en['energy'][0])
                elif isinstance(en, (int, float, np.floating, np.integer)):
                    raw_en = float(en)
                else:
                    raw_en = 140.5 * units.keV
                self.energy = float(raw_en / units.keV)
            except Exception:
                pass

            self.radiation_type = str(self.core_node.radiation_type)
            try:
                hl = float(self.core_node.half_life)
                self.half_life = hl / 3600.0 if hl > 0 and not np.isinf(hl) else 0.0
            except Exception:
                pass

            if self.core_node.distribution_path:
                self.file_path = str(self.core_node.distribution_path)

    @property
    def is_point_source(self) -> bool:
        return isinstance(self.core_node, PointSource)

    @property
    def dimensions(self) -> Tuple[int, ...]:
        if isinstance(self.core_node, Source) and self.core_node.distribution is not None:
            return tuple(self.core_node.distribution.shape)
        return (1, 1, 1)

    @property
    def size(self) -> np.ndarray:
        if isinstance(self.core_node, Source):
            return np.asarray(self.core_node.size, dtype=float)
        return np.array([20.0, 20.0, 20.0], dtype=float)

    @property
    def voxel_size(self) -> float:
        """Шаг вокселей источника в мм."""
        if isinstance(self.core_node, Source):
            return float(self.core_node.voxel_size)
        return 4.0

    @voxel_size.setter
    def voxel_size(self, val: float) -> None:
        if isinstance(self.core_node, Source):
            v = float(val)
            self.core_node.voxel_size = Float(v)
            self.property_changed.emit('voxel_size', v)
            self.property_changed.emit('size', self.size)

    def reload_distribution(self, path: str, shape: Optional[Tuple[int, ...]] = None, order: str = 'F') -> bool:
        """
        Перезагрузка матрицы активности источника из файла (.npy, .dat, .raw).
        """
        p = Path(path)
        if not p.exists():
            return False
        try:
            if p.suffix.lower() == '.npy':
                data = np.load(p, allow_pickle=True)
            else:
                s = shape or self.dimensions
                try:
                    data = np.loadtxt(p).reshape(s, order=order)
                except (ValueError, OSError):
                    data = np.fromfile(p, dtype=np.float32).reshape(s, order=order)
            if isinstance(self.core_node, Source):
                self.core_node.distribution_path = str(p)
                self.core_node.distribution = data.astype(float)
            self.file_path = str(p)
            self.property_changed.emit('file_path', self.file_path)
            self.property_changed.emit('size', self.size)
            self.property_changed.emit('distribution', data)
            return True
        except (OSError, ValueError, TypeError, KeyError) as e:
            _logger.warning(f"Ошибка загрузки распределения источника: {e}")
            return False


class DoseGridViewModel(NodeViewModel):
    """
    Модель представления узла сетки дозы DoseGridNode (паттерн MVVM).
    Предоставляет реактивные свойства size, dose_voxel_size, grid_shape, memory_mb,
    а также сигналы обновления параметров для PropertyInspector и VTKViewport.
    """

    color = gui_field(default=(0.2, 0.9, 0.2, 0.8))
    wireframe_visible = gui_field(default=True)
    dose_visible = gui_field(default=True)

    def __init__(self, core_node: DoseGridNode, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)

    @property
    def size(self) -> np.ndarray:
        """Габаритные размеры сетки (Lx, Ly, Lz) в мм."""
        return np.asarray(self.core_node.size, dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        arr = np.asarray(new_size, dtype=float)
        self.core_node.size = arr
        self.property_changed.emit('size', arr)
        self.property_changed.emit('grid_shape', self.grid_shape)
        self.property_changed.emit('memory_mb', self.memory_mb)

    @property
    def dose_voxel_size(self) -> float:
        """Размер стороны вокселя в мм."""
        return float(self.core_node.dose_voxel_size)

    @dose_voxel_size.setter
    def dose_voxel_size(self, val: float) -> None:
        v = float(val)
        self.core_node.dose_voxel_size = v
        self.property_changed.emit('dose_voxel_size', v)
        self.property_changed.emit('grid_shape', self.grid_shape)
        self.property_changed.emit('memory_mb', self.memory_mb)

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Количество вокселей по осям (Nx, Ny, Nz)."""
        return self.core_node.grid_shape

    @property
    def memory_mb(self) -> float:
        """Расход памяти RAM для сетки типа float64 в МБ."""
        return float(self.core_node.memory_mb)

    @property
    def is_active(self) -> bool:
        """Флаг активности сетки для накопления дозы."""
        return self.core_node.is_active

    @is_active.setter
    def is_active(self, val: bool) -> None:
        b = bool(val)
        self.core_node.is_active = b
        self.property_changed.emit('is_active', b)


def create_node_viewmodel(core_node: SpatialNode, parent_vm: Optional[NodeViewModel] = None) -> NodeViewModel:
    """
    Фабричная функция для инстанцирования специализированных ViewModel по типу core_node.
    """
    if isinstance(core_node, DoseGridNode):
        return DoseGridViewModel(core_node, parent_vm)
    if isinstance(core_node, ParametricParallelCollimator):
        return ParametricParallelCollimatorViewModel(core_node, parent_vm)
    if isinstance(core_node, ParametricParallelSquareCollimator):
        return ParametricParallelSquareCollimatorViewModel(core_node, parent_vm)
    if isinstance(core_node, WoodcockVoxelVolume):
        return VoxelVolumeViewModel(core_node, parent_vm)
    if isinstance(core_node, GammaCamera):
        return GammaCameraViewModel(core_node, parent_vm)
    if isinstance(core_node, (Source, PointSource)) or core_node.__class__.__name__ in ('Source', 'PointSource', 'I123'):
        return SourceViewModel(core_node, parent_vm)
    if core_node.__class__.__name__ in ('PetScanner', 'PETScanner'):
        return PetScannerViewModel(core_node, parent_vm)
    if isinstance(core_node, Volume):
        return VolumeViewModel(core_node, parent_vm)
    return NodeViewModel(core_node, parent_vm)


