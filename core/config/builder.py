import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import re

import settings.database_setting as database_setting
from core.config.models import (
    AnyNodeConfig, VolumeConfig, GammaCameraConfig, WoodcockVoxelVolumeConfig,
    ParametricParallelCollimatorConfig, ParametricParallelSquareCollimatorConfig,
    SourceConfig, BoxConfig, SimulationConfig, TranslateConfig, RotateConfig,
    NumpyDistributionConfig, RawDistributionConfig, AnyDistributionConfig,
    DoseGridNodeConfig
)
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.parametric_collimators import ParametricParallelCollimator, ParametricParallelSquareCollimator
from core.materials.materials import MaterialArray
from core.source.sources import Source
from core.scene.nodes import SpatialNode, CompositeNode
from core.scene.dose_grid_node import DoseGridNode

class SceneBuilder:
    def __init__(self, base_dir: Optional[Any] = None):
        self.base_dir = Path(base_dir) if base_dir else None
        self.factory_map: Dict[str, Callable[[AnyNodeConfig], SpatialNode]] = {
            'SpatialNode': self._build_spatial_node,
            'CompositeNode': self._build_composite_node,
            'Volume': self._build_volume,
            'GammaCamera': self._build_gamma_camera,
            'WoodcockVoxelVolume': self._build_woodcock_voxel_volume,
            'ParametricParallelCollimator': self._build_parametric_parallel_collimator,
            'ParametricParallelSquareCollimator': self._build_parametric_parallel_square_collimator,
            'Source': self._build_source,
            'DoseGridNode': self._build_dose_grid_node
        }
        self.node_cache: Dict[str, SpatialNode] = {}

    def _build_spatial_node(self, config) -> SpatialNode:
        node = SpatialNode()
        return node

    def _build_composite_node(self, config) -> SpatialNode:
        node = CompositeNode()
        return node

    def build_scene(self, config: AnyNodeConfig) -> SpatialNode:
        root_node = self._build_node(config)
        return root_node

    @staticmethod
    def _to_float(val: Any, check_positive: bool = False) -> float:
        """
        Строгая валидация и приведение значения к float.
        Значения из моделей конфигурации уже валидированы слоем Pydantic и приведены к HepUnits.
        """
        if isinstance(val, (int, float)):
            res = float(val)
        elif isinstance(val, str):
            if re.search(r'\$\{[^}]+\}', val):
                raise ValueError(f"Неразрешенный макрос интерполяции в значении: '{val}'")
            cleaned = val.strip()
            if not cleaned:
                raise ValueError("Пустое строковое значение недопустимо")
            res = float(cleaned)
        else:
            raise TypeError(f"Недопустимый тип значения для _to_float: {type(val)}. Ожидалось число.")

        if check_positive and res <= 0:
            raise ValueError(f"Значение {res} должно быть строго больше 0")
        return res

    def _build_node(self, config: AnyNodeConfig) -> SpatialNode:
        node_type = config.type
        if node_type not in self.factory_map:
            raise ValueError(f"Unknown node type: {node_type}")

        node = self.factory_map[node_type](config)

        # Применение трансформаций
        for transform in config.transformations:
            if isinstance(transform, TranslateConfig):
                x = self._to_float(transform.x)
                y = self._to_float(transform.y)
                z = self._to_float(transform.z)
                node.translate(x, y, z, transform.in_local)
            elif isinstance(transform, RotateConfig):
                alpha = self._to_float(transform.alpha)
                beta = self._to_float(transform.beta)
                gamma = self._to_float(transform.gamma)
                rot_center = tuple(self._to_float(c) for c in transform.rotation_center)
                node.rotate(alpha, beta, gamma, rot_center, transform.in_local)

        # Build children
        for child_config in config.children:
            child_node = self._build_node(child_config)
            node.add_child(child_node)

        return node

    def _get_material(self, name: str):
        """Получение объекта материала из базы данных по каноническому имени."""
        if name not in database_setting.material_database:
            raise ValueError(f"Материал '{name}' не найден в базе данных.")
        return database_setting.material_database[name]

    def _build_geometry(self, config):
        if isinstance(config, BoxConfig):
            x = self._to_float(config.x, check_positive=True)
            y = self._to_float(config.y, check_positive=True)
            z = self._to_float(config.z, check_positive=True)
            return Box(x, y, z)
        raise ValueError(f"Unknown geometry type: {config.type}")

    def _build_volume(self, config: VolumeConfig) -> Volume:
        geometry = self._build_geometry(config.geometry)
        material = self._get_material(config.material)
        return Volume(geometry=geometry, material=material, name=config.name)

    def _resolve_dist_path(self, path_str: str) -> str:
        # Разрешение путей распределения данных (с поддержкой относительных путей)
        file_path = Path(path_str)
        if file_path.is_absolute():
            if file_path.is_file():
                return str(file_path)
            raise FileNotFoundError(f"Файл распределения не найден: {file_path}")
        if self.base_dir:
            candidate_path = self.base_dir / file_path
            if candidate_path.is_file():
                return str(candidate_path)
        if file_path.is_file():
            return str(file_path)
        raise FileNotFoundError(f"Файл распределения не найден: {path_str}")

    def _load_raw_distribution(self, dist_config: AnyDistributionConfig) -> np.ndarray:
        resolved_path = self._resolve_dist_path(dist_config.path)
        if isinstance(dist_config, NumpyDistributionConfig):
            file_path = Path(resolved_path)
            if file_path.suffix.lower() in ('.txt', '.dat'):
                return np.loadtxt(resolved_path)
            elif file_path.suffix.lower() == '.npy':
                return np.load(resolved_path, allow_pickle=True)
            else:
                raise ValueError(f"Неподдерживаемое расширение для NumpyDistributionConfig: {file_path.suffix}")
        elif isinstance(dist_config, RawDistributionConfig):
            file_path = Path(resolved_path)
            if dist_config.encoding == 'binary':
                file_size = file_path.stat().st_size
                expected_size = int(np.prod(dist_config.shape) * 4)
                if file_size != expected_size:
                    raise ValueError(f"Размер бинарного файла {resolved_path} ({file_size} байт) не совпадает с ожидаемым ({expected_size} байт для float32)")
                data = np.fromfile(resolved_path, dtype=np.float32)
            else:
                data = np.loadtxt(resolved_path)

            expected_elements = int(np.prod(dist_config.shape))
            if data.size != expected_elements:
                raise ValueError(f"Количество элементов в файле {resolved_path} ({data.size}) не совпадает с требуемой формой {dist_config.shape} ({expected_elements})")

            return data.reshape(dist_config.shape, order=dist_config.order)
        raise ValueError(f"Unknown distribution format: {type(dist_config)}")

    def _build_woodcock_voxel_volume(self, config: WoodcockVoxelVolumeConfig) -> WoodcockVoxelVolume:
        dist_config = config.distribution
        raw_distribution = self._load_raw_distribution(dist_config)

        mat_arr = MaterialArray(raw_distribution.shape)

        if dist_config.fill_value is not None:
            mat = self._get_material(str(dist_config.fill_value))
            mat_arr[:] = mat

        if dist_config.mapping is not None:
            if not dist_config.mapping:
                raise ValueError("WoodcockVoxelVolumeConfig mapping cannot be empty if specified.")
            for map_val, mat_name in dist_config.mapping.items():
                mask = np.isclose(raw_distribution, map_val)
                mat = self._get_material(str(mat_name))
                indices = np.nonzero(mask)
                mat_arr[indices] = mat
        else:
            raise ValueError("WoodcockVoxelVolumeConfig mapping requires an explicit mapping, raw IDs are not currently supported by MaterialArray.")

        v_size_raw = config.voxel_size if config.voxel_size is not None else 1.0
        voxel_size = self._to_float(v_size_raw, check_positive=True)
        node = WoodcockVoxelVolume(voxel_size=voxel_size, material_distribution=mat_arr, name=config.name)
        node.distribution_config = dist_config
        if dist_config.path is not None:
            node.distribution_path = str(self._resolve_dist_path(dist_config.path))
        return node

    def _build_gamma_camera(self, config: GammaCameraConfig) -> GammaCamera:
        collimator = self._build_node(config.collimator)
        detector = self._build_node(config.detector)
        gap_raw = config.gap if config.gap is not None else 1.0
        gap = self._to_float(gap_raw, check_positive=True)
        shielding_raw = config.shielding_thickness if config.shielding_thickness is not None else 20.0
        shielding = self._to_float(shielding_raw, check_positive=True)
        glass_raw = config.glass_backend_thickness if config.glass_backend_thickness is not None else 50.0
        glass = self._to_float(glass_raw, check_positive=True)
        return GammaCamera(
            collimator=collimator,
            detector=detector,
            gap=gap,
            shielding_thickness=shielding,
            glass_backend_thickness=glass,
            name=config.name
        )

    def _build_parametric_parallel_collimator(self, config: ParametricParallelCollimatorConfig) -> ParametricParallelCollimator:
        material = self._get_material(config.material)
        size = [self._to_float(s, check_positive=True) for s in config.size] if isinstance(config.size, (list, tuple)) else self._to_float(config.size, check_positive=True)
        hole_diameter = self._to_float(config.hole_diameter, check_positive=True)
        septa = self._to_float(config.septa_thickness, check_positive=True)
        return ParametricParallelCollimator(
            size=size,
            hole_diameter=hole_diameter,
            septa=septa,
            material=material,
            name=config.name
        )

    def _build_parametric_parallel_square_collimator(self, config: ParametricParallelSquareCollimatorConfig) -> ParametricParallelSquareCollimator:
        material = self._get_material(config.material)
        size = [self._to_float(s, check_positive=True) for s in config.size] if isinstance(config.size, (list, tuple)) else self._to_float(config.size, check_positive=True)
        hole_size = [self._to_float(s, check_positive=True) for s in config.hole_size] if isinstance(config.hole_size, (list, tuple)) else self._to_float(config.hole_size, check_positive=True)
        septa = [self._to_float(s, check_positive=True) for s in config.septa_thickness] if isinstance(config.septa_thickness, (list, tuple)) else self._to_float(config.septa_thickness, check_positive=True)
        return ParametricParallelSquareCollimator(
            size=size,
            hole_width=hole_size,
            septa=septa,
            material=material,
            name=config.name
        )

    def _build_source(self, config: SourceConfig) -> Source:
        dist_config = config.distribution
        raw_distribution = self._load_raw_distribution(dist_config)

        distribution = np.array(raw_distribution, dtype=float, copy=True)

        if dist_config.fill_value is not None:
            distribution.fill(float(dist_config.fill_value))

        if dist_config.mapping is not None:
            for map_val, act_val in dist_config.mapping.items():
                mask = np.isclose(raw_distribution, map_val)
                distribution[mask] = float(act_val)

        activity = self._to_float(config.activity, check_positive=True) if config.activity is not None else None
        voxel_size = self._to_float(config.voxel_size, check_positive=True)
        energy = self._to_float(config.energy, check_positive=True)
        half_life = self._to_float(config.half_life, check_positive=True)

        node = Source(
            distribution=distribution,
            activity=activity,
            voxel_size=voxel_size,
            radiation_type=config.radiation_type,
            energy=energy,
            half_life=half_life
        )
        node.distribution_config = dist_config
        if config.name is not None:
            node.name = config.name
        if dist_config.path is not None:
            node.distribution_path = str(self._resolve_dist_path(dist_config.path))
        return node

    def _build_dose_grid_node(self, config: DoseGridNodeConfig) -> DoseGridNode:
        size = [
            self._to_float(config.size[0], check_positive=True),
            self._to_float(config.size[1], check_positive=True),
            self._to_float(config.size[2], check_positive=True),
        ]
        dose_voxel_size = self._to_float(config.dose_voxel_size, check_positive=True)
        node = DoseGridNode(
            name=config.name,
            size=size,
            dose_voxel_size=dose_voxel_size,
            is_active=config.is_active,
        )
        return node
