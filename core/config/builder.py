import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import re

import settings.database_setting as database_setting
import hepunits as units
from core.config.models import (
    AnyNodeConfig, VolumeConfig, GammaCameraConfig, GammaCameraSlotsConfig, WoodcockVoxelVolumeConfig,
    ParametricParallelCollimatorConfig,
    DirectParallelCollimatorConfig,
    PointSpatialConfig, VoxelSpatialConfig,
    DiscreteSpectrumConfig, ContinuousSpectrumConfig,
    IsotropicAngularConfig, ConeAngularConfig,
    RadiationChannelConfig,
    SourceConfig, BoxConfig, GeometryConfig, TranslateConfig, RotateConfig,
    MatrixTransformConfig,
    NumpyDistributionConfig, RawDistributionConfig, AnyDistributionConfig,
    DoseGridNodeConfig, GantryConfig, CompositeNodeConfig, BaseSpatialNodeConfig, BaseCompositeNodeConfig,
    SimulationManagerConfig,
)
from core.geometry.geometries import Box, Geometry
from core.geometry.volumes import Volume
from core.scene.gamma_camera_node import GammaCameraNode
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.direct_collimators import DirectParallelCollimator, CollimatorHoleShape
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Float, ParticleSpecies
from core.source.sources import (
    Source,
    SpatialDistribution,
    PointDistribution,
    VoxelDistribution,
    SpectralLine,
    EnergySpectrum,
    DiscreteSpectrum,
    ContinuousSpectrum,
    AngularDistribution,
    IsotropicAngular,
    ConeAngular,
    RadiationChannel,
)
from core.scene.nodes import SpatialNode, CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from core.scene.gantry_node import GantryNode
from core.transport.simulation_managers import SimulationManager

class SceneBuilder:
    def __init__(self, base_dir: Optional[Any] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else None
        self.distribution_registry: Dict[SpatialNode, AnyDistributionConfig] = {}
        self.slots_registry: Dict[SpatialNode, GammaCameraSlotsConfig] = {}
        self.factory_map: Dict[str, Callable[[Any], SpatialNode]] = {
            'SpatialNode': self._build_spatial_node,
            'CompositeNode': self._build_composite_node,
            'Volume': self._build_volume,
            'GammaCamera': self._build_gamma_camera,
            'WoodcockVoxelVolume': self._build_woodcock_voxel_volume,
            'ParametricParallelCollimator': self._build_parallel_collimator,
            'DirectParallelCollimator': self._build_parallel_collimator,
            'Source': self._build_source,
            'DoseGridNode': self._build_dose_grid_node,
            'Gantry': self._build_gantry,
            'GantryNode': self._build_gantry,
        }
        self.node_cache: Dict[str, SpatialNode] = {}

    def _build_spatial_node(self, config: BaseSpatialNodeConfig) -> SpatialNode:
        node = SpatialNode(name=config.name)
        return node

    def _build_composite_node(self, config: BaseCompositeNodeConfig) -> SpatialNode:
        node = CompositeNode(name=config.name)
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
            raise ValueError(f"Неизвестный тип узла сцены: {node_type}")

        node = self.factory_map[node_type](config)
        if config.tags:
            node.tags = list(config.tags)

        # Применение трансформаций
        for transform in config.transformations:
            if isinstance(transform, TranslateConfig):
                trans_x = self._to_float(transform.x)
                trans_y = self._to_float(transform.y)
                trans_z = self._to_float(transform.z)
                node.translate(trans_x, trans_y, trans_z, transform.in_local)
            elif isinstance(transform, RotateConfig):
                alpha = self._to_float(transform.alpha)
                beta = self._to_float(transform.beta)
                gamma = self._to_float(transform.gamma)
                rot_center = tuple(self._to_float(coord_val) for coord_val in transform.rotation_center)
                node.rotate(alpha, beta, gamma, rot_center, transform.in_local)
            elif isinstance(transform, MatrixTransformConfig):
                mat = np.array(transform.matrix, dtype=float)
                if transform.in_local:
                    node.local_matrix = node.local_matrix @ mat
                else:
                    node.local_matrix = mat @ node.local_matrix


        # Build children
        if isinstance(config, CompositeNodeConfig) and isinstance(node, CompositeNode):
            for child_config in config.children:
                child_node = self._build_node(child_config)
                node.add_child(child_node)

        return node

    def _get_material(self, name: str) -> Material:
        """Получение объекта материала из базы данных по каноническому имени."""
        if name not in database_setting.material_database:
            raise ValueError(f"Материал '{name}' не найден в базе данных.")
        return database_setting.material_database[name]

    def _build_geometry(self, config: GeometryConfig) -> Geometry:
        if isinstance(config, BoxConfig):
            size_x = Float(self._to_float(config.x, check_positive=True))
            size_y = Float(self._to_float(config.y, check_positive=True))
            size_z = Float(self._to_float(config.z, check_positive=True))
            return Box(size_x, size_y, size_z)
        raise ValueError(f"Неизвестный тип геометрии: {config.type}")

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
            parent_candidate = self.base_dir.parent / file_path
            if parent_candidate.is_file():
                return str(parent_candidate)
        if file_path.is_file():
            return str(file_path)
        raise FileNotFoundError(f"Файл распределения не найден: {path_str}")

    def _load_raw_distribution(self, dist_config: AnyDistributionConfig) -> np.ndarray:
        resolved_path = self._resolve_dist_path(dist_config.path)
        if isinstance(dist_config, NumpyDistributionConfig):
            file_path = Path(resolved_path)
            if file_path.suffix.lower() in ('.txt', '.dat'):
                loaded_txt: np.ndarray = np.loadtxt(resolved_path)
                return loaded_txt
            elif file_path.suffix.lower() == '.npy':
                loaded_npy: np.ndarray = np.load(resolved_path, allow_pickle=True)
                return loaded_npy
            else:
                raise ValueError(f"Неподдерживаемое расширение для NumpyDistributionConfig: {file_path.suffix}")
        elif isinstance(dist_config, RawDistributionConfig):
            file_path = Path(resolved_path)
            data: np.ndarray
            element_dtype = np.dtype(dist_config.dtype)
            if dist_config.encoding == 'binary':
                file_size = file_path.stat().st_size
                expected_size = int(np.prod(dist_config.shape) * element_dtype.itemsize)
                if file_size != expected_size:
                    raise ValueError(
                        f"Размер бинарного файла {resolved_path} ({file_size} байт) "
                        f"не совпадает с ожидаемым ({expected_size} байт для {element_dtype.name})"
                    )
                data = np.fromfile(resolved_path, dtype=element_dtype)
            else:
                data = np.loadtxt(resolved_path, dtype=element_dtype)

            expected_elements = int(np.prod(dist_config.shape))
            if data.size != expected_elements:
                raise ValueError(f"Количество элементов в файле {resolved_path} ({data.size}) не совпадает с требуемой формой {dist_config.shape} ({expected_elements})")

            return data.reshape(dist_config.shape, order=dist_config.order)
        raise ValueError(f"Неизвестный формат распределения: {type(dist_config)}")

    def _build_woodcock_voxel_volume(self, config: WoodcockVoxelVolumeConfig) -> WoodcockVoxelVolume:
        dist_config = config.distribution
        raw_distribution = self._load_raw_distribution(dist_config)

        mat_arr = MaterialArray(raw_distribution.shape)

        if dist_config.fill_value is not None:
            mat = self._get_material(str(dist_config.fill_value))
            mat_arr[:] = mat

        if not dist_config.mapping:
            raise ValueError("Для WoodcockVoxelVolumeConfig необходимо явно задать непустой словарь сопоставления материалов (mapping).")
        for map_val, mat_name in dist_config.mapping.items():
            mask = np.isclose(raw_distribution, map_val)
            mat = self._get_material(str(mat_name))
            indices = np.nonzero(mask)
            mat_arr[indices] = mat

        voxel_size = Float(self._to_float(config.voxel_size, check_positive=True))
        node = WoodcockVoxelVolume(voxel_size=voxel_size, material_distribution=mat_arr, name=config.name)
        self.distribution_registry[node] = dist_config
        return node

    def _build_gamma_camera(self, config: GammaCameraConfig) -> GammaCameraNode:
        slots_dict = {
            slot_name: slot_value
            for slot_name, slot_value in config.slots.model_dump().items()
            if slot_value is not None
        }
        camera_node = GammaCameraNode(name=config.name, slots=slots_dict)
        self.slots_registry[camera_node] = config.slots
        return camera_node

    def _build_parallel_collimator(
        self, config: ParametricParallelCollimatorConfig | DirectParallelCollimatorConfig
    ) -> ParametricParallelCollimator | DirectParallelCollimator:
        material = self._get_material(config.material)
        size = (
            [Float(self._to_float(dimension_value, check_positive=True)) for dimension_value in config.size]
            if isinstance(config.size, (list, tuple))
            else [Float(self._to_float(config.size, check_positive=True))] * 3
        )
        hole_diameter = Float(self._to_float(config.hole_diameter, check_positive=True))
        septa = Float(self._to_float(config.septa, check_positive=True))
        hole_shape = CollimatorHoleShape(config.hole_shape.lower())
        if isinstance(config, DirectParallelCollimatorConfig):
            hole_material = self._get_material(config.hole_material) if config.hole_material is not None else None
            return DirectParallelCollimator(
                size=size,
                hole_diameter=hole_diameter,
                septa=septa,
                material=material,
                hole_material=hole_material,
                hole_shape=hole_shape,
                name=config.name,
            )
        return ParametricParallelCollimator(
            size=size,
            hole_diameter=hole_diameter,
            septa=septa,
            material=material,
            hole_shape=hole_shape,
            name=config.name,
        )

    def _build_source(self, config: SourceConfig) -> Source:
        activity = Float(self._to_float(config.activity, check_positive=True)) if config.activity is not None else None
        half_life_raw = self._to_float(config.half_life, check_positive=False)
        if half_life_raw < 0.0:
            raise ValueError(f"Значение half_life ({half_life_raw}) не может быть отрицательным")
        half_life = Float(half_life_raw) if half_life_raw > 0.0 else Float(np.inf)

        spatial_cfg = config.spatial
        dist_config: Optional[AnyDistributionConfig] = None
        if isinstance(spatial_cfg, PointSpatialConfig):
            jitter_x = Float(self._to_float(spatial_cfg.cell_jitter[0], check_positive=False))
            jitter_y = Float(self._to_float(spatial_cfg.cell_jitter[1], check_positive=False))
            spatial_dist: SpatialDistribution = PointDistribution(cell_jitter=(jitter_x, jitter_y))
            effective_activity: Optional[Float] = activity if activity is not None else Float(1.0 * units.MBq)
        elif isinstance(spatial_cfg, VoxelSpatialConfig):
            voxel_size = Float(self._to_float(spatial_cfg.voxel_size, check_positive=True))
            dist_config = spatial_cfg.distribution
            raw_distribution = self._load_raw_distribution(dist_config)
            distribution = np.array(raw_distribution, dtype=float, copy=True)

            if dist_config.fill_value is not None:
                distribution.fill(float(dist_config.fill_value))

            if dist_config.mapping is not None:
                for map_val, act_val in dist_config.mapping.items():
                    mask = np.isclose(raw_distribution, map_val)
                    distribution[mask] = float(act_val)

            spatial_dist = VoxelDistribution(distribution=distribution, voxel_size=voxel_size)
            effective_activity = activity
        else:
            raise TypeError(f"Неизвестный тип пространственной конфигурации источника: {type(spatial_cfg)}")

        species_map = {
            'photon': ParticleSpecies.PHOTON,
            'electron': ParticleSpecies.ELECTRON,
            'positron': ParticleSpecies.POSITRON,
            'alpha': ParticleSpecies.ALPHA,
        }
        built_channels = []
        for ch_cfg in config.channels:
            species = species_map[ch_cfg.species]

            if isinstance(ch_cfg.spectrum, DiscreteSpectrumConfig):
                lines_tuple = tuple(
                    SpectralLine(
                        energy=Float(self._to_float(line_e, check_positive=True)),
                        probability=Float(line_p),
                    )
                    for line_e, line_p in ch_cfg.spectrum.lines
                )
                spectrum_obj: EnergySpectrum = DiscreteSpectrum(lines=lines_tuple)
            elif isinstance(ch_cfg.spectrum, ContinuousSpectrumConfig):
                grid_arr = np.array(
                    [self._to_float(e_val, check_positive=False) for e_val in ch_cfg.spectrum.energy_grid],
                    dtype=Float,
                )
                pdf_arr = np.array([float(p_val) for p_val in ch_cfg.spectrum.pdf_values], dtype=Float)
                spectrum_obj = ContinuousSpectrum(energy_grid=grid_arr, pdf_values=pdf_arr)
            else:
                raise TypeError(f"Неизвестный тип спектра канала источника: {type(ch_cfg.spectrum)}")

            if isinstance(ch_cfg.angular, IsotropicAngularConfig):
                angular_obj: AngularDistribution = IsotropicAngular()
            elif isinstance(ch_cfg.angular, ConeAngularConfig):
                max_angle = Float(self._to_float(ch_cfg.angular.max_polar_angle, check_positive=False))
                angular_obj = ConeAngular(max_polar_angle=max_angle)
            else:
                raise TypeError(f"Неизвестный тип углового распределения канала источника: {type(ch_cfg.angular)}")

            built_channels.append(
                RadiationChannel(
                    species=species,
                    spectrum=spectrum_obj,
                    angular=angular_obj,
                    branching_ratio=Float(ch_cfg.branching_ratio),
                )
            )

        node = Source(
            spatial=spatial_dist,
            channels=tuple(built_channels),
            activity=effective_activity,
            half_life=half_life,
            name=config.name,
        )
        if dist_config is not None:
            self.distribution_registry[node] = dist_config
        return node

    def get_distribution_path(self, node: SpatialNode) -> Optional[str]:
        """
        Возвращает абсолютный путь к файлу распределения для указанного узла из реестра сборок.
        """
        dist_config = self.distribution_registry.get(node)
        if dist_config is not None and isinstance(dist_config, (NumpyDistributionConfig, RawDistributionConfig)):
            if dist_config.path is not None:
                return str(self._resolve_dist_path(dist_config.path))
        return None

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
            dtype=config.dtype,
        )
        return node

    def _build_gantry(self, config: GantryConfig) -> GantryNode:
        """Сборка поворотной станины томографа GantryNode."""
        return GantryNode(name=config.name)

    def build_simulation_manager(
        self,
        config: SimulationManagerConfig,
        scene: CompositeNode,
        **kwargs: Any,
    ) -> SimulationManager:
        """Создает экземпляр SimulationManager из декларативной конфигурации SimulationManagerConfig."""
        return SimulationManager(
            scene=scene,
            start_time=self._to_float(config.start_time),
            stop_time=None if config.stop_time is None else self._to_float(config.stop_time),
            particles_number=int(config.particles_number),
            batch_capacity=None if config.batch_capacity is None else int(config.batch_capacity),
            buffer_capacity=None if config.buffer_capacity is None else int(config.buffer_capacity),
            min_energy=self._to_float(config.min_energy, check_positive=True),
            enable_atomic_relaxation=bool(config.enable_atomic_relaxation),
            photon_production_cut=self._to_float(config.photon_production_cut, check_positive=True),
            electron_production_cut=self._to_float(config.electron_production_cut, check_positive=True),
            seed=config.seed if "seed" not in kwargs else kwargs.pop("seed"),
            **kwargs,
        )


__all__ = ["SceneBuilder"]

