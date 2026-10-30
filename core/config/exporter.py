import os
import math
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

import numpy as np
from scipy.spatial.transform import Rotation
import hepunits as units

from core.config.models import (
    AnyAngularDistributionConfig,
    AnyDistributionConfig,
    AnyNodeConfig,
    AnyProtocolConfig,
    AnySpatialDistributionConfig,
    AnySpectrumConfig,
    BaseCompositeNodeConfig,
    BaseSpatialNodeConfig,
    BoxConfig,
    CollimatorHoleShapeLiteral,
    ConeAngularConfig,
    ContinuousSpectrumConfig,
    DataManagerConfig,
    DirectStreamHandlerConfig,
    DiscreteSpectrumConfig,
    GammaCameraConfig,
    GammaCameraSlotsConfig,
    IsotropicAngularConfig,
    LoggingConfig,
    NumpyDistributionConfig,
    ParametricParallelCollimatorConfig,
    DirectParallelCollimatorConfig,
    PointSpatialConfig,
    RadiationChannelConfig,
    RotateConfig,
    SimulationConfig,
    SimulationManagerConfig,
    SourceConfig,
    TransformConfig,
    TranslateConfig,
    MatrixTransformConfig,
    VolumeConfig,
    VoxelSpatialConfig,
    WoodcockVoxelVolumeConfig,
    DoseGridNodeConfig,
    GantryConfig,
    RawDistributionConfig,
)
from core.config.yaml_dumper import dump_simulation_config
from core.geometry.geometries import Box
from core.scene.gamma_camera_node import GammaCameraNode
from core.geometry.parametric_collimators import (
    ParametricParallelCollimator,
)
from core.geometry.direct_collimators import (
    DirectParallelCollimator,
    CollimatorHoleShape,
)
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.scene.nodes import CompositeNode, SpatialNode
from core.scene.dose_grid_node import DoseGridNode
from core.scene.gantry_node import GantryNode
from core.source.sources import (
    ConeAngular,
    ContinuousSpectrum,
    DiscreteSpectrum,
    IsotropicAngular,
    PointDistribution,
    Source,
    VoxelDistribution,
)
from core.transport.simulation_managers import SimulationManager

logger = logging.getLogger(__name__)


class SceneExporter:
    """
    Класс для обратной сериализации чистого графа сцены (SpatialNode, Volume)
    в Pydantic-конфигурации SimulationConfig и сохранения в формат YAML.
    """

    @classmethod
    def decompose_matrix(cls, matrix: np.ndarray, as_matrix: bool = True) -> List[TransformConfig]:
        """
        Преобразует матрицу трансформации 4x4 в список TransformConfig.
        По умолчанию при as_matrix=True выгружает полную матрицу 4x4 (MatrixTransformConfig),
        что гарантирует 100% стабильность сохранения/загрузки без потери знаков и Gimbal lock.
        При as_matrix=False выполняет декомпозицию на RotateConfig и TranslateConfig
        (в порядке [RotateConfig, TranslateConfig], так как SpatialNode умножает матрицы слева: T @ R = [R | t]).
        """
        transforms: List[TransformConfig] = []
        if matrix is None or np.allclose(matrix, np.eye(4), atol=1e-7):
            return transforms

        if as_matrix:
            mat_4x4 = [
                [float(np.round(matrix[i, j], 8)) for j in range(4)]
                for i in range(4)
            ]
            transforms.append(MatrixTransformConfig(matrix=mat_4x4))
            return transforms

        # 1. Извлечение вращения (применяется первым, чтобы последующая трансляция не поворачивалась)
        rot_mat = matrix[:3, :3]
        if not np.allclose(rot_mat, np.eye(3), atol=1e-5):
            try:
                rotation_obj = Rotation.from_matrix(rot_mat)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*Gimbal lock detected.*')
                    euler = rotation_obj.as_euler('zyx', degrees=False)
                transforms.append(RotateConfig(
                    alpha=float(euler[0]),
                    beta=float(euler[1]),
                    gamma=float(euler[2])
                ))
            except ValueError as error:
                logger.warning(f"Ошибка при декомпозиции матрицы вращения: {error}")

        # 2. Извлечение трансляции (применяется после вращения: M = T @ R = [R | t])
        translation_x = float(matrix[0, 3])
        translation_y = float(matrix[1, 3])
        translation_z = float(matrix[2, 3])
        if not (math.isclose(translation_x, 0.0, abs_tol=1e-6) and math.isclose(translation_y, 0.0, abs_tol=1e-6) and math.isclose(translation_z, 0.0, abs_tol=1e-6)):
            transforms.append(TranslateConfig(x=translation_x, y=translation_y, z=translation_z))

        return transforms

    @staticmethod
    def _extract_material_mapping(node: WoodcockVoxelVolume) -> Optional[Dict[float, Union[float, str]]]:
        """Извлекает словарь сопоставления индексов и канонических имен материалов из воксельного объема."""
        if node.material_distribution is not None and len(node.material_distribution.element_list) > 0:
            return {
                float(element_idx): material_item.name
                for element_idx, material_item in enumerate(node.material_distribution.element_list)
            }
        return None

    @classmethod
    def export_node(
        cls,
        node: SpatialNode,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
    ) -> AnyNodeConfig:
        """
        Рекурсивно экспортирует узел сцены в Pydantic-модель конфигурации.
        """
        node_name = node.name
        node_tags = list(node.tags) if node.tags else []
        transforms = cls.decompose_matrix(node.local_matrix)
        child_nodes: Sequence[SpatialNode] = (
            [c for c in node.childs if c is not node.lead_body and c is not node.channels]
            if isinstance(node, DirectParallelCollimator)
            else (node.childs if isinstance(node, CompositeNode) else ())
        )
        children_cfgs = [
            cls.export_node(
                child_node,
                distribution_registry=distribution_registry,
                slots_registry=slots_registry,
            )
            for child_node in child_nodes
        ]

        # 1. GammaCameraNode
        if isinstance(node, GammaCameraNode):
            slots_cfg = None
            if slots_registry is not None:
                slots_cfg = slots_registry.get(node)
            if slots_cfg is None:
                slots_cfg = GammaCameraSlotsConfig(
                    casing=node.slots.get("casing"),
                    detector_box=node.slots.get("detector_box"),
                    collimator=node.slots.get("collimator"),
                    crystal=node.slots.get("crystal"),
                    glass_backend=node.slots.get("glass_backend"),
                )
            return GammaCameraConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                slots=slots_cfg,
                children=children_cfgs,
            )

        # 2. ParametricParallelCollimator
        if isinstance(node, ParametricParallelCollimator):
            size_tuple = (float(node.size[0]), float(node.size[1]), float(node.size[2]))
            mat_name = node.material.name
            hole_shape_val = cast(
                CollimatorHoleShapeLiteral,
                node.hole_shape.value if isinstance(node.hole_shape, CollimatorHoleShape) else str(node.hole_shape),
            )
            return ParametricParallelCollimatorConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                size=size_tuple,
                hole_diameter=float(node.hole_diameter),
                septa=float(node.septa),
                material=mat_name,
                hole_shape=hole_shape_val,
                children=children_cfgs,
            )

        # 3.1. DirectParallelCollimator
        if isinstance(node, DirectParallelCollimator):
            size_tuple = (float(node.size[0]), float(node.size[1]), float(node.size[2]))
            mat_name = node.material.name
            hole_mat_name = node.explicit_hole_material.name if node.explicit_hole_material is not None else None
            hole_shape_val = cast(
                CollimatorHoleShapeLiteral,
                node.hole_shape.value if isinstance(node.hole_shape, CollimatorHoleShape) else str(node.hole_shape),
            )
            return DirectParallelCollimatorConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                size=size_tuple,
                hole_diameter=float(node.hole_diameter),
                septa=float(node.septa),
                material=mat_name,
                hole_material=hole_mat_name,
                hole_shape=hole_shape_val,
                children=children_cfgs,
            )

        # 4. WoodcockVoxelVolume
        if isinstance(node, WoodcockVoxelVolume):
            v_size = float(np.mean(node.voxel_size)) if isinstance(node.voxel_size, (list, tuple, np.ndarray, Sequence)) else float(node.voxel_size)
            dist_cfg = distribution_registry.get(node) if distribution_registry is not None else None
            actual_mapping = cls._extract_material_mapping(node)

            if dist_cfg is None:
                dist_cfg = NumpyDistributionConfig(
                    path=f"{node_name or 'phantom'}.npy",
                    mapping=actual_mapping if actual_mapping is not None else {0.0: 'Vacuum'},
                )
            elif isinstance(dist_cfg, (NumpyDistributionConfig, RawDistributionConfig)) and dist_cfg.mapping is None and actual_mapping is not None:
                dist_cfg = dist_cfg.model_copy(update={'mapping': actual_mapping})

            return WoodcockVoxelVolumeConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                voxel_size=v_size,
                distribution=dist_cfg,
                children=children_cfgs,
            )

        # 5. Volume
        if isinstance(node, Volume):
            geo = node.geometry
            if isinstance(geo, Box):
                geo_cfg = BoxConfig(x=float(geo.size[0]), y=float(geo.size[1]), z=float(geo.size[2]))
            else:
                bounds = node.local_bound
                geo_cfg = BoxConfig(
                    x=float(bounds[1, 0] - bounds[0, 0]),
                    y=float(bounds[1, 1] - bounds[0, 1]),
                    z=float(bounds[1, 2] - bounds[0, 2]),
                )

            return VolumeConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                geometry=geo_cfg,
                material=node.material.name,
                children=children_cfgs,
            )

        # 6. Source
        if isinstance(node, Source):
            act = float(node.initial_activity)
            scaled_activity = float(act) / float(units.MBq)
            act_val = f"{scaled_activity:.6g} MBq"
            half_life = 0.0 if np.isinf(node.half_life) else float(node.half_life)

            if isinstance(node.spatial, PointDistribution):
                spatial_cfg: AnySpatialDistributionConfig = PointSpatialConfig(
                    cell_jitter=(float(node.spatial.cell_jitter[0]), float(node.spatial.cell_jitter[1])),
                )
            elif isinstance(node.spatial, VoxelDistribution):
                v_sz = float(np.mean(node.spatial.voxel_size))
                dist_cfg = distribution_registry.get(node) if distribution_registry is not None else None
                if dist_cfg is None:
                    dist_cfg = NumpyDistributionConfig(path=f"{node_name or 'source_dist'}.npy")
                spatial_cfg = VoxelSpatialConfig(
                    voxel_size=v_sz,
                    distribution=dist_cfg,
                )
            else:
                raise TypeError(f"Неизвестный тип пространственного распределения источника: {type(node.spatial)}")

            exported_channels: List[RadiationChannelConfig] = []
            for ch in node.channels:
                species_str = cast(Literal['photon', 'electron', 'positron', 'alpha'], ch.species.name.lower())

                if isinstance(ch.spectrum, DiscreteSpectrum):
                    spec_cfg: AnySpectrumConfig = DiscreteSpectrumConfig(
                        lines=[
                            (f"{float(line.energy) / float(units.keV):.6g} keV", float(line.probability))
                            for line in ch.spectrum.lines
                        ]
                    )
                elif isinstance(ch.spectrum, ContinuousSpectrum):
                    spec_cfg = ContinuousSpectrumConfig(
                        energy_grid=[
                            f"{float(e_val) / float(units.keV):.6g} keV"
                            for e_val in ch.spectrum.energy_grid
                        ],
                        pdf_values=[float(p_val) for p_val in ch.spectrum.pdf_values],
                    )
                else:
                    raise TypeError(f"Неизвестный тип спектра канала источника: {type(ch.spectrum)}")

                if isinstance(ch.angular, IsotropicAngular):
                    ang_cfg: AnyAngularDistributionConfig = IsotropicAngularConfig()
                elif isinstance(ch.angular, ConeAngular):
                    ang_cfg = ConeAngularConfig(max_polar_angle=float(ch.angular.max_polar_angle))
                else:
                    raise TypeError(f"Неизвестный тип углового распределения канала источника: {type(ch.angular)}")

                exported_channels.append(
                    RadiationChannelConfig(
                        species=species_str,
                        branching_ratio=float(ch.branching_ratio),
                        spectrum=spec_cfg,
                        angular=ang_cfg,
                    )
                )

            return SourceConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                activity=act_val,
                half_life=half_life,
                spatial=spatial_cfg,
                channels=exported_channels,
                children=children_cfgs,
            )

        # 7. DoseGridNode
        if isinstance(node, DoseGridNode):
            return DoseGridNodeConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                size=(float(node.size[0]), float(node.size[1]), float(node.size[2])),
                dose_voxel_size=float(node.dose_voxel_size),
                is_active=bool(node.is_active),
                dtype=node.dtype,
                children=children_cfgs,
            )

        # 7.1. GantryNode
        if isinstance(node, GantryNode):
            gantry_transforms: List[TransformConfig] = []
            trans_x = float(node.local_matrix[0, 3])
            trans_y = float(node.local_matrix[1, 3])
            trans_z = float(node.local_matrix[2, 3])
            has_translation = not (
                math.isclose(trans_x, 0.0, abs_tol=1e-6)
                and math.isclose(trans_y, 0.0, abs_tol=1e-6)
                and math.isclose(trans_z, 0.0, abs_tol=1e-6)
            )
            if has_translation:
                gantry_transforms.append(TranslateConfig(x=trans_x, y=trans_y, z=trans_z))
            gantry_transforms.append(
                RotateConfig(
                    alpha=f"{float(np.degrees(node.gantry_angle)):.12g} deg",
                    in_local=has_translation,
                )
            )

            return GantryConfig(
                name=node_name,
                tags=node_tags,
                transformations=gantry_transforms,
                children=children_cfgs,
            )

        # 8. CompositeNode
        if isinstance(node, CompositeNode):
            return BaseCompositeNodeConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                children=children_cfgs,
            )

        # 9. SpatialNode (базовый)
        return BaseSpatialNodeConfig(
            name=node_name,
            tags=node_tags,
            transformations=transforms,
        )

    @classmethod
    def export_scene(
        cls,
        root_node: SpatialNode,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
    ) -> AnyNodeConfig:
        """
        Экспортирует корневой узел графа сцены.
        """
        return cls.export_node(
            root_node,
            distribution_registry=distribution_registry,
            slots_registry=slots_registry,
        )

    @classmethod
    def export_simulation_manager(
        cls,
        manager: SimulationManager,
        seed: Optional[int] = None,
    ) -> SimulationManagerConfig:
        """Экспортирует экземпляр SimulationManager в декларативную конфигурацию SimulationManagerConfig."""
        return SimulationManagerConfig(
            start_time=float(manager.start_time),
            stop_time=float(manager.stop_time),
            particles_number=int(manager.particles_number),
            batch_capacity=(
                None if int(manager.batch_capacity) == int(manager.particles_number) else int(manager.batch_capacity)
            ),
            buffer_capacity=(
                None
                if int(manager.data_buffer.interactions.capacity) == int(manager.batch_capacity)
                else int(manager.data_buffer.interactions.capacity)
            ),
            min_energy=float(manager.min_energy),
            enable_atomic_relaxation=bool(manager.enable_atomic_relaxation),
            photon_production_cut=float(manager.photon_production_cut),
            electron_production_cut=float(manager.electron_production_cut),
            seed=manager.seed if seed is None else int(seed),
        )

    @classmethod
    def export_to_config(
        cls,
        root_node: SpatialNode,
        simulation_manager_cfg: Optional[Union[SimulationManagerConfig, SimulationManager]] = None,
        data_manager_cfg: Optional[DataManagerConfig] = None,
        pool_size: int = 1,
        protocol: Optional[AnyProtocolConfig] = None,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
        logging_cfg: Optional[LoggingConfig] = None,
    ) -> SimulationConfig:
        """
        Формирует полный SimulationConfig из корневого узла сцены и опциональных параметров.
        """
        scene_cfg = cls.export_scene(
            root_node,
            distribution_registry=distribution_registry,
            slots_registry=slots_registry,
        )
        if isinstance(simulation_manager_cfg, SimulationManager):
            sim_mgr = cls.export_simulation_manager(simulation_manager_cfg)
        else:
            sim_mgr = simulation_manager_cfg or SimulationManagerConfig()
        data_mgr = data_manager_cfg or DataManagerConfig(
            filename="output.h5",
            handlers=[DirectStreamHandlerConfig()]
        )

        return SimulationConfig(
            pool_size=pool_size,
            protocol=protocol,
            simulation_manager=sim_mgr,
            data_manager=data_mgr,
            scene=scene_cfg,
            logging=logging_cfg,
        )

    @classmethod
    def prepare_distribution_registry_for_yaml(
        cls,
        root_node: SpatialNode,
        base_dir: Path,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
    ) -> Dict[SpatialNode, AnyDistributionConfig]:
        """
        Подготавливает реестр распределений перед сохранением в YAML:
        1. Приводит пути существующих файлов распределений к относительным относительно base_dir.
        2. Для воксельных узлов (WoodcockVoxelVolume, Source), созданных в памяти без внешнего файла
           на диске, автоматически сохраняет 3D-массив в .npy-файл в каталоге base_dir.
        """
        base_dir.mkdir(parents=True, exist_ok=True)
        adjusted_registry: Dict[SpatialNode, AnyDistributionConfig] = {}

        if distribution_registry is not None:
            for reg_node, orig_dist_cfg in distribution_registry.items():
                if orig_dist_cfg is not None and isinstance(orig_dist_cfg, (NumpyDistributionConfig, RawDistributionConfig)):
                    orig_path_str = str(orig_dist_cfg.path)
                    try:
                        resolved_file_path = Path(orig_path_str)
                        if not resolved_file_path.is_absolute():
                            resolved_file_path = (base_dir / resolved_file_path).resolve()
                        relative_path_str = os.path.relpath(resolved_file_path, base_dir)
                        normalized_path = relative_path_str.replace('\\', '/')
                    except (ValueError, OSError):
                        normalized_path = orig_path_str.replace('\\', '/')
                    adjusted_registry[reg_node] = orig_dist_cfg.model_copy(
                        update={'path': normalized_path}
                    )
                elif orig_dist_cfg is not None:
                    adjusted_registry[reg_node] = orig_dist_cfg

        def _ensure_node_distribution_saved(curr_node: SpatialNode) -> None:
            existing_orig = distribution_registry.get(curr_node) if distribution_registry is not None else None
            file_exists = False
            if existing_orig is not None and existing_orig.path:
                cand_path = Path(existing_orig.path)
                if cand_path.is_file() or (base_dir / cand_path).is_file():
                    file_exists = True

            if not file_exists:
                if isinstance(curr_node, WoodcockVoxelVolume) and curr_node.material_distribution is not None:
                    raw_name = (
                        Path(existing_orig.path).name
                        if (existing_orig is not None and existing_orig.path)
                        else f"{curr_node.name or 'phantom'}.npy"
                    )
                    if not raw_name.lower().endswith('.npy'):
                        raw_name = f"{Path(raw_name).stem}.npy"
                    target_npy = base_dir / raw_name
                    raw_indices = np.asarray(curr_node.material_distribution.view(np.ndarray), dtype=float)
                    np.save(target_npy, raw_indices)
                    actual_mapping: Optional[Dict[float, Union[float, str]]] = (
                        dict(existing_orig.mapping)
                        if (existing_orig is not None and existing_orig.mapping is not None)
                        else cls._extract_material_mapping(curr_node)
                    )
                    adjusted_registry[curr_node] = NumpyDistributionConfig(
                        path=raw_name,
                        mapping=actual_mapping if actual_mapping else {0.0: 'Vacuum'},
                    )
                elif isinstance(curr_node, Source):
                    if isinstance(curr_node.spatial, VoxelDistribution) and curr_node.spatial.distribution.size > 0:
                        raw_name = (
                            Path(existing_orig.path).name
                            if (existing_orig is not None and existing_orig.path)
                            else f"{curr_node.name or 'source_dist'}.npy"
                        )
                        if not raw_name.lower().endswith('.npy'):
                            raw_name = f"{Path(raw_name).stem}.npy"
                        target_npy = base_dir / raw_name
                        np.save(target_npy, np.asarray(curr_node.spatial.distribution, dtype=float))
                        adjusted_registry[curr_node] = NumpyDistributionConfig(path=raw_name)

            if isinstance(curr_node, CompositeNode):
                for child_item in curr_node.childs:
                    _ensure_node_distribution_saved(child_item)

        _ensure_node_distribution_saved(root_node)
        return adjusted_registry

    @classmethod
    def export_to_yaml(
        cls,
        root_node: SpatialNode,
        filepath: Union[str, Path],
        simulation_manager_cfg: Optional[SimulationManagerConfig] = None,
        data_manager_cfg: Optional[DataManagerConfig] = None,
        pool_size: int = 1,
        protocol: Optional[AnyProtocolConfig] = None,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
        logging_cfg: Optional[LoggingConfig] = None,
    ) -> None:
        """
        Экспортирует граф сцены напрямую в YAML-файл конфигурации.
        Пути к файлам распределений приводятся к относительным относительно каталога сохранения YAML.
        """
        target_path = Path(filepath).resolve()
        base_dir = target_path.parent
        adjusted_distribution_registry = cls.prepare_distribution_registry_for_yaml(
            root_node=root_node,
            base_dir=base_dir,
            distribution_registry=distribution_registry,
        )

        config = cls.export_to_config(
            root_node=root_node,
            simulation_manager_cfg=simulation_manager_cfg,
            data_manager_cfg=data_manager_cfg,
            pool_size=pool_size,
            protocol=protocol,
            distribution_registry=adjusted_distribution_registry,
            slots_registry=slots_registry,
            logging_cfg=logging_cfg,
        )
        dump_simulation_config(config, target_path)


__all__ = ["SceneExporter"]
