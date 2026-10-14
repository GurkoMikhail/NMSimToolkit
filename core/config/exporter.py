import os
import math
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from weakref import WeakKeyDictionary

import numpy as np
from scipy.spatial.transform import Rotation
import hepunits as units

from core.config.models import (
    AnyDistributionConfig,
    AnyNodeConfig,
    AnyProtocolConfig,
    BaseCompositeNodeConfig,
    BaseSpatialNodeConfig,
    BoxConfig,
    DataManagerConfig,
    DirectStreamHandlerConfig,
    GammaCameraConfig,
    GammaCameraSlotsConfig,
    LoggingConfig,
    NumpyDistributionConfig,
    ParametricParallelCollimatorConfig,
    DirectParallelCollimatorConfig,
    RotateConfig,
    SimulationConfig,
    SimulationManagerConfig,
    SourceMetadataConfig,
    SourceConfig,
    TransformConfig,
    TranslateConfig,
    MatrixTransformConfig,
    VolumeConfig,
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
from core.source.sources import Source

logger = logging.getLogger(__name__)


class SceneExporter:
    """
    Класс для обратной сериализации чистого графа сцены (SpatialNode, Volume)
    в Pydantic-конфигурации SimulationConfig и сохранения в формат YAML.
    """

    _source_metadata_registry: WeakKeyDictionary[SpatialNode, SourceMetadataConfig] = WeakKeyDictionary()

    @classmethod
    def register_source_metadata(cls, node: SpatialNode, metadata: SourceMetadataConfig) -> None:
        """Регистрирует декларативные метаданные источника во внешнем реестре без модификации узла ядра."""
        cls._source_metadata_registry[node] = metadata

    @classmethod
    def get_source_metadata(cls, node: SpatialNode) -> Optional[SourceMetadataConfig]:
        """Возвращает зарегистрированные декларативные метаданные источника из внешнего реестра."""
        return cls._source_metadata_registry.get(node)

    @classmethod
    def decompose_matrix(cls, matrix: np.ndarray, as_matrix: bool = True) -> List[TransformConfig]:
        """
        Преобразует матрицу трансформации 4x4 в список TransformConfig.
        По умолчанию при as_matrix=True выгружает полную матрицу 4x4 (MatrixTransformConfig),
        что гарантирует 100% стабильность сохранения/загрузки без потери знаков и Gimbal lock.
        При as_matrix=False выполняет декомпозицию на TranslateConfig и RotateConfig.
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

        # 1. Извлечение трансляции
        translation_x = float(matrix[0, 3])
        translation_y = float(matrix[1, 3])
        translation_z = float(matrix[2, 3])
        if not (math.isclose(translation_x, 0.0, abs_tol=1e-6) and math.isclose(translation_y, 0.0, abs_tol=1e-6) and math.isclose(translation_z, 0.0, abs_tol=1e-6)):
            transforms.append(TranslateConfig(x=translation_x, y=translation_y, z=translation_z))

        # 2. Извлечение вращения
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

        return transforms

    @classmethod
    def export_node(
        cls,
        node: SpatialNode,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
        source_metadata_registry: Optional[Dict[SpatialNode, SourceMetadataConfig]] = None,
    ) -> AnyNodeConfig:
        """
        Рекурсивно экспортирует узел сцены в Pydantic-модель конфигурации.
        """
        node_name = node.name
        node_tags = list(node.tags) if node.tags else []
        transforms = cls.decompose_matrix(node.local_matrix)

        # 1. GammaCameraNode
        if isinstance(node, GammaCameraNode):
            children_cfgs = [
                cls.export_node(
                    child_node,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_node in node.childs
            ]
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
            hole_shape_val = node.hole_shape.value if isinstance(node.hole_shape, CollimatorHoleShape) else str(node.hole_shape)
            children_cfgs = [
                cls.export_node(
                    child_node,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_node in node.childs
            ]
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
            hole_shape_val = node.hole_shape.value if isinstance(node.hole_shape, CollimatorHoleShape) else str(node.hole_shape)
            children_cfgs = [
                cls.export_node(
                    child_node,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_node in node.childs
                if child_node is not node.lead_body and child_node is not node.channels
            ]
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
            dist_cfg = None
            if distribution_registry is not None:
                dist_cfg = distribution_registry.get(node)

            if dist_cfg is None:
                actual_mapping: Optional[Dict[float, Union[float, str]]] = None
                if node.material_distribution is not None and len(node.material_distribution.element_list) > 0:
                    actual_mapping = {
                        float(element_idx): material_item.name
                        for element_idx, material_item in enumerate(node.material_distribution.element_list)
                    }
                dist_cfg = NumpyDistributionConfig(
                    path=f"{node_name or 'phantom'}.npy",
                    mapping=actual_mapping if actual_mapping is not None else {0.0: 'Vacuum'},
                )
            elif isinstance(dist_cfg, (NumpyDistributionConfig, RawDistributionConfig)) and dist_cfg.mapping is None:
                actual_mapping = None
                if node.material_distribution is not None and len(node.material_distribution.element_list) > 0:
                    actual_mapping = {
                        float(element_idx): material_item.name
                        for element_idx, material_item in enumerate(node.material_distribution.element_list)
                    }
                if actual_mapping is not None:
                    dist_cfg = dist_cfg.model_copy(update={'mapping': actual_mapping})

            children_cfgs = [
                cls.export_node(
                    child_node,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_node in node.childs
            ]
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

            mat_name = node.material.name
            children_cfgs = [
                cls.export_node(
                    child_node,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_node in node.childs
            ]

            return VolumeConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                geometry=geo_cfg,
                material=mat_name,
                children=children_cfgs,
            )

        # 6. Source
        if isinstance(node, Source):
            act = float(np.sum(node.initial_activity)) if node.initial_activity is not None else None
            act_val = None
            if act is not None:
                scaled_activity = float(act) / float(units.MBq)
                act_val = f"{scaled_activity:.6g} MBq"

            v_sz = float(np.mean(node.voxel_size)) if isinstance(node.voxel_size, (list, tuple, np.ndarray, Sequence)) else float(node.voxel_size)
            dist_cfg = None
            if distribution_registry is not None:
                dist_cfg = distribution_registry.get(node)
            if dist_cfg is None:
                dist_cfg = NumpyDistributionConfig(path=f"{node_name or 'source_dist'}.npy")
            children_cfgs = [
                cls.export_node(
                    child_node,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_node in node.childs
            ]
            half_life = float(node.half_life)
            rad_type = node.radiation_type
            source_meta: Optional[SourceMetadataConfig] = None
            if source_metadata_registry is not None:
                source_meta = source_metadata_registry.get(node)
            if source_meta is None:
                source_meta = cls._source_metadata_registry.get(node)
            radionuclide_val = source_meta.radionuclide if source_meta is not None else None
            phantom_type_val = source_meta.phantom_type if source_meta is not None else None

            energy_val = [
                [f"{float(energy_value) / float(units.keV):.6g} keV", float(probability_value)]
                for energy_value, probability_value in zip(node.energy["energy"], node.energy["probability"])
            ]
            return SourceConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                distribution=dist_cfg,
                activity=act_val,
                voxel_size=v_sz,
                radiation_type=str(rad_type),
                energy=energy_val,
                half_life=half_life,
                radionuclide=radionuclide_val,
                phantom_type=phantom_type_val,
                children=children_cfgs,
            )

        # 7. DoseGridNode
        if isinstance(node, DoseGridNode):
            children_cfgs = [
                cls.export_node(
                    child_item,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_item in node.childs
            ]
            return DoseGridNodeConfig(
                name=node_name,
                tags=node_tags,
                transformations=transforms,
                size=(float(node.size[0]), float(node.size[1]), float(node.size[2])),
                dose_voxel_size=float(node.dose_voxel_size),
                is_active=bool(node.is_active),
                children=children_cfgs,
            )

        # 7.1. GantryNode
        if isinstance(node, GantryNode):
            children_cfgs = [
                cls.export_node(
                    child_item,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_item in node.childs
            ]
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
            children_cfgs = [
                cls.export_node(
                    child_item,
                    distribution_registry=distribution_registry,
                    slots_registry=slots_registry,
                    source_metadata_registry=source_metadata_registry,
                )
                for child_item in node.childs
            ]
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
        source_metadata_registry: Optional[Dict[SpatialNode, SourceMetadataConfig]] = None,
    ) -> AnyNodeConfig:
        """
        Экспортирует корневой узел графа сцены.
        """
        return cls.export_node(
            root_node,
            distribution_registry=distribution_registry,
            slots_registry=slots_registry,
            source_metadata_registry=source_metadata_registry,
        )

    @classmethod
    def export_to_config(
        cls,
        root_node: SpatialNode,
        simulation_manager_cfg: Optional[SimulationManagerConfig] = None,
        data_manager_cfg: Optional[DataManagerConfig] = None,
        pool_size: int = 1,
        protocol: Optional[AnyProtocolConfig] = None,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
        source_metadata_registry: Optional[Dict[SpatialNode, SourceMetadataConfig]] = None,
        logging_cfg: Optional[LoggingConfig] = None,
    ) -> SimulationConfig:
        """
        Формирует полный SimulationConfig из корневого узла сцены и опциональных параметров.
        """
        scene_cfg = cls.export_scene(
            root_node,
            distribution_registry=distribution_registry,
            slots_registry=slots_registry,
            source_metadata_registry=source_metadata_registry,
        )
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
        source_metadata_registry: Optional[Dict[SpatialNode, SourceMetadataConfig]] = None,
        logging_cfg: Optional[LoggingConfig] = None,
    ) -> None:
        """
        Экспортирует граф сцены напрямую в YAML-файл конфигурации.
        Пути к файлам распределений приводятся к относительным относительно каталога сохранения YAML.
        """
        target_path = Path(filepath).resolve()
        base_dir = target_path.parent

        adjusted_distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None
        if distribution_registry is not None:
            adjusted_distribution_registry = {}
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
                    adjusted_distribution_registry[reg_node] = orig_dist_cfg.model_copy(
                        update={'path': normalized_path}
                    )
                else:
                    adjusted_distribution_registry[reg_node] = orig_dist_cfg

        config = cls.export_to_config(
            root_node=root_node,
            simulation_manager_cfg=simulation_manager_cfg,
            data_manager_cfg=data_manager_cfg,
            pool_size=pool_size,
            protocol=protocol,
            distribution_registry=adjusted_distribution_registry,
            slots_registry=slots_registry,
            source_metadata_registry=source_metadata_registry,
            logging_cfg=logging_cfg,
        )
        dump_simulation_config(config, target_path)


__all__ = ["SceneExporter"]
