import math
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation
import hepunits as units

from core.config.models import (
    AnyDistributionConfig,
    AnyNodeConfig,
    BaseCompositeNodeConfig,
    BaseSpatialNodeConfig,
    BoxConfig,
    DataManagerConfig,
    DirectStreamHandlerConfig,
    GammaCameraConfig,
    GammaCameraSlotsConfig,
    NumpyDistributionConfig,
    ParametricParallelCollimatorConfig,
    DirectParallelCollimatorConfig,
    RotateConfig,
    SimulationConfig,
    SimulationManagerConfig,
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
    ) -> AnyNodeConfig:
        """
        Рекурсивно экспортирует узел сцены в Pydantic-модель конфигурации.
        """
        node_name = node.name
        transforms = cls.decompose_matrix(node.local_matrix)

        # 1. GammaCameraNode
        if isinstance(node, GammaCameraNode):
            children_cfgs = [
                cls.export_node(child_node, distribution_registry=distribution_registry, slots_registry=slots_registry)
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
                transformations=transforms,
                slots=slots_cfg,
                children=children_cfgs,
            )


        # 2. ParametricParallelCollimator
        if isinstance(node, ParametricParallelCollimator):
            size_tuple = (float(node.size[0]), float(node.size[1]), float(node.size[2]))
            mat_name = node.material.name
            hole_shape_val = node.hole_shape.value if isinstance(node.hole_shape, CollimatorHoleShape) else str(node.hole_shape)
            return ParametricParallelCollimatorConfig(
                name=node_name,
                transformations=transforms,
                size=size_tuple,
                hole_diameter=float(node.hole_diameter),
                septa=float(node.septa),
                material=mat_name,
                hole_shape=hole_shape_val,
            )

        # 3.1. DirectParallelCollimator
        if isinstance(node, DirectParallelCollimator):
            size_tuple = (float(node.size[0]), float(node.size[1]), float(node.size[2]))
            mat_name = node.material.name
            hole_mat_name = node.explicit_hole_material.name if node.explicit_hole_material is not None else None
            hole_shape_val = node.hole_shape.value if isinstance(node.hole_shape, CollimatorHoleShape) else str(node.hole_shape)
            return DirectParallelCollimatorConfig(
                name=node_name,
                transformations=transforms,
                size=size_tuple,
                hole_diameter=float(node.hole_diameter),
                septa=float(node.septa),
                material=mat_name,
                hole_material=hole_mat_name,
                hole_shape=hole_shape_val,
            )

        # 4. WoodcockVoxelVolume
        if isinstance(node, WoodcockVoxelVolume):
            v_size = float(np.mean(node.voxel_size)) if isinstance(node.voxel_size, (list, tuple, np.ndarray, Sequence)) else float(node.voxel_size)
            dist_cfg = None
            if distribution_registry is not None:
                dist_cfg = distribution_registry.get(node)
            if dist_cfg is None:
                dist_cfg = NumpyDistributionConfig(path=f"{node_name or 'phantom'}.npy")
            children_cfgs = [cls.export_node(child_node, distribution_registry=distribution_registry, slots_registry=slots_registry) for child_node in node.childs]
            return WoodcockVoxelVolumeConfig(
                name=node_name,
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
                geo_cfg = BoxConfig(x=float(node.size[0]), y=float(node.size[1]), z=float(node.size[2]))

            mat_name = node.material.name
            children_cfgs = [cls.export_node(child_node, distribution_registry=distribution_registry, slots_registry=slots_registry) for child_node in node.childs]

            return VolumeConfig(
                name=node_name,
                transformations=transforms,
                geometry=geo_cfg,
                material=mat_name,
                children=children_cfgs,
            )

        # 6. Source
        if isinstance(node, Source):
            act = float(np.sum(node.initial_activity)) if node.initial_activity is not None else None
            v_sz = float(np.mean(node.voxel_size)) if isinstance(node.voxel_size, (list, tuple, np.ndarray, Sequence)) else float(node.voxel_size)
            dist_cfg = None
            if distribution_registry is not None:
                dist_cfg = distribution_registry.get(node)
            if dist_cfg is None:
                dist_cfg = NumpyDistributionConfig(path=f"{node_name or 'source_dist'}.npy")
            children_cfgs = [cls.export_node(child_node, distribution_registry=distribution_registry, slots_registry=slots_registry) for child_node in node.childs]
            half_life = float(node.half_life)
            rad_type = node.radiation_type
            if len(node.energy) == 1:
                energy_val: Union[float, List[List[float]]] = float(node.energy["energy"][0])
            else:
                energy_val = [[float(energy_value), float(probability_value)] for energy_value, probability_value in zip(node.energy["energy"], node.energy["probability"])]
            return SourceConfig(
                name=node_name,
                transformations=transforms,
                distribution=dist_cfg,
                activity=act,
                voxel_size=v_sz,
                radiation_type=str(rad_type),
                energy=energy_val,
                half_life=half_life,
                children=children_cfgs,
            )

        # 7. DoseGridNode
        if isinstance(node, DoseGridNode):
            children_cfgs = [cls.export_node(child_item, distribution_registry=distribution_registry, slots_registry=slots_registry) for child_item in node.childs]
            return DoseGridNodeConfig(
                name=node_name,
                transformations=transforms,
                size=(float(node.size[0]), float(node.size[1]), float(node.size[2])),
                dose_voxel_size=float(node.dose_voxel_size),
                is_active=bool(node.is_active),
                children=children_cfgs,
            )

        # 7.1. GantryNode
        if isinstance(node, GantryNode):
            children_cfgs = [cls.export_node(child_item, distribution_registry=distribution_registry, slots_registry=slots_registry) for child_item in node.childs]
            return GantryConfig(
                name=node_name,
                transformations=transforms,
                children=children_cfgs,
            )

        # 8. CompositeNode
        if isinstance(node, CompositeNode):
            children_cfgs = [cls.export_node(child_item, distribution_registry=distribution_registry, slots_registry=slots_registry) for child_item in node.childs]
            return BaseCompositeNodeConfig(
                name=node_name,
                transformations=transforms,
                children=children_cfgs,
            )

        # 9. SpatialNode (базовый)
        return BaseSpatialNodeConfig(
            name=node_name,
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
        return cls.export_node(root_node, distribution_registry=distribution_registry, slots_registry=slots_registry)

    @classmethod
    def export_to_config(
        cls,
        root_node: SpatialNode,
        simulation_manager_cfg: Optional[SimulationManagerConfig] = None,
        data_manager_cfg: Optional[DataManagerConfig] = None,
        pool_size: int = 1,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
    ) -> SimulationConfig:
        """
        Формирует полный SimulationConfig из корневого узла сцены и опциональных параметров.
        """
        scene_cfg = cls.export_scene(root_node, distribution_registry=distribution_registry, slots_registry=slots_registry)
        sim_mgr = simulation_manager_cfg or SimulationManagerConfig()
        data_mgr = data_manager_cfg or DataManagerConfig(
            filename="output.h5",
            handlers=[DirectStreamHandlerConfig()]
        )

        return SimulationConfig(
            pool_size=pool_size,
            simulation_manager=sim_mgr,
            data_manager=data_mgr,
            scene=scene_cfg,
        )

    @classmethod
    def export_to_yaml(
        cls,
        root_node: SpatialNode,
        filepath: Union[str, Path],
        simulation_manager_cfg: Optional[SimulationManagerConfig] = None,
        data_manager_cfg: Optional[DataManagerConfig] = None,
        pool_size: int = 1,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
    ) -> None:
        """
        Экспортирует граф сцены напрямую в YAML-файл конфигурации.
        """
        config = cls.export_to_config(
            root_node=root_node,
            simulation_manager_cfg=simulation_manager_cfg,
            data_manager_cfg=data_manager_cfg,
            pool_size=pool_size,
            distribution_registry=distribution_registry,
            slots_registry=slots_registry,
        )
        dump_simulation_config(config, filepath)
