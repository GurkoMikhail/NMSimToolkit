import math
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation
import hepunits as units

from core.config.models import (
    AnyNodeConfig,
    BaseCompositeNodeConfig,
    BaseSpatialNodeConfig,
    BoxConfig,
    DataManagerConfig,
    DirectStreamHandlerConfig,
    GammaCameraConfig,
    NumpyDistributionConfig,
    ParametricParallelCollimatorConfig,
    ParametricParallelSquareCollimatorConfig,
    RotateConfig,
    SimulationConfig,
    SimulationManagerConfig,
    SourceConfig,
    TransformConfig,
    TranslateConfig,
    VolumeConfig,
    WoodcockVoxelVolumeConfig,
    DoseGridNodeConfig,
    RawDistributionConfig,
)
from core.config.yaml_dumper import dump_simulation_config
from core.geometry.geometries import Box
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.parametric_collimators import (
    ParametricParallelCollimator,
    ParametricParallelSquareCollimator,
)
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.scene.nodes import CompositeNode, SpatialNode
from core.scene.dose_grid_node import DoseGridNode
from core.source.sources import Source

logger = logging.getLogger(__name__)


class SceneExporter:
    """
    Класс для обратной сериализации чистого графа сцены (SpatialNode, Volume)
    в Pydantic-конфигурации SimulationConfig и сохранения в формат YAML.
    """

    @classmethod
    def decompose_matrix(cls, matrix: np.ndarray) -> List[TransformConfig]:
        """
        Декомпозирует матрицу трансформации 4x4 на трансляцию и поворот (углы Эйлера).
        """
        transforms: List[TransformConfig] = []
        if matrix is None or np.allclose(matrix, np.eye(4)):
            return transforms

        # 1. Извлечение трансляции
        tx = float(matrix[0, 3])
        ty = float(matrix[1, 3])
        tz = float(matrix[2, 3])
        if not (math.isclose(tx, 0.0, abs_tol=1e-6) and math.isclose(ty, 0.0, abs_tol=1e-6) and math.isclose(tz, 0.0, abs_tol=1e-6)):
            transforms.append(TranslateConfig(x=tx, y=ty, z=tz))

        # 2. Извлечение вращения
        rot_mat = matrix[:3, :3]
        if not np.allclose(rot_mat, np.eye(3), atol=1e-5):
            try:
                r = Rotation.from_matrix(rot_mat)
                euler = r.as_euler('xyz', degrees=False)
                transforms.append(RotateConfig(
                    alpha=float(euler[0]),
                    beta=float(euler[1]),
                    gamma=float(euler[2])
                ))
            except ValueError as e:
                logger.warning(f"Ошибка при декомпозиции матрицы вращения: {e}")

        return transforms

    @classmethod
    def export_node(cls, node: SpatialNode) -> AnyNodeConfig:
        """
        Рекурсивно экспортирует узел сцены в Pydantic-модель конфигурации.
        """
        node_name = node.name
        transforms = cls.decompose_matrix(node.local_matrix)

        # 1. GammaCamera
        if isinstance(node, GammaCamera):
            collimator_cfg = cls.export_node(node.collimator)
            detector_cfg = cls.export_node(node.detector)
            return GammaCameraConfig(
                name=node_name,
                transformations=transforms,
                collimator=collimator_cfg,
                detector=detector_cfg,
            )

        # 2. ParametricParallelCollimator
        if isinstance(node, ParametricParallelCollimator):
            size_tuple = (float(node.size[0]), float(node.size[1]), float(node.size[2]))
            mat_name = node.material.name
            return ParametricParallelCollimatorConfig(
                name=node_name,
                transformations=transforms,
                size=size_tuple,
                hole_diameter=float(node.hole_diameter),
                septa_thickness=float(node.septa),
                material=mat_name,
            )

        # 3. ParametricParallelSquareCollimator
        if isinstance(node, ParametricParallelSquareCollimator):
            size_tuple = (float(node.size[0]), float(node.size[1]), float(node.size[2]))
            mat_name = node.material.name
            return ParametricParallelSquareCollimatorConfig(
                name=node_name,
                transformations=transforms,
                size=size_tuple,
                hole_size=float(node.hole_width),
                septa_thickness=float(node.septa),
                material=mat_name,
            )

        # 4. WoodcockVoxelVolume
        if isinstance(node, WoodcockVoxelVolume):
            v_size = float(np.mean(node.voxel_size)) if isinstance(node.voxel_size, (list, tuple, np.ndarray, Sequence)) else float(node.voxel_size)
            dist_path = node.distribution_path or 'phantom.npy'
            if node.distribution_config is not None:
                dist_cfg = node.distribution_config
            else:
                p_str = str(dist_path).lower()
                if p_str.endswith(('.dat', '.raw', '.txt')):
                    sh = (int(node.material_distribution.shape[0]), int(node.material_distribution.shape[1]), int(node.material_distribution.shape[2]))
                    dist_cfg = RawDistributionConfig(path=str(dist_path), shape=sh, order='F')
                else:
                    dist_cfg = NumpyDistributionConfig(path=str(dist_path))
            children_cfgs = [cls.export_node(c) for c in node.childs]
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
                # По умолчанию Box с размерами объекта
                geo_cfg = BoxConfig(x=float(node.size[0]), y=float(node.size[1]), z=float(node.size[2]))

            mat_name = node.material.name
            children_cfgs = [cls.export_node(c) for c in node.childs]

            return VolumeConfig(
                name=node_name,
                transformations=transforms,
                geometry=geo_cfg,
                material=mat_name,
                children=children_cfgs,
            )

        # 6. Source
        if isinstance(node, Source):
            dist_path = node.distribution_path or 'source_dist.npy'
            act = float(np.sum(node.initial_activity)) if node.initial_activity is not None else None
            v_sz = float(np.mean(node.voxel_size)) if isinstance(node.voxel_size, (list, tuple, np.ndarray, Sequence)) else float(node.voxel_size)
            if node.distribution_config is not None:
                dist_cfg = node.distribution_config
            else:
                p_str = str(dist_path).lower()
                if p_str.endswith(('.dat', '.raw', '.txt')):
                    sh = (int(node.distribution.shape[0]), int(node.distribution.shape[1]), int(node.distribution.shape[2]))
                    dist_cfg = RawDistributionConfig(path=str(dist_path), shape=sh, order='F')
                else:
                    dist_cfg = NumpyDistributionConfig(path=str(dist_path))
            children_cfgs = [cls.export_node(c) for c in node.childs]
            half_life = float(node.half_life)
            rad_type = node.radiation_type
            if len(node.energy) == 1:
                energy_val: Union[float, List[List[float]]] = float(node.energy["energy"][0])
            else:
                energy_val = [[float(e), float(p)] for e, p in zip(node.energy["energy"], node.energy["probability"])]
            return SourceConfig(
                name=node_name,
                transformations=transforms,
                distribution=dist_cfg,
                activity=act,
                voxel_size=v_sz,
                radiation_type=rad_type,
                energy=energy_val,
                half_life=half_life,
                children=children_cfgs,
            )

        # 7. DoseGridNode
        if isinstance(node, DoseGridNode):
            children_cfgs = [cls.export_node(c) for c in node.childs]
            return DoseGridNodeConfig(
                name=node_name,
                transformations=transforms,
                size=(float(node.size[0]), float(node.size[1]), float(node.size[2])),
                dose_voxel_size=float(node.dose_voxel_size),
                is_active=bool(node.is_active),
                children=children_cfgs,
            )

        # 8. CompositeNode
        if isinstance(node, CompositeNode):
            children_cfgs = [cls.export_node(c) for c in node.childs]
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
    def export_scene(cls, root_node: SpatialNode) -> AnyNodeConfig:
        """
        Экспортирует корневой узел графа сцены.
        """
        return cls.export_node(root_node)

    @classmethod
    def export_to_config(
        cls,
        root_node: SpatialNode,
        simulation_manager_cfg: Optional[SimulationManagerConfig] = None,
        data_manager_cfg: Optional[DataManagerConfig] = None,
        pool_size: int = 1,
    ) -> SimulationConfig:
        """
        Формирует полный SimulationConfig из корневого узла сцены и опциональных параметров.
        """
        scene_cfg = cls.export_scene(root_node)
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
    ) -> None:
        """
        Экспортирует граф сцены напрямую в YAML-файл конфигурации.
        """
        config = cls.export_to_config(
            root_node=root_node,
            simulation_manager_cfg=simulation_manager_cfg,
            data_manager_cfg=data_manager_cfg,
            pool_size=pool_size,
        )
        dump_simulation_config(config, filepath)
