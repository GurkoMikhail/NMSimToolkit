import numpy as np
from typing import Dict, Any, Callable

import settings.database_setting as database_setting
from core.config.models import (
    AnyNodeConfig, VolumeConfig, GammaCameraConfig, WoodcockVoxelVolumeConfig,
    ParametricParallelCollimatorConfig, ParametricParallelSquareCollimatorConfig,
    SourceConfig, BoxConfig, SimulationConfig, TranslateConfig, RotateConfig
)
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.parametric_collimators import ParametricParallelCollimator, ParametricParallelSquareCollimator
from core.source.sources import Source
from core.scene.nodes import SpatialNode

class SceneBuilder:
    def __init__(self):
        self.factory_map: Dict[str, Callable[[AnyNodeConfig], SpatialNode]] = {
            'Volume': self._build_volume,
            'GammaCamera': self._build_gamma_camera,
            'WoodcockVoxelVolume': self._build_woodcock_voxel_volume,
            'ParametricParallelCollimator': self._build_parametric_parallel_collimator,
            'ParametricParallelSquareCollimator': self._build_parametric_parallel_square_collimator,
            'Source': self._build_source
        }
        self.node_cache: Dict[str, SpatialNode] = {}

    def build_scene(self, config: AnyNodeConfig) -> SpatialNode:
        root_node = self._build_node(config)
        return root_node

    def _build_node(self, config: AnyNodeConfig) -> SpatialNode:
        node_type = config.type
        if node_type not in self.factory_map:
            raise ValueError(f"Unknown node type: {node_type}")

        node = self.factory_map[node_type](config)

        # Apply transformations
        for transform in config.transformations:
            if isinstance(transform, TranslateConfig):
                node.translate(transform.x, transform.y, transform.z, transform.in_local)
            elif isinstance(transform, RotateConfig):
                node.rotate(transform.alpha, transform.beta, transform.gamma, transform.rotation_center, transform.in_local)

        # Build children
        for child_config in config.children:
            child_node = self._build_node(child_config)
            node.add_child(child_node)

        return node

    def _get_material(self, name: str):
        return database_setting.material_database[name]

    def _build_geometry(self, config):
        if isinstance(config, BoxConfig):
            return Box(config.x, config.y, config.z)
        raise ValueError(f"Unknown geometry type: {config.type}")

    def _build_volume(self, config: VolumeConfig) -> Volume:
        geometry = self._build_geometry(config.geometry)
        material = self._get_material(config.material)
        return Volume(geometry=geometry, material=material, name=config.name)

    def _build_woodcock_voxel_volume(self, config: WoodcockVoxelVolumeConfig) -> WoodcockVoxelVolume:
        material_distribution = np.load(config.material_distribution_path)
        # Note: Depending on how MaterialArray is initialized from numpy array, this might need adaptation
        from core.materials.materials import MaterialArray
        mat_arr = MaterialArray(material_distribution.shape)
        mat_arr.ID = material_distribution # Simplified assuming distribution is IDs
        return WoodcockVoxelVolume(voxel_size=config.voxel_size, material_distribution=mat_arr, name=config.name)

    def _build_gamma_camera(self, config: GammaCameraConfig) -> GammaCamera:
        collimator = self._build_node(config.collimator)
        detector = self._build_node(config.detector)
        return GammaCamera(
            collimator=collimator,
            detector=detector,
            gap=config.gap,
            shielding_thickness=config.shielding_thickness,
            glass_backend_thickness=config.glass_backend_thickness,
            name=config.name
        )

    def _build_parametric_parallel_collimator(self, config: ParametricParallelCollimatorConfig) -> ParametricParallelCollimator:
        material = self._get_material(config.material)
        return ParametricParallelCollimator(
            size=config.size,
            hole_diameter=config.hole_diameter,
            septa_thickness=config.septa_thickness,
            material=material,
            name=config.name
        )

    def _build_parametric_parallel_square_collimator(self, config: ParametricParallelSquareCollimatorConfig) -> ParametricParallelSquareCollimator:
        material = self._get_material(config.material)
        return ParametricParallelSquareCollimator(
            size=config.size,
            hole_size=config.hole_size,
            septa_thickness=config.septa_thickness,
            material=material,
            name=config.name
        )

    def _build_source(self, config: SourceConfig) -> Source:
        distribution = np.load(config.distribution_path)
        return Source(
            distribution=distribution,
            activity=config.activity,
            voxel_size=config.voxel_size,
            radiation_type=config.radiation_type,
            energy=config.energy,
            half_life=config.half_life
        )
