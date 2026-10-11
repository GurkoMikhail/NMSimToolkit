"""
Модуль унифицированного сбора процедурных метаданных симуляции (Metadata Collector).

Предоставляет модульную систему провайдеров для извлечения информации о модальности
протокола исследования, кинематике подвижных частей (станины томографа), геометрии
детекторов (кристаллов и коллиматоров), источнике и характеристиках симуляции
в строгом соответствии с 4 каноническими группами NeXus/DICOM NM:
metadata/gantry, metadata/geometry, metadata/source, metadata/simulation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import hepunits as units

from core.config.models import (
    BaseProtocolConfig,
    CompositeNodeConfig,
    CustomSweepProtocolConfig,
    SimulationManagerConfig,
    SourceConfig,
    StepAndShootProtocolConfig,
)
from core.geometry.direct_collimators import DirectParallelCollimator
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.physics.processes import get_process_mapping
from core.scene.gantry_node import GantryNode
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.nodes import CompositeNode, SpatialNode
from core.source.sources import Source


def _find_nodes_recursive(current_node: SpatialNode, target_class: type) -> List[Any]:
    """
    Рекурсивно выполняет обход графа сцены и возвращает список узлов заданного типа.
    """
    matching_nodes: List[Any] = []
    if isinstance(current_node, target_class):
        matching_nodes.append(current_node)
    if isinstance(current_node, CompositeNode):
        for child_node in current_node.childs:
            matching_nodes.extend(_find_nodes_recursive(child_node, target_class))
    return matching_nodes


def _find_node_by_name(current_node: SpatialNode, target_name: str) -> Optional[SpatialNode]:
    """
    Рекурсивно ищет узел в поддереве по его имени.
    """
    if current_node.name == target_name:
        return current_node
    if isinstance(current_node, CompositeNode):
        for child_node in current_node.childs:
            found_node = _find_node_by_name(child_node, target_name)
            if found_node is not None:
                return found_node
    return None


def _find_source_config(current_config: Any) -> Optional[SourceConfig]:
    """
    Рекурсивно выполняет поиск SourceConfig в дереве конфигурации сцены
    со строгой статической типизацией без утиной типизации.
    """
    if isinstance(current_config, SourceConfig):
        return current_config
    if isinstance(current_config, CompositeNodeConfig):
        for child_config in current_config.children:
            found_source = _find_source_config(child_config)
            if found_source is not None:
                return found_source
    return None


class ProtocolMetadataProvider:
    """
    Провайдер метаданных протокола сканирования.
    Инспектирует конфигурацию протокола исследования со строгой проверкой типов через isinstance.
    """

    def collect(
        self,
        protocol: Optional[BaseProtocolConfig],
        context: Dict[str, Any],
        task_id: int,
    ) -> Dict[str, Any]:
        """
        Формирует словарь метаданных протокола сканирования.
        """
        view_index: int = int(context.get("view_index", task_id))

        if isinstance(protocol, StepAndShootProtocolConfig):
            if isinstance(protocol.radius, list):
                orbit_radius_value = float(protocol.radius[view_index % len(protocol.radius)]) if protocol.radius else 0.0
            else:
                orbit_radius_value = float(protocol.radius) if protocol.radius is not None else 0.0
            return {
                "modality": "StepAndShoot",
                "view_index": view_index,
                "views_total": int(protocol.steps),
                "exposure_time": float(protocol.time_per_step),
                "orbit_radius": orbit_radius_value,
            }

        if isinstance(protocol, CustomSweepProtocolConfig):
            views_count: int = 1
            if protocol.zipped_variables:
                first_variable_values = next(iter(protocol.zipped_variables.values()))
                views_count = len(first_variable_values)
            return {
                "modality": "CustomSweep",
                "view_index": view_index,
                "views_total": views_count,
                "exposure_time": 0.0,
                "orbit_radius": 0.0,
            }

        return {
            "modality": "CustomSweep" if protocol is None else protocol.__class__.__name__,
            "view_index": view_index,
            "views_total": 1,
            "exposure_time": 0.0,
            "orbit_radius": 0.0,
        }


class KinematicsMetadataProvider:
    """
    Провайдер метаданных пространственной кинематики подвижных узлов сцены (GantryNode).
    """

    def collect(
        self,
        root_scene: SpatialNode,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Инспектирует станину томографа в графе сцены и возвращает ее кинематические параметры.
        """
        gantry_nodes: List[GantryNode] = _find_nodes_recursive(root_scene, GantryNode)

        if gantry_nodes:
            primary_gantry = gantry_nodes[0]
            kinematics_data: Dict[str, Any] = {
                "gantry_name": primary_gantry.name,
                "gantry_angle": float(primary_gantry.gantry_angle),
            }
            if len(gantry_nodes) > 1:
                kinematics_data["all_gantries"] = {
                    gantry_node.name: float(gantry_node.gantry_angle)
                    for gantry_node in gantry_nodes
                }
            return kinematics_data

        context_gantry_angle = 0.0
        if context is not None and "gantry_angle" in context:
            context_gantry_angle = float(context["gantry_angle"])
        elif context is not None and "current_angle" in context:
            context_gantry_angle = float(context["current_angle"])

        return {
            "gantry_name": "None",
            "gantry_angle": context_gantry_angle,
        }


class DetectorMetadataProvider:
    """
    Провайдер метаданных детекторных головок (GammaCameraNode) и сцинтилляционных кристаллов.
    """

    def collect(self, root_scene: SpatialNode) -> Dict[str, Any]:
        """
        Инспектирует гамма-камеры сцены, разрешает слоты кристаллов и сохраняет их матрицы.
        """
        camera_nodes: List[GammaCameraNode] = _find_nodes_recursive(root_scene, GammaCameraNode)
        detectors_dict: Dict[str, Any] = {}

        for camera_node in camera_nodes:
            crystal_name: Optional[str] = camera_node.slots.get("crystal")
            crystal_node: Optional[SpatialNode] = None

            if crystal_name is not None:
                crystal_node = _find_node_by_name(camera_node, crystal_name)
                if crystal_node is None:
                    crystal_node = _find_node_by_name(root_scene, crystal_name)

            if crystal_node is not None:
                target_matrix = np.array(crystal_node.global_matrix, dtype=np.float64)
                effective_crystal_name = crystal_node.name
                effective_tags = list(crystal_node.tags)
            else:
                target_matrix = np.array(camera_node.global_matrix, dtype=np.float64)
                effective_crystal_name = crystal_name if crystal_name is not None else camera_node.name
                effective_tags = list(camera_node.tags)

            camera_record: Dict[str, Any] = {
                "camera_name": camera_node.name,
                "crystal_name": effective_crystal_name,
                "global_matrix": target_matrix,
            }
            if effective_tags:
                camera_record["tags"] = np.array(
                    [tag_value.encode("utf-8") for tag_value in effective_tags],
                    dtype="S50",
                )

            detectors_dict[camera_node.name] = camera_record

        return detectors_dict


class GantryMetadataProvider:
    """
    Провайдер канонических метаданных кинематики и ориентации гантри (metadata/gantry).
    """

    def collect(
        self,
        root_scene: SpatialNode,
        protocol: Optional[BaseProtocolConfig] = None,
        context: Optional[Dict[str, Any]] = None,
        task_id: int = 0,
    ) -> Dict[str, Any]:
        context_dict: Dict[str, Any] = context if context is not None else {}
        gantry_nodes: List[GantryNode] = _find_nodes_recursive(root_scene, GantryNode)

        if gantry_nodes:
            gantry_angle_rad = float(gantry_nodes[0].gantry_angle)
        elif "gantry_angle" in context_dict:
            gantry_angle_rad = float(context_dict["gantry_angle"])
        elif "current_angle" in context_dict:
            gantry_angle_rad = float(context_dict["current_angle"])
        else:
            gantry_angle_rad = 0.0

        gantry_angle_deg = float(np.rad2deg(gantry_angle_rad))

        view_index = int(context_dict.get("view_index", task_id))
        orbit_radius_val = 0.0
        rotation_direction_val = "CW"
        frame_duration_val = 0.0
        angular_step_rad = 0.0
        angular_step_deg = 0.0

        if isinstance(protocol, StepAndShootProtocolConfig):
            rotation_direction_val = str(protocol.rotation_direction)
            frame_duration_val = float(protocol.time_per_step) / float(units.s)
            if isinstance(protocol.radius, list):
                orbit_radius_val = float(protocol.radius[view_index % len(protocol.radius)])
            elif protocol.radius is not None:
                orbit_radius_val = float(protocol.radius)

            steps_count = int(protocol.steps)
            if steps_count > 1:
                denominator = (steps_count - 1) if protocol.endpoint else steps_count
                angular_step_rad = abs(float(protocol.end_angle) - float(protocol.start_angle)) / float(denominator)
                angular_step_deg = float(np.rad2deg(angular_step_rad))
        else:
            if "radius" in context_dict:
                orbit_radius_val = float(context_dict["radius"])
            if "step_time" in context_dict:
                frame_duration_val = float(context_dict["step_time"]) / float(units.s)
            elif "current_time" in context_dict:
                frame_duration_val = float(context_dict["current_time"]) / float(units.s)

        gantry_data: Dict[str, Any] = {
            "gantry_angle": gantry_angle_rad,
            "gantry_angle_deg": gantry_angle_deg,
            "orbit_radius": orbit_radius_val,
            "rotation_direction": rotation_direction_val,
            "frame_duration": frame_duration_val,
            "angular_step": angular_step_deg,
            "angular_step_rad": angular_step_rad,
        }

        # Матрицы поз детекторов
        camera_nodes: List[GammaCameraNode] = _find_nodes_recursive(root_scene, GammaCameraNode)
        for cam_index, camera_node in enumerate(camera_nodes):
            crystal_name = camera_node.slots.get("crystal")
            crystal_node = None
            if crystal_name is not None:
                crystal_node = _find_node_by_name(camera_node, crystal_name) or _find_node_by_name(root_scene, crystal_name)

            if crystal_node is not None:
                pose_matrix = np.array(crystal_node.global_matrix, dtype=np.float64)
            else:
                pose_matrix = np.array(camera_node.global_matrix, dtype=np.float64)

            cam_name = camera_node.name or f"detector_{cam_index + 1}"
            gantry_data[f"pose_matrix_{cam_name}"] = pose_matrix

            canonical_alias = f"pose_matrix_detector_{cam_index + 1}"
            if canonical_alias not in gantry_data:
                gantry_data[canonical_alias] = pose_matrix

        return gantry_data


class GeometryMetadataProvider:
    """
    Провайдер метаданных геометрии гамма-камеры, кристаллов и коллиматоров (metadata/geometry),
    а также материального состава физических объемов сцены.
    """

    def collect(self, root_scene: SpatialNode) -> Dict[str, Any]:
        camera_nodes: List[GammaCameraNode] = _find_nodes_recursive(root_scene, GammaCameraNode)
        heads_count = len(camera_nodes)

        geometry_data: Dict[str, Any] = {
            "detector_heads_count": heads_count,
        }

        if not camera_nodes:
            geometry_data["crystal_dimensions"] = np.zeros(3, dtype=np.float64)
            geometry_data["collimator_type"] = "None"
            geometry_data["collimator_hole_diameter"] = 0.0
            geometry_data["collimator_hole_length"] = 0.0
            geometry_data["collimator_septa_thickness"] = 0.0
            geometry_data["collimator_hole_shape"] = "None"
            geometry_data["collimator_material"] = "None"
        else:
            for cam_index, camera_node in enumerate(camera_nodes):
                cam_name = camera_node.name or f"Detector_{cam_index + 1}"

                # Разрешение кристалла
                crystal_name = camera_node.slots.get("crystal")
                crystal_node = None
                if crystal_name is not None:
                    crystal_node = _find_node_by_name(camera_node, crystal_name) or _find_node_by_name(root_scene, crystal_name)

                if isinstance(crystal_node, Volume) and isinstance(crystal_node.geometry, Box):
                    crystal_dims = np.array([float(dim) for dim in crystal_node.geometry.size], dtype=np.float64)
                else:
                    crystal_dims = np.zeros(3, dtype=np.float64)

                # Разрешение коллиматора
                collimator_name = camera_node.slots.get("collimator")
                collimator_node = None
                if collimator_name is not None:
                    collimator_node = _find_node_by_name(camera_node, collimator_name) or _find_node_by_name(root_scene, collimator_name)

                coll_type = "None"
                coll_hole_diam = 0.0
                coll_hole_len = 0.0
                coll_septa = 0.0
                coll_shape = "None"
                coll_mat = "None"

                if isinstance(collimator_node, ParametricParallelCollimator):
                    coll_type = "Parallel Hole"
                    coll_hole_diam = float(collimator_node.hole_diameter) / float(units.mm)
                    coll_hole_len = float(collimator_node.geometry.size[2]) / float(units.mm)
                    coll_septa = float(collimator_node.septa) / float(units.mm)
                    coll_shape = str(collimator_node.hole_shape.value)
                    coll_mat = str(collimator_node.material.name)
                elif isinstance(collimator_node, DirectParallelCollimator):
                    coll_type = "Parallel Hole"
                    coll_hole_diam = float(collimator_node.hole_diameter) / float(units.mm)
                    coll_hole_len = float(collimator_node.size[2]) / float(units.mm)
                    coll_septa = float(collimator_node.septa) / float(units.mm)
                    coll_shape = str(collimator_node.hole_shape.value)
                    coll_mat = str(collimator_node.material.name)
                elif isinstance(collimator_node, Volume):
                    coll_type = collimator_node.geometry.__class__.__name__
                    coll_mat = str(collimator_node.material.name)
                    if isinstance(collimator_node.geometry, Box):
                        coll_hole_len = float(collimator_node.geometry.size[2]) / float(units.mm)

                if cam_index == 0:
                    geometry_data["crystal_dimensions"] = crystal_dims
                    geometry_data["collimator_type"] = coll_type
                    geometry_data["collimator_hole_diameter"] = coll_hole_diam
                    geometry_data["collimator_hole_length"] = coll_hole_len
                    geometry_data["collimator_septa_thickness"] = coll_septa
                    geometry_data["collimator_hole_shape"] = coll_shape
                    geometry_data["collimator_material"] = coll_mat

                geometry_data[f"crystal_dimensions_{cam_name}"] = crystal_dims
                geometry_data[f"collimator_type_{cam_name}"] = coll_type
                geometry_data[f"collimator_hole_diameter_{cam_name}"] = coll_hole_diam
                geometry_data[f"collimator_hole_length_{cam_name}"] = coll_hole_len
                geometry_data[f"collimator_septa_thickness_{cam_name}"] = coll_septa
                geometry_data[f"collimator_hole_shape_{cam_name}"] = coll_shape
                geometry_data[f"collimator_material_{cam_name}"] = coll_mat

        # Сбор параметров всех физических объемов сцены для сохранения обратной совместимости
        volume_nodes: List[Volume] = _find_nodes_recursive(root_scene, Volume)
        for volume_node in volume_nodes:
            record: Dict[str, Any] = {
                "name": volume_node.name,
                "material": volume_node.material.name,
                "geometry_type": volume_node.geometry.__class__.__name__,
                "global_matrix": np.array(volume_node.global_matrix, dtype=np.float64),
            }
            if volume_node.tags:
                record["tags"] = np.array(
                    [tag_item.encode("utf-8") for tag_item in volume_node.tags],
                    dtype="S50",
                )
            geometry_data[volume_node.name] = record

        return geometry_data


class SourceMetadataProvider:
    """
    Провайдер паспорта источника и модели фантома (metadata/source).
    """

    def collect(
        self,
        root_scene: SpatialNode,
        scene_config: Optional[CompositeNodeConfig] = None,
    ) -> Dict[str, Any]:
        sources: List[Source] = _find_nodes_recursive(root_scene, Source)
        if sources:
            primary_source = sources[0]
            half_life_val = float(primary_source.half_life) / float(units.s)
            simulated_activity_val = float(primary_source.initial_activity) / float(units.Bq)
        else:
            half_life_val = 0.0
            simulated_activity_val = 0.0

        radionuclide_name = "Custom / Unspecified"
        phantom_model_name = "Custom / Unspecified"

        if scene_config is not None:
            source_cfg = _find_source_config(scene_config)
            if source_cfg is not None:
                if source_cfg.radionuclide:
                    radionuclide_name = source_cfg.radionuclide
                if source_cfg.phantom_type:
                    phantom_model_name = source_cfg.phantom_type

        return {
            "radionuclide": radionuclide_name,
            "half_life": half_life_val,
            "simulated_activity": simulated_activity_val,
            "phantom_type": phantom_model_name,
        }


class SimulationMetadataProvider:
    """
    Провайдер паспорта вычислений Монте-Карло (metadata/simulation).
    """

    def collect(
        self,
        sim_config: Optional[SimulationManagerConfig] = None,
        random_seed: Optional[int] = None,
        precision_mode: str = "single",
        active_processes: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        if sim_config is not None and sim_config.physics_list is not None:
            physics_list_str = sim_config.physics_list
        elif active_processes:
            process_names = [proc.name for proc in active_processes]
            physics_list_str = ", ".join(process_names)
        else:
            physics_list_str = ", ".join(get_process_mapping().values())

        simulated_histories_val = int(sim_config.particles_number) if sim_config is not None else 0
        seed_value = int(random_seed) if random_seed is not None else 0

        return {
            "code_name": "NMSimToolkit",
            "code_version": "0.2.0",
            "physics_list": physics_list_str,
            "simulated_histories": simulated_histories_val,
            "random_seed": seed_value,
            "precision_mode": precision_mode,
        }


class ProcedureMetadataCollector:
    """
    Фасадный класс для централизованного сбора процедурных метаданных симуляции.
    """

    def __init__(self) -> None:
        self.protocol_provider = ProtocolMetadataProvider()
        self.kinematics_provider = KinematicsMetadataProvider()
        self.detector_provider = DetectorMetadataProvider()
        self.geometry_provider = GeometryMetadataProvider()
        self.gantry_provider = GantryMetadataProvider()
        self.source_provider = SourceMetadataProvider()
        self.simulation_provider = SimulationMetadataProvider()

    def collect(
        self,
        root_scene: SpatialNode,
        protocol: Optional[BaseProtocolConfig] = None,
        context: Optional[Dict[str, Any]] = None,
        task_id: int = 0,
        sim_config: Optional[SimulationManagerConfig] = None,
        scene_config: Optional[CompositeNodeConfig] = None,
        random_seed: Optional[int] = None,
        precision_mode: str = "single",
        active_processes: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Собирает полный набор процедурных и канонических метаданных для текущей задачи симуляции.

        :param root_scene: Корневой узел графа сцены.
        :param protocol: Конфигурация протокола исследования.
        :param context: Словарь контекстных переменных симуляции.
        :param task_id: Идентификатор задачи.
        :param sim_config: Конфигурация SimulationManager.
        :param scene_config: Конфигурация сцены Pydantic.
        :param random_seed: Числовое зерно генератора случайных чисел.
        :param precision_mode: Режим точности float ('single' / 'double').
        :param active_processes: Список активных процессов симулятора.
        :return: Структурированный словарь процедурных и канонических метаданных.
        """
        context_dict: Dict[str, Any] = context if context is not None else {}
        protocol_meta = self.protocol_provider.collect(protocol=protocol, context=context_dict, task_id=task_id)
        kinematics_meta = self.kinematics_provider.collect(root_scene=root_scene, context=context_dict)
        detectors_meta = self.detector_provider.collect(root_scene=root_scene)
        geometry_meta = self.geometry_provider.collect(root_scene=root_scene)
        gantry_meta = self.gantry_provider.collect(root_scene=root_scene, protocol=protocol, context=context_dict, task_id=task_id)
        source_meta = self.source_provider.collect(root_scene=root_scene, scene_config=scene_config)
        simulation_meta = self.simulation_provider.collect(
            sim_config=sim_config,
            random_seed=random_seed,
            precision_mode=precision_mode,
            active_processes=active_processes,
        )

        return {
            "gantry": gantry_meta,
            "geometry": geometry_meta,
            "source": source_meta,
            "simulation": simulation_meta,
            # Сохранение ключей верхнего уровня для обратной совместимости с тестами
            "protocol": protocol_meta,
            "kinematics": kinematics_meta,
            "detectors": detectors_meta,
            "acquisition": {
                "protocol": protocol_meta,
                "kinematics": kinematics_meta,
                "detectors": detectors_meta,
                "geometry": geometry_meta,
            },
        }


__all__ = [
    "ProtocolMetadataProvider",
    "KinematicsMetadataProvider",
    "DetectorMetadataProvider",
    "GeometryMetadataProvider",
    "GantryMetadataProvider",
    "SourceMetadataProvider",
    "SimulationMetadataProvider",
    "ProcedureMetadataCollector",
]
