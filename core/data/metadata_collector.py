"""
Модуль унифицированного сбора процедурных метаданных симуляции (Metadata Collector).

Предоставляет модульную систему провайдеров для извлечения информации о модальности
протокола исследования, кинематике подвижных частей (станины томографа) и пространственном
расположении детекторов (гамма-камер и кристаллов) без использования утиной типизации.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from core.config.models import (
    BaseProtocolConfig,
    CustomSweepProtocolConfig,
    StepAndShootProtocolConfig,
    SpectProtocolConfig,
)
from core.geometry.volumes import Volume
from core.scene.gantry_node import GantryNode
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.nodes import CompositeNode, SpatialNode


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

        if isinstance(protocol, SpectProtocolConfig):
            orbit_radius_value = float(protocol.radius) if protocol.radius is not None else 0.0
            return {
                "modality": "SPECT",
                "view_index": view_index,
                "views_total": int(protocol.views),
                "exposure_time": float(protocol.time_per_view),
                "orbit_radius": orbit_radius_value,
            }

        if isinstance(protocol, StepAndShootProtocolConfig):
            orbit_radius_value = float(protocol.radius) if protocol.radius is not None else 0.0
            return {
                "modality": "StepAndShoot",
                "view_index": view_index,
                "views_total": int(protocol.views),
                "exposure_time": float(protocol.time_per_view),
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


class GeometryMetadataProvider:
    """
    Провайдер метаданных геометрии и материального состава сцены моделирования.
    """

    def collect(self, root_scene: SpatialNode) -> Dict[str, Any]:
        """
        Рекурсивно инспектирует сцену и возвращает параметры ключевых геометрических объемов (Volume).
        """
        volume_nodes: List[Volume] = _find_nodes_recursive(root_scene, Volume)
        geometry_dict: Dict[str, Any] = {}

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
            geometry_dict[volume_node.name] = record

        return geometry_dict


class ProcedureMetadataCollector:
    """
    Фасадный класс для централизованного сбора процедурных метаданных симуляции.
    """

    def __init__(self) -> None:
        self.protocol_provider = ProtocolMetadataProvider()
        self.kinematics_provider = KinematicsMetadataProvider()
        self.detector_provider = DetectorMetadataProvider()
        self.geometry_provider = GeometryMetadataProvider()

    def collect(
        self,
        root_scene: SpatialNode,
        protocol: Optional[BaseProtocolConfig],
        context: Dict[str, Any],
        task_id: int,
    ) -> Dict[str, Any]:
        """
        Собирает полный набор процедурных метаданных для текущей задачи симуляции.

        :param root_scene: Корневой узел графа сцены.
        :param protocol: Конфигурация протокола исследования.
        :param context: Словарь контекстных переменных симуляции.
        :param task_id: Идентификатор задачи.
        :return: Структурированный словарь процедурных метаданных.
        """
        protocol_meta = self.protocol_provider.collect(protocol=protocol, context=context, task_id=task_id)
        kinematics_meta = self.kinematics_provider.collect(root_scene=root_scene, context=context)
        detectors_meta = self.detector_provider.collect(root_scene=root_scene)
        geometry_meta = self.geometry_provider.collect(root_scene=root_scene)

        return {
            "protocol": protocol_meta,
            "kinematics": kinematics_meta,
            "detectors": detectors_meta,
            "geometry": geometry_meta,
        }


__all__ = [
    "ProtocolMetadataProvider",
    "KinematicsMetadataProvider",
    "DetectorMetadataProvider",
    "GeometryMetadataProvider",
    "ProcedureMetadataCollector",
]
