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
    CustomSweepProtocolConfig,
    SimulationManagerConfig,
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


def _resolve_camera_slot(
    camera_node: GammaCameraNode, root_scene: SpatialNode, slot_key: str
) -> Optional[SpatialNode]:
    """Разрешает узел слота гамма-камеры по имени в поддереве камеры или корневой сцене."""
    slot_name = camera_node.slots.get(slot_key)
    if slot_name is None:
        return None
    return _find_node_by_name(camera_node, slot_name) or _find_node_by_name(root_scene, slot_name)


def _are_head_specs_equal(first_spec: Dict[str, Any], second_spec: Dict[str, Any]) -> bool:
    """
    Проверяет эквивалентность словарей характеристик двух детекторных головок
    с учетом сравнения массивов numpy.
    """
    if set(first_spec.keys()) != set(second_spec.keys()):
        return False
    for spec_key, first_val in first_spec.items():
        second_val = second_spec[spec_key]
        if isinstance(first_val, np.ndarray) and isinstance(second_val, np.ndarray):
            if not np.array_equal(first_val, second_val):
                return False
        elif first_val != second_val:
            return False
    return True


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
        Формирует словарь метаданных протокола сканирования без фиктивных нулевых значений.
        """
        view_index: int = int(context.get("view_index", task_id))

        if isinstance(protocol, StepAndShootProtocolConfig):
            result: Dict[str, Any] = {
                "modality": "StepAndShoot",
                "view_index": view_index,
                "views_total": int(protocol.steps),
                "exposure_time": round(float(protocol.time_per_step) / float(units.s), 6),
            }
            if isinstance(protocol.radius, list):
                if protocol.radius:
                    result["orbit_radius"] = float(protocol.radius[view_index % len(protocol.radius)]) / float(units.mm)
            elif protocol.radius is not None:
                result["orbit_radius"] = float(protocol.radius) / float(units.mm)
            return result

        if isinstance(protocol, CustomSweepProtocolConfig):
            views_count: int = 1
            if protocol.zipped_variables:
                first_variable_values = next(iter(protocol.zipped_variables.values()))
                views_count = len(first_variable_values)
            return {
                "modality": "CustomSweep",
                "view_index": view_index,
                "views_total": views_count,
            }

        return {
            "modality": "Static" if protocol is None else protocol.__class__.__name__,
            "view_index": view_index,
            "views_total": 1,
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
        Если узел GantryNode отсутствует и угол не задан в контексте, возвращает пустой словарь.
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

        if context is not None:
            if "gantry_angle" in context:
                return {"gantry_angle": float(context["gantry_angle"])}
            if "current_angle" in context:
                return {"gantry_angle": float(context["current_angle"])}

        return {}


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
            crystal_node = _resolve_camera_slot(camera_node, root_scene, "crystal")

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
        sim_config: Optional[SimulationManagerConfig] = None,
    ) -> Dict[str, Any]:
        context_dict: Dict[str, Any] = context if context is not None else {}
        gantry_nodes: List[GantryNode] = _find_nodes_recursive(root_scene, GantryNode)
        gantry_data: Dict[str, Any] = {}

        # 1. Угол гантри (только если узел GantryNode присутствует или угол явно задан)
        gantry_angle: Optional[float] = None
        if gantry_nodes:
            gantry_angle = float(gantry_nodes[0].gantry_angle)
        elif "gantry_angle" in context_dict:
            gantry_angle = float(context_dict["gantry_angle"])
        elif "current_angle" in context_dict:
            gantry_angle = float(context_dict["current_angle"])
        elif isinstance(protocol, StepAndShootProtocolConfig):
            gantry_angle = float(protocol.start_angle)

        if gantry_angle is not None:
            gantry_data["gantry_angle"] = gantry_angle

        # 2. Специфичные параметры круговой томографии StepAndShoot
        view_index = int(context_dict.get("view_index", task_id))
        if isinstance(protocol, StepAndShootProtocolConfig):
            gantry_data["rotation_direction"] = str(protocol.rotation_direction)
            if isinstance(protocol.radius, list):
                if protocol.radius:
                    gantry_data["orbit_radius"] = (
                        float(protocol.radius[view_index % len(protocol.radius)]) / float(units.mm)
                    )
            elif protocol.radius is not None:
                gantry_data["orbit_radius"] = float(protocol.radius) / float(units.mm)

            steps_count = int(protocol.steps)
            if steps_count > 1:
                denominator = (steps_count - 1) if protocol.endpoint else steps_count
                angular_step = abs(float(protocol.end_angle) - float(protocol.start_angle)) / float(denominator)
                gantry_data["angular_step"] = float(np.rad2deg(angular_step))

        # 3. Время экспозиции кадра (frame_duration) в секундах (записывается только если > 0.0)
        frame_duration_val: Optional[float] = None
        if isinstance(protocol, StepAndShootProtocolConfig):
            frame_duration_val = round(float(protocol.time_per_step) / float(units.s), 6)
        elif "step_time" in context_dict:
            frame_duration_val = round(float(context_dict["step_time"]) / float(units.s), 6)
        elif "current_time" in context_dict:
            frame_duration_val = round(float(context_dict["current_time"]) / float(units.s), 6)
        elif "stop_time" in context_dict:
            frame_duration_val = round(
                (float(context_dict["stop_time"]) - float(context_dict.get("start_time", 0.0))) / float(units.s),
                6,
            )
        elif (
            sim_config is not None
            and isinstance(sim_config.stop_time, (int, float))
            and isinstance(sim_config.start_time, (int, float))
        ):
            frame_duration_val = round(
                (float(sim_config.stop_time) - float(sim_config.start_time)) / float(units.s),
                6,
            )

        if frame_duration_val is not None and frame_duration_val > 0.0:
            gantry_data["frame_duration"] = frame_duration_val

        # 4. Матрицы поз детекторов под каноническими ключами pose_matrix_detector_{i}
        camera_nodes: List[GammaCameraNode] = _find_nodes_recursive(root_scene, GammaCameraNode)
        for cam_index, camera_node in enumerate(camera_nodes):
            crystal_node = _resolve_camera_slot(camera_node, root_scene, "crystal")
            target_node = crystal_node if crystal_node is not None else camera_node
            gantry_data[f"pose_matrix_detector_{cam_index + 1}"] = np.array(target_node.global_matrix, dtype=np.float64)
            if camera_node.name:
                gantry_data[f"detector_{cam_index + 1}_name"] = str(camera_node.name)

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

        if camera_nodes:
            head_specs: List[Dict[str, Any]] = []
            for camera_node in camera_nodes:
                head_dict: Dict[str, Any] = {}

                crystal_node = _resolve_camera_slot(camera_node, root_scene, "crystal")
                if isinstance(crystal_node, Volume) and isinstance(crystal_node.geometry, Box):
                    head_dict["crystal_dimensions"] = np.array(
                        [float(dim) / float(units.mm) for dim in crystal_node.geometry.size],
                        dtype=np.float64,
                    )

                collimator_node = _resolve_camera_slot(camera_node, root_scene, "collimator")
                if isinstance(collimator_node, (ParametricParallelCollimator, DirectParallelCollimator)):
                    head_dict["collimator_type"] = "Parallel Hole"
                    head_dict["collimator_hole_diameter"] = float(collimator_node.hole_diameter) / float(units.mm)
                    head_dict["collimator_hole_length"] = float(collimator_node.size[2]) / float(units.mm)
                    head_dict["collimator_septa_thickness"] = float(collimator_node.septa) / float(units.mm)
                    head_dict["collimator_hole_shape"] = str(collimator_node.hole_shape.value)
                    head_dict["collimator_material"] = str(collimator_node.material.name)
                elif isinstance(collimator_node, Volume):
                    head_dict["collimator_type"] = collimator_node.geometry.__class__.__name__
                    head_dict["collimator_material"] = str(collimator_node.material.name)
                    if isinstance(collimator_node.geometry, Box):
                        head_dict["collimator_hole_length"] = float(collimator_node.geometry.size[2]) / float(units.mm)

                head_specs.append(head_dict)

            all_identical = all(_are_head_specs_equal(head_specs[0], spec) for spec in head_specs[1:])
            geometry_data.update(head_specs[0])
            if not all_identical:
                for cam_index, spec in enumerate(head_specs):
                    geometry_data[f"detector_{cam_index + 1}"] = dict(spec)

        # Изоляция физических объемов сцены во вложенном словаре volumes
        volume_nodes: List[Volume] = _find_nodes_recursive(root_scene, Volume)
        if volume_nodes:
            volumes_dict: Dict[str, Any] = {}
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
                volumes_dict[volume_node.name] = record
            geometry_data["volumes"] = volumes_dict

        return geometry_data


class SourceMetadataProvider:
    """
    Провайдер паспорта источника и модели фантома (metadata/source).
    """

    def collect(
        self,
        root_scene: SpatialNode,
        protocol: Optional[BaseProtocolConfig] = None,
    ) -> Dict[str, Any]:
        sources: List[Source] = _find_nodes_recursive(root_scene, Source)
        if not sources:
            return {}

        primary_source = sources[0]
        half_life_val = round(float(primary_source.half_life) / float(units.s), 6)
        simulated_activity_val = round(float(primary_source.initial_activity) / float(units.Bq), 6)

        radionuclide_name: Optional[str] = None
        phantom_model_name: Optional[str] = None

        if protocol is not None and protocol.metadata is not None:
            if protocol.metadata.radionuclide:
                radionuclide_name = protocol.metadata.radionuclide
            if protocol.metadata.phantom_type:
                phantom_model_name = protocol.metadata.phantom_type

        source_data: Dict[str, Any] = {
            "half_life": half_life_val,
            "simulated_activity": simulated_activity_val,
        }
        if radionuclide_name:
            source_data["radionuclide"] = radionuclide_name
        if phantom_model_name:
            source_data["phantom_type"] = phantom_model_name

        return source_data


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
        if active_processes:
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
        random_seed: Optional[int] = None,
        precision_mode: str = "single",
        active_processes: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Собирает полный набор процедурных и канонических метаданных для текущей задачи симуляции.
        """
        context_dict: Dict[str, Any] = context if context is not None else {}
        return {
            "gantry": self.gantry_provider.collect(
                root_scene=root_scene,
                protocol=protocol,
                context=context_dict,
                task_id=task_id,
                sim_config=sim_config,
            ),
            "geometry": self.geometry_provider.collect(root_scene=root_scene),
            "source": self.source_provider.collect(root_scene=root_scene, protocol=protocol),
            "simulation": self.simulation_provider.collect(
                sim_config=sim_config,
                random_seed=random_seed,
                precision_mode=precision_mode,
                active_processes=active_processes,
            ),
            "protocol": self.protocol_provider.collect(protocol=protocol, context=context_dict, task_id=task_id),
            "kinematics": self.kinematics_provider.collect(root_scene=root_scene, context=context_dict),
            "detectors": self.detector_provider.collect(root_scene=root_scene),
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

