"""
Модуль сервиса шаблонизации конфигураций и извлечения макросов переменных (ConfigTemplateBinder).
Изолирует механизмы параметризации конфигурационных файлов (макросы ${var})
от расчетного физического ядра и детерминированных строителей сцены.
"""

from copy import deepcopy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.config.models import (
    SimulationConfig,
    SimulationManagerConfig,
    DataManagerConfig,
    AnyProtocolConfig,
    AnyDistributionConfig,
    GammaCameraSlotsConfig,
)
from core.config.exporter import SceneExporter
from core.config.yaml_dumper import dump_simulation_config
from core.scene.nodes import SpatialNode


class ConfigTemplateBinder:
    """
    Сервис извлечения и наложения строковых макросов (${variable}) на структуры конфигурации сцены.
    Обеспечивает изоляцию шаблонизации от базового графа сцены и строителей ядра.
    """

    @classmethod
    def extract_bindings(cls, scene_or_config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Рекурсивно обходит конфигурацию или словарь сцены и извлекает соответствия
        между узлами/разделами конфигурации и шаблонными переменными:
        {target_name: {property_name: variable_name}}
        """
        bindings_by_target: Dict[str, Dict[str, str]] = {}

        # 1. Проверка simulation_manager
        sim_mgr = scene_or_config.get("simulation_manager")
        if isinstance(sim_mgr, dict):
            sim_bindings: Dict[str, str] = {}
            for field_name in ("particles_number", "min_energy", "start_time", "stop_time"):
                val = sim_mgr.get(field_name)
                if isinstance(val, str):
                    match = re.search(r'\$\{([^}]+)\}', val)
                    if match:
                        sim_bindings[field_name] = match.group(1)
            if sim_bindings:
                bindings_by_target["simulation_manager"] = sim_bindings

        # 2. Проверка data_manager
        data_mgr = scene_or_config.get("data_manager")
        if isinstance(data_mgr, dict):
            dm_bindings: Dict[str, str] = {}
            filename_val = data_mgr.get("filename")
            if isinstance(filename_val, str):
                match = re.search(r'\$\{([^}]+)\}', filename_val)
                if match:
                    dm_bindings["filename"] = match.group(1)
            if dm_bindings:
                bindings_by_target["data_manager"] = dm_bindings

        # 3. Обход графа сцены
        scene_dict = scene_or_config.get("scene") if "scene" in scene_or_config and isinstance(scene_or_config["scene"], dict) else scene_or_config

        def _traverse(node_dict: Dict[str, Any]) -> None:
            node_name = node_dict.get("name")
            node_type = str(node_dict.get("type", "")).lower()
            node_name_lower = str(node_name).lower() if node_name else ""
            is_gantry_or_camera = any(
                substr in node_type or substr in node_name_lower
                for substr in ("gantry", "camera", "detector", "head")
            )
            node_bindings: Dict[str, str] = {}

            # Пространственные трансформации
            for transform in node_dict.get("transformations", []):
                if not isinstance(transform, dict):
                    continue

                # Трансляция
                for coord_name in ("x", "y", "z"):
                    coord_val = transform.get(coord_name)
                    if isinstance(coord_val, str):
                        match = re.search(r'\$\{([^}]+)\}', coord_val)
                        if match:
                            variable_name = match.group(1)
                            node_bindings[coord_name] = variable_name
                            node_bindings[f"translation.{coord_name}"] = variable_name
                            if coord_name == "y" and is_gantry_or_camera:
                                node_bindings["radius"] = variable_name

                # Вращение
                for rot_name in ("alpha", "beta", "gamma"):
                    rot_val = transform.get(rot_name)
                    if isinstance(rot_val, str):
                        match = re.search(r'\$\{([^}]+)\}', rot_val)
                        if match:
                            variable_name = match.group(1)
                            node_bindings[rot_name] = variable_name
                            node_bindings[f"rotation.{rot_name}"] = variable_name
                            if rot_name == "alpha" and is_gantry_or_camera:
                                node_bindings["gantry_angle"] = variable_name

            # Активность источника
            activity_val = node_dict.get("activity")
            if isinstance(activity_val, str):
                match = re.search(r'\$\{([^}]+)\}', activity_val)
                if match:
                    node_bindings["activity"] = match.group(1)

            # Энергия источника
            energy_val = node_dict.get("energy")
            if isinstance(energy_val, str):
                match = re.search(r'\$\{([^}]+)\}', energy_val)
                if match:
                    node_bindings["energy"] = match.group(1)
            elif isinstance(energy_val, list) and len(energy_val) > 0:
                first_item = energy_val[0]
                if isinstance(first_item, (list, tuple)) and len(first_item) >= 1 and isinstance(first_item[0], str):
                    match = re.search(r'\$\{([^}]+)\}', first_item[0])
                    if match:
                        node_bindings["energy"] = match.group(1)

            if node_name and node_bindings:
                bindings_by_target[node_name] = node_bindings

            # Рекурсивный обход дочерних узлов
            for child_dict in node_dict.get("children", []):
                if isinstance(child_dict, dict):
                    _traverse(child_dict)

        if isinstance(scene_dict, dict) and (scene_dict.get("name") or scene_dict.get("children")):
            _traverse(scene_dict)

        return bindings_by_target

    @classmethod
    def apply_bindings(
        cls,
        scene_or_config: Union[Dict[str, Any], SimulationConfig],
        bindings_registry: Optional[Dict[Any, Any]] = None,
    ) -> Union[Dict[str, Any], SimulationConfig]:
        """
        Накладывает макросы ${variable} на поля словаря сцены или SimulationConfig перед экспортом или запуском.
        Поддерживает привязки как к узлам сцены, так и к параметрам SimulationManager / DataManager.
        """
        if not bindings_registry:
            return scene_or_config

        if isinstance(scene_or_config, SimulationConfig):
            raw_dict = scene_or_config.model_dump(exclude_none=True)
            cls._apply_bindings_to_dict(raw_dict, bindings_registry)
            return SimulationConfig.model_validate(raw_dict)

        cls._apply_bindings_to_dict(scene_or_config, bindings_registry)
        return scene_or_config

    @classmethod
    def _apply_bindings_to_dict(
        cls,
        data_dict: Dict[str, Any],
        bindings_registry: Dict[Any, Any],
    ) -> None:
        """
        Применяет реестр привязок к словарю полной конфигурации или словарю сцены.
        """
        # Нормализация реестра привязок
        normalized_bindings: Dict[str, Dict[str, str]] = {}
        for key, prop_map in bindings_registry.items():
            if not prop_map:
                continue
            if isinstance(key, str):
                if "." in key:
                    target_name, prop_name = key.split(".", 1)
                    normalized_bindings.setdefault(target_name, {})[prop_name] = str(prop_map)
                    continue
                node_name = key
            elif isinstance(key, SpatialNode):
                node_name = key.name
            else:
                raise TypeError(f"Ключ реестра привязок должен быть строкой или SpatialNode, получено: {type(key)}")

            if isinstance(prop_map, dict):
                normalized_bindings.setdefault(node_name, {}).update(
                    {str(k): str(v) for k, v in prop_map.items()}
                )
            elif isinstance(prop_map, str):
                normalized_bindings.setdefault(node_name, {})[str(key)] = str(prop_map)

        # 1. Привязки simulation_manager
        if "simulation_manager" in data_dict and isinstance(data_dict["simulation_manager"], dict):
            sim_dict = data_dict["simulation_manager"]
            sim_bindings = normalized_bindings.get("simulation_manager", {})
            if "particles_number" in sim_bindings:
                sim_dict["particles_number"] = f"${{{sim_bindings['particles_number']}}}"
            if "min_energy" in sim_bindings:
                sim_dict["min_energy"] = f"${{{sim_bindings['min_energy']}}} keV"
            if "start_time" in sim_bindings:
                sim_dict["start_time"] = f"${{{sim_bindings['start_time']}}} s"
            if "stop_time" in sim_bindings:
                sim_dict["stop_time"] = f"${{{sim_bindings['stop_time']}}} s"

        # 2. Привязки data_manager
        if "data_manager" in data_dict and isinstance(data_dict["data_manager"], dict):
            dm_dict = data_dict["data_manager"]
            dm_bindings = normalized_bindings.get("data_manager", {})
            if "filename" in dm_bindings:
                dm_dict["filename"] = f"${{{dm_bindings['filename']}}}"

        # 3. Привязки узлов сцены
        scene_dict = data_dict.get("scene", data_dict) if isinstance(data_dict, dict) else data_dict

        def _traverse(node_dict: Dict[str, Any]) -> None:
            node_name = node_dict.get("name")
            node_bindings = normalized_bindings.get(node_name, {}) if node_name else {}

            has_trans = any(
                k in node_bindings for k in ("translation.x", "translation.y", "translation.z", "x", "y", "z", "radius")
            )
            has_rot = any(
                k in node_bindings for k in ("rotation.alpha", "rotation.beta", "rotation.gamma", "alpha", "beta", "gamma", "gantry_angle")
            )

            if has_trans or has_rot:
                transforms_list: List[Dict[str, Any]] = node_dict.setdefault("transformations", [])

                # 1.1. Трансляция
                bx = node_bindings.get("translation.x") or node_bindings.get("x")
                by = node_bindings.get("translation.y") or node_bindings.get("y") or node_bindings.get("radius")
                bz = node_bindings.get("translation.z") or node_bindings.get("z")

                # 1.2. Вращение
                b_alpha = node_bindings.get("rotation.alpha") or node_bindings.get("alpha") or node_bindings.get("gantry_angle")
                b_beta = node_bindings.get("rotation.beta") or node_bindings.get("beta")
                b_gamma = node_bindings.get("rotation.gamma") or node_bindings.get("gamma")

                trans_applied = False
                rot_applied = False

                for transform in transforms_list:
                    if not isinstance(transform, dict):
                        continue
                    t_type = transform.get("type", "").lower()
                    if (t_type == "translate" or any(k in transform for k in ("x", "y", "z"))) and not trans_applied and has_trans:
                        if bx:
                            transform["x"] = f"${{{bx}}} mm"
                        if by:
                            transform["y"] = f"${{{by}}} mm"
                        if bz:
                            transform["z"] = f"${{{bz}}} mm"
                        transform.setdefault("type", "translate")
                        trans_applied = True
                    elif (t_type == "rotate" or any(k in transform for k in ("alpha", "beta", "gamma"))) and not rot_applied and has_rot:
                        if b_alpha:
                            transform["alpha"] = f"${{{b_alpha}}} deg"
                        if b_beta:
                            transform["beta"] = f"${{{b_beta}}} deg"
                        if b_gamma:
                            transform["gamma"] = f"${{{b_gamma}}} deg"
                        transform.setdefault("type", "rotate")
                        rot_applied = True

                if has_trans and not trans_applied:
                    new_trans: Dict[str, Any] = {"type": "translate", "x": 0.0, "y": 0.0, "z": 0.0}
                    if bx:
                        new_trans["x"] = f"${{{bx}}} mm"
                    if by:
                        new_trans["y"] = f"${{{by}}} mm"
                    if bz:
                        new_trans["z"] = f"${{{bz}}} mm"
                    transforms_list.append(new_trans)

                if has_rot and not rot_applied:
                    new_rot: Dict[str, Any] = {"type": "rotate", "alpha": 0.0, "beta": 0.0, "gamma": 0.0}
                    if b_alpha:
                        new_rot["alpha"] = f"${{{b_alpha}}} deg"
                    if b_beta:
                        new_rot["beta"] = f"${{{b_beta}}} deg"
                    if b_gamma:
                        new_rot["gamma"] = f"${{{b_gamma}}} deg"
                    transforms_list.append(new_rot)

            # 2. Активность источника
            b_act = node_bindings.get("activity")
            if b_act:
                node_dict["activity"] = f"${{{b_act}}} MBq"

            # 3. Энергия источника
            b_energy = node_bindings.get("energy")
            if b_energy and "energy" in node_dict:
                current_energy = node_dict["energy"]
                if isinstance(current_energy, list):
                    new_energy_list = []
                    for item in current_energy:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            new_energy_list.append([f"${{{b_energy}}} keV", float(item[1])])
                        else:
                            new_energy_list.append(item)
                    node_dict["energy"] = new_energy_list
                else:
                    node_dict["energy"] = [[f"${{{b_energy}}} keV", 1.0]]

            # Рекурсивный обход детей
            for child_dict in node_dict.get("children", []):
                if isinstance(child_dict, dict):
                    _traverse(child_dict)

        if isinstance(scene_dict, dict) and (scene_dict.get("name") or scene_dict.get("children")):
            _traverse(scene_dict)
        return scene_dict

    @classmethod
    def export_templated_yaml(
        cls,
        root_node: SpatialNode,
        filepath: Union[str, Path],
        bindings_registry: Optional[Dict[Any, Dict[str, str]]] = None,
        simulation_manager_cfg: Optional[SimulationManagerConfig] = None,
        data_manager_cfg: Optional[DataManagerConfig] = None,
        pool_size: int = 1,
        protocol: Optional[AnyProtocolConfig] = None,
        distribution_registry: Optional[Dict[SpatialNode, AnyDistributionConfig]] = None,
        slots_registry: Optional[Dict[SpatialNode, GammaCameraSlotsConfig]] = None,
    ) -> None:
        """
        Экспортирует сцену и конфигурацию в YAML с наложением шаблонных макросов ${variable}.
        Сохраняет архитектурную чистоту ядра, изолируя подстановку шаблонов от SceneExporter.
        """
        # 1. Чистый экспорт ядра
        config = SceneExporter.export_to_config(
            root_node,
            simulation_manager_cfg=simulation_manager_cfg,
            data_manager_cfg=data_manager_cfg,
            pool_size=pool_size,
            protocol=protocol,
            distribution_registry=distribution_registry,
            slots_registry=slots_registry,
        )

        # 2. Сериализация в raw dict
        raw_dict = config.model_dump(exclude_none=True)

        # 3. Наложение шаблонов макросов
        if bindings_registry:
            cls.apply_bindings(raw_dict, bindings_registry)

        # 4. Дамп в YAML через канонический дамппер с поддержкой якорей материалов
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dump_simulation_config(raw_dict, target_path)


__all__ = ["ConfigTemplateBinder"]
