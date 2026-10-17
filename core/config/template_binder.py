"""
Модуль сервиса шаблонизации конфигураций и извлечения макросов переменных (ConfigTemplateBinder).
Изолирует механизмы параметризации конфигурационных файлов (макросы ${var})
от расчетного физического ядра и детерминированных строителей сцены.
"""

from copy import deepcopy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from core.config.models import (
    SimulationConfig,
    SimulationManagerConfig,
    DataManagerConfig,
    LoggingConfig,
    AnyProtocolConfig,
    AnyDistributionConfig,
    GammaCameraSlotsConfig,
    StepAndShootProtocolConfig,
    TranslateConfig,
    RotateConfig,
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

        protocol_val = scene_or_config.get("protocol")
        is_step_and_shoot = (
            (isinstance(protocol_val, dict) and protocol_val.get("type") == "StepAndShoot")
            or isinstance(protocol_val, StepAndShootProtocolConfig)
        )

        # 1. Проверка simulation_manager
        sim_mgr = scene_or_config.get("simulation_manager")
        if isinstance(sim_mgr, dict):
            sim_bindings: Dict[str, str] = {}
            for field_name in ("particles_number", "min_energy", "start_time", "stop_time"):
                val = sim_mgr.get(field_name)
                if isinstance(val, str):
                    match = re.search(r'\$\{([^}]+)\}', val)
                    if match:
                        sim_bindings[field_name] = match.group(1).strip().split()[0]
            if is_step_and_shoot:
                if "stop_time" not in sim_mgr or sim_mgr["stop_time"] is None:
                    sim_bindings["stop_time"] = "stop_time"
                if "start_time" not in sim_mgr or sim_mgr["start_time"] is None:
                    if sim_bindings.get("stop_time") == "stop_time":
                        sim_bindings["start_time"] = "start_time"
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
                    dm_bindings["filename"] = match.group(1).strip().split()[0]
                    dm_bindings["filename_template"] = filename_val
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
                            variable_name = match.group(1).strip().split()[0]
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
                            variable_name = match.group(1).strip().split()[0]
                            node_bindings[rot_name] = variable_name
                            node_bindings[f"rotation.{rot_name}"] = variable_name
                            if rot_name == "alpha" and is_gantry_or_camera:
                                node_bindings["gantry_angle"] = variable_name

            # Активность источника
            activity_val = node_dict.get("activity")
            if isinstance(activity_val, str):
                match = re.search(r'\$\{([^}]+)\}', activity_val)
                if match:
                    node_bindings["activity"] = match.group(1).strip().split()[0]

            # Энергия источника
            energy_val = node_dict.get("energy")
            if isinstance(energy_val, str):
                match = re.search(r'\$\{([^}]+)\}', energy_val)
                if match:
                    node_bindings["energy"] = match.group(1).strip().split()[0]
            elif isinstance(energy_val, list) and len(energy_val) > 0:
                first_item = energy_val[0]
                if isinstance(first_item, (list, tuple)) and len(first_item) >= 1 and isinstance(first_item[0], str):
                    match = re.search(r'\$\{([^}]+)\}', first_item[0])
                    if match:
                        node_bindings["energy"] = match.group(1).strip().split()[0]

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
                    {str(prop_key): str(prop_val) for prop_key, prop_val in prop_map.items()}
                )
            elif isinstance(prop_map, str):
                normalized_bindings.setdefault(node_name, {})[str(key)] = str(prop_map)

        protocol_val = data_dict.get("protocol")
        is_step_and_shoot = (
            (isinstance(protocol_val, dict) and protocol_val.get("type") == "StepAndShoot")
            or isinstance(protocol_val, StepAndShootProtocolConfig)
        )

        # 1. Привязки simulation_manager
        if "simulation_manager" in data_dict and isinstance(data_dict["simulation_manager"], dict):
            sim_dict = data_dict["simulation_manager"]
            sim_bindings = normalized_bindings.get("simulation_manager", {})
            if "particles_number" in sim_bindings:
                sim_dict["particles_number"] = f"${{{sim_bindings['particles_number']}}}"
            if "min_energy" in sim_bindings:
                sim_dict["min_energy"] = f"${{{sim_bindings['min_energy']}}} keV"
            if "start_time" in sim_bindings:
                start_var = sim_bindings["start_time"].strip().split()[0]
                if is_step_and_shoot and start_var in ("start_time", "stop_time", "step_time", "current_time"):
                    sim_dict["start_time"] = f"${{{start_var}}}"
                else:
                    sim_dict["start_time"] = f"${{{start_var}}} s"
            if "stop_time" in sim_bindings:
                stop_var = sim_bindings["stop_time"].strip().split()[0]
                if is_step_and_shoot and stop_var in ("start_time", "stop_time", "step_time", "current_time"):
                    sim_dict["stop_time"] = f"${{{stop_var}}}"
                else:
                    sim_dict["stop_time"] = f"${{{stop_var}}} s"

        # 2. Привязки data_manager
        if "data_manager" in data_dict and isinstance(data_dict["data_manager"], dict):
            dm_dict = data_dict["data_manager"]
            dm_bindings = normalized_bindings.get("data_manager", {})
            if "filename" in dm_bindings:
                bound_val = dm_bindings["filename"]
                template_val = dm_bindings.get("filename_template", "")
                current_filename = str(dm_dict.get("filename", ""))
                if "${" in current_filename:
                    # Текущий filename уже содержит шаблон макроса — сохраняем его без перезаписи
                    pass
                elif "${" in template_val:
                    dm_dict["filename"] = template_val
                elif "${" in bound_val:
                    dm_dict["filename"] = bound_val
                else:
                    clean_var = bound_val.strip().split()[0]
                    if is_step_and_shoot and clean_var in ("gantry_angle", "current_angle"):
                        macro_expr = f"{clean_var} deg"
                    elif is_step_and_shoot and clean_var in ("start_time", "stop_time", "step_time", "current_time"):
                        macro_expr = f"{clean_var} s"
                    else:
                        macro_expr = clean_var
                    file_path = Path(current_filename) if current_filename else Path("output.h5")
                    suffix = file_path.suffix or ".h5"
                    stem = file_path.stem or "output"
                    new_name = f"{stem}_${{{macro_expr}}}{suffix}"
                    dm_dict["filename"] = (
                        str(file_path.with_name(new_name)).replace("\\", "/")
                        if file_path.parent != Path(".")
                        else new_name
                    )

        # 3. Привязки узлов сцены
        scene_dict = data_dict.get("scene", data_dict) if isinstance(data_dict, dict) else data_dict

        def _format_angle_macro(var_name: str) -> str:
            clean_var = var_name.strip().split()[0]
            if is_step_and_shoot and clean_var in ("gantry_angle", "current_angle"):
                return f"${{{clean_var}}}"
            return f"${{{clean_var}}} deg"

        def _traverse(node_dict: Dict[str, Any]) -> None:
            node_name = node_dict.get("name")
            node_bindings = normalized_bindings.get(node_name, {}) if node_name else {}

            has_trans = any(
                prop_key in node_bindings for prop_key in ("translation.x", "translation.y", "translation.z", "x", "y", "z", "radius")
            )
            has_rot = any(
                prop_key in node_bindings for prop_key in ("rotation.alpha", "rotation.beta", "rotation.gamma", "alpha", "beta", "gamma", "gantry_angle")
            )

            if has_trans or has_rot:
                transforms_list: List[Dict[str, Any]] = node_dict.setdefault("transformations", [])

                has_explicit_tr_or_rot = any(
                    isinstance(tr_item, dict)
                    and (
                        tr_item.get("type", "").lower() in ("translate", "rotate")
                        or any(coord_key in tr_item for coord_key in ("x", "y", "z", "alpha", "beta", "gamma"))
                    )
                    for tr_item in transforms_list
                )
                has_matrix_transform = any(
                    isinstance(tr_item, dict) and tr_item.get("type", "").lower() == "matrix" and "matrix" in tr_item
                    for tr_item in transforms_list
                )
                if has_matrix_transform and not has_explicit_tr_or_rot:
                    expanded_transforms: List[Dict[str, Any]] = []
                    for tr_item in transforms_list:
                        if isinstance(tr_item, dict) and tr_item.get("type", "").lower() == "matrix" and "matrix" in tr_item:
                            decomposed_cfgs = SceneExporter.decompose_matrix(
                                np.array(tr_item["matrix"], dtype=float),
                                as_matrix=False,
                            )
                            for dec_cfg in decomposed_cfgs:
                                if isinstance(dec_cfg, RotateConfig):
                                    expanded_transforms.append({
                                        "type": "rotate",
                                        "alpha": f"{float(np.degrees(dec_cfg.alpha)):.6g} deg",
                                        "beta": f"{float(np.degrees(dec_cfg.beta)):.6g} deg",
                                        "gamma": f"{float(np.degrees(dec_cfg.gamma)):.6g} deg",
                                        "rotation_center": list(dec_cfg.rotation_center),
                                        "in_local": bool(tr_item.get("in_local", False)),
                                    })
                                elif isinstance(dec_cfg, TranslateConfig):
                                    expanded_transforms.append({
                                        "type": "translate",
                                        "x": float(dec_cfg.x),
                                        "y": float(dec_cfg.y),
                                        "z": float(dec_cfg.z),
                                        "in_local": bool(tr_item.get("in_local", False)),
                                    })
                        else:
                            expanded_transforms.append(tr_item)
                    transforms_list[:] = expanded_transforms

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
                    if (t_type == "translate" or any(coord_key in transform for coord_key in ("x", "y", "z"))) and not trans_applied and has_trans:
                        if bx:
                            transform["x"] = f"${{{bx}}} mm"
                        if by:
                            transform["y"] = f"${{{by}}} mm"
                        if bz:
                            transform["z"] = f"${{{bz}}} mm"
                        transform.setdefault("type", "translate")
                        trans_applied = True
                    elif (t_type == "rotate" or any(rot_key in transform for rot_key in ("alpha", "beta", "gamma"))) and not rot_applied and has_rot:
                        if b_alpha:
                            transform["alpha"] = _format_angle_macro(b_alpha)
                        if b_beta:
                            transform["beta"] = _format_angle_macro(b_beta)
                        if b_gamma:
                            transform["gamma"] = _format_angle_macro(b_gamma)
                        transform.setdefault("type", "rotate")
                        rot_applied = True

                if has_rot and not rot_applied:
                    new_rot: Dict[str, Any] = {"type": "rotate", "alpha": 0.0, "beta": 0.0, "gamma": 0.0}
                    if b_alpha:
                        new_rot["alpha"] = _format_angle_macro(b_alpha)
                    if b_beta:
                        new_rot["beta"] = _format_angle_macro(b_beta)
                    if b_gamma:
                        new_rot["gamma"] = _format_angle_macro(b_gamma)
                    transforms_list.insert(0, new_rot)

                if has_trans and not trans_applied:
                    new_trans: Dict[str, Any] = {"type": "translate", "x": 0.0, "y": 0.0, "z": 0.0}
                    if bx:
                        new_trans["x"] = f"${{{bx}}} mm"
                    if by:
                        new_trans["y"] = f"${{{by}}} mm"
                    if bz:
                        new_trans["z"] = f"${{{bz}}} mm"
                    transforms_list.append(new_trans)

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
        source_metadata_registry: Optional[Dict[SpatialNode, Any]] = None,
        logging_cfg: Optional[LoggingConfig] = None,
    ) -> None:
        """
        Экспортирует сцену и конфигурацию в YAML с наложением шаблонных макросов ${variable}.
        Сохраняет архитектурную чистоту ядра, изолируя подстановку шаблонов от SceneExporter.
        """
        target_path = Path(filepath).resolve()
        base_dir = target_path.parent
        adjusted_distribution_registry = SceneExporter.prepare_distribution_registry_for_yaml(
            root_node=root_node,
            base_dir=base_dir,
            distribution_registry=distribution_registry,
        )

        # 1. Чистый экспорт ядра
        config = SceneExporter.export_to_config(
            root_node,
            simulation_manager_cfg=simulation_manager_cfg,
            data_manager_cfg=data_manager_cfg,
            pool_size=pool_size,
            protocol=protocol,
            distribution_registry=adjusted_distribution_registry,
            slots_registry=slots_registry,
            source_metadata_registry=source_metadata_registry,
            logging_cfg=logging_cfg,
        )

        # 2. Сериализация в raw dict
        raw_dict = config.model_dump(exclude_none=True)

        # 3. Наложение шаблонов макросов
        if bindings_registry:
            cls.apply_bindings(raw_dict, bindings_registry)

        # 4. Дамп в YAML через канонический дамппер с поддержкой якорей материалов
        dump_simulation_config(raw_dict, target_path)


__all__ = ["ConfigTemplateBinder"]
