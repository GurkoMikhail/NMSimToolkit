"""
Модуль кастомного сериализатора конфигураций симуляции в формат YAML.
Поддерживает формирование YAML-якорей (&anchor) и ссылок (*alias) для материалов сцены,
исключая избыточное дублирование определений материалов.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Dict, Set, Union
import yaml

from core.config.models import SimulationConfig


class AnchorStr(str):
    """Строковый тип-маркер для принудительного использования YAML-якорей и псевдонимов."""
    pass


class MaterialAnchorDumper(yaml.SafeDumper):
    """
    Кастомный YAML-дампер, сохраняющий псевдонимы исключительно для объектов AnchorStr.
    """

    def ignore_aliases(self, data: Any) -> bool:
        if isinstance(data, AnchorStr):
            return False
        return super().ignore_aliases(data)


def represent_anchor_str(yaml_dumper: yaml.SafeDumper, str_data: AnchorStr) -> yaml.Node:
    """
    Генерирует детерминированный якорь для строки материала с добавлением короткого хеша.
    """
    node = yaml.representer.SafeRepresenter.represent_str(yaml_dumper, str_data)
    # Формируем читаемое имя якоря без спецсимволов с добавлением детерминированного хеша
    safe_anchor_name = str_data.split(',')[0].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
    deterministic_hash = hashlib.md5(str_data.encode('utf-8')).hexdigest()[:4]
    safe_anchor_name = f"{safe_anchor_name}_{deterministic_hash}"
    yaml_dumper.anchors[node] = safe_anchor_name
    return node


MaterialAnchorDumper.add_representer(AnchorStr, represent_anchor_str)


def dump_simulation_config(config: Union[SimulationConfig, Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """
    Сохраняет конфигурацию SimulationConfig или структурированный словарь в файл YAML
    с автоматическим якорным связыванием повторяющихся названий материалов сцены.
    """
    if isinstance(config, SimulationConfig):
        raw_dict = config.model_dump(exclude_none=True)
    elif isinstance(config, dict):
        raw_dict = deepcopy(config)
    else:
        raise TypeError(f"Ожидался SimulationConfig или dict, получено {type(config)}")
    unique_materials: Set[str] = set()

    def extract_materials(node_dict: Dict[str, Any]) -> None:
        if 'material' in node_dict and isinstance(node_dict['material'], str):
            unique_materials.add(node_dict['material'])
        if 'hole_material' in node_dict and isinstance(node_dict['hole_material'], str):
            unique_materials.add(node_dict['hole_material'])
        if 'children' in node_dict:
            for child_node in node_dict['children']:
                extract_materials(child_node)
        if 'collimator' in node_dict:
            extract_materials(node_dict['collimator'])
        if 'detector' in node_dict:
            extract_materials(node_dict['detector'])
        if 'distribution' in node_dict:
            distribution_params = node_dict['distribution']
            if distribution_params.get('mapping') is not None:
                for mat_val in distribution_params['mapping'].values():
                    if isinstance(mat_val, str):
                        unique_materials.add(mat_val)
            if distribution_params.get('fill_value') is not None:
                fill_material = distribution_params['fill_value']
                if isinstance(fill_material, str):
                    unique_materials.add(fill_material)

    extract_materials(raw_dict['scene'])

    # Создаем единые объекты AnchorStr для одинаковых строк материалов,
    # чтобы PyYAML создал якорь в секции Materials и псевдонимы (*alias) в графе сцены.
    anchored_materials_list: list[AnchorStr] = []
    anchored_materials_map: Dict[str, AnchorStr] = {}
    for material_name in sorted(list(unique_materials)):
        anchor_object = AnchorStr(material_name)
        anchored_materials_list.append(anchor_object)
        anchored_materials_map[material_name] = anchor_object

    def inject_anchors(node_dict: Dict[str, Any]) -> None:
        if 'material' in node_dict and node_dict['material'] in anchored_materials_map:
            node_dict['material'] = anchored_materials_map[node_dict['material']]
        if 'hole_material' in node_dict and node_dict['hole_material'] in anchored_materials_map:
            node_dict['hole_material'] = anchored_materials_map[node_dict['hole_material']]
        if 'children' in node_dict:
            for child_node in node_dict['children']:
                inject_anchors(child_node)
        if 'collimator' in node_dict:
            inject_anchors(node_dict['collimator'])
        if 'detector' in node_dict:
            inject_anchors(node_dict['detector'])
        if 'distribution' in node_dict:
            distribution_params = node_dict['distribution']
            if distribution_params.get('mapping') is not None:
                for label_val, mat_name in distribution_params['mapping'].items():
                    if isinstance(mat_name, str) and mat_name in anchored_materials_map:
                        distribution_params['mapping'][label_val] = anchored_materials_map[mat_name]
            if distribution_params.get('fill_value') is not None:
                fill_val = distribution_params['fill_value']
                if isinstance(fill_val, str) and fill_val in anchored_materials_map:
                    distribution_params['fill_value'] = anchored_materials_map[fill_val]

    inject_anchors(raw_dict['scene'])

    final_dict: Dict[str, Any] = {}
    if 'pool_size' in raw_dict:
        final_dict['pool_size'] = raw_dict['pool_size']
    if 'protocol' in raw_dict:
        final_dict['protocol'] = raw_dict['protocol']
    if anchored_materials_list:
        final_dict['Materials'] = anchored_materials_list
    final_dict['simulation_manager'] = raw_dict['simulation_manager']
    final_dict['data_manager'] = raw_dict['data_manager']
    final_dict['scene'] = raw_dict['scene']
    for config_key, config_val in raw_dict.items():
        if config_key not in final_dict:
            final_dict[config_key] = config_val

    with open(filepath, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(final_dict, yaml_file, Dumper=MaterialAnchorDumper, default_flow_style=False, sort_keys=False)
