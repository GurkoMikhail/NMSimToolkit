import yaml
import hashlib
from pathlib import Path
from core.config.models import SimulationConfig

class MaterialAnchorDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        if isinstance(data, AnchorStr):
            return False
        return super().ignore_aliases(data)

class AnchorStr(str):
    pass

def represent_anchor_str(dumper, data):
    node = yaml.representer.SafeRepresenter.represent_str(dumper, data)
    # We use a deterministic hash to guarantee no collisions while avoiding noisy git diffs
    safe_anchor_name = data.split(',')[0].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
    deterministic_hash = hashlib.md5(data.encode('utf-8')).hexdigest()[:4]
    safe_anchor_name = f"{safe_anchor_name}_{deterministic_hash}"
    dumper.anchors[node] = safe_anchor_name
    return node

MaterialAnchorDumper.add_representer(AnchorStr, represent_anchor_str)

def dump_simulation_config(config: SimulationConfig, filepath: str | Path):
    raw_dict = config.model_dump(exclude_none=True)

    unique_materials = set()

    def extract_materials(node: dict):
        if 'material' in node and isinstance(node['material'], str):
            unique_materials.add(node['material'])
        if 'children' in node:
            for child in node['children']:
                extract_materials(child)
        if 'collimator' in node:
            extract_materials(node['collimator'])
        if 'detector' in node:
            extract_materials(node['detector'])
        if 'distribution' in node:
            dist = node['distribution']
            if dist.get('mapping') is not None:
                for mat in dist['mapping'].values():
                    if isinstance(mat, str):
                        unique_materials.add(mat)
            if dist.get('fill_value') is not None:
                if isinstance(dist['fill_value'], str):
                    unique_materials.add(dist['fill_value'])

    extract_materials(raw_dict['scene'])

    # Python caches small strings and identity might be shared.
    # To force YAML to use aliases, the exact SAME string instance must be referenced
    # in the list AND in the dict structure.
    anchored_materials_list = []
    anchored_materials_map = {}
    for m in sorted(list(unique_materials)):
        anchor_obj = AnchorStr(m)
        anchored_materials_list.append(anchor_obj)
        anchored_materials_map[m] = anchor_obj

    def inject_anchors(node: dict):
        if 'material' in node and node['material'] in anchored_materials_map:
            node['material'] = anchored_materials_map[node['material']]
        if 'children' in node:
            for child in node['children']:
                inject_anchors(child)
        if 'collimator' in node:
            inject_anchors(node['collimator'])
        if 'detector' in node:
            inject_anchors(node['detector'])
        if 'distribution' in node:
            dist = node['distribution']
            if dist.get('mapping') is not None:
                for val, mat in dist['mapping'].items():
                    if isinstance(mat, str) and mat in anchored_materials_map:
                        dist['mapping'][val] = anchored_materials_map[mat]
            if dist.get('fill_value') is not None:
                fv = dist['fill_value']
                if isinstance(fv, str) and fv in anchored_materials_map:
                    dist['fill_value'] = anchored_materials_map[fv]

    inject_anchors(raw_dict['scene'])

    if anchored_materials_list:
        final_dict = {
            'Materials': anchored_materials_list,
            'simulation_manager': raw_dict['simulation_manager'],
            'data_manager': raw_dict['data_manager'],
            'scene': raw_dict['scene']
        }
    else:
        final_dict = raw_dict

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(final_dict, f, Dumper=MaterialAnchorDumper, default_flow_style=False, sort_keys=False)
