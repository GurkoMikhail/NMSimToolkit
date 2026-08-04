import string
import math
from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field

class RingRepeaterConfig(BaseModel):
    num_nodes: int
    radius: float
    start_angle: float = 0.0
    angular_span: float = 360.0
    axis: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    center: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_axis: List[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0])
    template: Dict[str, Any]

class GridRepeaterConfig(BaseModel):
    count_x: int = 1
    count_y: int = 1
    count_z: int = 1
    pitch_x: float = 0.0
    pitch_y: float = 0.0
    pitch_z: float = 0.0
    origin: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    angle: float = 0.0
    template: Dict[str, Any]

class ZipRepeaterConfig(BaseModel):
    values: Dict[str, List[Any]]
    template: Dict[str, Any]

def _smart_cast(value: str) -> Union[str, int, float]:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

def _apply_template(node: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(node, dict):
        return {k: _apply_template(v, variables) for k, v in node.items()}
    elif isinstance(node, list):
        return [_apply_template(v, variables) for v in node]
    elif isinstance(node, str) and "$" in node:
        tmpl = string.Template(node)
        substituted = tmpl.safe_substitute(variables)
        return _smart_cast(substituted)
    else:
        return node

def _process_ring_repeater(config: Dict[str, Any]) -> Dict[str, Any]:
    repeater = RingRepeaterConfig(**config)
    children = []

    if math.isclose(repeater.angular_span, 360.0):
        step = 360.0 / repeater.num_nodes if repeater.num_nodes > 0 else 0.0
    else:
        step = repeater.angular_span / max(1, repeater.num_nodes - 1)

    for i in range(repeater.num_nodes):
        current_angle = repeater.start_angle + i * step

        variables = {
            "index": i,
            "angle": current_angle,
        }

        child = _apply_template(repeater.template, variables)

        transformations = child.get("transformations", [])
        if not isinstance(transformations, list):
            transformations = []

        t1 = {
            "type": "translate",
            "x": repeater.radius * repeater.base_axis[0],
            "y": repeater.radius * repeater.base_axis[1],
            "z": repeater.radius * repeater.base_axis[2]
        }
        t2 = {"type": "rotate", "axis": repeater.axis, "angle": f"{current_angle} deg"}
        t3 = {
            "type": "translate",
            "x": repeater.center[0],
            "y": repeater.center[1],
            "z": repeater.center[2]
        }

        transformations.extend([t1, t2, t3])
        child["transformations"] = transformations

        # Recursively expand any repeaters inside this generated child
        child = expand_repeaters(child)
        children.append(child)

    return {"type": "CompositeNode", "children": children}

def _process_grid_repeater(config: Dict[str, Any]) -> Dict[str, Any]:
    repeater = GridRepeaterConfig(**config)
    children = []

    start_x = - (repeater.count_x - 1) * repeater.pitch_x / 2.0
    start_y = - (repeater.count_y - 1) * repeater.pitch_y / 2.0
    start_z = - (repeater.count_z - 1) * repeater.pitch_z / 2.0

    idx = 0
    for iz in range(repeater.count_z):
        for iy in range(repeater.count_y):
            for ix in range(repeater.count_x):
                x = start_x + ix * repeater.pitch_x
                y = start_y + iy * repeater.pitch_y
                z = start_z + iz * repeater.pitch_z

                variables = {
                    "index": idx,
                    "ix": ix,
                    "iy": iy,
                    "iz": iz,
                    "x": x,
                    "y": y,
                    "z": z
                }

                child = _apply_template(repeater.template, variables)

                transformations = child.get("transformations", [])
                if not isinstance(transformations, list):
                    transformations = []

                t1 = {"type": "translate", "x": x, "y": y, "z": z}

                transforms_to_add = [t1]

                if not math.isclose(repeater.angle, 0.0):
                    t2 = {"type": "rotate", "axis": repeater.axis, "angle": f"{repeater.angle} deg"}
                    transforms_to_add.append(t2)

                t3 = {
                    "type": "translate",
                    "x": repeater.origin[0],
                    "y": repeater.origin[1],
                    "z": repeater.origin[2]
                }
                transforms_to_add.append(t3)

                transformations.extend(transforms_to_add)
                child["transformations"] = transformations

                # Recursively expand
                child = expand_repeaters(child)
                children.append(child)
                idx += 1

    return {"type": "CompositeNode", "children": children}

def _process_zip_repeater(config: Dict[str, Any]) -> Dict[str, Any]:
    repeater = ZipRepeaterConfig(**config)
    children = []

    if not repeater.values:
        return {"type": "CompositeNode", "children": []}

    num_nodes = len(next(iter(repeater.values.values())))
    for key, val_list in repeater.values.items():
        if len(val_list) != num_nodes:
            raise ValueError(f"ZipRepeater values list for '{key}' has length {len(val_list)}, expected {num_nodes}")

    for i in range(num_nodes):
        variables = {"index": i}
        for key, val_list in repeater.values.items():
            variables[key] = val_list[i]

        child = _apply_template(repeater.template, variables)
        child = expand_repeaters(child)
        children.append(child)

    return {"type": "CompositeNode", "children": children}

def expand_repeaters(config_dict: Any) -> Any:
    if isinstance(config_dict, dict):
        node_type = config_dict.get("type")
        if node_type == "RingRepeater":
            return _process_ring_repeater(config_dict)
        elif node_type == "GridRepeater":
            return _process_grid_repeater(config_dict)
        elif node_type == "ZipRepeater":
            return _process_zip_repeater(config_dict)
        else:
            return {k: expand_repeaters(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [expand_repeaters(item) for item in config_dict]
    else:
        return config_dict
