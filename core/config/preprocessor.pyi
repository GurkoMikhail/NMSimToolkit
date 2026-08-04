from typing import Any, Dict, List, Union
from pydantic import BaseModel

class RingRepeaterConfig(BaseModel):
    num_nodes: int
    radius: float
    start_angle: float
    angular_span: float
    axis: List[float]
    center: List[float]
    base_axis: List[float]
    template: Dict[str, Any]

class GridRepeaterConfig(BaseModel):
    count_x: int
    count_y: int
    count_z: int
    pitch_x: float
    pitch_y: float
    pitch_z: float
    origin: List[float]
    axis: List[float]
    angle: float
    template: Dict[str, Any]

class ZipRepeaterConfig(BaseModel):
    values: Dict[str, List[Any]]
    template: Dict[str, Any]

def _smart_cast(value: str) -> Union[str, int, float]: ...
def _apply_template(node: Any, variables: Dict[str, Any]) -> Any: ...
def _process_ring_repeater(config: Dict[str, Any]) -> Dict[str, Any]: ...
def _process_grid_repeater(config: Dict[str, Any]) -> Dict[str, Any]: ...
def _process_zip_repeater(config: Dict[str, Any]) -> Dict[str, Any]: ...
def expand_repeaters(config_dict: Any) -> Any: ...
