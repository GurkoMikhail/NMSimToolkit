import numpy as np
from typing import Tuple, List, Dict
from core.geometry.volumes import ElementaryVolume, TransformableVolume, VolumeWithChilds
from core.geometry.geometries import Box
from core.materials.materials import Material
from core.other.typing_definitions import Float, Index
from core.geometry.geometry_buffer_soa import GeometryBuffer

class SceneCompiler:
    def __init__(self) -> None: ...
    def compile(self, root_volume: ElementaryVolume) -> Tuple[GeometryBuffer, List[Material]]: ...
    def _traverse(self, volume: ElementaryVolume) -> int: ...
