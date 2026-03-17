from typing import List

from core.geometry.volumes import Volume
from core.physics.processes import Process
from core.materials.material_bank import MaterialBank
from core.physics.physics_buffer import PhysicsBuffer
from core.materials.materials import Material

class PhysicsCompiler:
    def _build_material_bank(self, materials_list: List[Material], processes_list: List[Process]) -> MaterialBank: ...
    def compile_scene(self, root_volume: Volume, processes_list: List[Process]) -> PhysicsBuffer: ...
