import numpy as np
from typing import Dict, Any, Union
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Energy

class AttenuationFunction(Dict[Material, Any]):
    def __init__(self, process: Any, attenuation_database: Any, kind: str = 'linear') -> None: ...
    def __call__(self, material: Union[Material, MaterialArray], energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]: ...
