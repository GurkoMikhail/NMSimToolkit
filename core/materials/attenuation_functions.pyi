import numpy as np
from typing import Dict, Any, Union, overload
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Energy, Float
from numpy.typing import NDArray

class AttenuationFunction(Dict[Material, Any]):
    def __init__(self, process: Any, attenuation_database: Any, kind: str = 'linear') -> None: ...

    @overload
    def __call__(self, material: Material, energy: Float) -> Float: ...

    @overload
    def __call__(self, material: Union[Material, MaterialArray], energy: NDArray[Float]) -> NDArray[Float]: ...
