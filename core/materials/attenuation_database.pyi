import numpy as np
from typing import Dict, Any, Iterable, Optional, Union
from core.materials.materials import Material

class AttenuationDataBase(Dict[Material, np.ndarray]):
    _base_name: str
    _elements_MAC: Dict[str, np.ndarray]
    def __init__(self, base_name: str = 'NIST XCOM Elements MAC') -> None: ...
    @property
    def base_name(self) -> str: ...
    @base_name.setter
    def base_name(self, value: str) -> None: ...
    def add_material(self, material: Union[Material, Iterable[Material]]) -> None: ...

processes_names: Dict[str, str]
processes_dtype: np.dtype
MAC_dtype: np.dtype
