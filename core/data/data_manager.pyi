import numpy as np
from pathlib import Path
from typing import List, Any, Optional, Dict, Tuple, Generic
from core.geometry.volumes import ElementaryVolume
from core.other.typing_definitions import Precision

class SimulationDataManager(Generic[Precision]):
    filename: Path
    sensitive_volumes: List[ElementaryVolume[Precision]]
    lock: Optional[Any]
    save_emission_distribution: bool
    save_dose_distribution: bool
    distribution_voxel_size: float
    iteraction_buffer_size: int
    _buffered_interaction_number: int
    interaction_data: Dict[str, List[np.recarray]]
    args: List[str]

    def __init__(self, filename: str, sensitive_volumes: List[ElementaryVolume[Precision]] = ..., lock: Optional[Any] = None, **kwds: Any) -> None: ...
    def check_progress_in_file(self) -> Tuple[Optional[float], Optional[Any]]: ...
    def add_interaction_data(self, interaction_data: np.recarray) -> None: ...
    def concatenate_interaction_data(self) -> None: ...
    def clear_interaction_data(self) -> None: ...
    def save_interaction_data(self) -> None: ...
    def _save_interaction_data(self) -> None: ...

interaction_data_dtype: np.dtype
