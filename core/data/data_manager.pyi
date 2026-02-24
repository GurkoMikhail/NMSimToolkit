import numpy as np
from pathlib import Path
from typing import List, Any, Optional, Dict, Tuple, Union
from core.geometry.volumes import ElementaryVolume
from core.other.typing_definitions import Float
from core.data.interaction_data import InteractionArray

class SimulationDataManager:
    filename: Path
    sensitive_volumes: List[ElementaryVolume]
    lock: Optional[Any]
    save_emission_distribution: bool
    save_dose_distribution: bool
    distribution_voxel_size: Float
    interaction_buffer_size: int
    _buffered_interaction_number: int
    interaction_data: Dict[str, Union[List[InteractionArray], InteractionArray]]
    args: List[str]

    def __init__(self, filename: str, sensitive_volumes: List[ElementaryVolume] = ..., lock: Optional[Any] = None, **kwds: Any) -> None: ...
    def check_progress_in_file(self) -> Tuple[Optional[Float], Optional[Any]]: ...
    def add_interaction_data(self, interaction_data: InteractionArray) -> None: ...
    def concatenate_interaction_data(self) -> None: ...
    def clear_interaction_data(self) -> None: ...
    def save_interaction_data(self) -> None: ...
    def _save_interaction_data(self) -> None: ...
