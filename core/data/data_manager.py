import logging
from pathlib import Path
from hepunits import*
from h5py import File
import numpy as np
from typing import List, Any, Optional, Dict, Tuple, cast
from numpy.typing import NDArray
from core.other.typing_definitions import Float
from core.geometry.volumes import ElementaryVolume
from core.data.interaction_data import InteractionArray

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class SimulationDataManager:
    """ 
    Основной класс менеджера данных получаемых при моделировании
    """
    filename: Path
    sensitive_volumes: List[ElementaryVolume]
    lock: Optional[Any]
    save_emission_distribution: bool
    save_dose_distribution: bool
    distribution_voxel_size: float
    iteraction_buffer_size: int
    _buffered_interaction_number: int
    interaction_data: Dict[str, List[InteractionArray]]

    def __init__(self, filename: str, sensitive_volumes: List[ElementaryVolume] = [], lock: Optional[Any] = None, **kwds: Any) -> None:
        self.filename = Path(f'output data/{filename}')
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.sensitive_volumes = sensitive_volumes
        self.lock = lock
        self.save_emission_distribution = True
        self.save_dose_distribution = True
        self.distribution_voxel_size = 4.*mm
        self.clear_interaction_data()
        self.iteraction_buffer_size = int(10**3)
        self._buffered_interaction_number = 0
        self.args = [
            'save_emission_distribution',
            'save_dose_distribution',
            'distribution_voxel_size',
            'iteraction_buffer_size'
            ]

        for arg in self.args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

    def check_progress_in_file(self) -> Tuple[Optional[float], Optional[Any]]:
        try:
            file = File(self.filename, 'r')
        except Exception:
            last_time = None
            state = None
        else:
            try:
                last_time = file['Source timer']
                # state = file['Source state']
                state = None
                last_time = float(np.array(last_time))
            except Exception:
                print(f'\tНе удалось восстановить прогресс')
                last_time = None
                state = None
            finally:
                print(f'\tПрогресс восстановлен')
                file.close()
        finally:
            print(f'\tSource timer: {last_time}')
            return last_time, state

    def add_interaction_data(self, interaction_data: InteractionArray) -> None:
        from core.geometry.volumes import TransformableVolume
        for volume in self.sensitive_volumes:
            in_volume = volume.check_inside(interaction_data.position)
            interaction_data_in_volume = cast(InteractionArray, interaction_data[in_volume])
            if interaction_data_in_volume.size > 0:
                # Определение точности на основе одного из вещественных полей
                precision = interaction_data.dtype['energy_deposit'].type
                interaction_data_for_save = InteractionArray(interaction_data_in_volume.shape, precision=precision)

                interaction_data_for_save.global_position = interaction_data_in_volume.position
                interaction_data_for_save.global_direction = interaction_data_in_volume.direction
                if isinstance(volume, TransformableVolume):
                    interaction_data_for_save.local_position = volume.convert_to_local_position(interaction_data_in_volume.position, as_parent=False)
                    interaction_data_for_save.local_direction = volume.convert_to_local_direction(interaction_data_in_volume.direction, as_parent=False)

                for field in interaction_data_in_volume.dtype.names: # type: ignore
                    if field in interaction_data_for_save.dtype.names and field not in ['global_position', 'global_direction', 'local_position', 'local_direction']: # type: ignore
                        interaction_data_for_save[field] = interaction_data_in_volume[field]

                self.interaction_data[volume.name].append(interaction_data_for_save)
                self._buffered_interaction_number += interaction_data_for_save.size
        if self._buffered_interaction_number > self.iteraction_buffer_size:
            self.save_interaction_data()
            self._buffered_interaction_number = 0

    def concatenate_interaction_data(self) -> None:
        for volume in self.sensitive_volumes:
            volume_name = volume.name
            if self.interaction_data[volume_name]:
                self.interaction_data[volume_name] = np.concatenate(self.interaction_data[volume_name]).view(InteractionArray) # type: ignore

    def clear_interaction_data(self) -> None:
        self.interaction_data = {volume.name: [] for volume in self.sensitive_volumes}

    def save_interaction_data(self):
        if self.lock is None:
            self._save_interaction_data()
        else:
            with self.lock:
                self._save_interaction_data()

    def _save_interaction_data(self):
        self.concatenate_interaction_data()
        try:
            file = File(self.filename, 'a')
        except Exception:
            _logger.exception(f'Не удалось сохранить данные в {self.filename}!')
        else:
            if not 'interaction_data' in file:
                group = file.create_group('interaction_data')
                for volume_name, interaction_data in self.interaction_data.items():
                    volume_group = group.create_group(volume_name)
                    for field in interaction_data.dtype.fields:
                        maxshape = list(interaction_data[field].shape)
                        maxshape[0] = None
                        volume_group.create_dataset(
                            field,
                            data=interaction_data[field],
                            compression="gzip",
                            chunks=True,
                            maxshape=maxshape
                            )
            else:
                group = file['interaction_data']
                for volume_name, interaction_data in self.interaction_data.items():
                    volume_group = group[volume_name]
                    for key in volume_group.keys():
                        volume_group[key].resize(
                            (volume_group[key].shape[0] + interaction_data[key].shape[0]),
                            axis=0
                            )
                        volume_group[key][-interaction_data[key].shape[0]:] = interaction_data[key]
            _logger.info(f'{self._buffered_interaction_number} events saved to {self.filename}')
            file.close()
        self.clear_interaction_data()
