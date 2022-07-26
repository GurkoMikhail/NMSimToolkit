from pathlib import Path
from hepunits import*
from h5py import File
import numpy as np


class SimulationDataManager:
    """ 
    Основной класс менеджера данных получаемых при моделировании
    """

    def __init__(self, filename, sensitive_volumes=[], **kwds):
        self.filename = Path(filename)
        self.sensitive_volumes = sensitive_volumes
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

    def check_progress_in_file(self):
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

    def add_interaction_data(self, interaction_data):
        for volume in self.sensitive_volumes:
            in_volume = volume.check_inside(interaction_data.position)
            interaction_data_in_volume = interaction_data[in_volume]
            if interaction_data_in_volume.size > 0:
                interaction_data_for_save = np.recarray(interaction_data_in_volume.shape, dtype=interaction_data_dtype)

                interaction_data_for_save.global_position = interaction_data_in_volume.position
                interaction_data_for_save.global_direction = interaction_data_in_volume.direction
                interaction_data_for_save.local_position = volume.convert_to_local_position(interaction_data_in_volume.position, as_parent=False)
                interaction_data_for_save.local_direction = volume.convert_to_local_direction(interaction_data_in_volume.direction, as_parent=False)

                for field in interaction_data_in_volume.dtype.fields:
                    if field in interaction_data_for_save.dtype.fields:
                        interaction_data_for_save[field] = interaction_data_in_volume[field]

                self.interaction_data[volume.name].append(interaction_data_for_save)
                self._buffered_interaction_number += interaction_data_for_save.size
        if self._buffered_interaction_number > self.iteraction_buffer_size:
            self.save_interaction_data()
            self._buffered_interaction_number = 0

    def concatenate_interaction_data(self):
        for volume in self.sensitive_volumes:
            volume_name = volume.name
            self.interaction_data[volume_name] = np.concatenate(self.interaction_data[volume_name]).view(np.recarray)

    def clear_interaction_data(self):
        self.interaction_data = {volume.name: [] for volume in self.sensitive_volumes}

    def save_interaction_data(self):
        self.concatenate_interaction_data()
        try:
            file = File(f'output data/{self.filename}', 'a')
        except Exception:
            print(f'Не удалось сохранить {self.filename.name}!')
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
            print(f'{self.filename.name} generated {self._buffered_interaction_number} events')
            file.close()
        self.clear_interaction_data()


interaction_data_dtype = np.dtype([
        ('global_position', '3d'),
        ('global_direction', '3d'),
        ('local_position', '3d'),
        ('local_direction', '3d'),
        ('process_name', 'S30'),
        ('particle_type', 'S30'),
        ('particle_ID', 'u8'),
        ('energy_deposit', 'd'),
        ('material_density', 'd'),
        ('scattering_angles', '2d'),
        ('emission_time', 'd'),
        ('emission_position', '3d'),
        ('distance_traveled', 'd'),
])

