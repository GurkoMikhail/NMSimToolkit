import numpy as np
from typing import Optional, Any, Union, Tuple, cast, Dict
from numpy.typing import NDArray
from core.other.typing_definitions import Float, ID

def get_interaction_dtype() -> np.dtype:
    """ Генерирует dtype для данных взаимодействия """
    return np.dtype([
        ('global_position', (Float, 3)),
        ('global_direction', (Float, 3)),
        ('local_position', (Float, 3)),
        ('local_direction', (Float, 3)),
        ('process_name', 'U30'),
        ('particle_type', 'U30'),
        ('particle_ID', 'u8'),
        ('energy_deposit', Float),
        ('material_density', Float),
        ('scattering_angles', (Float, 2)),
        ('emission_time', Float),
        ('emission_energy', Float),
        ('emission_position', (Float, 3)),
        ('emission_direction', (Float, 3)),
        ('distance_traveled', Float),
    ])

class InteractionArray(np.recarray):
    """
    Класс массива данных взаимодействий
    """
    def __new__(cls, shape: Union[int, Tuple[int, ...]]) -> 'InteractionArray':
        dtype = get_interaction_dtype()
        # Создаем массив и преобразуем его в recarray и затем в наш класс
        obj = np.ndarray.__new__(cls, shape, dtype=dtype).view(cls)
        return cast('InteractionArray', obj)

    # Псевдонимы для совместимости со старым кодом процессов
    @property
    def position(self) -> NDArray[Float]: # type: ignore
        return self.global_position

    @position.setter
    def position(self, value: Any) -> None:
        self.global_position = value

    @property
    def direction(self) -> NDArray[Float]: # type: ignore
        return self.global_direction

    @direction.setter
    def direction(self, value: Any) -> None:
        self.global_direction = value
