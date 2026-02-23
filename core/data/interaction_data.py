import numpy as np
from typing import Optional, Any, Union, Tuple, cast, Dict
from numpy.typing import NDArray
from core.other.typing_definitions import Float

# Define internal precision for default operations from typing_definitions
from core.other.typing_definitions import Float as DEFAULT_PRECISION

def get_interaction_dtype(precision: Any = DEFAULT_PRECISION) -> np.dtype:
    """ Генерирует dtype для данных взаимодействия с заданной точностью """
    p_char = 'd' if precision == np.float64 else 'f'
    return np.dtype([
        ('global_position', f'3{p_char}'),
        ('global_direction', f'3{p_char}'),
        ('local_position', f'3{p_char}'),
        ('local_direction', f'3{p_char}'),
        ('process_name', 'U30'),
        ('particle_type', 'U30'),
        ('particle_ID', 'u8'),
        ('energy_deposit', p_char),
        ('material_density', p_char),
        ('scattering_angles', f'2{p_char}'),
        ('emission_time', p_char),
        ('emission_energy', p_char),
        ('emission_position', f'3{p_char}'),
        ('emission_direction', f'3{p_char}'),
        ('distance_traveled', p_char),
    ])

class InteractionArray(np.recarray):
    """
    Класс массива данных взаимодействий
    """
    def __new__(cls, shape: Union[int, Tuple[int, ...]], precision: Any = DEFAULT_PRECISION) -> 'InteractionArray':
        dtype = get_interaction_dtype(precision)
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
