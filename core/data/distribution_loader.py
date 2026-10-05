import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np

_logger = logging.getLogger(__name__)


class DistributionLoader:
    """
    Сервис детерминированной загрузки матриц распределения вокселей и активности.
    Реализует строгую диспетчеризацию по расширению файла без угадывания форматов через подавление исключений
    и выполняет обязательную LBYL-валидацию целостности бинарных буферов (file_size == count * sizeof(dtype)).
    """

    SUPPORTED_EXTENSIONS: Sequence[str] = ('.npy', '.raw', '.bin', '.dat', '.txt', '.csv')

    @classmethod
    def inspect_metadata(cls, file_path: Union[str, Path]) -> dict[str, Any]:
        """
        Легковесная инспекция метаданных файла распределения без полной загрузки всего массива в память.
        """
        path_object = Path(file_path)
        if not path_object.is_file():
            raise FileNotFoundError(f"Файл распределения не существует: {path_object}")

        suffix = path_object.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Неподдерживаемое расширение файла распределения '{suffix}'. "
                f"Поддерживаемые форматы: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        file_size_bytes = path_object.stat().st_size
        metadata_result: dict[str, Any] = {
            'suffix': suffix,
            'file_size': file_size_bytes,
            'is_npy': (suffix == '.npy'),
            'shape': None,
            'order': None,
            'dtype': None,
        }

        if suffix == '.npy':
            try:
                with open(path_object, 'rb') as file_descriptor:
                    version = np.lib.format.read_magic(file_descriptor)
                    if version == (1, 0):
                        shape, is_fortran_order, data_type = np.lib.format.read_array_header_1_0(file_descriptor)
                    elif version == (2, 0):
                        shape, is_fortran_order, data_type = np.lib.format.read_array_header_2_0(file_descriptor)
                    else:
                        shape, is_fortran_order, data_type = np.lib.format._read_array_header(file_descriptor, version)
                    metadata_result['shape'] = tuple(shape)
                    metadata_result['order'] = 'F' if is_fortran_order else 'C'
                    metadata_result['dtype'] = np.dtype(data_type)
            except Exception:
                memory_mapped_array = np.load(path_object, mmap_mode='r')
                metadata_result['shape'] = tuple(memory_mapped_array.shape)
                metadata_result['order'] = 'F' if np.isfortran(memory_mapped_array) else 'C'
                metadata_result['dtype'] = memory_mapped_array.dtype

        return metadata_result

    @classmethod
    def load(
        cls,
        file_path: Union[str, Path],
        target_shape: Optional[Tuple[int, ...]] = None,
        order: str = 'F',
        dtype: Any = np.float32,
        encoding: Optional[str] = None
    ) -> np.ndarray:
        """
        Загрузка матрицы распределения из файла.

        :param file_path: Путь к файлу на диске.
        :param target_shape: Требуемая форма массива (обязательна для .raw, .bin, .dat, .txt, .csv).
        :param order: Порядок развертки многомерного массива ('C' или 'F').
        :param dtype: Тип данных элементов для бинарных файлов.
        :param encoding: Явное указание кодировки ('binary' или 'text'). При None определяется по расширению/LBYL-размеру.
        :return: Загруженный массив numpy.
        :raises FileNotFoundError: Если файл не существует на диске.
        :raises ValueError: Если формат не поддерживается, размеры не заданы или размер буфера не совпадает.
        """
        path_object = Path(file_path)
        if not path_object.is_file():
            raise FileNotFoundError(f"Файл распределения не существует: {path_object}")

        suffix = path_object.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Неподдерживаемое расширение файла распределения '{suffix}'. "
                f"Поддерживаемые форматы: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        # 1. NumPy бинарный формат
        if suffix == '.npy':
            loaded_data = np.load(path_object, allow_pickle=True)
            return loaded_data

        # Для не-.npy форматов target_shape обязателен
        if target_shape is None or any(dim_size <= 0 for dim_size in target_shape):
            raise ValueError(
                f"Для формата '{suffix}' необходимо указать положительные размеры target_shape, получено: {target_shape}"
            )

        element_dtype = np.dtype(dtype)
        expected_elements_count = int(np.prod(target_shape))
        expected_bytes_count = expected_elements_count * element_dtype.itemsize
        actual_bytes_count = path_object.stat().st_size

        # Определение режима: бинарный или текстовый
        is_binary = False
        if encoding == 'binary':
            is_binary = True
        elif encoding == 'text':
            is_binary = False
        elif suffix in ('.raw', '.bin'):
            is_binary = True
        elif suffix in ('.txt', '.csv'):
            is_binary = False
        elif suffix == '.dat':
            # Для .dat: если режим не задан явно, проверяем совпадение байтового объема
            is_binary = (actual_bytes_count == expected_bytes_count)

        if is_binary:
            if actual_bytes_count != expected_bytes_count:
                raise ValueError(
                    f"Размер бинарного файла ({actual_bytes_count} байт) не соответствует "
                    f"требуемой форме {target_shape} для типа {element_dtype} ({expected_bytes_count} байт)"
                )
            raw_buffer = np.fromfile(path_object, dtype=element_dtype)
            return raw_buffer.reshape(target_shape, order=order)
        else:
            text_data = np.loadtxt(path_object, dtype=element_dtype)
            if text_data.shape != tuple(target_shape):
                if text_data.size != expected_elements_count:
                    raise ValueError(
                        f"Количество элементов в текстовом файле ({text_data.size}) не совпадает "
                        f"с требуемой формой {target_shape} ({expected_elements_count})"
                    )
                return text_data.reshape(target_shape, order=order)
            return text_data
