"""
Модуль базы данных массовых коэффициентов ослабления (MAC) по таблицам NIST XCOM.
"""

from pathlib import Path
import math
from typing import Any, Dict, Iterable, Tuple

import h5py
import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.materials.atomic_properties import element_symbol
from core.materials.materials import Material
from core.other.typing_definitions import Float

processes_names: Dict[str, str] = {
    'Photoelectric absorption': 'PhotoelectricEffect',
    'Incoherent scattering': 'ComptonScattering',
    'Coherent scattering': 'CoherentScattering',
}
processes_dtype = np.dtype([(name, Float) for name in processes_names.values()])
MAC_dtype = np.dtype([('Energy', Float), ('Coefficient', processes_dtype)])


def _interpolate_element_on_material_grid(
    material_energy: NDArray[Float],
    element_energy: NDArray[Float],
    element_coeff: NDArray[Float],
) -> NDArray[Float]:
    """
    Вычисляет массовый коэффициент ослабления отдельного элемента на объединенной сетке материала
    методом кусочной log-log интерполяции между краями поглощения с учетом односторонних пределов
    на парах узлов краев.
    """
    mat_size = material_energy.size
    el_size = element_energy.size
    result = np.zeros(mat_size, dtype=Float)

    for node_idx in range(mat_size):
        query_e = float(material_energy[node_idx])
        mat_left_pos = int(np.searchsorted(material_energy, query_e, side='left'))
        offset = node_idx - mat_left_pos

        left_pos = int(np.searchsorted(element_energy, query_e, side='left'))
        right_pos = int(np.searchsorted(element_energy, query_e, side='right'))
        multiplicity = right_pos - left_pos

        if multiplicity >= 2:
            if offset == 0:
                result[node_idx] = element_coeff[left_pos]
            else:
                result[node_idx] = element_coeff[right_pos - 1]
        elif multiplicity == 1:
            result[node_idx] = element_coeff[left_pos]
        else:
            if right_pos <= 0:
                result[node_idx] = element_coeff[0]
            elif right_pos >= el_size:
                result[node_idx] = element_coeff[el_size - 1]
            else:
                e0 = float(element_energy[right_pos - 1])
                e1 = float(element_energy[right_pos])
                v0 = float(element_coeff[right_pos - 1])
                v1 = float(element_coeff[right_pos])
                if v0 > 0.0 and v1 > 0.0 and e0 > 0.0 and e1 > e0:
                    log_fraction = (math.log(query_e) - math.log(e0)) / (math.log(e1) - math.log(e0))
                    result[node_idx] = Float(
                        math.exp(math.log(v0) + log_fraction * (math.log(v1) - math.log(v0)))
                    )
                elif e1 > e0:
                    lin_fraction = (query_e - e0) / (e1 - e0)
                    result[node_idx] = Float(v0 + lin_fraction * (v1 - v0))
                else:
                    result[node_idx] = Float(v0)

    return result


class AttenuationDataBase(Dict[Material, NDArray[Any]]):
    """База данных массовых коэффициентов ослабления для материалов и химических элементов."""

    _base_name: str
    _elements_MAC: Dict[str, NDArray[Any]]
    _elements_edges: Dict[str, Tuple[NDArray[Float], NDArray[np.str_]]]
    _energy_range: NDArray[Float]

    def __init__(self, base_name: str = 'NIST XCOM Elements MAC') -> None:
        super().__init__()
        self._base_name = base_name
        self._elements_MAC = {}
        self._elements_edges = {}
        self._energy_range = np.array([1.0 * units.keV, 20.0 * units.MeV], dtype=Float)
        self._load_elements_MAC()

    @property
    def base_name(self) -> str:
        """Имя файла HDF5-базы массовых коэффициентов ослабления."""
        return self._base_name

    @base_name.setter
    def base_name(self, value: str) -> None:
        self._base_name = value
        self._load_elements_MAC()

    @property
    def energy_range(self) -> NDArray[Float]:
        """Канонический энергетический диапазон загруженной базы данных сечений."""
        return self._energy_range

    @property
    def elements_MAC(self) -> Dict[str, NDArray[Any]]:
        """Словарь таблиц массовых коэффициентов ослабления химических элементов."""
        return self._elements_MAC

    @property
    def elements_edges(self) -> Dict[str, Tuple[NDArray[Float], NDArray[np.str_]]]:
        """Словарь краев поглощения химических элементов: {символ: (energies, labels)}."""
        return self._elements_edges

    def get_element_edges(self, element: int | str) -> Tuple[NDArray[Float], NDArray[np.str_]]:
        """
        Возвращает кортеж (edge_energies, edge_labels) краев поглощения для элемента
        по атомному номеру Z или химическому символу.
        """
        if isinstance(element, (int, np.integer)):
            atomic_z = int(element)
            if atomic_z not in element_symbol:
                raise ValueError(f"Неизвестный атомный номер элемента: Z={atomic_z}.")
            symbol = element_symbol[atomic_z]
        elif isinstance(element, str):
            symbol = element
        else:
            raise TypeError(
                f"Аргумент element должен быть атомным номером (int) или символом (str), получено: {type(element)}."
            )

        if symbol not in self._elements_edges:
            raise ValueError(f"Химический элемент '{symbol}' отсутствует в базе краев поглощения.")
        return self._elements_edges[symbol]

    @staticmethod
    def _validate_and_extract_element(
        element: str,
        element_group: Any,
    ) -> Tuple[NDArray[Any], NDArray[Float], NDArray[np.str_]]:
        """
        Валидирует схему группы элемента в HDF5 (монотонность сетки, наличие и согласованность
        подгруппы Edges, скачки фотоэффекта на краях) и возвращает структурированный массив MAC_dtype
        вместе с метаданными краев.
        """
        if not isinstance(element_group, h5py.Group):
            raise ValueError(f"Объект '{element}' в HDF5-базе не является группой h5py.Group.")

        if 'Energy' not in element_group:
            raise ValueError(f"В группе элемента '{element}' отсутствует обязательный датасет 'Energy'.")

        if 'Edges' not in element_group or not isinstance(element_group['Edges'], h5py.Group):
            raise ValueError(f"В группе элемента '{element}' отсутствует обязательная подгруппа 'Edges'.")

        edges_group = element_group['Edges']
        if 'energy' not in edges_group or 'label' not in edges_group:
            raise ValueError(
                f"В подгруппе 'Edges' элемента '{element}' должны присутствовать датасеты 'energy' и 'label'."
            )

        energy = np.asarray(element_group['Energy'][:], dtype=Float)
        if energy.ndim != 1 or energy.size < 2:
            raise ValueError(
                f"Сетка энергий элемента '{element}' должна быть одномерным массивом длины >= 2."
            )
        if not np.all(np.isfinite(energy)) or float(energy[0]) <= 0.0:
            raise ValueError(
                f"Сетка энергий элемента '{element}' должна содержать строго положительные конечные числа."
            )

        if np.any(np.diff(energy) < 0.0):
            raise ValueError(f"Сетка энергий элемента '{element}' не является монотонно неубывающей.")

        unique_energies, counts = np.unique(energy, return_counts=True)

        edge_energy = np.asarray(edges_group['energy'][:], dtype=Float)
        raw_labels = edges_group['label'][:]
        edge_label = np.asarray(
            [lbl.decode('utf-8') if isinstance(lbl, (bytes, np.bytes_)) else str(lbl) for lbl in raw_labels],
            dtype=np.str_,
        )

        if edge_energy.ndim != 1 or edge_label.ndim != 1 or edge_energy.size != edge_label.size:
            raise ValueError(
                f"Датасеты 'Edges/energy' и 'Edges/label' элемента '{element}' имеют несогласованные размеры."
            )
        if edge_energy.size > 1 and np.any(np.diff(edge_energy) <= 0.0):
            raise ValueError(f"Массив 'Edges/energy' элемента '{element}' должен строго возрастать.")

        duplicate_energies = unique_energies[counts >= 2]
        if duplicate_energies.size != edge_energy.size or not np.allclose(
            duplicate_energies, edge_energy, rtol=1e-12, atol=0.0
        ):
            raise ValueError(
                f"Дублирующиеся узлы краев в 'Energy' элемента '{element}' не совпадают с 'Edges/energy'."
            )

        mapped_processes: Dict[str, NDArray[Float]] = {}
        for hdf_name, internal_name in processes_names.items():
            if hdf_name not in element_group:
                raise ValueError(
                    f"В группе элемента '{element}' отсутствует обязательный датасет процесса '{hdf_name}'."
                )
            proc_arr = np.asarray(element_group[hdf_name][:], dtype=Float)
            if proc_arr.ndim != 1 or proc_arr.size != energy.size:
                raise ValueError(
                    f"Размер датасета '{hdf_name}' элемента '{element}' ({proc_arr.shape}) "
                    f"не совпадает с размером сетки 'Energy' ({energy.shape})."
                )
            if not np.all(np.isfinite(proc_arr)) or np.any(proc_arr < 0.0):
                raise ValueError(
                    f"Значения процесса '{hdf_name}' элемента '{element}' должны быть конечными и неотрицательными."
                )
            mapped_processes[internal_name] = proc_arr

        pe_values = mapped_processes['PhotoelectricEffect']
        for edge_val in duplicate_energies:
            left_pos = int(np.searchsorted(energy, edge_val, side='left'))
            right_pos = int(np.searchsorted(energy, edge_val, side='right'))
            if np.any(pe_values[left_pos + 1 : right_pos] <= pe_values[left_pos]):
                raise ValueError(
                    f"На паре узлов края E={edge_val} элемента '{element}' значение выше края "
                    f"({pe_values[right_pos - 1]}) должно быть строго больше значения ниже края ({pe_values[left_pos]})."
                )

        mac_table: NDArray[Any] = np.zeros(energy.size, dtype=MAC_dtype)
        mac_table['Energy'] = energy
        for process_name, value in mapped_processes.items():
            mac_table['Coefficient'][process_name] = value

        return mac_table, edge_energy, edge_label

    def _load_elements_MAC(self) -> None:
        candidate_path = Path(self._base_name)
        if candidate_path.suffix == '.h5' and candidate_path.is_file():
            table_path = candidate_path
        else:
            table_path = Path(f'tables/{self._base_name}.h5')
        if not table_path.is_file():
            raise ValueError(f"Файл базы данных коэффициентов ослабления не найден: '{table_path}'.")

        loaded_mac: Dict[str, NDArray[Any]] = {}
        loaded_edges: Dict[str, Tuple[NDArray[Float], NDArray[np.str_]]] = {}
        min_energies = []
        max_energies = []

        with h5py.File(table_path, 'r') as file:
            if len(file.keys()) == 0:
                raise ValueError(f"Файл базы данных '{table_path}' не содержит групп элементов.")

            for element, element_group in file.items():
                element_name = str(element)
                mac_table, edge_energy, edge_label = self._validate_and_extract_element(
                    element_name, element_group
                )
                loaded_mac[element_name] = mac_table
                loaded_edges[element_name] = (edge_energy, edge_label)
                min_energies.append(float(mac_table['Energy'][0]))
                max_energies.append(float(mac_table['Energy'][-1]))

            if 'energy_range' in file.attrs:
                attr_range = np.asarray(file.attrs['energy_range'], dtype=Float)
                if attr_range.shape != (2,) or attr_range[0] <= 0.0 or attr_range[0] >= attr_range[1]:
                    raise ValueError(f"Некорректный атрибут 'energy_range' в файле '{table_path}'.")
                self._energy_range = attr_range
            else:
                self._energy_range = np.array([max(min_energies), min(max_energies)], dtype=Float)

        self._elements_MAC = loaded_mac
        self._elements_edges = loaded_edges

    def add_materials(self, materials: Iterable[Material]) -> None:
        """Пакетно добавляет коллекцию материалов в базу данных."""
        for mat in materials:
            self.add_material(mat)

    def add_material(self, material: Material) -> None:
        """Добавляет единичный материал в базу данных коэффициентов ослабления."""
        if len(material.composition_dict) == 0:
            raise ValueError(f"Материал '{material.name}' имеет пустой элементный состав.")

        process_field_names = tuple(processes_names.values())
        max_multiplicity: Dict[float, int] = {}

        for element in material.composition_dict.keys():
            if element not in self._elements_MAC:
                raise ValueError(
                    f"Химический элемент '{element}' материала '{material.name}' отсутствует в базе сечений."
                )
            el_energy = self._elements_MAC[element]['Energy']
            unique_vals, counts = np.unique(el_energy, return_counts=True)
            for e_val, count in zip(unique_vals, counts):
                key_val = float(e_val)
                prev_count = max_multiplicity.get(key_val, 0)
                if int(count) > prev_count:
                    max_multiplicity[key_val] = int(count)

        merged_energy_list = []
        for e_val in sorted(max_multiplicity.keys()):
            for _ in range(max_multiplicity[e_val]):
                merged_energy_list.append(e_val)

        array_of_energy = np.asarray(merged_energy_list, dtype=Float)
        array_of_mac: NDArray[Any] = np.zeros(array_of_energy.size, dtype=MAC_dtype)
        array_of_mac['Energy'] = array_of_energy

        for process_name in process_field_names:
            total_coeff = np.zeros(array_of_energy.size, dtype=Float)
            for element, weight in material.composition_dict.items():
                el_table = self._elements_MAC[element]
                el_energy = el_table['Energy']
                el_coeff = el_table['Coefficient'][process_name]
                interpolated_el = _interpolate_element_on_material_grid(
                    array_of_energy,
                    el_energy,
                    el_coeff,
                )
                total_coeff += Float(weight) * interpolated_el
            array_of_mac['Coefficient'][process_name] = total_coeff

        self.update({material: array_of_mac})


__all__ = [
    "_interpolate_element_on_material_grid",
    "AttenuationDataBase",
    "processes_names",
    "processes_dtype",
    "MAC_dtype",
]