"""
Воспроизводимый офлайн-скрипт генерации HDF5-базы массовых коэффициентов ослабления
`tables/NIST XCOM Elements MAC.h5` на компактной родной сетке NIST XCOM с парами узлов
краев фотоэлектрического поглощения и метаданными электронных оболочек.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import hepunits as units
import numpy as np
import requests
from numpy.typing import NDArray
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.materials.atomic_properties import atomic_number, element_symbol
from core.materials.attenuation_database import AttenuationDataBase
from core.other.typing_definitions import Float

SCHEMA_VERSION: str = "2.0"
DATA_SOURCE: str = "NIST XCOM Photon Cross Sections Database (NBSIR 87-3597, v3.1)"
NIST_XCOM_URL: str = "https://physics.nist.gov/cgi-bin/Xcom/xcom3_1"

SCRIPT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH: Path = SCRIPT_DIR / "nist_xcom_v3_1_reference.npz"
DEFAULT_OUTPUT_PATH: Path = SCRIPT_DIR.parent / "NIST XCOM Elements MAC.h5"

PROCESS_DATASET_NAMES: Tuple[str, ...] = (
    "Photoelectric absorption",
    "Incoherent scattering",
    "Coherent scattering",
    "Pair production in nuclear field",
    "Pair production in electron field",
)

TABLE_MIN_ENERGY: Float = Float(1.0 * units.keV)
TABLE_MAX_ENERGY: Float = Float(20.0 * units.MeV)
RAW_MIN_ENERGY: float = 0.001
RAW_MAX_ENERGY: float = 20.0

_ROW_PATTERN: re.Pattern[str] = re.compile(
    r'<tr\s+BGColor="#FFFFCC"[^>]*>(.*?)</tr>',
    re.IGNORECASE | re.DOTALL,
)
_CELL_PATTERN: re.Pattern[str] = re.compile(
    r"<td[^>]*>(.*?)</td>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_PATTERN: re.Pattern[str] = re.compile(r"<[^>]+>")


def generate_nist_xcom_elements_hdf5(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
) -> Path:
    """
    Формирует файл `NIST XCOM Elements MAC.h5` по эталонному архиву стандартных таблиц NIST XCOM v3.1.
    Все величины записываются в канонических единицах `hepunits` (`MeV` для энергии, `cm2 / g` для MAC).
    """
    if not reference_path.is_file():
        raise FileNotFoundError(f"Эталонный архив NIST XCOM не найден: '{reference_path}'.")

    energy_scale = float(units.MeV)
    mac_scale = float(units.cm2 / units.g)
    utf8_string_dtype = h5py.string_dtype(encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(reference_path, allow_pickle=False) as reference_archive:
        with h5py.File(output_path, "w") as h5_file:
            h5_file.attrs["schema_version"] = SCHEMA_VERSION
            h5_file.attrs["data_source"] = DATA_SOURCE
            h5_file.attrs["energy_min"] = float(TABLE_MIN_ENERGY)
            h5_file.attrs["energy_max"] = float(TABLE_MAX_ENERGY)
            h5_file.attrs["energy_range"] = np.array(
                [float(TABLE_MIN_ENERGY), float(TABLE_MAX_ENERGY)],
                dtype=Float,
            )

            for atomic_z in sorted(element_symbol.keys()):
                symbol = element_symbol[atomic_z]
                energy_key = f"{symbol}_Energy"
                if energy_key not in reference_archive:
                    raise ValueError(f"В эталонном архиве отсутствует элемент '{symbol}' (Z={atomic_z}).")

                energy_arr = np.asarray(reference_archive[energy_key], dtype=Float) * energy_scale
                edge_energy_arr = (
                    np.asarray(reference_archive[f"{symbol}_Edges_energy"], dtype=Float) * energy_scale
                )
                edge_label_arr = np.asarray(reference_archive[f"{symbol}_Edges_label"], dtype=np.str_)

                processes_map: Dict[str, NDArray[Float]] = {}
                for process_key in PROCESS_DATASET_NAMES:
                    archive_proc_key = f"{symbol}_{process_key}"
                    if archive_proc_key not in reference_archive:
                        raise ValueError(
                            f"В эталонном архиве отсутствует процесс '{process_key}' для элемента '{symbol}'."
                        )
                    processes_map[process_key] = (
                        np.asarray(reference_archive[archive_proc_key], dtype=Float) * mac_scale
                    )

                element_group = h5_file.create_group(symbol)
                element_group.attrs["Z"] = int(atomic_number[symbol])
                element_group.create_dataset("Energy", data=energy_arr, dtype=Float)

                for process_key, proc_values in processes_map.items():
                    element_group.create_dataset(process_key, data=proc_values, dtype=Float)

                edges_group = element_group.create_group("Edges")
                edges_group.create_dataset("energy", data=edge_energy_arr, dtype=Float)
                edges_group.create_dataset(
                    "label",
                    data=np.asarray([str(lbl) for lbl in edge_label_arr], dtype=object),
                    dtype=utf8_string_dtype,
                )
                AttenuationDataBase._validate_and_extract_element(symbol, element_group)

    return output_path


def parse_nist_xcom_html(
    html_text: str,
) -> Tuple[NDArray[Float], Dict[str, NDArray[Float]], NDArray[Float], NDArray[np.str_]]:
    """
    Извлекает из HTML-ответа официального сервиса NIST XCOM v3.1 (`xcom3_1`)
    энергии узлов, 5 парциальных сечений и метки краев поглощения в диапазоне `[1 кэВ, 20 МэВ]`.
    """
    row_matches = _ROW_PATTERN.findall(html_text)
    if not row_matches:
        raise ValueError("В HTML-ответе NIST XCOM не найдено ни одной строки таблицы сечений.")

    energies: List[float] = []
    coherent: List[float] = []
    incoherent: List[float] = []
    photoelectric: List[float] = []
    pair_nuclear: List[float] = []
    pair_electron: List[float] = []
    edge_energies: List[float] = []
    edge_labels: List[str] = []

    for row_html in row_matches:
        cells = _CELL_PATTERN.findall(row_html)
        if len(cells) != 9:
            continue

        energy_val = float(cells[1].strip())
        if energy_val < RAW_MIN_ENERGY or energy_val > RAW_MAX_ENERGY:
            continue

        edge_text = (
            _TAG_PATTERN.sub("", cells[0])
            .replace("&#160;", " ")
            .replace("&nbsp;", " ")
            .strip()
        )
        if edge_text:
            tokens = edge_text.split()
            edge_energies.append(energy_val)
            edge_labels.append(tokens[-1])

        energies.append(energy_val)
        coherent.append(float(cells[2].strip()))
        incoherent.append(float(cells[3].strip()))
        photoelectric.append(float(cells[4].strip()))
        pair_nuclear.append(float(cells[5].strip()))
        pair_electron.append(float(cells[6].strip()))

    energy_arr = np.asarray(energies, dtype=Float)
    processes_map: Dict[str, NDArray[Float]] = {
        "Coherent scattering": np.asarray(coherent, dtype=Float),
        "Incoherent scattering": np.asarray(incoherent, dtype=Float),
        "Photoelectric absorption": np.asarray(photoelectric, dtype=Float),
        "Pair production in nuclear field": np.asarray(pair_nuclear, dtype=Float),
        "Pair production in electron field": np.asarray(pair_electron, dtype=Float),
    }
    edge_energy_arr = np.asarray(edge_energies, dtype=Float)
    edge_label_arr = np.asarray(edge_labels, dtype=np.str_)

    return energy_arr, processes_map, edge_energy_arr, edge_label_arr


def _create_nist_retry_session() -> requests.Session:
    """Создает HTTP-сессию с автоматическими повторными попытками при временных сетевых сбоях."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("POST",),
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_nist_xcom_element(
    atomic_z: int,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> Tuple[NDArray[Float], Dict[str, NDArray[Float]], NDArray[Float], NDArray[np.str_]]:
    """
    Запрашивает таблицу стандартной сетки (`Standard Grid`) для элемента с атомным номером `atomic_z`
    напрямую с официального веб-сервиса NIST XCOM v3.1 (`https://physics.nist.gov/cgi-bin/Xcom/xcom3_1`).
    """
    if atomic_z not in element_symbol:
        raise ValueError(f"Недопустимый атомный номер элемента для NIST XCOM: Z={atomic_z}.")
    if timeout <= 0.0:
        raise ValueError(f"Таймаут запроса должен быть строго положительным, получено: {timeout}.")

    payload = {
        "ZNum": str(atomic_z),
        "ZSym": "",
        "OutOpt": "PIC",
        "NumAdd": "1",
        "Energies": "",
        "Output": "on",
        "WindowXmin": "0.001",
        "WindowXmax": "20",
        "ResizeFlag": "on",
    }

    if session is not None:
        response = session.post(NIST_XCOM_URL, data=payload, timeout=timeout)
    else:
        with _create_nist_retry_session() as temp_session:
            response = temp_session.post(NIST_XCOM_URL, data=payload, timeout=timeout)
    response.raise_for_status()

    return parse_nist_xcom_html(html_text=response.text)


def fetch_and_save_nist_xcom_reference(
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    timeout: float = 30.0,
) -> Path:
    """
    Скачивает таблицы NIST XCOM v3.1 на стандартной сетке для всех 92 элементов (Z=1..92)
    и сохраняет эталонный архив `nist_xcom_v3_1_reference.npz`.
    """
    if timeout <= 0.0:
        raise ValueError(f"Таймаут запроса должен быть строго положительным, получено: {timeout}.")

    archive_dict: Dict[str, NDArray[Float] | NDArray[np.str_]] = {}

    with _create_nist_retry_session() as session:
        for atomic_z in sorted(element_symbol.keys()):
            symbol = element_symbol[atomic_z]
            energy_arr, processes_map, edge_energy_arr, edge_label_arr = fetch_nist_xcom_element(
                atomic_z=atomic_z,
                session=session,
                timeout=timeout,
            )
            archive_dict[f"{symbol}_Energy"] = energy_arr
            archive_dict[f"{symbol}_Edges_energy"] = edge_energy_arr
            archive_dict[f"{symbol}_Edges_label"] = edge_label_arr
            for proc_name, proc_vals in processes_map.items():
                archive_dict[f"{symbol}_{proc_name}"] = proc_vals

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(reference_path, **archive_dict)
    return reference_path


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа командной строки для перегенерации `tables/NIST XCOM Elements MAC.h5`."""
    parser = argparse.ArgumentParser(
        description="Генерация HDF5-таблицы NIST XCOM Elements MAC.h5 с парами узлов краев поглощения."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Путь к выходному HDF5-файлу (по умолчанию tables/NIST XCOM Elements MAC.h5).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Путь к эталонному офлайн-архиву NIST XCOM v3.1 (.npz).",
    )
    parser.add_argument(
        "--fetch-nist",
        action="store_true",
        help="Скачать таблицы для Z=1..92 напрямую с официального сервера physics.nist.gov и обновить эталон .npz.",
    )
    args = parser.parse_args(argv)
    if args.fetch_nist:
        updated_ref = fetch_and_save_nist_xcom_reference(reference_path=args.reference)
        print(f"Успешно обновлен эталонный архив NIST XCOM: {updated_ref}")

    generated_path = generate_nist_xcom_elements_hdf5(
        output_path=args.output,
        reference_path=args.reference,
    )
    print(f"Успешно сгенерирован файл: {generated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
