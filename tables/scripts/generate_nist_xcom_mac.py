"""
Воспроизводимый офлайн-скрипт генерации HDF5-базы массовых коэффициентов ослабления
и атомной релаксации `tables/NIST XCOM Elements MAC.h5` по гибридной схеме:
- сетка энергий, парциальные сечения, пороги подоболочек и скачки фотопоглощения — из NIST XCOM v3.1;
- выходы флуоресценции, постоянные распада вакансий и спектры радиационных/Оже-переходов — из ENDF/B-VIII.0 EADL (EPICS2017).
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import hepunits as units
import numpy as np
import requests
from numpy.typing import NDArray
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.materials.atomic_properties import atomic_number, element_symbol
from core.materials.attenuation_database import (
    AttenuationDataBase,
    EADL_DESIGNATOR_TO_LABEL,
    EADL_LABEL_TO_DESIGNATOR,
    ElementRelaxationData,
)
from core.other.typing_definitions import Float, Index

SCHEMA_VERSION: str = "2.1"
DATA_SOURCE: str = (
    "NIST XCOM Photon Cross Sections Database (NBSIR 87-3597, v3.1) + "
    "ENDF/B-VIII.0 Atomic Relaxation Sublibrary (EADL / EPICS2017)"
)
NIST_XCOM_URL: str = "https://physics.nist.gov/cgi-bin/Xcom/xcom3_1"
EADL_EPICS2017_URL: str = "https://gitlab.cern.ch/mbandier/epics2017/-/raw/master/EADL/EADL2017.ALL.txt"

SCRIPT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH: Path = SCRIPT_DIR / "nist_xcom_v3_1_reference.npz"
DEFAULT_EADL_REFERENCE_PATH: Path = SCRIPT_DIR / "eadl_epics2017_reference.npz"
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
_ENDL_FLOAT_PATTERN: re.Pattern[str] = re.compile(r"^([+-]?\d*\.?\d+)([+-]\d+)$")


def _parse_endl_float(token: str) -> float:
    """Преобразует числовую строку в формате ENDL (например, '1.2345-6') в стандартный float."""
    cleaned = token.strip().replace("D", "E").replace("d", "e")
    match = _ENDL_FLOAT_PATTERN.match(cleaned)
    if match is not None:
        return float(f"{match.group(1)}E{match.group(2)}")
    return float(cleaned)


def parse_eadl_epics2017_text(eadl_text: str) -> Dict[str, NDArray[Any]]:
    """
    Извлекает из текста файла библиотеки EADL (EPICS2017 / ENDF/B-VIII.0, формат ENDL)
    параметры электронных подоболочек и переходов релаксации для Z = 1..100:
    - I=912: число электронов на подоболочке;
    - I=913: энергия связи подоболочки (МэВ);
    - I=921: радиационная ширина уровня Gamma_rad (МэВ);
    - I=922: безызлучательная ширина уровня Gamma_nonrad (МэВ);
    - I=931: радиационные переходы (дизайгнатор вторичной вакансии, вероятность, энергия в МэВ);
    - I=932: безызлучательные Оже-переходы (первичная и вторичная вакансии, вероятность, энергия в МэВ).
    """
    sections: List[List[str]] = []
    current_lines: List[str] = []
    for raw_line in eadl_text.splitlines():
        if len(raw_line) >= 72 and raw_line[:71].strip() == "" and raw_line[71:72] == "1":
            if current_lines:
                sections.append(current_lines)
                current_lines = []
        elif raw_line.strip():
            current_lines.append(raw_line)

    if not sections:
        raise ValueError("В тексте EADL не найдено ни одной валидной секции данных.")

    electrons_by_z: Dict[int, Dict[int, float]] = {}
    binding_by_z: Dict[int, Dict[int, float]] = {}
    rad_width_by_z: Dict[int, Dict[int, float]] = {}
    nonrad_width_by_z: Dict[int, Dict[int, float]] = {}
    rad_trans_by_z: Dict[int, Dict[int, List[Tuple[int, float, float]]]] = {}
    auger_trans_by_z: Dict[int, Dict[int, List[Tuple[int, int, float, float]]]] = {}

    for sec in sections:
        if len(sec) < 2:
            continue
        header1 = sec[0]
        header2 = sec[1]
        za_val = int(header1[0:6].strip())
        atomic_z = za_val // 1000
        i_code = int(header2[2:5].strip())
        data_lines = sec[2:]

        if i_code == 912:
            el_map = electrons_by_z.setdefault(atomic_z, {})
            for line in data_lines:
                tokens = line.split()
                if len(tokens) >= 2:
                    desig = int(round(_parse_endl_float(tokens[0])))
                    el_map[desig] = _parse_endl_float(tokens[1])
        elif i_code == 913:
            bind_map = binding_by_z.setdefault(atomic_z, {})
            for line in data_lines:
                tokens = line.split()
                if len(tokens) >= 2:
                    desig = int(round(_parse_endl_float(tokens[0])))
                    bind_map[desig] = _parse_endl_float(tokens[1])
        elif i_code == 921:
            rw_map = rad_width_by_z.setdefault(atomic_z, {})
            for line in data_lines:
                tokens = line.split()
                if len(tokens) >= 2:
                    desig = int(round(_parse_endl_float(tokens[0])))
                    rw_map[desig] = _parse_endl_float(tokens[1])
        elif i_code == 922:
            nrw_map = nonrad_width_by_z.setdefault(atomic_z, {})
            for line in data_lines:
                tokens = line.split()
                if len(tokens) >= 2:
                    desig = int(round(_parse_endl_float(tokens[0])))
                    nrw_map[desig] = _parse_endl_float(tokens[1])
        elif i_code == 931:
            h2_tokens = header2.split()
            shell_desig = int(round(_parse_endl_float(h2_tokens[3])))
            rad_list = rad_trans_by_z.setdefault(atomic_z, {}).setdefault(shell_desig, [])
            for line in data_lines:
                tokens = line.split()
                if len(tokens) >= 3:
                    sec_desig = int(round(_parse_endl_float(tokens[0])))
                    prob = _parse_endl_float(tokens[1])
                    trans_energy = _parse_endl_float(tokens[2])
                    rad_list.append((sec_desig, prob, trans_energy))
        elif i_code == 932:
            h2_tokens = header2.split()
            shell_desig = int(round(_parse_endl_float(h2_tokens[3])))
            auger_list = auger_trans_by_z.setdefault(atomic_z, {}).setdefault(shell_desig, [])
            for line in data_lines:
                tokens = line.split()
                if len(tokens) >= 4:
                    vac1_desig = int(round(_parse_endl_float(tokens[0])))
                    vac2_desig = int(round(_parse_endl_float(tokens[1])))
                    prob = _parse_endl_float(tokens[2])
                    auger_energy = _parse_endl_float(tokens[3])
                    auger_list.append((vac1_desig, vac2_desig, prob, auger_energy))

    if not binding_by_z:
        raise ValueError("В тексте EADL отсутствуют таблицы энергий связи подоболочек (I=913).")

    archive_dict: Dict[str, NDArray[Any]] = {}
    for atomic_z in sorted(binding_by_z.keys()):
        bind_map = binding_by_z[atomic_z]
        el_map = electrons_by_z.get(atomic_z, {})
        rw_map = rad_width_by_z.get(atomic_z, {})
        nrw_map = nonrad_width_by_z.get(atomic_z, {})
        rad_map = rad_trans_by_z.get(atomic_z, {})
        auger_map = auger_trans_by_z.get(atomic_z, {})

        designators = sorted(bind_map.keys())
        for desig in designators:
            if desig not in EADL_DESIGNATOR_TO_LABEL:
                raise ValueError(f"Неизвестный дизайгнатор подоболочки EADL {desig} для Z={atomic_z}.")

        num_shells = len(designators)
        desig_arr = np.asarray(designators, dtype=np.int32)
        el_arr = np.asarray([el_map.get(desig, 0.0) for desig in designators], dtype=Float)
        bind_arr = np.asarray([bind_map[desig] for desig in designators], dtype=Float)
        rw_arr = np.asarray([rw_map.get(desig, 0.0) for desig in designators], dtype=Float)
        nrw_arr = np.asarray([nrw_map.get(desig, 0.0) for desig in designators], dtype=Float)

        rad_offsets = np.zeros(num_shells + 1, dtype=Index)
        rad_vac_list: List[int] = []
        rad_prob_list: List[float] = []
        rad_energy_list: List[float] = []

        auger_offsets = np.zeros(num_shells + 1, dtype=Index)
        auger_vac1_list: List[int] = []
        auger_vac2_list: List[int] = []
        auger_prob_list: List[float] = []
        auger_energy_list: List[float] = []

        for idx, desig in enumerate(designators):
            r_items = rad_map.get(desig, [])
            for sec_d, r_prob, r_e in r_items:
                rad_vac_list.append(sec_d)
                rad_prob_list.append(r_prob)
                rad_energy_list.append(r_e)
            rad_offsets[idx + 1] = Index(len(rad_vac_list))

            a_items = auger_map.get(desig, [])
            for v1_d, v2_d, a_prob, a_e in a_items:
                auger_vac1_list.append(v1_d)
                auger_vac2_list.append(v2_d)
                auger_prob_list.append(a_prob)
                auger_energy_list.append(a_e)
            auger_offsets[idx + 1] = Index(len(auger_vac1_list))

        prefix = f"Z{atomic_z}"
        archive_dict[f"{prefix}_subshell_designator"] = desig_arr
        archive_dict[f"{prefix}_electron_number"] = el_arr
        archive_dict[f"{prefix}_binding_energy"] = bind_arr
        archive_dict[f"{prefix}_radiative_width"] = rw_arr
        archive_dict[f"{prefix}_nonradiative_width"] = nrw_arr
        archive_dict[f"{prefix}_rad_offsets"] = rad_offsets
        archive_dict[f"{prefix}_rad_vacancy_designator"] = np.asarray(rad_vac_list, dtype=np.int32)
        archive_dict[f"{prefix}_rad_probability"] = np.asarray(rad_prob_list, dtype=Float)
        archive_dict[f"{prefix}_rad_energy"] = np.asarray(rad_energy_list, dtype=Float)
        archive_dict[f"{prefix}_auger_offsets"] = auger_offsets
        archive_dict[f"{prefix}_auger_vacancy_primary"] = np.asarray(auger_vac1_list, dtype=np.int32)
        archive_dict[f"{prefix}_auger_vacancy_secondary"] = np.asarray(auger_vac2_list, dtype=np.int32)
        archive_dict[f"{prefix}_auger_probability"] = np.asarray(auger_prob_list, dtype=Float)
        archive_dict[f"{prefix}_auger_energy"] = np.asarray(auger_energy_list, dtype=Float)

    return archive_dict


def _build_element_relaxation_data(
    atomic_z: int,
    symbol: str,
    energy_arr: NDArray[Float],
    pe_arr: NDArray[Float],
    edge_energy_arr: NDArray[Float],
    edge_label_arr: NDArray[np.str_],
    eadl_archive: Any,
) -> ElementRelaxationData:
    """
    Формирует самосогласованную структуру ElementRelaxationData для элемента `symbol` (Z=`atomic_z`):
    - пороги подоболочек и скачки поглощения извлекаются напрямую из краёв NIST XCOM;
    - выходы флуоресценции, постоянные распада и спектры переходов извлекаются из эталона EADL
      с приведением к единицам `hepunits` и ограничением энергии переходов E_line <= E_bind(Z, s).
    """
    prefix = f"Z{atomic_z}"
    desig_key = f"{prefix}_subshell_designator"
    if desig_key not in eadl_archive:
        raise ValueError(f"В эталонном архиве EADL отсутствуют данные для элемента '{symbol}' (Z={atomic_z}).")

    energy_scale = float(units.MeV)
    hbar_planck = float(units.hbar_Planck)

    eadl_designators = np.asarray(eadl_archive[desig_key], dtype=np.int32)
    eadl_rad_width = np.asarray(eadl_archive[f"{prefix}_radiative_width"], dtype=Float) * energy_scale
    eadl_nonrad_width = np.asarray(eadl_archive[f"{prefix}_nonradiative_width"], dtype=Float) * energy_scale
    eadl_rad_offsets = np.asarray(eadl_archive[f"{prefix}_rad_offsets"], dtype=Index)
    eadl_rad_vac = np.asarray(eadl_archive[f"{prefix}_rad_vacancy_designator"], dtype=np.int32)
    eadl_rad_prob = np.asarray(eadl_archive[f"{prefix}_rad_probability"], dtype=Float)
    eadl_rad_energy = np.asarray(eadl_archive[f"{prefix}_rad_energy"], dtype=Float) * energy_scale
    eadl_auger_offsets = np.asarray(eadl_archive[f"{prefix}_auger_offsets"], dtype=Index)
    eadl_auger_vac1 = np.asarray(eadl_archive[f"{prefix}_auger_vacancy_primary"], dtype=np.int32)
    eadl_auger_vac2 = np.asarray(eadl_archive[f"{prefix}_auger_vacancy_secondary"], dtype=np.int32)
    eadl_auger_prob = np.asarray(eadl_archive[f"{prefix}_auger_probability"], dtype=Float)
    eadl_auger_energy = np.asarray(eadl_archive[f"{prefix}_auger_energy"], dtype=Float) * energy_scale

    desig_to_eadl_idx: Dict[int, int] = {
        int(d_val): idx for idx, d_val in enumerate(eadl_designators)
    }

    num_shells = int(edge_energy_arr.size)
    subshell_designator = np.zeros(num_shells, dtype=np.int32)
    jump_ratio = np.zeros(num_shells, dtype=Float)
    jump_cumulative_weight = np.ones(num_shells + 1, dtype=Float)
    fluorescence_yield = np.zeros(num_shells, dtype=Float)
    decay_constant = np.zeros(num_shells, dtype=Float)

    rad_offsets = np.zeros(num_shells + 1, dtype=Index)
    rad_vac_list: List[int] = []
    rad_energy_list: List[float] = []
    rad_prob_list: List[float] = []

    auger_offsets = np.zeros(num_shells + 1, dtype=Index)
    auger_vac1_list: List[int] = []
    auger_vac2_list: List[int] = []
    auger_energy_list: List[float] = []
    auger_prob_list: List[float] = []

    for shell_idx in range(num_shells):
        lbl_str = str(edge_label_arr[shell_idx])
        if lbl_str not in EADL_LABEL_TO_DESIGNATOR:
            raise ValueError(
                f"Неизвестная метка края поглощения '{lbl_str}' у элемента '{symbol}' (Z={atomic_z})."
            )
        desig_val = EADL_LABEL_TO_DESIGNATOR[lbl_str]
        if desig_val not in desig_to_eadl_idx:
            raise ValueError(
                f"Подоболочка '{lbl_str}' (дизайгнатор={desig_val}) элемента '{symbol}' (Z={atomic_z}) "
                f"отсутствует в таблице EADL."
            )

        subshell_designator[shell_idx] = np.int32(desig_val)
        bind_e = float(edge_energy_arr[shell_idx])

        left_pos = int(np.searchsorted(energy_arr, bind_e, side="left"))
        right_pos = int(np.searchsorted(energy_arr, bind_e, side="right"))
        jump_val = float(pe_arr[right_pos - 1]) / float(pe_arr[left_pos])
        jump_ratio[shell_idx] = Float(jump_val)
        jump_cumulative_weight[shell_idx + 1] = Float(
            float(jump_cumulative_weight[shell_idx]) * jump_val
        )

        eadl_idx = desig_to_eadl_idx[desig_val]
        total_width = float(eadl_rad_width[eadl_idx]) + float(eadl_nonrad_width[eadl_idx])
        if total_width <= 0.0:
            raise ValueError(
                f"Неположительная полная ширина уровня Gamma_tot={total_width} для оболочки '{lbl_str}' "
                f"элемента '{symbol}' (Z={atomic_z})."
            )
        decay_constant[shell_idx] = Float(total_width / hbar_planck)

        er_start = int(eadl_rad_offsets[eadl_idx])
        er_end = int(eadl_rad_offsets[eadl_idx + 1])
        r_vacs = eadl_rad_vac[er_start:er_end]
        r_probs = eadl_rad_prob[er_start:er_end]
        r_energies = np.minimum(eadl_rad_energy[er_start:er_end], bind_e)
        r_valid = (r_probs > 0.0) & (r_energies > 0.0)
        r_vacs = r_vacs[r_valid]
        r_probs = r_probs[r_valid]
        r_energies = r_energies[r_valid]

        ea_start = int(eadl_auger_offsets[eadl_idx])
        ea_end = int(eadl_auger_offsets[eadl_idx + 1])
        a_vac1 = eadl_auger_vac1[ea_start:ea_end]
        a_vac2 = eadl_auger_vac2[ea_start:ea_end]
        a_probs = eadl_auger_prob[ea_start:ea_end]
        a_energies = np.minimum(eadl_auger_energy[ea_start:ea_end], bind_e)
        a_valid = (a_probs > 0.0) & (a_energies > 0.0)
        a_vac1 = a_vac1[a_valid]
        a_vac2 = a_vac2[a_valid]
        a_probs = a_probs[a_valid]
        a_energies = a_energies[a_valid]

        sum_rad = float(np.sum(r_probs))
        sum_auger = float(np.sum(a_probs))
        total_prob = sum_rad + sum_auger
        if total_prob <= 0.0:
            raise ValueError(
                f"Нулевая суммарная вероятность релаксации для оболочки '{lbl_str}' "
                f"элемента '{symbol}' (Z={atomic_z})."
            )

        omega = sum_rad / total_prob
        fluorescence_yield[shell_idx] = Float(min(max(omega, 0.0), 1.0))

        if sum_rad > 0.0:
            norm_r_probs = r_probs / sum_rad
            for vac_d, e_val, p_val in zip(r_vacs, r_energies, norm_r_probs):
                rad_vac_list.append(int(vac_d))
                rad_energy_list.append(float(e_val))
                rad_prob_list.append(float(p_val))
        rad_offsets[shell_idx + 1] = Index(len(rad_vac_list))

        if sum_auger > 0.0:
            norm_a_probs = a_probs / sum_auger
            for v1_d, v2_d, e_val, p_val in zip(a_vac1, a_vac2, a_energies, norm_a_probs):
                auger_vac1_list.append(int(v1_d))
                auger_vac2_list.append(int(v2_d))
                auger_energy_list.append(float(e_val))
                auger_prob_list.append(float(p_val))
        auger_offsets[shell_idx + 1] = Index(len(auger_vac1_list))

    return ElementRelaxationData(
        subshell_label=np.asarray(edge_label_arr, dtype=np.str_),
        subshell_designator=subshell_designator,
        binding_energy=np.asarray(edge_energy_arr, dtype=Float),
        jump_ratio=jump_ratio,
        jump_cumulative_weight=jump_cumulative_weight,
        fluorescence_yield=fluorescence_yield,
        decay_constant=decay_constant,
        rad_offsets=rad_offsets,
        rad_vacancy_designator=np.asarray(rad_vac_list, dtype=np.int32),
        rad_energy=np.asarray(rad_energy_list, dtype=Float),
        rad_probability=np.asarray(rad_prob_list, dtype=Float),
        auger_offsets=auger_offsets,
        auger_vacancy_primary=np.asarray(auger_vac1_list, dtype=np.int32),
        auger_vacancy_secondary=np.asarray(auger_vac2_list, dtype=np.int32),
        auger_energy=np.asarray(auger_energy_list, dtype=Float),
        auger_probability=np.asarray(auger_prob_list, dtype=Float),
    )


def generate_nist_xcom_elements_hdf5(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    eadl_reference_path: Path = DEFAULT_EADL_REFERENCE_PATH,
) -> Path:
    """
    Формирует файл `NIST XCOM Elements MAC.h5` по эталонным архивам NIST XCOM v3.1 и EADL (EPICS2017).
    Все величины записываются в канонических единицах `hepunits` (`MeV` для энергии, `cm2 / g` для MAC,
    `1 / ns` для постоянных распада вакансий).
    """
    if not reference_path.is_file():
        raise FileNotFoundError(f"Эталонный архив NIST XCOM не найден: '{reference_path}'.")
    if not eadl_reference_path.is_file():
        raise FileNotFoundError(f"Эталонный архив EADL не найден: '{eadl_reference_path}'.")

    energy_scale = float(units.MeV)
    mac_scale = float(units.cm2 / units.g)
    utf8_string_dtype = h5py.string_dtype(encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        np.load(reference_path, allow_pickle=False) as reference_archive,
        np.load(eadl_reference_path, allow_pickle=False) as eadl_archive,
    ):
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

                relaxation_data = _build_element_relaxation_data(
                    atomic_z=atomic_z,
                    symbol=symbol,
                    energy_arr=energy_arr,
                    pe_arr=processes_map["Photoelectric absorption"],
                    edge_energy_arr=edge_energy_arr,
                    edge_label_arr=edge_label_arr,
                    eadl_archive=eadl_archive,
                )

                rel_group = element_group.create_group("Relaxation")
                rel_group.create_dataset(
                    "subshell_label",
                    data=np.asarray([str(lbl) for lbl in relaxation_data.subshell_label], dtype=object),
                    dtype=utf8_string_dtype,
                )
                rel_group.create_dataset(
                    "subshell_designator",
                    data=relaxation_data.subshell_designator,
                    dtype=np.int32,
                )
                rel_group.create_dataset(
                    "binding_energy",
                    data=relaxation_data.binding_energy,
                    dtype=Float,
                )
                rel_group.create_dataset(
                    "jump_ratio",
                    data=relaxation_data.jump_ratio,
                    dtype=Float,
                )
                rel_group.create_dataset(
                    "jump_cumulative_weight",
                    data=relaxation_data.jump_cumulative_weight,
                    dtype=Float,
                )
                rel_group.create_dataset(
                    "fluorescence_yield",
                    data=relaxation_data.fluorescence_yield,
                    dtype=Float,
                )
                rel_group.create_dataset(
                    "decay_constant",
                    data=relaxation_data.decay_constant,
                    dtype=Float,
                )
                rel_group.create_dataset(
                    "rad_offsets",
                    data=relaxation_data.rad_offsets,
                    dtype=Index,
                )
                rel_group.create_dataset(
                    "rad_vacancy_designator",
                    data=relaxation_data.rad_vacancy_designator,
                    dtype=np.int32,
                )
                rel_group.create_dataset(
                    "rad_energy",
                    data=relaxation_data.rad_energy,
                    dtype=Float,
                )
                rel_group.create_dataset(
                    "rad_probability",
                    data=relaxation_data.rad_probability,
                    dtype=Float,
                )
                rel_group.create_dataset(
                    "auger_offsets",
                    data=relaxation_data.auger_offsets,
                    dtype=Index,
                )
                rel_group.create_dataset(
                    "auger_vacancy_primary",
                    data=relaxation_data.auger_vacancy_primary,
                    dtype=np.int32,
                )
                rel_group.create_dataset(
                    "auger_vacancy_secondary",
                    data=relaxation_data.auger_vacancy_secondary,
                    dtype=np.int32,
                )
                rel_group.create_dataset(
                    "auger_energy",
                    data=relaxation_data.auger_energy,
                    dtype=Float,
                )
                rel_group.create_dataset(
                    "auger_probability",
                    data=relaxation_data.auger_probability,
                    dtype=Float,
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
        allowed_methods=("GET", "POST"),
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


def fetch_and_save_eadl_reference(
    eadl_reference_path: Path = DEFAULT_EADL_REFERENCE_PATH,
    timeout: float = 60.0,
) -> Path:
    """
    Скачивает оцененную библиотеку атомной релаксации EADL (EPICS2017 / ENDF/B-VIII.0, Z=1..100)
    и сохраняет сжатый эталонный архив `eadl_epics2017_reference.npz`.
    """
    if timeout <= 0.0:
        raise ValueError(f"Таймаут запроса должен быть строго положительным, получено: {timeout}.")

    with _create_nist_retry_session() as session:
        response = session.get(EADL_EPICS2017_URL, timeout=timeout)
        response.raise_for_status()
        eadl_text = response.text

    archive_dict = parse_eadl_epics2017_text(eadl_text)
    eadl_reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(eadl_reference_path, **archive_dict)
    return eadl_reference_path


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа командной строки для перегенерации `tables/NIST XCOM Elements MAC.h5`."""
    parser = argparse.ArgumentParser(
        description=(
            "Генерация HDF5-таблицы NIST XCOM Elements MAC.h5 с парами узлов краев поглощения "
            "и параметрами атомной релаксации ENDF/B-VIII.0 EADL."
        )
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
        "--eadl-reference",
        type=Path,
        default=DEFAULT_EADL_REFERENCE_PATH,
        help="Путь к эталонному офлайн-архиву атомной релаксации EADL EPICS2017 (.npz).",
    )
    parser.add_argument(
        "--fetch-nist",
        action="store_true",
        help="Скачать таблицы для Z=1..92 напрямую с официального сервера physics.nist.gov и обновить эталон .npz.",
    )
    parser.add_argument(
        "--fetch-eadl",
        action="store_true",
        help="Скачать библиотеку атомной релаксации EADL EPICS2017 (Z=1..100) и обновить эталон .npz.",
    )
    args = parser.parse_args(argv)
    if args.fetch_nist:
        updated_ref = fetch_and_save_nist_xcom_reference(reference_path=args.reference)
        print(f"Успешно обновлен эталонный архив NIST XCOM: {updated_ref}")

    if args.fetch_eadl:
        updated_eadl = fetch_and_save_eadl_reference(eadl_reference_path=args.eadl_reference)
        print(f"Успешно обновлен эталонный архив EADL EPICS2017: {updated_eadl}")

    generated_path = generate_nist_xcom_elements_hdf5(
        output_path=args.output,
        reference_path=args.reference,
        eadl_reference_path=args.eadl_reference,
    )
    print(f"Успешно сгенерирован файл: {generated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
