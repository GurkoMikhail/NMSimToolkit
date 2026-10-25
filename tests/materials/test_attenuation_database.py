"""Тесты валидации таблиц NIST XCOM, смешивания материалов и интерполяции с краями поглощения."""

from pathlib import Path
import tempfile

import h5py
import hepunits as units
import numpy as np
import pytest

from core.materials.attenuation_database import AttenuationDataBase
from core.materials.attenuation_functions import AttenuationFunction
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Float, Index
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_kernels import (
    _compute_log_energy_and_bin,
    _get_macroscopic_cross_sections,
)
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PhotoelectricEffect,
)
from settings.database_setting import attenuation_database, material_database
from tables.scripts.generate_nist_xcom_mac import (
    fetch_and_save_nist_xcom_reference,
    fetch_nist_xcom_element,
    generate_nist_xcom_elements_hdf5,
    parse_nist_xcom_html,
)


def _write_minimal_element_group(
    h5_file: h5py.File,
    symbol: str,
    energies: np.ndarray,
    pe_values: np.ndarray,
    edge_energies: np.ndarray,
    edge_labels: list[str],
) -> None:
    """Вспомогательная запись одного элемента в тестовый HDF5-файл."""
    grp = h5_file.create_group(symbol)
    grp.create_dataset("Energy", data=energies.astype(np.float64))
    grp.create_dataset("Photoelectric absorption", data=pe_values.astype(np.float64))
    ones = np.ones_like(energies, dtype=np.float64)
    grp.create_dataset("Coherent scattering", data=ones)
    grp.create_dataset("Incoherent scattering", data=ones)
    edges_grp = grp.create_group("Edges")
    edges_grp.create_dataset("energy", data=edge_energies.astype(np.float64))
    str_dt = h5py.string_dtype(encoding="utf-8")
    edges_grp.create_dataset(
        "label",
        shape=(len(edge_labels),),
        dtype=str_dt,
        data=np.asarray(edge_labels, dtype=object),
    )


def test_element_schema_validation_on_production_table() -> None:
    """Проверка схемы эталонной таблицы NIST XCOM Elements MAC.h5 для всех 92 элементов."""
    el_mac = attenuation_database.elements_MAC
    el_edges = attenuation_database.elements_edges
    assert len(el_mac) == 92
    assert len(el_edges) == 92
    assert len(el_mac["H"]["Energy"]) == 46

    e_min, e_max = attenuation_database.energy_range
    assert np.isclose(e_min, 0.001 * units.MeV, rtol=1e-12)
    assert np.isclose(e_max, 20.0 * units.MeV, rtol=1e-12)

    for symbol, table in el_mac.items():
        energy = table["Energy"]
        assert 46 <= len(energy) <= 76, f"Неожиданное число узлов для {symbol}: {len(energy)}"
        diffs = np.diff(energy)
        assert np.all(diffs >= 0.0)

        unique_vals, counts = np.unique(energy, return_counts=True)
        dup_energies = unique_vals[counts >= 2]
        edge_energy_arr, edge_label_arr = el_edges[symbol]
        np.testing.assert_allclose(dup_energies, edge_energy_arr, rtol=0.0, atol=1e-12)
        assert len(edge_label_arr) == len(dup_energies)

        # На каждом крае поглощения все узлы выше края строго превышают левый предел ниже края
        pe = table["Coefficient"]["PhotoelectricEffect"]
        for edge_val in dup_energies:
            left_pos = int(np.searchsorted(energy, edge_val, side="left"))
            right_pos = int(np.searchsorted(energy, edge_val, side="right"))
            assert np.all(pe[left_pos + 1 : right_pos] > pe[left_pos]), (
                f"Нет скачка вверх на крае элемента {symbol} при E={edge_val}"
            )


def test_corrupted_tables_raise_value_error() -> None:
    """Проверка Fail-Fast выброса ValueError при загрузке повреждённых таблиц."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Немонотонная сетка энергий
        bad_monotonic = tmp_path / "bad_monotonic.h5"
        with h5py.File(bad_monotonic, "w") as h5f:
            _write_minimal_element_group(
                h5f,
                "H",
                energies=np.array([0.001, 0.005, 0.003, 20.0]),
                pe_values=np.array([10.0, 2.0, 3.0, 0.001]),
                edge_energies=np.array([]),
                edge_labels=[],
            )
        with pytest.raises(ValueError, match="не является монотонно неубывающей"):
            AttenuationDataBase(base_name=str(bad_monotonic))

        # 2. Отсутствие подгруппы Edges
        bad_no_edges = tmp_path / "bad_no_edges.h5"
        with h5py.File(bad_no_edges, "w") as h5f:
            _write_minimal_element_group(
                h5f,
                "H",
                energies=np.array([0.001, 0.01, 20.0]),
                pe_values=np.array([10.0, 1.0, 0.001]),
                edge_energies=np.array([]),
                edge_labels=[],
            )
            del h5f["H"]["Edges"]
        with pytest.raises(ValueError, match="отсутствует обязательная подгруппа 'Edges'"):
            AttenuationDataBase(base_name=str(bad_no_edges))

        # 3. Несоответствие дублей в Energy и массива Edges/energy
        bad_edge_mismatch = tmp_path / "bad_edge_mismatch.h5"
        with h5py.File(bad_edge_mismatch, "w") as h5f:
            _write_minimal_element_group(
                h5f,
                "Pb",
                energies=np.array([0.001, 0.088, 0.088, 20.0]),
                pe_values=np.array([1000.0, 1.91, 7.72, 0.01]),
                edge_energies=np.array([0.085]),
                edge_labels=["K"],
            )
        with pytest.raises(ValueError, match="не совпадают с 'Edges/energy'"):
            AttenuationDataBase(base_name=str(bad_edge_mismatch))

        # 4. Инвертированный или нулевой скачок фотоэффекта на паре края
        bad_jump = tmp_path / "bad_jump.h5"
        with h5py.File(bad_jump, "w") as h5f:
            _write_minimal_element_group(
                h5f,
                "Pb",
                energies=np.array([0.001, 0.088, 0.088, 20.0]),
                pe_values=np.array([1000.0, 7.72, 1.91, 0.01]),
                edge_energies=np.array([0.088]),
                edge_labels=["K"],
            )
        with pytest.raises(ValueError, match="выше края"):
            AttenuationDataBase(base_name=str(bad_jump))

        # 5. Поврежденный правый предел на тройном узле края (pe[left_pos+1] > pe[left_pos], но pe[right_pos-1] <= pe[left_pos])
        bad_triple_edge = tmp_path / "bad_triple_edge.h5"
        with h5py.File(bad_triple_edge, "w") as h5f:
            _write_minimal_element_group(
                h5f,
                "Mo",
                energies=np.array([0.001, 0.02, 0.02, 0.02, 20.0]),
                pe_values=np.array([1000.0, 12.0, 78.46, 12.0, 0.01]),
                edge_energies=np.array([0.02]),
                edge_labels=["K"],
            )
        with pytest.raises(ValueError, match="выше края"):
            AttenuationDataBase(base_name=str(bad_triple_edge))


def test_offline_generator_reproducibility() -> None:
    """Проверка детерминированной генерации HDF5-таблицы оффлайн-скриптом."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "generated_mac.h5"
        generate_nist_xcom_elements_hdf5(output_path=out_file)
        db = AttenuationDataBase(base_name=str(out_file))
        assert len(db.elements_MAC) == 92
        np.testing.assert_array_equal(
            db.elements_MAC["Pb"]["Energy"],
            attenuation_database.elements_MAC["Pb"]["Energy"],
        )


def test_compound_material_grid_deduplication_and_edge_preservation() -> None:
    """
    Проверка смешивания составных материалов (Sodium Iodide и Tissue, Soft (ICRU-44)):
    регулярные узлы дедуплицируются, а парные узлы краёв поглощения сохраняются как пары.
    """
    nai: Material = material_database["Sodium Iodide"]
    nai_table = attenuation_database[nai]
    nai_energy = nai_table["Energy"]

    # Сетка NaI должна оставаться компактной (58 узлов) и монотонно неубывающей
    assert len(nai_energy) == 58
    assert np.all(np.diff(nai_energy) >= 0.0)

    unique_vals, counts = np.unique(nai_energy, return_counts=True)
    # Ни один узел не может иметь кратность > 2 (даже при совпадении K-края Na и M1-края I на 1.072 кэВ)
    assert np.all(counts <= 2)

    dup_energies = unique_vals[counts == 2]
    na_edges = attenuation_database.elements_edges["Na"][0]
    i_edges = attenuation_database.elements_edges["I"][0]
    expected_union_edges = np.union1d(na_edges, i_edges)
    np.testing.assert_allclose(dup_energies, expected_union_edges, rtol=0.0, atol=1e-12)

    # Для мягкой ткани регулярные узлы также дедуплицированы (57 узлов), а края сохранены как пары
    soft_tissue: Material = material_database["Tissue, Soft (ICRU-44)"]
    st_table = attenuation_database[soft_tissue]
    st_energy = st_table["Energy"]
    assert len(st_energy) == 57
    _, st_counts = np.unique(st_energy, return_counts=True)
    assert np.all(st_counts <= 2)


def test_k_and_l_edge_jump_ratios_match_nist_xcom_reference() -> None:
    """
    Проверка точности скачка фотоэлектрического сечения на K- и L-краях поглощения
    свинца (Pb), йода (I) и вольфрама (W) относительно эталона NIST XCOM (допуск < 0.1%).
    """
    mac_unit = units.cm2 / units.g

    # Эталонные значения фотоэффекта NIST XCOM v3.1 (в см^2/г) непосредственно ниже и выше краев:
    # Pb K-edge (88.00 кэВ): 1.547 -> 7.321 см^2/г
    # Pb L1-edge (15.86 кэВ): 131.2 -> 151.7 см^2/г
    # Pb L2-edge (15.20 кэВ): 104.5 -> 145.2 см^2/г
    # Pb L3-edge (13.04 кэВ): 63.10 -> 158.2 см^2/г
    # I K-edge (33.17 кэВ): 5.808 -> 35.08 см^2/г
    # W K-edge (69.53 кэВ): 2.118 -> 10.80 см^2/г
    reference_jumps = [
        ("Pb", 0.08800 * units.MeV, 1.547 * mac_unit, 7.321 * mac_unit),
        ("Pb", 0.01586 * units.MeV, 131.2 * mac_unit, 151.7 * mac_unit),
        ("Pb", 0.01520 * units.MeV, 104.5 * mac_unit, 145.2 * mac_unit),
        ("Pb", 0.01304 * units.MeV, 63.10 * mac_unit, 158.2 * mac_unit),
        ("I", 0.03317 * units.MeV, 5.808 * mac_unit, 35.08 * mac_unit),
        ("W", 0.06953 * units.MeV, 2.118 * mac_unit, 10.80 * mac_unit),
    ]

    mismatches = []
    for symbol, edge_energy, expected_minus, expected_plus in reference_jumps:
        table = attenuation_database.elements_MAC[symbol]
        energy = table["Energy"]
        pe = table["Coefficient"]["PhotoelectricEffect"]

        indices = np.flatnonzero(np.isclose(energy, edge_energy, rtol=1e-6, atol=0.0))
        assert len(indices) == 2, f"Край {edge_energy} для {symbol} должен быть представлен парой узлов"
        idx_minus, idx_plus = int(indices[0]), int(indices[1])
        assert idx_plus == idx_minus + 1

        # Ни один узел вблизи края не сдвинут на -1 эВ (1e-6 МэВ): положительные шаги сетки > 10 эВ
        positive_diffs = np.diff(energy)[np.diff(energy) > 0.0]
        assert np.all(positive_diffs > 10.0 * units.eV), (
            f"Обнаружен искусственно сдвинутый узел вблизи края для {symbol}: min diff = {np.min(positive_diffs)}"
        )

        actual_minus = float(pe[idx_minus])
        actual_plus = float(pe[idx_plus])
        if not np.isclose(actual_minus, expected_minus, rtol=1e-3, atol=0.0) or not np.isclose(
            actual_plus, expected_plus, rtol=1e-3, atol=0.0
        ):
            mismatches.append(
                f"{symbol} {edge_energy}: actual=({actual_minus / mac_unit:.4g}, {actual_plus / mac_unit:.4g}), "
                f"expected=({expected_minus / mac_unit:.4g}, {expected_plus / mac_unit:.4g})"
            )

        expected_ratio = expected_plus / expected_minus
        actual_ratio = actual_plus / actual_minus
        if not np.isclose(actual_ratio, expected_ratio, rtol=1e-3, atol=0.0):
            mismatches.append(f"{symbol} ratio={actual_ratio:.6g} vs {expected_ratio:.6g}")

    assert not mismatches, "; ".join(mismatches)

    # Проверка метода get_element_edges по атомному номеру Z и символу
    for atomic_z, sym in ((53, "I"), (74, "W"), (82, "Pb")):
        edges_by_z, labels_by_z = attenuation_database.get_element_edges(atomic_z)
        edges_by_sym, labels_by_sym = attenuation_database.get_element_edges(sym)
        np.testing.assert_array_equal(edges_by_z, edges_by_sym)
        np.testing.assert_array_equal(labels_by_z, labels_by_sym)
        assert "K" in labels_by_z

    with pytest.raises(ValueError, match="Неизвестный атомный номер"):
        attenuation_database.get_element_edges(999)
    with pytest.raises(ValueError, match="отсутствует в базе"):
        attenuation_database.get_element_edges("Unobtainium")


def test_log_log_interpolation_accuracy_between_grid_nodes() -> None:
    """
    Проверка точности кусочной лог-лог интерполяции между узлами сетки
    на примере Pb, I, W и составного материала NaI (погрешность <= 0.1%).
    """
    pb_mat: Material = material_database["Pb"]
    nai_mat: Material = material_database["Sodium Iodide"]

    pe_process = PhotoelectricEffect(attenuation_database=attenuation_database)
    pb_func: AttenuationFunction = pe_process.attenuation_function

    pb_energy, pb_pe_lac = pb_func[pb_mat]

    # Выбираем гладкий интервал между 150 кэВ и 200 кэВ (выше K-края свинца 88 кэВ)
    idx_left = int(np.flatnonzero(np.isclose(pb_energy, 0.150 * units.MeV, rtol=1e-7, atol=0.0))[0])
    idx_right = idx_left + 1
    e0, e1 = float(pb_energy[idx_left]), float(pb_energy[idx_right])
    assert e1 > e0
    lac0, lac1 = float(pb_pe_lac[idx_left]), float(pb_pe_lac[idx_right])
    query_energy = 0.5 * (e0 + e1)

    # Аналитическое значение лог-лог интерполяции
    log_fraction = (np.log(query_energy) - np.log(e0)) / (np.log(e1) - np.log(e0))
    expected_lac = float(np.exp(np.log(lac0) + log_fraction * (np.log(lac1) - np.log(lac0))))

    interp_scalar_val = float(pb_func(pb_mat, query_energy))
    interp_vec_val = float(pb_func(pb_mat, np.array([query_energy], dtype=Float))[0])
    assert np.isclose(interp_scalar_val, expected_lac, rtol=1e-12, atol=0.0)
    assert np.isclose(interp_vec_val, expected_lac, rtol=1e-12, atol=0.0)

    # Проверка сохранения односторонней лог-лог интерполяции для Na в точке K-края йода (33.17 кэВ) при смешивании NaI
    i_k_edge = 0.03317 * units.MeV
    na_table = attenuation_database.elements_MAC["Na"]
    na_energy = na_table["Energy"]
    na_pe = na_table["Coefficient"]["PhotoelectricEffect"]
    pos_right = int(np.searchsorted(na_energy, i_k_edge, side="right"))
    na_e0, na_e1 = float(na_energy[pos_right - 1]), float(na_energy[pos_right])
    na_pe0, na_pe1 = float(na_pe[pos_right - 1]), float(na_pe[pos_right])
    na_log_fraction = (np.log(i_k_edge) - np.log(na_e0)) / (np.log(na_e1) - np.log(na_e0))
    expected_na_pe = float(np.exp(np.log(na_pe0) + na_log_fraction * (np.log(na_pe1) - np.log(na_pe0))))

    i_table = attenuation_database.elements_MAC["I"]
    i_edge_indices = np.flatnonzero(np.isclose(i_table["Energy"], i_k_edge, rtol=1e-6, atol=0.0))
    i_pe_minus = float(i_table["Coefficient"]["PhotoelectricEffect"][i_edge_indices[0]])
    i_pe_plus = float(i_table["Coefficient"]["PhotoelectricEffect"][i_edge_indices[1]])

    weight_na = float(nai_mat.composition_dict["Na"])
    weight_i = float(nai_mat.composition_dict["I"])
    expected_nai_mac_minus = weight_na * expected_na_pe + weight_i * i_pe_minus
    expected_nai_mac_plus = weight_na * expected_na_pe + weight_i * i_pe_plus

    nai_table = attenuation_database[nai_mat]
    nai_edge_indices = np.flatnonzero(np.isclose(nai_table["Energy"], i_k_edge, rtol=1e-6, atol=0.0))
    assert len(nai_edge_indices) == 2
    actual_nai_mac_minus = float(
        nai_table["Coefficient"]["PhotoelectricEffect"][nai_edge_indices[0]]
    )
    actual_nai_mac_plus = float(
        nai_table["Coefficient"]["PhotoelectricEffect"][nai_edge_indices[1]]
    )

    assert np.isclose(actual_nai_mac_minus, expected_nai_mac_minus, rtol=1e-12, atol=0.0)
    assert np.isclose(actual_nai_mac_plus, expected_nai_mac_plus, rtol=1e-12, atol=0.0)


def test_one_sided_limits_at_absorption_edges_in_kernel_and_function() -> None:
    """
    Проверка односторонних пределов на краях поглощения:
    при E_edge - 1 эВ (1e-6 МэВ) и E_edge - 0.1 эВ (1e-7 МэВ) выбирается левая ветвь (ниже края),
    при E_edge и E_edge + 1 эВ выбирается правая ветвь (выше края), включая элементы Mo и Fr
    с тройным повторением узла энергии на крае поглощения в родной сетке NIST XCOM.
    """
    pb_mat: Material = material_database["Pb"]
    nai_mat: Material = material_database["Sodium Iodide"]
    mo_mat: Material = material_database["Mo"]
    fr_mat: Material = material_database["Fr"]

    compiler = PhysicsCompiler()
    processes = [
        PhotoelectricEffect(attenuation_database=attenuation_database),
        ComptonScattering(attenuation_database=attenuation_database),
        CoherentScattering(attenuation_database=attenuation_database),
    ]
    mat_bank = compiler._build_material_bank([pb_mat, nai_mat, mo_mat, fr_mat], processes)

    pe_func = processes[0].attenuation_function

    delta_energy = 1.0 * units.eV
    fine_delta_energy = 1e-7 * units.MeV
    out_lacs = np.zeros(3, dtype=Float)

    cases = [
        (pb_mat, 0.08800 * units.MeV, 2, 2.0),
        (pb_mat, 0.01304 * units.MeV, 2, 2.0),
        (nai_mat, 0.03317 * units.MeV, 2, 5.0),
        (mo_mat, 0.02000 * units.MeV, 3, 1.2),
        (fr_mat, 0.00300 * units.MeV, 3, 1.2),
    ]
    for target_mat, edge_energy, expected_mult, min_ratio in cases:
        mat_id = Index(target_mat.ID)
        mat_offset = int(mat_bank.mat_pointers[mat_id]["start_idx"])
        mat_len = int(mat_bank.mat_pointers[mat_id]["length"])
        mat_grid = mat_bank.physics_energy_grid[mat_offset : mat_offset + mat_len]
        mat_lacs = mat_bank.physics_lac_table[mat_offset : mat_offset + mat_len]

        edge_indices = np.flatnonzero(np.isclose(mat_grid, edge_energy, rtol=1e-7, atol=0.0))
        assert len(edge_indices) == expected_mult
        lac_minus = float(mat_lacs[int(edge_indices[0]), 0])
        lac_plus = float(mat_lacs[int(edge_indices[-1]), 0])
        assert lac_plus > min_ratio * lac_minus

        for step_below, tol in ((fine_delta_energy, 5e-4), (delta_energy, 3e-3)):
            e_below = Float(edge_energy - step_below)
            log_e_below, log_bin_below = _compute_log_energy_and_bin(e_below, mat_bank)
            _get_macroscopic_cross_sections(e_below, log_e_below, log_bin_below, mat_id, mat_bank, out_lacs)
            assert np.isclose(float(out_lacs[0]), lac_minus, rtol=tol, atol=0.0)
            assert np.isclose(float(pe_func(target_mat, float(e_below))), lac_minus, rtol=tol, atol=0.0)

        for step_above, tol in ((0.0, 1e-12), (fine_delta_energy, 5e-4), (delta_energy, 3e-3)):
            e_above = Float(edge_energy + step_above)
            log_e_above, log_bin_above = _compute_log_energy_and_bin(e_above, mat_bank)
            _get_macroscopic_cross_sections(e_above, log_e_above, log_bin_above, mat_id, mat_bank, out_lacs)
            assert np.isclose(float(out_lacs[0]), lac_plus, rtol=tol, atol=0.0)
            assert np.isclose(float(pe_func(target_mat, float(e_above))), lac_plus, rtol=tol, atol=0.0)


def test_attenuation_function_energy_range_slicing_preserves_edge_pairs_and_bounds() -> None:
    """
    Проверка того, что обрезка по energy_range в AttenuationFunction не разрывает пары узлов
    краев поглощения на границах диапазона, проверяет границы базы данных и корректно работает с MaterialArray.
    """
    pb_mat: Material = material_database["Pb"]
    nai_mat: Material = material_database["Sodium Iodide"]
    pb_edges, pb_labels = attenuation_database.elements_edges["Pb"]
    l3_edge = float(pb_edges[pb_labels == "L3"][0])
    k_edge = float(pb_edges[pb_labels == "K"][0])

    pe_proc = PhotoelectricEffect(attenuation_database=attenuation_database)
    pe_proc.energy_range = np.array([l3_edge, k_edge], dtype=Float)

    sliced_energy, sliced_lac = pe_proc.attenuation_function[pb_mat]
    # Оба граничных края (L3 и K) обязаны войти целиком как пары из 2 узлов
    assert np.sum(sliced_energy == l3_edge) == 2
    assert np.sum(sliced_energy == k_edge) == 2
    assert sliced_lac[1] > sliced_lac[0]
    assert sliced_lac[-1] > sliced_lac[-2]

    # Выход за пределы диапазона базы данных [1 кэВ, 20 МэВ] вызывает ValueError
    with pytest.raises(ValueError, match="выходит за границы"):
        pe_proc.energy_range = np.array([0.5 * units.keV, 10.0 * units.MeV], dtype=Float)
    with pytest.raises(ValueError, match="выходит за границы"):
        pe_proc.energy_range = np.array([1.0 * units.keV, 25.0 * units.MeV], dtype=Float)
    with pytest.raises(ValueError, match="Некорректный энергетический диапазон"):
        pe_proc.energy_range = np.array([10.0 * units.MeV, 1.0 * units.keV], dtype=Float)

    # Неподдерживаемый режим интерполяции вызывает ValueError
    with pytest.raises(ValueError, match="только кусочную log-log"):
        AttenuationFunction(pe_proc, attenuation_database, kind="linear")

    # Проверка векторного вызова на MaterialArray
    pe_full = PhotoelectricEffect(attenuation_database=attenuation_database)
    mat_arr = MaterialArray((2,))
    mat_arr[0] = pb_mat
    mat_arr[1] = nai_mat
    energies = np.array([k_edge - 1.0 * units.eV, k_edge + 1.0 * units.eV], dtype=Float)
    res_arr = pe_full.attenuation_function(mat_arr, energies)
    assert res_arr.shape == (2,)
    assert np.isclose(res_arr[0], float(pe_full.attenuation_function(pb_mat, energies[0])), rtol=1e-12)
    assert np.isclose(res_arr[1], float(pe_full.attenuation_function(nai_mat, energies[1])), rtol=1e-12)


def test_parse_nist_xcom_html() -> None:
    """Проверка парсера HTML-таблицы NIST XCOM v3.1 и LBYL-предусловий клиентских функций."""
    sample_html = """
    <table border=1>
    <tr BGColor="#FFFFCC">
    <td></td><td>1.000E-03</td><td>6.942E+00</td><td>6.428E-03</td><td>4.936E+03</td><td>0.000E+00</td><td>0.000E+00</td><td>4.943E+03</td><td>4.936E+03</td></tr>
    <tr BGColor="#FFFFCC">
    <td></td><td>2.520E-03</td><td>5.666E+00</td><td>2.045E-02</td><td>5.359E+02</td><td>0.000E+00</td><td>0.000E+00</td><td>5.416E+02</td><td>5.359E+02</td></tr>
    <tr BGColor="#FFFFCC">
    <td>42&#160;L<sub>3</sub></td><td> 2.520E-03</td><td> 5.666E+00</td><td> 2.045E-02</td><td> 1.973E+03</td><td> 0.000E+00</td><td> 0.000E+00</td><td> 1.979E+03</td><td> 1.973E+03</td></tr>
    <tr BGColor="#FFFFCC">
    <td></td><td>2.000E-02</td><td>9.893E-01</td><td>9.510E-02</td><td>1.200E+01</td><td>0.000E+00</td><td>0.000E+00</td><td>1.308E+01</td><td>1.209E+01</td></tr>
    <tr BGColor="#FFFFCC">
    <td>42&#160;K</td><td> 2.000E-02</td><td> 9.893E-01</td><td> 9.510E-02</td><td> 7.846E+01</td><td> 0.000E+00</td><td> 0.000E+00</td><td> 7.955E+01</td><td> 7.856E+01</td></tr>
    <tr BGColor="#FFFFCC">
    <td></td><td>2.000E-02</td><td>9.893E-01</td><td>9.510E-02</td><td>7.846E+01</td><td>0.000E+00</td><td>0.000E+00</td><td>7.955E+01</td><td>7.856E+01</td></tr>
    <tr BGColor="#FFFFCC">
    <td></td><td>2.000E+01</td><td>4.455E-06</td><td>1.275E-02</td><td>5.479E-05</td><td>2.697E-02</td><td>5.650E-04</td><td>4.034E-02</td><td>4.034E-02</td></tr>
    <tr BGColor="#FFFFCC">
    <td></td><td>3.000E+01</td><td>1.980E-06</td><td>9.400E-03</td><td>2.300E-05</td><td>3.010E-02</td><td>6.800E-04</td><td>4.020E-02</td><td>4.020E-02</td></tr>
    </table>
    """
    energy, processes_map, edge_energy, edge_label = parse_nist_xcom_html(sample_html)
    assert len(energy) == 7
    assert np.isclose(energy[0], 0.001)
    assert np.isclose(energy[-1], 20.0)
    np.testing.assert_allclose(edge_energy, [0.00252, 0.02])
    assert list(edge_label) == ["L3", "K"]
    assert np.isclose(processes_map["Photoelectric absorption"][3], 12.0)
    assert np.isclose(processes_map["Photoelectric absorption"][4], 78.46)

    with pytest.raises(ValueError, match="не найдено ни одной строки"):
        parse_nist_xcom_html("<html><body>Empty</body></html>")
    with pytest.raises(ValueError, match="Недопустимый атомный номер"):
        fetch_nist_xcom_element(atomic_z=0)
    with pytest.raises(ValueError, match="Недопустимый атомный номер"):
        fetch_nist_xcom_element(atomic_z=999)
    with pytest.raises(ValueError, match="Таймаут запроса"):
        fetch_nist_xcom_element(atomic_z=1, timeout=0.0)
    with pytest.raises(ValueError, match="Таймаут запроса"):
        fetch_and_save_nist_xcom_reference(timeout=-5.0)



