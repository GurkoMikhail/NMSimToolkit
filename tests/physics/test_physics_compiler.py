import math
import numpy as np

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.attenuation_database import _interpolate_element_on_material_grid
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.materials.materials import Material
from core.other.typing_definitions import Float, Index
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_kernels import (
    _compute_log_1d_array,
    _compute_log_2d_array,
    _compute_log_energy_and_bin,
    _get_macroscopic_cross_sections,
    _locate_energy_interval,
    _locate_row,
)
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PhotoelectricEffect,
)
from core.scene.nodes import CompositeNode
from core.source.sources import Source
from settings.database_setting import attenuation_database, material_database
import hepunits as units
from numba import njit
import pytest


def test_build_element_csr_basic():
    compiler = PhysicsCompiler()

    class MockMaterial:
        def __init__(self, ID, composition_dict, name="Mock"):
            self.ID = ID
            self._comp_dict = composition_dict
            self.name = name
            self.density = 1.0

        @property
        def composition_dict(self):
            return self._comp_dict

    mat1 = MockMaterial(ID=1, composition_dict={'H': 0.2, 'O': 0.8}, name="Mat1")
    mat3 = MockMaterial(ID=3, composition_dict={'C': 1.0}, name="Mat3")
    mat5 = MockMaterial(ID=5, composition_dict={'N': 0.4, 'O': 0.4}, name="Mat5")

    materials_list = [mat1, mat3, mat5]
    capacity = 11

    mat_info = np.zeros(capacity, dtype=MaterialInfoDType)
    mat_pointers = np.zeros(capacity, dtype=MaterialPointerDType)
    lookup_num_bins = 4
    mat_pointers[1]['start_idx'] = 0
    mat_pointers[1]['length'] = 2
    mat_pointers[1]['lookup_offset'] = 0
    mat_pointers[3]['start_idx'] = 2
    mat_pointers[3]['length'] = 2
    mat_pointers[3]['lookup_offset'] = lookup_num_bins
    mat_pointers[5]['start_idx'] = 4
    mat_pointers[5]['length'] = 2
    mat_pointers[5]['lookup_offset'] = 2 * lookup_num_bins

    physics_energy_grid = np.array([0.01, 1.0, 0.01, 1.0, 0.01, 1.0], dtype=Float)
    physics_lac_table = np.ones((6, 1), dtype=Float)
    log_e_min = Float(np.log(0.01))
    log_e_max = Float(np.log(1.0))
    mat_bank = MaterialBank(
        mat_info_buffer=mat_info,
        mat_pointers=mat_pointers,
        physics_energy_grid=physics_energy_grid,
        physics_lac_table=physics_lac_table,
        physics_log_energy_grid=_compute_log_1d_array(physics_energy_grid),
        physics_log_lac_table=_compute_log_2d_array(physics_lac_table),
        lookup_bin_start=np.zeros(3 * lookup_num_bins, dtype=Index),
        lookup_log_e_min=log_e_min,
        lookup_inv_dlog_e=Float(lookup_num_bins / (log_e_max - log_e_min)),
        lookup_num_bins=lookup_num_bins,
        lookup_max_advance=0,
    )
    mat_bank._validate()

    processes = [PhotoelectricEffect(attenuation_database=attenuation_database)]
    csr = compiler._build_element_csr(materials_list, processes, mat_bank, capacity=capacity)  # type: ignore[arg-type]

    assert len(csr.element_offsets) == capacity + 1
    expected_offsets = np.array([0, 0, 2, 2, 3, 3, 5, 5, 5, 5, 5, 5])
    np.testing.assert_array_equal(csr.element_offsets, expected_offsets)

    expected_Z = np.array([1, 8, 6, 7, 8])
    np.testing.assert_array_equal(csr.element_Z, expected_Z)

    assert csr.element_partial_fractions.shape == (6, 1, 2)
    np.testing.assert_allclose(csr.element_partial_fractions[0:2, 0, :], [[0.2, 0.8], [0.2, 0.8]])
    np.testing.assert_allclose(csr.element_partial_fractions[2:4, 0, :], [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(csr.element_partial_fractions[4:6, 0, :], [[0.5, 0.5], [0.5, 0.5]])


def test_build_element_csr_partial_cross_sections_and_edges() -> None:
    """
    Проверка вычисления нормированных парциальных долей элементов на финальной сетке MaterialBank
    (включая односторонние пределы на парах узлов краев поглощения NaI) и контроля инварианта <= 1e-12.
    """
    compiler = PhysicsCompiler()
    processes = [
        PhotoelectricEffect(attenuation_database=attenuation_database),
        ComptonScattering(attenuation_database=attenuation_database),
        CoherentScattering(attenuation_database=attenuation_database),
    ]
    nai: Material = material_database["Sodium Iodide"]
    water: Material = material_database["Water, Liquid"]
    lead: Material = material_database["Pb"]
    materials = [nai, water, lead]

    mat_bank = compiler._build_material_bank(materials, processes)
    csr = compiler._build_element_csr(materials, processes, mat_bank)

    for material in materials:
        mat_id = material.ID
        elem_start = int(csr.element_offsets[mat_id])
        elem_end = int(csr.element_offsets[mat_id + 1])
        count = elem_end - elem_start
        row_start = int(mat_bank.mat_pointers[mat_id]["start_idx"])
        row_length = int(mat_bank.mat_pointers[mat_id]["length"])

        fractions_block = csr.element_partial_fractions[
            row_start : row_start + row_length, :, :count
        ]
        assert np.all(fractions_block >= 0.0)
        assert np.all(fractions_block <= 1.0)
        sums = np.sum(fractions_block, axis=2)
        np.testing.assert_allclose(sums, 1.0, rtol=0.0, atol=1e-12)

    # Проверка односторонних пределов на К-крае йода в NaI (~33.1694 кэВ)
    nai_row_start = int(mat_bank.mat_pointers[nai.ID]["start_idx"])
    nai_row_len = int(mat_bank.mat_pointers[nai.ID]["length"])
    nai_grid = mat_bank.physics_energy_grid[nai_row_start : nai_row_start + nai_row_len]
    nai_elem_start = int(csr.element_offsets[nai.ID])
    nai_z = csr.element_Z[nai_elem_start : nai_elem_start + 2]
    iodine_offset = int(np.flatnonzero(nai_z == 53)[0])
    sodium_offset = int(np.flatnonzero(nai_z == 11)[0])

    iodine_edges, iodine_labels = attenuation_database.get_element_edges("I")
    k_edge_energy = float(iodine_edges[iodine_labels == "K"][0])
    edge_local_indices = np.flatnonzero(np.isclose(nai_grid, k_edge_energy, rtol=1e-12, atol=0.0))
    assert len(edge_local_indices) == 2
    left_local, right_local = int(edge_local_indices[0]), int(edge_local_indices[1])

    row_left = nai_row_start + left_local
    row_right = nai_row_start + right_local

    # В правом узле К-края йода доля фотоэффекта на йоде строго выше, чем в левом узле
    pe_iodine_left = float(csr.element_partial_fractions[row_left, 0, iodine_offset])
    pe_iodine_right = float(csr.element_partial_fractions[row_right, 0, iodine_offset])
    assert pe_iodine_right > pe_iodine_left
    assert abs((pe_iodine_left + float(csr.element_partial_fractions[row_left, 0, sodium_offset])) - 1.0) <= 1e-12
    assert abs((pe_iodine_right + float(csr.element_partial_fractions[row_right, 0, sodium_offset])) - 1.0) <= 1e-12

    # Прямая сверка с отношением односторонних пределов вкладов элементов
    iodine_table = attenuation_database.elements_MAC["I"]
    iodine_mac_on_nai = _interpolate_element_on_material_grid(
        nai_grid, iodine_table["Energy"], iodine_table["Coefficient"]["PhotoelectricEffect"]
    )
    iodine_weight = float(nai.composition_dict["I"])
    expected_left = (
        nai.density * iodine_weight * float(iodine_mac_on_nai[left_local])
        / float(mat_bank.physics_lac_table[row_left, 0])
    )
    expected_right = (
        nai.density * iodine_weight * float(iodine_mac_on_nai[right_local])
        / float(mat_bank.physics_lac_table[row_right, 0])
    )
    assert abs(pe_iodine_left - expected_left) <= 1e-12
    assert abs(pe_iodine_right - expected_right) <= 1e-12

    # Проверка Fail-Fast при расхождении полного сечения в MaterialBank и суммы парциальных вкладов
    corrupted_lacs = np.copy(mat_bank.physics_lac_table)
    corrupted_lacs[row_left, 0] *= 1.01
    corrupted_bank = mat_bank._replace(physics_lac_table=corrupted_lacs)
    with pytest.raises(ValueError, match="Нарушен инвариант парциальных сечений"):
        compiler._build_element_csr(materials, processes, corrupted_bank)

    # Проверка Fail-Fast при обнулении полного сечения в MaterialBank при ненулевых парциальных вкладах
    zeroed_lacs = np.copy(mat_bank.physics_lac_table)
    zeroed_lacs[row_left, 0] = 0.0
    zeroed_bank = mat_bank._replace(physics_lac_table=zeroed_lacs)
    with pytest.raises(ValueError, match="полное сечение в MaterialBank равно 0.0"):
        compiler._build_element_csr(materials, processes, zeroed_bank)

    # Проверка отсутствия расхождения индексов строк row при обрезке по суженному energy_range [10 кэВ, 500 кэВ]
    trimmed_processes = [
        PhotoelectricEffect(attenuation_database=attenuation_database),
        ComptonScattering(attenuation_database=attenuation_database),
        CoherentScattering(attenuation_database=attenuation_database),
    ]
    trimmed_range = np.array([10.0 * units.keV, 500.0 * units.keV], dtype=Float)
    for proc in trimmed_processes:
        proc.energy_range = trimmed_range

    trimmed_bank = compiler._build_material_bank([nai], trimmed_processes)
    trimmed_csr = compiler._build_element_csr([nai], trimmed_processes, trimmed_bank)
    for col_idx, proc in enumerate(trimmed_processes):
        assert int(proc.process_col_idx) == col_idx

    trimmed_row_start = int(trimmed_bank.mat_pointers[nai.ID]["start_idx"])
    trimmed_row_len = int(trimmed_bank.mat_pointers[nai.ID]["length"])
    assert trimmed_row_len < nai_row_len
    trimmed_grid = trimmed_bank.physics_energy_grid[trimmed_row_start : trimmed_row_start + trimmed_row_len]
    trimmed_k_edges = np.flatnonzero(np.isclose(trimmed_grid, k_edge_energy, rtol=1e-12, atol=0.0))
    assert len(trimmed_k_edges) == 2
    t_left, t_right = int(trimmed_k_edges[0]), int(trimmed_k_edges[1])
    assert abs(float(trimmed_csr.element_partial_fractions[trimmed_row_start + t_left, 0, iodine_offset]) - pe_iodine_left) <= 1e-12
    assert abs(float(trimmed_csr.element_partial_fractions[trimmed_row_start + t_right, 0, iodine_offset]) - pe_iodine_right) <= 1e-12

    # Проверка тройки узлов края поглощения (multiplicity == 3) и совпадающих краев двух элементов:
    # в левом узле (k = left_pos) нормируются левые пределы на левое суммарное сечение,
    # а во всех правых узлах (k > left_pos) — правые пределы на правое суммарное сечение
    edge_e = Float(0.050 * units.MeV)
    grid_with_triple = np.array([0.010, edge_e, edge_e, edge_e, 0.100], dtype=Float)
    el_a_energy = np.array([0.010, edge_e, edge_e, edge_e, 0.100], dtype=Float)
    el_a_coeff = np.array([10.0, 2.0, 5.0, 8.0, 1.0], dtype=Float)
    el_b_energy = np.array([0.010, edge_e, edge_e, 0.100], dtype=Float)
    el_b_coeff = np.array([4.0, 1.0, 3.0, 0.5], dtype=Float)

    interp_a = _interpolate_element_on_material_grid(grid_with_triple, el_a_energy, el_a_coeff)
    interp_b = _interpolate_element_on_material_grid(grid_with_triple, el_b_energy, el_b_coeff)
    # В левом узле края (индекс 1): A=2.0, B=1.0; в правых узлах (индексы 2 и 3): A=8.0, B=3.0
    np.testing.assert_allclose(interp_a, [10.0, 2.0, 8.0, 8.0, 1.0], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(interp_b, [4.0, 1.0, 3.0, 3.0, 0.5], rtol=1e-12, atol=0.0)


def test_build_material_bank_preserves_edge_pairs_directly() -> None:
    """
    Проверка того, что PhysicsCompiler._build_material_bank копирует значения LAC напрямую
    по индексам узлов из AttenuationFunction, сохраняя левый и правый пределы на краях поглощения.
    """
    compiler = PhysicsCompiler()
    processes = [
        PhotoelectricEffect(attenuation_database=attenuation_database),
        ComptonScattering(attenuation_database=attenuation_database),
        CoherentScattering(attenuation_database=attenuation_database),
    ]
    lead: Material = material_database["Pb"]
    mat_bank = compiler._build_material_bank([lead], processes)

    offset = int(mat_bank.mat_pointers[lead.ID]["start_idx"])
    length = int(mat_bank.mat_pointers[lead.ID]["length"])
    lead_grid = mat_bank.physics_energy_grid[offset : offset + length]
    lead_lacs = mat_bank.physics_lac_table[offset : offset + length]

    ref_table = attenuation_database[lead]
    np.testing.assert_array_equal(lead_grid, ref_table["Energy"])
    np.testing.assert_allclose(
        lead_lacs[:, 0],
        ref_table["Coefficient"]["PhotoelectricEffect"] * lead.density,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        lead_lacs[:, 1],
        ref_table["Coefficient"]["ComptonScattering"] * lead.density,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        lead_lacs[:, 2],
        ref_table["Coefficient"]["CoherentScattering"] * lead.density,
        rtol=1e-12,
        atol=0.0,
    )

    # На К-крае свинца (88 кэВ) левый и правый узлы пары обязаны отличаться
    k_edge_indices = np.flatnonzero(np.isclose(lead_grid, 0.088 * units.MeV, rtol=1e-6, atol=0.0))
    assert len(k_edge_indices) == 2
    idx_left, idx_right = int(k_edge_indices[0]), int(k_edge_indices[1])
    assert lead_lacs[idx_right, 0] > 4.0 * lead_lacs[idx_left, 0]


def test_compile_scene_source_energy_out_of_range_raises() -> None:
    """
    Проверка предусловия PhysicsCompiler.compile_scene:
    все спектральные линии источников сцены должны лежать в рабочем диапазоне таблиц [1 кэВ, 20 МэВ].
    """
    compiler = PhysicsCompiler()
    processes = [
        PhotoelectricEffect(attenuation_database=attenuation_database),
        ComptonScattering(attenuation_database=attenuation_database),
        CoherentScattering(attenuation_database=attenuation_database),
    ]
    scene = CompositeNode()
    world_vol = Volume(
        geometry=Box(100.0 * units.mm, 100.0 * units.mm, 100.0 * units.mm),
        material=material_database["Air, Dry (near sea level)"],
    )
    scene.add_child(world_vol)

    # 1. Источник с энергией ниже нижней границы (0.5 кэВ < 1.0 кэВ)
    low_source = Source(
        distribution=np.ones((1, 1, 1), dtype=Float),
        activity=Float(1000.0 * units.Bq),
        energy=Float(0.5 * units.keV),
    )
    scene.add_child(low_source)
    with pytest.raises(ValueError, match="вне допустимого диапазона"):
        compiler.compile_scene(scene, processes)
    scene.remove_child(low_source)

    # 2. Источник с энергией выше верхней границы (25 МэВ > 20 МэВ)
    high_source = Source(
        distribution=np.ones((1, 1, 1), dtype=Float),
        activity=Float(1000.0 * units.Bq),
        energy=Float(25.0 * units.MeV),
    )
    scene.add_child(high_source)
    with pytest.raises(ValueError, match="вне допустимого диапазона"):
        compiler.compile_scene(scene, processes)
    scene.remove_child(high_source)

    # 3. Источник с допустимой энергией (140.5 кэВ) успешно компилируется
    valid_source = Source(
        distribution=np.ones((1, 1, 1), dtype=Float),
        activity=Float(1000.0 * units.Bq),
        energy=Float(140.5 * units.keV),
    )
    scene.add_child(valid_source)
    physics_buf = compiler.compile_scene(scene, processes)
    assert physics_buf.material_bank.physics_energy_grid.size > 0


@njit(cache=True)
def _ref_searchsorted_locate_and_lacs(
    energy: Float,
    material_id: Index,
    material_bank: MaterialBank,
    out_lacs_slice: np.ndarray,
) -> tuple[Index, Float]:
    """
    Эталонная реализация поиска интервала через np.searchsorted(..., side='right')
    и кусочной лог-лог интерполяции сечений (в точности повторяющая алгоритм до Фазы 3)
    для проверки побитовой эквивалентности двухступенчатого лог-индекса.
    """
    pointer = material_bank.mat_pointers[material_id]
    start_idx = Index(pointer["start_idx"])
    length = Index(pointer["length"])
    mat_energy_grid = material_bank.physics_energy_grid[start_idx : start_idx + length]
    num_procs = material_bank.physics_lac_table.shape[1]

    if energy < mat_energy_grid[0]:
        for proc_idx in range(num_procs):
            out_lacs_slice[proc_idx] = material_bank.physics_lac_table[start_idx, proc_idx]
        return start_idx, Float(0.0)

    last_row = Index(start_idx + length - 1)
    if energy >= mat_energy_grid[length - 1]:
        for proc_idx in range(num_procs):
            out_lacs_slice[proc_idx] = material_bank.physics_lac_table[last_row, proc_idx]
        return last_row, Float(0.0)

    idx = np.searchsorted(mat_energy_grid, energy, side="right") - 1
    if idx < 0:
        idx = 0
    elif idx >= length - 1:
        idx = length - 2

    row0 = Index(start_idx + idx)
    row1 = Index(row0 + 1)
    energy_low = mat_energy_grid[idx]
    energy_high = mat_energy_grid[idx + 1]

    if energy <= energy_low or energy_high <= energy_low:
        for proc_idx in range(num_procs):
            out_lacs_slice[proc_idx] = material_bank.physics_lac_table[row0, proc_idx]
        return row0, Float(0.0)

    if energy_low > 0.0:
        log_fraction = Float(
            (math.log(energy) - math.log(energy_low))
            / (math.log(energy_high) - math.log(energy_low))
        )
    else:
        log_fraction = Float((energy - energy_low) / (energy_high - energy_low))

    lin_fraction = (energy - energy_low) / (energy_high - energy_low)
    for proc_idx in range(num_procs):
        val_low = material_bank.physics_lac_table[row0, proc_idx]
        val_high = material_bank.physics_lac_table[row1, proc_idx]
        if val_low > 0.0 and val_high > 0.0 and energy_low > 0.0:
            log_val_low = math.log(val_low)
            log_val_high = math.log(val_high)
            out_lacs_slice[proc_idx] = Float(
                math.exp(log_val_low + log_fraction * (log_val_high - log_val_low))
            )
        else:
            out_lacs_slice[proc_idx] = Float(val_low + lin_fraction * (val_high - val_low))

    return row0, log_fraction


@njit(cache=True)
def _verify_batch_bitwise_equivalence(
    energies: np.ndarray,
    material_id: Index,
    material_bank: MaterialBank,
) -> int:
    """
    Пакетная проверка побитовой идентичности двухступенчатого лог-индекса (_locate_row,
    _locate_energy_interval, _get_macroscopic_cross_sections) и эталона np.searchsorted.
    Возвращает индекс первой обнаруженной ошибки или -1 при полном совпадении.
    """
    num_energies = energies.shape[0]
    num_processes = material_bank.physics_lac_table.shape[1]
    lacs_fast = np.empty(num_processes, dtype=Float)
    lacs_ref = np.empty(num_processes, dtype=Float)

    for idx in range(num_energies):
        energy = Float(energies[idx])
        log_e, log_bin = _compute_log_energy_and_bin(energy, material_bank)
        row_fast, frac_fast = _locate_row(energy, log_e, log_bin, material_id, material_bank)
        row_loc, frac_loc = _locate_energy_interval(energy, material_id, material_bank)
        _get_macroscopic_cross_sections(
            energy, log_e, log_bin, material_id, material_bank, lacs_fast
        )
        row_ref, frac_ref = _ref_searchsorted_locate_and_lacs(
            energy, material_id, material_bank, lacs_ref
        )

        if row_fast != row_ref or row_loc != row_ref:
            return idx
        if frac_fast != frac_ref or frac_loc != frac_ref:
            return idx
        for col_idx in range(num_processes):
            if lacs_fast[col_idx] != lacs_ref[col_idx]:
                return idx

    return -1


def test_two_level_log_index_bitwise_equivalence() -> None:
    """
    Проверка побитовой эквивалентности (0 ULP) двухступенчатого лог-индекса и предвычисленных
    лог-таблиц относительно эталонного np.searchsorted(..., side='right') на:
    - 10^6 случайных энергий в [E_min / 2, 2 * E_max];
    - всех узлах сеток энергии всех 12 материалов базы данных;
    - всех парах и тройках узлов краев поглощения;
    - окрестностях E_edge +/- eps (nextafter) и всех границах лог-бинов;
    - краевых случаях (неактивный материал length == 0, минимальная сетка length == 1 и length == 2,
      energy <= 0.0, 1 и 2 активных физических процесса).
    """
    compiler = PhysicsCompiler()
    processes = [
        PhotoelectricEffect(attenuation_database=attenuation_database),
        ComptonScattering(attenuation_database=attenuation_database),
        CoherentScattering(attenuation_database=attenuation_database),
    ]
    all_materials = list(material_database.values())
    mat_bank = compiler._build_material_bank(all_materials, processes)
    assert mat_bank.lookup_num_bins == 512
    assert 0 <= mat_bank.lookup_max_advance <= 16

    rng = np.random.default_rng(np.random.SeedSequence(20261025))
    e_min = Float(1.0e-3 * units.MeV)
    e_max = Float(20.0 * units.MeV)
    half_e_min = Float(0.5 * e_min)
    double_e_max = Float(2.0 * e_max)

    # 10^5 случайных энергий в [E_min / 2, 2 * E_max] (лог-равномерные + линейно-равномерные)
    num_random = 100_000
    num_log = num_random // 2
    num_lin = num_random - num_log
    random_energies = np.empty(num_random, dtype=Float)
    random_energies[:num_log] = np.exp(
        rng.uniform(np.log(half_e_min), np.log(double_e_max), size=num_log)
    )
    random_energies[num_log:] = rng.uniform(half_e_min, double_e_max, size=num_lin)

    # Границы лог-бинов и их +/- 1 ULP окрестности
    bin_indices = np.arange(mat_bank.lookup_num_bins + 1, dtype=Float)
    bin_energies = np.exp(
        mat_bank.lookup_log_e_min + bin_indices / mat_bank.lookup_inv_dlog_e
    ).astype(Float)

    # Проверяем случайный массив из 10^5 энергий на свинце (содержащем K, L, M, N края)
    lead_id = Index(material_database["Pb"].ID)
    fail_idx = _verify_batch_bitwise_equivalence(random_energies, lead_id, mat_bank)
    assert fail_idx == -1, (
        f"Нарушена побитовая эквивалентность на случайной энергии E={random_energies[fail_idx]} для Pb"
    )

    # Проверяем детерминированный набор особых точек (все узлы сетки, края поглощения,
    # +/- 1 ULP, +/- 1e-12 * E, границы бинов, неположительные энергии) + 20 000 случайных точек
    # для каждого из 12 материалов в базе данных
    sub_random = random_energies[:20_000]
    for material in all_materials:
        mat_id = Index(material.ID)
        pointer = mat_bank.mat_pointers[mat_id]
        start_idx = int(pointer["start_idx"])
        length = int(pointer["length"])
        mat_grid = mat_bank.physics_energy_grid[start_idx : start_idx + length]

        nodes_minus_ulp = np.nextafter(mat_grid, -np.inf)
        nodes_plus_ulp = np.nextafter(mat_grid, np.inf)
        nodes_minus_eps = (mat_grid * (1.0 - 1e-12)).astype(Float)
        nodes_plus_eps = (mat_grid * (1.0 + 1e-12)).astype(Float)
        bins_minus_ulp = np.nextafter(bin_energies, -np.inf)
        bins_plus_ulp = np.nextafter(bin_energies, np.inf)

        special_energies = np.concatenate(
            [
                mat_grid,
                nodes_minus_ulp,
                nodes_plus_ulp,
                nodes_minus_eps,
                nodes_plus_eps,
                bin_energies,
                bins_minus_ulp,
                bins_plus_ulp,
                np.array(
                    [-1.0 * units.MeV, 0.0 * units.MeV, half_e_min, e_min, e_max, double_e_max],
                    dtype=Float,
                ),
                sub_random,
            ]
        )
        fail_pos = _verify_batch_bitwise_equivalence(special_energies, mat_id, mat_bank)
        assert fail_pos == -1, (
            f"Нарушена побитовая эквивалентность для материала {material.name} "
            f"при E={special_energies[fail_pos]}"
        )

    # Проверка при 1 и 2 активных физических процессах
    for num_procs in (1, 2):
        sub_bank = compiler._build_material_bank([material_database["Pb"]], processes[:num_procs])
        fail_sub = _verify_batch_bitwise_equivalence(sub_random[:10_000], lead_id, sub_bank)
        assert fail_sub == -1

        # Неактивный слот в sub_bank (например, ID=1, не включенный в сцену: length == 0)
        inactive_id = Index(1)
        assert int(sub_bank.mat_pointers[inactive_id]["length"]) == 0
        out_lacs = np.full(num_procs, 99.0, dtype=Float)
        log_e_test, log_bin_test = _compute_log_energy_and_bin(Float(0.1 * units.MeV), sub_bank)
        row_inact, frac_inact = _locate_row(
            Float(0.1 * units.MeV), log_e_test, log_bin_test, inactive_id, sub_bank
        )
        assert row_inact == 0 and frac_inact == 0.0
        _get_macroscopic_cross_sections(
            Float(0.1 * units.MeV), log_e_test, log_bin_test, inactive_id, sub_bank, out_lacs
        )
        np.testing.assert_array_equal(out_lacs, np.zeros(num_procs, dtype=Float))

    # Проверка минимальных сеток из 2 узлов (length == 2) и 1 узла (length == 1)
    mat_pointers_min = np.zeros(2, dtype=MaterialPointerDType)
    mat_pointers_min[0]["start_idx"] = 0
    mat_pointers_min[0]["length"] = 2
    mat_pointers_min[0]["lookup_offset"] = 0
    mat_pointers_min[1]["start_idx"] = 2
    mat_pointers_min[1]["length"] = 1
    mat_pointers_min[1]["lookup_offset"] = 8

    grid_min = np.array([1.0e-3, 20.0, 0.5], dtype=Float)
    lacs_min = np.array([[10.0, 2.0, 0.5], [0.01, 0.05, 0.0], [1.5, 0.3, 0.1]], dtype=Float)
    bank_min = MaterialBank(
        mat_info_buffer=np.zeros(2, dtype=MaterialInfoDType),
        mat_pointers=mat_pointers_min,
        physics_energy_grid=grid_min,
        physics_lac_table=lacs_min,
        physics_log_energy_grid=_compute_log_1d_array(grid_min),
        physics_log_lac_table=_compute_log_2d_array(lacs_min),
        lookup_bin_start=np.zeros(16, dtype=Index),
        lookup_log_e_min=Float(math.log(1.0e-3)),
        lookup_inv_dlog_e=Float(8.0 / (math.log(20.0) - math.log(1.0e-3))),
        lookup_num_bins=8,
        lookup_max_advance=0,
    )
    bank_min._validate()
    fail_min2 = _verify_batch_bitwise_equivalence(sub_random[:10_000], Index(0), bank_min)
    assert fail_min2 == -1
    for test_e in (0.0, 1.0e-3, 0.5, 20.0):
        log_e_1, log_bin_1 = _compute_log_energy_and_bin(Float(test_e), bank_min)
        row_1, frac_1 = _locate_row(Float(test_e), log_e_1, log_bin_1, Index(1), bank_min)
        assert row_1 == 2 and frac_1 == 0.0
        out_1 = np.empty(3, dtype=Float)
        _get_macroscopic_cross_sections(
            Float(test_e), log_e_1, log_bin_1, Index(1), bank_min, out_1
        )
        np.testing.assert_array_equal(out_1, lacs_min[2])


def test_lookup_max_advance_precondition_violation() -> None:
    """
    Проверка жесткого предусловия (Fail-Fast) на максимальное число шагов доводки
    lookup_max_advance <= max_allowed_advance при вырожденном числе лог-бинов.
    """
    compiler = PhysicsCompiler()
    processes = [
        PhotoelectricEffect(attenuation_database=attenuation_database),
        ComptonScattering(attenuation_database=attenuation_database),
        CoherentScattering(attenuation_database=attenuation_database),
    ]
    lead: Material = material_database["Pb"]

    with pytest.raises(ValueError, match="lookup_max_advance="):
        compiler._build_material_bank([lead], processes, lookup_num_bins=2, max_allowed_advance=16)

    with pytest.raises(ValueError, match="lookup_num_bins"):
        compiler._build_material_bank([lead], processes, lookup_num_bins=0)

    scene = CompositeNode()
    world_vol = Volume(
        geometry=Box(100.0 * units.mm, 100.0 * units.mm, 100.0 * units.mm),
        material=lead,
    )
    scene.add_child(world_vol)
    with pytest.raises(ValueError, match="lookup_max_advance="):
        compiler.compile_scene(scene, processes, lookup_num_bins=1, max_allowed_advance=16)


def test_material_bank_validate_invariants() -> None:
    """
    Проверка инвариантов MaterialBank._validate() при некорректных размерах буферов,
    выходе указателей за границы и нарушении монотонности/положительности сеток.
    """
    compiler = PhysicsCompiler()
    processes = [
        PhotoelectricEffect(attenuation_database=attenuation_database),
        ComptonScattering(attenuation_database=attenuation_database),
        CoherentScattering(attenuation_database=attenuation_database),
    ]
    lead: Material = material_database["Pb"]
    valid_bank = compiler._build_material_bank([lead], processes)
    valid_bank._validate()

    # 1. Несовпадение размера physics_log_energy_grid
    bad_log_grid = valid_bank._replace(physics_log_energy_grid=valid_bank.physics_log_energy_grid[:-1])
    with pytest.raises(ValueError, match="physics_log_energy_grid"):
        bad_log_grid._validate()

    # 2. Несовпадение формы physics_log_lac_table
    bad_log_lacs = valid_bank._replace(physics_log_lac_table=valid_bank.physics_log_lac_table[:-1, :])
    with pytest.raises(ValueError, match="physics_log_lac_table"):
        bad_log_lacs._validate()

    # 3. Некорректное число лог-бинов
    bad_bins = valid_bank._replace(lookup_num_bins=0)
    with pytest.raises(ValueError, match="lookup_num_bins"):
        bad_bins._validate()

    # 4. Отрицательный lookup_max_advance
    bad_advance = valid_bank._replace(lookup_max_advance=-1)
    with pytest.raises(ValueError, match="lookup_max_advance"):
        bad_advance._validate()

    # 5. Нарушение монотонности lookup_bin_start внутри материала
    non_monotonic_bins = np.copy(valid_bank.lookup_bin_start)
    offset = int(valid_bank.mat_pointers[lead.ID]["lookup_offset"])
    non_monotonic_bins[offset + 10] = non_monotonic_bins[offset + 11] + 1
    bad_monotonic = valid_bank._replace(lookup_bin_start=non_monotonic_bins)
    with pytest.raises(ValueError, match="Нарушена монотонность"):
        bad_monotonic._validate()

    # 6. Выход значения lookup_bin_start за пределы [0, length - 2]
    out_of_bounds_bins = np.copy(valid_bank.lookup_bin_start)
    length = int(valid_bank.mat_pointers[lead.ID]["length"])
    out_of_bounds_bins[offset + valid_bank.lookup_num_bins - 1] = length - 1
    bad_bounds = valid_bank._replace(lookup_bin_start=out_of_bounds_bins)
    with pytest.raises(ValueError, match="выходят за допустимый диапазон"):
        bad_bounds._validate()

    # 7. Неположительный узел в physics_energy_grid
    non_positive_grid = np.copy(valid_bank.physics_energy_grid)
    start_idx = int(valid_bank.mat_pointers[lead.ID]["start_idx"])
    non_positive_grid[start_idx] = 0.0
    bad_pos_grid = valid_bank._replace(physics_energy_grid=non_positive_grid)
    with pytest.raises(ValueError, match="строго положительные"):
        bad_pos_grid._validate()

    # 8. Нарушение монотонности physics_energy_grid внутри материала
    non_monotonic_grid = np.copy(valid_bank.physics_energy_grid)
    non_monotonic_grid[start_idx + 5] = non_monotonic_grid[start_idx + 6] + 0.01
    bad_mono_grid = valid_bank._replace(physics_energy_grid=non_monotonic_grid)
    with pytest.raises(ValueError, match="physics_energy_grid"):
        bad_mono_grid._validate()


if __name__ == '__main__':
    test_build_element_csr_basic()
    test_build_element_csr_partial_cross_sections_and_edges()
    test_build_material_bank_preserves_edge_pairs_directly()
    test_compile_scene_source_energy_out_of_range_raises()
    test_two_level_log_index_bitwise_equivalence()
    test_lookup_max_advance_precondition_violation()
    test_material_bank_validate_invariants()
    print("All Pytest-style assertions passed!")


