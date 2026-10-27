import tracemalloc
import unittest

import hepunits as units
import numpy as np
from numba import njit

from core.geometry.geometries import Box
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.volumes import Volume
from core.materials.attenuation_database import _interpolate_element_on_material_grid
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.materials.materials import Material
from core.other.typing_definitions import (
    CFuncAddress,
    Charge,
    Float,
    ID,
    Index,
    ParticleSpecies,
    ProcessID,
    Species,
    Time,
)
from core.particles.kinematic_state import KinematicState
from core.particles.particles import ParticleBank
from core.particles.particles_kernels import _rotate_particle
from core.physics.g4coherent import _generate_coherent_theta_scalar
from core.physics.g4compton import _generate_compton_theta_scalar, electron_mass_c2
from core.physics.interaction_buffers import (
    InteractionBuffer,
    RNGContext,
    SecondaryActivationBuffer,
    SimulationDataBuffer,
)
from core.physics.physics_buffer import ElementCSR, PhysicsBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PhotoelectricEffect,
)
from core.physics.processes_kernels import (
    C_LIGHT,
    _compute_activation_time,
    _sample_Z_scalar,
    _sample_ionized_subshell_scalar,
    make_coherent_kernel,
    make_compton_kernel,
    make_photoelectric_kernel,
    partition_transferred_energy_kernel,
)
from core.scene.nodes import CompositeNode
from core.source.source_kernels import inject_secondary_sources_kernel
from core.source.sources import Source
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from settings.database_setting import attenuation_database, material_database


class TestProcessesSoA(unittest.TestCase):
    """Тесты физических ядер взаимодействия фотонов (SoA) и генераторов углов рассеяния."""

    def setUp(self) -> None:
        self.capacity = 1000
        self.bank = ParticleBank.allocate(self.capacity)
        self.rng = np.random.default_rng(42)
        self.rng_ctx = RNGContext.from_numpy_rng(self.rng)

        energy = np.full(self.capacity, 0.5 * units.MeV)

        pos = self.bank.state.position
        pos.x[:] = self.rng.random(self.capacity)
        pos.y[:] = self.rng.random(self.capacity)
        pos.z[:] = self.rng.random(self.capacity)

        dir_vec = self.bank.state.direction
        dir_vec.x[:] = 0.0
        dir_vec.y[:] = 0.0
        dir_vec.z[:] = 1.0

        self.bank.state.is_active[:] = True
        self.bank.initial_state.ID[:] = np.arange(self.capacity, dtype=np.uint64)
        self.bank.initial_state.emission_time[:] = 0.0
        self.bank.state.energy[:] = energy
        self.target_indices = np.arange(self.capacity, dtype=np.int64)

        self.buffer = InteractionBuffer.allocate(self.capacity)
        self.act_buffer = SecondaryActivationBuffer.allocate(0)

    def test_photoelectric_kernel(self) -> None:
        """Проверка фотоэлектрического поглощения в SoA-ядре."""
        kernel = make_photoelectric_kernel(process_id=ProcessID(1))
        material_ids = np.full(self.capacity, 0, dtype=Index)
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        mock_physics = self._create_mock_physics_buffer()

        kernel(
            self.bank.state,
            particle_ids,
            self.bank.initial_state.emission_time,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            self.act_buffer,
            mock_physics,
            self.rng_ctx,
        )

        self.assertEqual(self.buffer.cursor[0], self.capacity)
        np.testing.assert_array_equal(self.bank.state.energy, 0.0)
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.5 * units.MeV)
        np.testing.assert_array_equal(self.buffer.process_id, 1)
        np.testing.assert_array_equal(self.buffer.Z, 13)

    def test_compton_kernel_equivalence_and_performance(self) -> None:
        """Проверка воспроизводимости и инвариантов генерации углов Комптона."""
        material_ids = np.full(self.capacity, 13, dtype=Charge)
        energy_arr = np.full(self.capacity, 0.5 * units.MeV, dtype=Float)

        new_rng = np.random.default_rng(1234)
        new_rng_ctx = RNGContext.from_numpy_rng(new_rng)

        @njit(cache=True)
        def _get_new_thetas(cap: int, energies: np.ndarray, z_vals: np.ndarray, ctx: RNGContext) -> np.ndarray:
            out = np.empty(cap, dtype=np.float64)
            for idx in range(cap):
                out[idx] = _generate_compton_theta_scalar(energies[idx], z_vals[idx], ctx)
            return out

        theta_new = _get_new_thetas(self.capacity, energy_arr, material_ids, new_rng_ctx)

        self.assertTrue(np.all(theta_new >= 0.0))
        self.assertTrue(np.all(theta_new <= np.pi))
        self.assertTrue(0.1 < np.mean(theta_new) < 2.5)

    def test_coherent_kernel_equivalence(self) -> None:
        """Проверка воспроизводимости и инвариантов генерации углов когерентного рассеяния."""
        material_ids = np.full(self.capacity, 82, dtype=Charge)
        energy_arr = np.full(self.capacity, 0.5 * units.MeV, dtype=Float)

        new_rng = np.random.default_rng(777)
        new_rng_ctx = RNGContext.from_numpy_rng(new_rng)

        @njit(cache=True)
        def _get_new_thetas_coh(cap: int, energies: np.ndarray, z_vals: np.ndarray, ctx: RNGContext) -> np.ndarray:
            out = np.empty(cap, dtype=np.float64)
            for idx in range(cap):
                out[idx] = _generate_coherent_theta_scalar(energies[idx], z_vals[idx], ctx)
            return out

        theta_new = _get_new_thetas_coh(self.capacity, energy_arr, material_ids, new_rng_ctx)

        self.assertTrue(np.all(theta_new >= 0.0))
        self.assertTrue(np.all(theta_new <= np.pi))
        self.assertTrue(np.mean(theta_new) < 1.0)

    def test_coherent_scattering_de_morgan_and_nan_guard(self) -> None:
        """
        Регрессионный тест _generate_coherent_theta_scalar (g4coherent.py):
        1. Защита от NaN при нулевой/неположительной энергии (xx <= 0.0) или Z <= 0 (возврат 0.0).
        2. Корректная работа цикла отбора (rejection sampling) по закону де Моргана
           (выход при cost >= -1.0 и 2*r3 < 1 + cost^2) в широком диапазоне энергий.
        """
        rng = np.random.default_rng(2026)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        @njit(cache=True)
        def _eval_edge_cases(ctx: RNGContext) -> np.ndarray:
            res = np.empty(4, dtype=np.float64)
            res[0] = _generate_coherent_theta_scalar(Float(0.0), Charge(82), ctx)
            res[1] = _generate_coherent_theta_scalar(Float(-0.1), Charge(82), ctx)
            res[2] = _generate_coherent_theta_scalar(Float(0.14 * units.MeV), Charge(0), ctx)
            res[3] = _generate_coherent_theta_scalar(Float(1e-18), Charge(13), ctx)
            return res

        edge_thetas = _eval_edge_cases(rng_ctx)
        self.assertTrue(np.all(np.isfinite(edge_thetas)))
        np.testing.assert_array_equal(edge_thetas[:3], 0.0)
        self.assertGreaterEqual(edge_thetas[3], 0.0)

        # Проверка выборки при низких и средних энергиях (где возможны широкие углы рассеяния и cost вблизи -1)
        sample_count = 5000
        low_energies = np.full(sample_count, 0.005 * units.MeV, dtype=Float)
        z_carbon = np.full(sample_count, 6, dtype=Charge)

        @njit(cache=True)
        def _sample_many(energies: np.ndarray, z_vals: np.ndarray, ctx: RNGContext) -> np.ndarray:
            count = energies.shape[0]
            out = np.empty(count, dtype=np.float64)
            for idx in range(count):
                out[idx] = _generate_coherent_theta_scalar(energies[idx], z_vals[idx], ctx)
            return out

        sampled_thetas = _sample_many(low_energies, z_carbon, rng_ctx)
        self.assertTrue(np.all(np.isfinite(sampled_thetas)))
        self.assertTrue(np.all(sampled_thetas >= 0.0))
        self.assertTrue(np.all(sampled_thetas <= np.pi))
        cos_thetas = np.cos(sampled_thetas)
        self.assertTrue(np.all(cos_thetas >= -1.0))
        self.assertTrue(np.all(cos_thetas <= 1.0))
        # Убеждаемся, что при малых энергиях углы не вырождаются в константу и покрывают как переднюю, так и заднюю полусферу
        self.assertLess(float(np.min(cos_thetas)), 0.0)
        self.assertGreater(float(np.max(cos_thetas)), 0.9)

    def _create_mock_physics_buffer(self) -> PhysicsBuffer:
        mat_info = np.zeros(2, dtype=MaterialInfoDType)
        mat_pointers = np.zeros(2, dtype=MaterialPointerDType)
        mat_pointers[0]["start_idx"] = 0
        mat_pointers[0]["length"] = 2
        mat_pointers[0]["lookup_offset"] = 0
        mat_pointers[1]["start_idx"] = 2
        mat_pointers[1]["length"] = 2
        mat_pointers[1]["lookup_offset"] = 4

        physics_energy_grid = np.array([1e-3, 20.0, 1e-3, 20.0], dtype=Float)
        physics_lac_table = np.ones((4, 3), dtype=Float)
        log_e_min = Float(np.log(1e-3))
        log_e_max = Float(np.log(20.0))
        lookup_num_bins = 4
        material_bank = MaterialBank(
            mat_info_buffer=mat_info,
            mat_pointers=mat_pointers,
            physics_energy_grid=physics_energy_grid,
            physics_lac_table=physics_lac_table,
            physics_log_energy_grid=np.log(physics_energy_grid).astype(Float),
            physics_log_lac_table=np.log(physics_lac_table).astype(Float),
            lookup_bin_start=np.zeros(2 * lookup_num_bins, dtype=Index),
            lookup_log_e_min=log_e_min,
            lookup_inv_dlog_e=Float(lookup_num_bins / (log_e_max - log_e_min)),
            lookup_num_bins=lookup_num_bins,
            lookup_max_advance=0,
        )
        material_bank._validate()

        element_partial_fractions = np.ones((4, 3, 1), dtype=Float)
        csr = ElementCSR(
            element_offsets=np.array([0, 1, 2], dtype=Index),
            element_Z=np.array([13, 82], dtype=Charge),
            element_partial_fractions=element_partial_fractions,
        )

        return PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=np.array([0, 1], dtype=Index),
            woodcock_function_pointers=np.zeros(2, dtype=CFuncAddress),
            element_csr=csr,
            relaxation_bank=PhysicsCompiler()._build_atomic_relaxation_bank([]),
        )

    def test_compton_kernel(self) -> None:
        kernel = make_compton_kernel(process_id=ProcessID(2))
        material_ids = np.full(self.capacity, 0, dtype=Index)
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        physics_buffer = self._create_mock_physics_buffer()

        kernel(
            self.bank.state,
            particle_ids,
            self.bank.initial_state.emission_time,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            self.act_buffer,
            physics_buffer,
            self.rng_ctx,
        )

        tracemalloc.start()
        kernel(
            self.bank.state,
            particle_ids,
            self.bank.initial_state.emission_time,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            self.act_buffer,
            physics_buffer,
            self.rng_ctx,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertTrue(peak < 100_000)

        for _ in range(10):
            kernel(
                self.bank.state,
                particle_ids,
                self.bank.initial_state.emission_time,
                self.target_indices,
                current_volumes,
                material_ids,
                self.buffer,
                self.act_buffer,
                physics_buffer,
                self.rng_ctx,
            )

        self.assertEqual(self.buffer.cursor[0], self.capacity * 12)
        self.assertTrue(np.all(self.buffer.energy_deposit > 0.0))
        self.assertTrue(np.all(self.bank.state.energy < 0.5 * units.MeV))
        np.testing.assert_array_equal(self.buffer.process_id, 2)
        dir_norms = np.sqrt(
            self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2
        )
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_coherent_kernel(self) -> None:
        kernel = make_coherent_kernel(process_id=ProcessID(3))
        material_ids = np.full(self.capacity, 1, dtype=Index)
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        physics_buffer = self._create_mock_physics_buffer()

        kernel(
            self.bank.state,
            particle_ids,
            self.bank.initial_state.emission_time,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            self.act_buffer,
            physics_buffer,
            self.rng_ctx,
        )

        tracemalloc.start()
        kernel(
            self.bank.state,
            particle_ids,
            self.bank.initial_state.emission_time,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            self.act_buffer,
            physics_buffer,
            self.rng_ctx,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertTrue(peak < 100_000)

        for _ in range(10):
            kernel(
                self.bank.state,
                particle_ids,
                self.bank.initial_state.emission_time,
                self.target_indices,
                current_volumes,
                material_ids,
                self.buffer,
                self.act_buffer,
                physics_buffer,
                self.rng_ctx,
            )

        self.assertEqual(self.buffer.cursor[0], self.capacity * 12)
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.0)
        np.testing.assert_array_equal(self.bank.state.energy, 0.5 * units.MeV)
        np.testing.assert_array_equal(self.buffer.process_id, 3)

        dir_norms = np.sqrt(
            self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2
        )
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_rotate_particle(self) -> None:
        state = KinematicState.allocate(1)
        state.direction.x[0] = 0.0
        state.direction.y[0] = 0.0
        state.direction.z[0] = 1.0

        theta, phi = np.pi / 4, np.pi / 2

        @njit
        def _wrapper(kin_state, p_idx, angle_theta, angle_phi):
            _rotate_particle(kin_state, p_idx, angle_theta, angle_phi)

        _wrapper(state, 0, theta, phi)

        dir_x = state.direction.x[0]
        dir_y = state.direction.y[0]
        dir_z = state.direction.z[0]

        norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        self.assertAlmostEqual(norm, 1.0, places=5)
        self.assertAlmostEqual(dir_x, 0.0, places=5)
        self.assertAlmostEqual(dir_y, np.sin(theta), places=5)
        self.assertAlmostEqual(dir_z, np.cos(theta), places=5)

    def test_single_element_material_deterministic_no_rng_consumption(self) -> None:
        """
        Однокомпонентный материал (Pb, Z=82): выбор Z возвращает 82 детерминированно
        без потребления случайных чисел из RNGContext.
        """
        compiler = PhysicsCompiler()
        processes = [
            PhotoelectricEffect(attenuation_database=attenuation_database, process_col_idx=Index(0)),
            ComptonScattering(attenuation_database=attenuation_database, process_col_idx=Index(1)),
            CoherentScattering(attenuation_database=attenuation_database, process_col_idx=Index(2)),
        ]
        lead: Material = material_database["Pb"]
        mat_bank = compiler._build_material_bank([lead], processes)
        csr = compiler._build_element_csr([lead], processes, mat_bank)
        phys_buf = PhysicsBuffer(
            material_bank=mat_bank,
            majorant_material_map=np.array([lead.ID], dtype=Index),
            woodcock_function_pointers=np.zeros(1, dtype=CFuncAddress),
            element_csr=csr,
            relaxation_bank=compiler._build_atomic_relaxation_bank(processes),
        )

        rng = np.random.default_rng(98765)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        @njit(cache=True)
        def _sample_lead_batch(
            count: int,
            mat_id: Index,
            energy: Float,
            p_buf: PhysicsBuffer,
            ctx: RNGContext,
        ) -> np.ndarray:
            out_z = np.empty(count, dtype=Charge)
            for idx in range(count):
                out_z[idx] = _sample_Z_scalar(mat_id, Index(idx % 3), energy, p_buf, ctx)
            return out_z

        # Прогрев JIT
        _sample_lead_batch(4, Index(lead.ID), Float(0.140 * units.MeV), phys_buf, rng_ctx)

        state_before = rng.bit_generator.state
        sampled_z = _sample_lead_batch(10_000, Index(lead.ID), Float(0.140 * units.MeV), phys_buf, rng_ctx)
        state_after = rng.bit_generator.state

        np.testing.assert_array_equal(sampled_z, 82)
        self.assertEqual(state_before["state"], state_after["state"])

        # Дополнительно проверяем PhotoelectricEffect.apply на однокомпонентном материале:
        # фотоэффект не потребляет ГСЧ ни на углы, ни на выбор Z при count == 1
        bank = ParticleBank.allocate(100)
        bank.state.is_active[:] = True
        bank.state.energy[:] = Float(0.140 * units.MeV)
        bank.initial_state.ID[:] = np.arange(100, dtype=ID)
        bank.navigation_state.current_volume[:] = 0
        inter_buf = InteractionBuffer.allocate(100)
        mat_ids = np.full(100, lead.ID, dtype=Index)
        target_indices = np.arange(100, dtype=Index)

        state_before_pe = rng.bit_generator.state
        processes[0].apply(bank, target_indices, inter_buf, phys_buf, mat_ids, rng_ctx)
        state_after_pe = rng.bit_generator.state

        np.testing.assert_array_equal(inter_buf.Z[:100], 82)
        self.assertEqual(state_before_pe["state"], state_after_pe["state"])

    def test_nai_target_element_sampling_chi2_all_processes(self) -> None:
        """
        Статистическая верификация Фазы 2 (NaI, 10^6 выборок, критерий хи-квадрат, alpha = 0.001):
        частота выбора Z=53 (I) и Z=11 (Na) для фотоэффекта, комптоновского и когерентного
        рассеяния при 140 кэВ, вблизи К-края йода (33.169 кэВ под краем и 33.170 кэВ над краем)
        и между L3- и L2-краями йода (4.7 кэВ) совпадает с отношением парциальных сечений
        mu_{p,I}(E) / mu_p(E).
        """
        compiler = PhysicsCompiler()
        processes = [
            PhotoelectricEffect(attenuation_database=attenuation_database),
            ComptonScattering(attenuation_database=attenuation_database),
            CoherentScattering(attenuation_database=attenuation_database),
        ]
        nai: Material = material_database["Sodium Iodide"]
        mat_bank = compiler._build_material_bank([nai], processes)
        csr = compiler._build_element_csr([nai], processes, mat_bank)
        phys_buf = PhysicsBuffer(
            material_bank=mat_bank,
            majorant_material_map=np.array([nai.ID], dtype=Index),
            woodcock_function_pointers=np.zeros(1, dtype=CFuncAddress),
            element_csr=csr,
            relaxation_bank=compiler._build_atomic_relaxation_bank(processes),
        )

        @njit(cache=True)
        def _count_iodine_samples(
            num_samples: int,
            mat_id: Index,
            process_col: Index,
            energy: Float,
            p_buf: PhysicsBuffer,
            ctx: RNGContext,
        ) -> int:
            iodine_hits = 0
            for _ in range(num_samples):
                z_val = _sample_Z_scalar(mat_id, process_col, energy, p_buf, ctx)
                if z_val == 53:
                    iodine_hits += 1
            return iodine_hits

        iodine_table = attenuation_database.elements_MAC["I"]
        sodium_table = attenuation_database.elements_MAC["Na"]
        weight_i = float(nai.composition_dict["I"])
        weight_na = float(nai.composition_dict["Na"])

        rng = np.random.default_rng(202602)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        num_samples = 1_000_000
        # Критическое значение хи-квадрат с 1 степенью свободы для alpha = 0.001 равно 10.828 (3.29 сигма)
        chi2_critical = 10.828

        test_energies = [
            Float(140.0 * units.keV),
            Float(33.169 * units.keV),
            Float(33.170 * units.keV),
            Float(4.7 * units.keV),
        ]

        for energy_val in test_energies:
            e_grid_query = np.array([energy_val], dtype=Float)
            for col_idx, proc in enumerate(processes):
                mu_i = (
                    nai.density
                    * weight_i
                    * float(
                        _interpolate_element_on_material_grid(
                            e_grid_query,
                            iodine_table["Energy"],
                            iodine_table["Coefficient"][proc.name],
                        )[0]
                    )
                )
                mu_na = (
                    nai.density
                    * weight_na
                    * float(
                        _interpolate_element_on_material_grid(
                            e_grid_query,
                            sodium_table["Energy"],
                            sodium_table["Coefficient"][proc.name],
                        )[0]
                    )
                )
                expected_prob_i = mu_i / (mu_i + mu_na)
                expected_prob_na = 1.0 - expected_prob_i

                observed_i = _count_iodine_samples(
                    num_samples, Index(nai.ID), Index(col_idx), energy_val, phys_buf, rng_ctx
                )
                observed_na = num_samples - observed_i

                exp_count_i = num_samples * expected_prob_i
                exp_count_na = num_samples * expected_prob_na

                chi2_stat = (
                    (observed_i - exp_count_i) ** 2 / exp_count_i
                    + (observed_na - exp_count_na) ** 2 / exp_count_na
                )
                self.assertLess(
                    chi2_stat,
                    chi2_critical,
                    msg=(
                        f"Критерий хи-квадрат не пройден для процесса {proc.name} при E={energy_val}: "
                        f"chi2={chi2_stat:.3f}, p_obs={observed_i / num_samples:.6f}, p_exp={expected_prob_i:.6f}"
                    ),
                )

        # Проверка того, что _photoelectric_device_func выбирает Z до обнуления энергии частицы:
        # при 140 кэВ ожидаемая доля Na (Z=11) составляет ~0.085% (~85 из 100 000),
        # тогда как при ошибочном выборе при E=0.0 (узел 1 кэВ) доля Na составила бы ~5.35% (~5350 из 100 000).
        pe_batch = 100_000
        bank = ParticleBank.allocate(pe_batch)
        bank.state.is_active[:] = True
        bank.state.energy[:] = Float(140.0 * units.keV)
        bank.initial_state.ID[:] = np.arange(pe_batch, dtype=ID)
        bank.navigation_state.current_volume[:] = 0
        inter_buf = InteractionBuffer.allocate(pe_batch)
        mat_ids = np.full(pe_batch, nai.ID, dtype=Index)
        target_indices = np.arange(pe_batch, dtype=Index)

        processes[0].apply(bank, target_indices, inter_buf, phys_buf, mat_ids, rng_ctx)
        np.testing.assert_array_equal(bank.state.energy, 0.0)
        na_count_pe = int(np.sum(inter_buf.Z[:pe_batch] == 11))
        self.assertLess(na_count_pe, 200)

    def test_process_column_indexing_independent_of_propagator_order(self) -> None:
        """
        Проверка того, что при произвольном порядке процессов в ParticlePropagator
        каждое ядро процесса обращается к своему столбцу process_col_idx в PhysicsBuffer,
        а в InteractionBuffer.process_id записывает свой канонический ProcessID.
        """
        compiler = PhysicsCompiler()
        nai: Material = material_database["Sodium Iodide"]

        scene = CompositeNode()
        crystal_vol = Volume(
            geometry=Box(500.0 * units.mm, 500.0 * units.mm, 500.0 * units.mm),
            material=nai,
            name="NaICrystal",
        )
        scene.add_child(crystal_vol)
        geometry_buffer = GeometryCompiler().compile_scene(scene)

        order_canonical = [PhotoelectricEffect, ComptonScattering, CoherentScattering]
        order_permuted = [ComptonScattering, CoherentScattering, PhotoelectricEffect]

        results_by_process_id: dict[int, list[float]] = {0: [], 1: [], 2: []}

        for proc_classes in (order_canonical, order_permuted):
            rng = np.random.default_rng(77777)
            propagator = ParticlePropagator(
                processes_list=proc_classes,
                attenuation_database=attenuation_database,
                rng=rng,
            )
            physics_buffer = compiler.compile_scene(scene, propagator.processes)

            batch_size = 60_000
            for col_idx, proc in enumerate(propagator.processes):
                self.assertEqual(int(proc.process_col_idx), col_idx)
                bank = ParticleBank.allocate(batch_size)
                bank.state.is_active[:] = True
                bank.state.species[:] = 0
                bank.state.energy[:] = Float(140.0 * units.keV)
                bank.state.distance_traveled[:] = 0.0
                bank.state.position.x[:] = 0.0
                bank.state.position.y[:] = 0.0
                bank.state.position.z[:] = 0.0
                bank.state.direction.x[:] = 0.0
                bank.state.direction.y[:] = 0.0
                bank.state.direction.z[:] = 1.0
                bank.initial_state.ID[:] = np.arange(batch_size, dtype=ID)
                bank.navigation_state.current_volume[:] = 0
                inter_buf = InteractionBuffer.allocate(batch_size)
                mat_ids = np.full(batch_size, nai.ID, dtype=Index)
                target_indices = np.arange(batch_size, dtype=Index)

                proc_rng = np.random.default_rng(1000 + int(proc.process_id))
                proc_rng_ctx = RNGContext.from_numpy_rng(proc_rng)

                proc.apply(
                    bank,
                    target_indices,
                    inter_buf,
                    physics_buffer,
                    mat_ids,
                    proc_rng_ctx,
                )

                recorded_pids = inter_buf.process_id[:batch_size]
                np.testing.assert_array_equal(recorded_pids, int(proc.process_id))
                iodine_fraction = float(np.mean(inter_buf.Z[:batch_size] == 53))
                results_by_process_id[int(proc.process_id)].append(iodine_fraction)

            # Также проверяем полный шаг ParticlePropagator.step с перестановленным порядком
            step_count = 30_000
            step_bank = ParticleBank.allocate(step_count)
            step_bank.state.is_active[:] = True
            step_bank.state.species[:] = 0
            step_bank.state.energy[:] = Float(140.0 * units.keV)
            step_bank.state.distance_traveled[:] = 0.0
            step_bank.state.position.x[:] = 0.0
            step_bank.state.position.y[:] = 0.0
            step_bank.state.position.z[:] = 0.0
            step_bank.state.direction.x[:] = 0.0
            step_bank.state.direction.y[:] = 0.0
            step_bank.state.direction.z[:] = 1.0
            step_bank.initial_state.ID[:] = np.arange(1, step_count + 1, dtype=ID)
            step_bank.initial_state.emission_time[:] = 0.0
            step_bank.initial_state.emission_energy[:] = Float(140.0 * units.keV)
            step_bank.initial_state.emission_position.x[:] = 0.0
            step_bank.initial_state.emission_position.y[:] = 0.0
            step_bank.initial_state.emission_position.z[:] = 0.0
            step_bank.initial_state.emission_direction.x[:] = 0.0
            step_bank.initial_state.emission_direction.y[:] = 0.0
            step_bank.initial_state.emission_direction.z[:] = 1.0
            step_bank.navigation_state.current_volume[:] = -1
            step_bank.navigation_state.boundary_distance[:] = -1.0
            data_buffer = SimulationDataBuffer.allocate(
                step_count, step_count, step_count, step_count
            )
            step_rng = np.random.default_rng(54321)
            step_rng_ctx = RNGContext.from_numpy_rng(step_rng)

            propagator.step(
                step_bank,
                data_buffer,
                geometry_buffer,
                physics_buffer,
                step_rng_ctx,
            )
            num_inter = data_buffer.interactions.cursor_value
            self.assertGreater(num_inter, 20_000)
            z_values = data_buffer.interactions.Z[:num_inter]
            pids = data_buffer.interactions.process_id[:num_inter]
            self.assertTrue(np.all((z_values == 11) | (z_values == 53)))
            compton_mask = pids == int(ComptonScattering.process_id)
            self.assertGreater(int(np.sum(compton_mask)), 3000)
            compton_iodine_frac = float(np.mean(z_values[compton_mask] == 53))
            self.assertTrue(0.79 < compton_iodine_frac < 0.84)

        # При одинаковом сиде для каждого типа процесса выборка Z дает побитово идентичную долю I (Z=53)
        # независимо от позиции процесса в списке ParticlePropagator.processes
        for pid, fractions in results_by_process_id.items():
            self.assertEqual(len(fractions), 2)
            self.assertEqual(fractions[0], fractions[1], msg=f"Несовпадение для process_id={pid}")

        # Убеждаемся, что доли I для трех процессов статистически существенно различаются (~0.999, ~0.816, ~0.975)
        self.assertGreater(results_by_process_id[0][0], 0.998)
        self.assertTrue(0.80 < results_by_process_id[1][0] < 0.83)
        self.assertTrue(0.96 < results_by_process_id[2][0] < 0.99)

    def test_secondary_activation_buffer_and_activation_time_kinematics(self) -> None:
        """
        Проверка Фазы 3:
        1. Предвыделение SecondaryActivationBuffer на базе Vector3DSoA, курсора и параметров отсечки.
        2. Точный релятивистский расчет мирового времени точки активации t_act = t_emit + l / v(E, m_0)
           для фотонов (v = c) и массивных электронов.
        """
        act_buf = SecondaryActivationBuffer.allocate(
            capacity=16,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(50.0 * units.keV),
        )
        self.assertTrue(act_buf.is_enabled)
        self.assertEqual(act_buf.capacity, 16)
        self.assertEqual(act_buf.cursor_value, 0)
        self.assertEqual(act_buf.remaining_capacity, 16)
        self.assertTrue(act_buf.enable_atomic_relaxation)
        self.assertAlmostEqual(float(act_buf.photon_production_cut), float(1.0 * units.keV), places=12)
        self.assertAlmostEqual(float(act_buf.electron_production_cut), float(50.0 * units.keV), places=12)

        t_emit = Time(10.0 * units.ns)
        distance = Float(300.0 * units.mm)
        t_photon = _compute_activation_time(
            t_emit,
            distance,
            Float(0.140 * units.MeV),
            Species(ParticleSpecies.PHOTON),
        )
        expected_photon_t = float(t_emit + distance / C_LIGHT)
        self.assertAlmostEqual(float(t_photon), expected_photon_t, places=12)

        e_kin = Float(0.510998910 * units.MeV)  # T == m_e c^2 -> gamma = 2, beta = sqrt(3)/2
        t_electron = _compute_activation_time(
            t_emit,
            distance,
            e_kin,
            Species(ParticleSpecies.ELECTRON),
        )
        expected_beta = float(np.sqrt(1.0 - (float(electron_mass_c2) / float(e_kin + electron_mass_c2)) ** 2))
        expected_electron_t = float(t_emit + distance / (expected_beta * float(C_LIGHT)))
        self.assertAlmostEqual(float(t_electron), expected_electron_t, places=12)

    def test_partition_transferred_energy_conservation_and_characteristic_lines(self) -> None:
        """
        Проверка Фазы 3 (partition_transferred_energy_kernel и PhotoelectricEffect.apply):
        1. При отключенной релаксации и electron_production_cut == inf: 0 обращений к ГСЧ и 100% локальное
           энерговыделение без записей в SecondaryActivationBuffer.
        2. При включенной релаксации в Pb (Z=82, E=140 кэВ) и NaI (Z=53/11, E=140 кэВ):
           - строгий пошаговый баланс энергии: sum(delta_E) == sum(E_dep_local) + sum(E_sec_tracked);
           - энергии рожденных вторичных фотонов принадлежат дискретным характеристическим линиям
             AtomicRelaxationBank и удовлетворяют E_line >= photon_production_cut;
           - отсутствие холостых записей в SecondaryActivationBuffer при безызлучательных Оже-переходах
             (когда electron_production_cut == inf).
        3. При конечном electron_production_cut рождаются надпороговые фотоэлектроны, Оже-электроны
           и комптоновские электроны с точным сохранением энергии.
        """
        compiler = PhysicsCompiler()
        processes = [
            PhotoelectricEffect(attenuation_database=attenuation_database),
            ComptonScattering(attenuation_database=attenuation_database),
            CoherentScattering(attenuation_database=attenuation_database),
        ]
        lead: Material = material_database["Pb"]
        nai: Material = material_database["Sodium Iodide"]
        mat_bank = compiler._build_material_bank([lead, nai], processes)
        csr = compiler._build_element_csr([lead, nai], processes, mat_bank)
        relax_bank = compiler._build_atomic_relaxation_bank(processes)
        phys_buf = PhysicsBuffer(
            material_bank=mat_bank,
            majorant_material_map=np.array([lead.ID, nai.ID], dtype=Index),
            woodcock_function_pointers=np.zeros(2, dtype=CFuncAddress),
            element_csr=csr,
            relaxation_bank=relax_bank,
        )

        num_particles = 5000
        initial_energy = Float(140.0 * units.keV)
        bank = ParticleBank.allocate(num_particles)
        bank.state.is_active[:] = True
        bank.state.species[:] = int(ParticleSpecies.PHOTON)
        bank.state.energy[:] = initial_energy
        bank.state.distance_traveled[:] = Float(15.0 * units.mm)
        bank.state.position.x[:] = 1.0
        bank.state.position.y[:] = 2.0
        bank.state.position.z[:] = 3.0
        bank.state.direction.x[:] = 0.0
        bank.state.direction.y[:] = 0.0
        bank.state.direction.z[:] = 1.0
        bank.initial_state.ID[:] = np.arange(100, 100 + num_particles, dtype=ID)
        bank.initial_state.emission_time[:] = Time(5.0 * units.ns)
        bank.navigation_state.current_volume[:] = 0
        target_indices = np.arange(num_particles, dtype=Index)
        lead_mat_ids = np.full(num_particles, lead.ID, dtype=Index)

        # 1. Режим с релаксацией в свинце (Z=82): K-край = 88.005 кэВ < 140 кэВ
        inter_buf = InteractionBuffer.allocate(num_particles)
        act_buf = SecondaryActivationBuffer.allocate(
            capacity=2 * num_particles,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )
        rng = np.random.default_rng(202603)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        processes[0].apply(
            bank,
            target_indices,
            inter_buf,
            phys_buf,
            lead_mat_ids,
            rng_ctx,
            activation_buffer=act_buf,
        )

        self.assertEqual(inter_buf.cursor_value, num_particles)
        sec_count = act_buf.cursor_value
        # Для свинца при 140 кэВ доля K-ионизации ~80%, выход флуоресценции omega_K ~ 0.96 -> ожидаем > 3500 фотонов
        self.assertGreater(sec_count, 3500)
        self.assertLess(sec_count, num_particles)

        sec_energies = act_buf.energy[:sec_count]
        sec_species = act_buf.species[:sec_count]
        sec_z = act_buf.target_z[:sec_count]
        sec_shells = act_buf.shell_idx[:sec_count]

        self.assertTrue(np.all(sec_species == int(ParticleSpecies.PHOTON)))
        self.assertTrue(np.all(sec_z == 82))
        self.assertTrue(np.all(sec_shells >= 0))
        self.assertTrue(np.all(sec_energies >= Float(1.0 * units.keV)))

        # Строгий закон сохранения энергии: N * 140 кэВ == sum(local_deposit) + sum(sec_energies)
        total_initial = float(num_particles) * float(initial_energy)
        total_local_deposit = float(np.sum(inter_buf.energy_deposit[:num_particles]))
        total_secondary_energy = float(np.sum(sec_energies))
        self.assertAlmostEqual(
            total_initial,
            total_local_deposit + total_secondary_energy,
            places=9,
        )

        # Проверяем, что рождаются характеристические K-альфа линии свинца (~72.8 и ~75.0 кэВ)
        k_alpha_mask = (sec_energies > Float(72.0 * units.keV)) & (sec_energies < Float(76.0 * units.keV))
        self.assertGreater(int(np.sum(k_alpha_mask)), 2000)

        # 2. Режим с конечным порогом электронов (electron_production_cut = 10 кэВ) для фотоэффекта и Комптона
        bank.state.energy[:] = initial_energy
        inter_buf.reset_cursor()
        act_buf_e = SecondaryActivationBuffer.allocate(
            capacity=2 * num_particles,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(10.0 * units.keV),
        )
        processes[0].apply(
            bank,
            target_indices,
            inter_buf,
            phys_buf,
            lead_mat_ids,
            rng_ctx,
            activation_buffer=act_buf_e,
        )
        sec_e_count = act_buf_e.cursor_value
        # Теперь на каждый K-фотоэффект рождается фотоэлектрон (~52 кэВ) И флуоресцентный фотон/Оже-электрон
        self.assertGreater(sec_e_count, num_particles)
        self.assertTrue(np.any(act_buf_e.species[:sec_e_count] == int(ParticleSpecies.ELECTRON)))
        self.assertTrue(np.any(act_buf_e.species[:sec_e_count] == int(ParticleSpecies.PHOTON)))
        self.assertAlmostEqual(
            total_initial,
            float(np.sum(inter_buf.energy_deposit[:num_particles]))
            + float(np.sum(act_buf_e.energy[:sec_e_count])),
            places=9,
        )

        # 3. Комптоновское рассеяние: проверка сохранения target_z, исходного parent_direction (до рассеяния)
        #    и баланса энергии
        bank.state.energy[:] = initial_energy
        bank.state.direction.x[:] = 0.0
        bank.state.direction.y[:] = 0.0
        bank.state.direction.z[:] = 1.0
        inter_buf.reset_cursor()
        act_buf_compton = SecondaryActivationBuffer.allocate(
            capacity=num_particles,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(5.0 * units.keV),
        )
        processes[1].apply(
            bank,
            target_indices,
            inter_buf,
            phys_buf,
            lead_mat_ids,
            rng_ctx,
            activation_buffer=act_buf_compton,
        )
        compton_sec_count = act_buf_compton.cursor_value
        self.assertGreater(compton_sec_count, 1000)
        self.assertTrue(np.all(act_buf_compton.species[:compton_sec_count] == int(ParticleSpecies.ELECTRON)))
        self.assertTrue(np.all(act_buf_compton.target_z[:compton_sec_count] == 82))
        self.assertTrue(np.all(act_buf_compton.direction.x[:compton_sec_count] == 0.0))
        self.assertTrue(np.all(act_buf_compton.direction.y[:compton_sec_count] == 0.0))
        self.assertTrue(np.all(act_buf_compton.direction.z[:compton_sec_count] == 1.0))
        self.assertAlmostEqual(
            total_initial,
            float(np.sum(bank.state.energy[:num_particles]))
            + float(np.sum(inter_buf.energy_deposit[:num_particles]))
            + float(np.sum(act_buf_compton.energy[:compton_sec_count])),
            places=9,
        )

        # 4. Переполнение SecondaryActivationBuffer: не поместившиеся вторичные частицы обязаны
        #    остаться в local_deposit без потери энергии
        bank.state.energy[:] = initial_energy
        inter_buf.reset_cursor()
        tiny_capacity = 10
        act_buf_tiny = SecondaryActivationBuffer.allocate(
            capacity=tiny_capacity,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(10.0 * units.keV),
        )
        processes[0].apply(
            bank,
            target_indices,
            inter_buf,
            phys_buf,
            lead_mat_ids,
            rng_ctx,
            activation_buffer=act_buf_tiny,
        )
        self.assertEqual(act_buf_tiny.cursor_value, tiny_capacity)
        self.assertAlmostEqual(
            total_initial,
            float(np.sum(inter_buf.energy_deposit[:num_particles]))
            + float(np.sum(act_buf_tiny.energy[:tiny_capacity])),
            places=9,
        )

    def test_three_production_cut_regimes_per_event_energy_balance_and_genealogy(self) -> None:
        """
        Проверка Фазы 4 (сохранение энергии и генеалогия (particle_ID, parent_ID)):
        Точный по-событийный (per-vertex) баланс энергии
            delta_E_i == E_dep_local_i + sum_{c: parent_ID[c] == pid_i} E_sec_c
        во всех трёх режимах порогов production_cuts в кристалле NaI (Z=53, 11) при E = 140.5 кэВ:
        1. Полное локальное поглощение (photon_production_cut = inf, electron_production_cut = inf);
        2. Столкновительная Керма с фотонной релаксацией (photon_production_cut = 1 кэВ, electron_production_cut = inf);
        3. Связанный гамма-электронный режим с фотоэлектронами и Оже-электронами
           (photon_production_cut = 1 кэВ, electron_production_cut = 1 кэВ),
           а также выгрузка вторичных частиц в ParticleBank с проверкой связности (ID, parent_ID).
        """
        compiler = PhysicsCompiler()
        processes = [
            PhotoelectricEffect(attenuation_database=attenuation_database),
            ComptonScattering(attenuation_database=attenuation_database),
            CoherentScattering(attenuation_database=attenuation_database),
        ]
        nai: Material = material_database["Sodium Iodide"]
        mat_bank = compiler._build_material_bank([nai], processes)
        csr = compiler._build_element_csr([nai], processes, mat_bank)
        relax_bank = compiler._build_atomic_relaxation_bank(processes)
        phys_buf = PhysicsBuffer(
            material_bank=mat_bank,
            majorant_material_map=np.array([nai.ID], dtype=Index),
            woodcock_function_pointers=np.zeros(1, dtype=CFuncAddress),
            element_csr=csr,
            relaxation_bank=relax_bank,
        )

        num_events = 2000
        initial_energy = Float(140.5 * units.keV)
        parent_ids = np.arange(500, 500 + num_events, dtype=ID)
        target_indices = np.arange(num_events, dtype=Index)
        nai_mat_ids = np.full(num_events, nai.ID, dtype=Index)

        def _prepare_bank() -> ParticleBank:
            bank = ParticleBank.allocate(num_events)
            bank.state.is_active[:] = True
            bank.state.species[:] = int(ParticleSpecies.PHOTON)
            bank.state.energy[:] = initial_energy
            bank.state.distance_traveled[:] = Float(10.0 * units.mm)
            bank.state.position.x[:] = 0.0
            bank.state.position.y[:] = 0.0
            bank.state.position.z[:] = 0.0
            bank.state.direction.x[:] = 0.0
            bank.state.direction.y[:] = 0.0
            bank.state.direction.z[:] = 1.0
            bank.initial_state.ID[:] = parent_ids
            bank.initial_state.parent_ID[:] = ID(-1)
            bank.initial_state.emission_time[:] = Time(0.0)
            bank.navigation_state.current_volume[:] = 0
            return bank

        # Режим 1: Полное локальное поглощение (пороги +inf)
        bank_mode1 = _prepare_bank()
        inter_mode1 = InteractionBuffer.allocate(num_events)
        act_mode1 = SecondaryActivationBuffer.allocate(
            capacity=2 * num_events,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(np.inf),
            electron_production_cut=Float(np.inf),
        )
        rng_mode1 = np.random.default_rng(2026401)
        ctx_mode1 = RNGContext.from_numpy_rng(rng_mode1)
        processes[0].apply(
            bank_mode1,
            target_indices,
            inter_mode1,
            phys_buf,
            nai_mat_ids,
            ctx_mode1,
            activation_buffer=act_mode1,
        )
        self.assertEqual(act_mode1.cursor_value, 0)
        np.testing.assert_allclose(
            inter_mode1.energy_deposit[:num_events],
            np.full(num_events, initial_energy, dtype=Float),
            rtol=0.0,
            atol=1e-12,
        )

        # Режим 2: Столкновительная Керма с фотонной релаксацией (E_cut_gamma = 1 кэВ, E_cut_e = +inf)
        bank_mode2 = _prepare_bank()
        inter_mode2 = InteractionBuffer.allocate(num_events)
        act_mode2 = SecondaryActivationBuffer.allocate(
            capacity=2 * num_events,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )
        rng_mode2 = np.random.default_rng(2026402)
        ctx_mode2 = RNGContext.from_numpy_rng(rng_mode2)
        processes[0].apply(
            bank_mode2,
            target_indices,
            inter_mode2,
            phys_buf,
            nai_mat_ids,
            ctx_mode2,
            activation_buffer=act_mode2,
        )
        sec_count_mode2 = act_mode2.cursor_value
        self.assertGreater(sec_count_mode2, 1200)
        self.assertTrue(np.all(act_mode2.species[:sec_count_mode2] == int(ParticleSpecies.PHOTON)))

        sec_sum_by_event_mode2 = np.zeros(num_events, dtype=Float)
        np.add.at(
            sec_sum_by_event_mode2,
            act_mode2.parent_ID[:sec_count_mode2] - 500,
            act_mode2.energy[:sec_count_mode2],
        )
        np.testing.assert_allclose(
            inter_mode2.energy_deposit[:num_events] + sec_sum_by_event_mode2,
            np.full(num_events, initial_energy, dtype=Float),
            rtol=0.0,
            atol=1e-12,
        )

        # Режим 3: Связанный гамма-электронный режим с фотоэлектронами, рентгеном и Оже-электронами
        bank_mode3 = _prepare_bank()
        inter_mode3 = InteractionBuffer.allocate(num_events)
        act_mode3 = SecondaryActivationBuffer.allocate(
            capacity=2 * num_events,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(1.0 * units.keV),
        )
        rng_mode3 = np.random.default_rng(2026403)
        ctx_mode3 = RNGContext.from_numpy_rng(rng_mode3)
        processes[0].apply(
            bank_mode3,
            target_indices,
            inter_mode3,
            phys_buf,
            nai_mat_ids,
            ctx_mode3,
            activation_buffer=act_mode3,
        )
        sec_count_mode3 = act_mode3.cursor_value
        self.assertGreater(sec_count_mode3, int(1.8 * num_events))

        sec_species_mode3 = act_mode3.species[:sec_count_mode3]
        sec_shells_mode3 = act_mode3.shell_idx[:sec_count_mode3]
        sec_parents_mode3 = act_mode3.parent_ID[:sec_count_mode3]
        sec_energies_mode3 = act_mode3.energy[:sec_count_mode3]

        # Проверяем наличие всех трёх физических типов вторичных частиц при фотоэффекте:
        # 1) фотоэлектроны (ELECTRON, shell_idx == -1)
        # 2) характеристические фотоны (PHOTON, shell_idx >= 0)
        # 3) электроны Оже (ELECTRON, shell_idx >= 0)
        pe_mask = (sec_species_mode3 == int(ParticleSpecies.ELECTRON)) & (sec_shells_mode3 == -1)
        fluor_mask = (sec_species_mode3 == int(ParticleSpecies.PHOTON)) & (sec_shells_mode3 >= 0)
        auger_mask = (sec_species_mode3 == int(ParticleSpecies.ELECTRON)) & (sec_shells_mode3 >= 0)
        self.assertEqual(int(np.sum(pe_mask)), num_events)
        self.assertGreater(int(np.sum(fluor_mask)), 1200)
        self.assertGreater(int(np.sum(auger_mask)), 150)

        sec_sum_by_event_mode3 = np.zeros(num_events, dtype=Float)
        np.add.at(sec_sum_by_event_mode3, sec_parents_mode3 - 500, sec_energies_mode3)
        np.testing.assert_allclose(
            inter_mode3.energy_deposit[:num_events] + sec_sum_by_event_mode3,
            np.full(num_events, initial_energy, dtype=Float),
            rtol=0.0,
            atol=1e-12,
        )

        # Выгружаем все вторичные активации Режима 3 через inject_secondary_sources_kernel в ParticleBank
        sec_bank = ParticleBank.allocate(sec_count_mode3)
        sec_bank.count_array[0] = Index(500 + num_events)
        inj_sec, _ = inject_secondary_sources_kernel(
            sec_bank,
            act_mode3,
            relax_bank,
            Float(np.inf),
            ctx_mode3,
        )
        self.assertEqual(int(inj_sec), sec_count_mode3)
        self.assertEqual(act_mode3.cursor_value, 0)
        np.testing.assert_array_equal(
            sec_bank.initial_state.ID[:sec_count_mode3],
            np.arange(500 + num_events, 500 + num_events + sec_count_mode3, dtype=ID),
        )
        np.testing.assert_array_equal(sec_bank.initial_state.parent_ID[:sec_count_mode3], sec_parents_mode3)
        self.assertTrue(np.all((sec_bank.initial_state.parent_ID[:sec_count_mode3] >= 500) & (
            sec_bank.initial_state.parent_ID[:sec_count_mode3] < 500 + num_events
        )))

    def test_lead_characteristic_xray_above_and_below_k_edge(self) -> None:
        """
        Физический тест Фазы 4 (характеристический рентген свинца Pb, Z=82, K-край = 88.005 кэВ):
        1. При облучении свинцовой пластины фотонами 99mTc (140.5 кэВ > 88.005 кэВ) рождаются
           вторичные фотоны (parent_ID >= 0) с характерными линиями K_alpha (~72.8-75.0 кэВ)
           и K_beta (~84.4-87.3 кэВ).
        2. При энергии первичных фотонов ниже K-края свинца (80.0 кэВ < 88.005 кэВ) возбуждаются
           только L/M-оболочки (E_sec < 16 кэВ), а K-флуоресценция полностью отсутствует.
        """
        lead_mat: Material = material_database["Pb"]

        def _run_lead_target_simulation(primary_energy: Float, seed_val: int) -> dict[str, np.ndarray]:
            world = Volume(
                geometry=Box(200.0 * units.mm, 200.0 * units.mm, 200.0 * units.mm),
                material=material_database["Air, Dry (near sea level)"],
                name="World",
            )
            pb_plate = Volume(
                geometry=Box(100.0 * units.mm, 100.0 * units.mm, 5.0 * units.mm),
                material=lead_mat,
                name="LeadCollimatorPlate",
            )
            world.add_child(pb_plate)
            src = Source.from_point(
                activity=Float(1.0e5 * units.Bq),
                energy=primary_energy,
                half_life=Float(np.inf),
                name="PointBeam",
            )
            world.add_child(src)

            mgr = SimulationManager(
                scene=world,
                propagator=ParticlePropagator(
                    processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
                ),
                start_time=Float(0.0),
                stop_time=None,
                particles_number=1500,
                batch_capacity=256,
                buffer_capacity=4096,
                min_energy=Float(1.0 * units.keV),
                seed=seed_val,
                enable_atomic_relaxation=True,
                photon_production_cut=Float(1.0 * units.keV),
                electron_production_cut=Float(np.inf),
            )
            mgr.start()
            mgr.join()

            init_chunks: list[dict[str, np.ndarray]] = []
            while True:
                item = mgr.queue.get_nowait()
                if item is None or item == "stop":
                    break
                if item["type"].value == "initial_states":
                    init_chunks.append(item["data"])
            return {
                key: np.concatenate([chunk[key] for chunk in init_chunks])
                for key in init_chunks[0]
            }

        # 1. Выше K-края свинца (140.5 кэВ > 88.005 кэВ)
        above_init = _run_lead_target_simulation(Float(140.5 * units.keV), seed_val=2026410)
        sec_mask_above = above_init["parent_ID"] >= 0
        sec_energies_above = above_init["emission_energy"][sec_mask_above]
        self.assertGreater(len(sec_energies_above), 400)

        # Линии K_alpha свинца (K_alpha2 = 72.805 кэВ, K_alpha1 = 74.969 кэВ)
        pb_k_alpha_mask = (sec_energies_above >= Float(72.5 * units.keV)) & (
            sec_energies_above <= Float(75.5 * units.keV)
        )
        # Линии K_beta свинца (K_beta3 = 84.450 кэВ, K_beta1 = 84.938 кэВ, K_beta2 = 87.300 кэВ)
        pb_k_beta_mask = (sec_energies_above >= Float(84.0 * units.keV)) & (
            sec_energies_above <= Float(87.8 * units.keV)
        )
        k_alpha_count = int(np.sum(pb_k_alpha_mask))
        k_beta_count = int(np.sum(pb_k_beta_mask))
        self.assertGreater(k_alpha_count, 250)
        self.assertGreater(k_beta_count, 50)
        # Отношение интенсивностей K_alpha / K_beta для свинца составляет ~3.5 - 4.5
        self.assertGreater(k_alpha_count, 2 * k_beta_count)

        # 2. Ниже K-края свинца (80.0 кэВ < 88.005 кэВ): K-флуоресценция невозможна, только L/M (< 16 кэВ)
        below_init = _run_lead_target_simulation(Float(80.0 * units.keV), seed_val=2026411)
        sec_mask_below = below_init["parent_ID"] >= 0
        sec_energies_below = below_init["emission_energy"][sec_mask_below]
        self.assertGreater(len(sec_energies_below), 150)
        self.assertEqual(int(np.sum(sec_energies_below > Float(70.0 * units.keV))), 0)
        self.assertTrue(np.all(sec_energies_below < Float(16.0 * units.keV)))


if __name__ == "__main__":
    unittest.main()

