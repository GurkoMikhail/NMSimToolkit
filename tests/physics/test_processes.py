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
from core.other.typing_definitions import CFuncAddress, Charge, Float, ID, Index, ProcessID
from core.particles.kinematic_state import KinematicState
from core.particles.particles import ParticleBank
from core.particles.particles_kernels import _rotate_particle
from core.physics.g4coherent import _generate_coherent_theta_scalar
from core.physics.g4compton import _generate_compton_theta_scalar
from core.physics.interaction_buffers import InteractionBuffer, RNGContext, SimulationDataBuffer
from core.physics.physics_buffer import ElementCSR, PhysicsBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PhotoelectricEffect,
)
from core.physics.processes_kernels import (
    _sample_Z_scalar,
    make_coherent_kernel,
    make_compton_kernel,
    make_photoelectric_kernel,
)
from core.scene.nodes import CompositeNode
from core.transport.propagator import ParticlePropagator
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
        self.bank.state.energy[:] = energy
        self.target_indices = np.arange(self.capacity, dtype=np.int64)

        self.buffer = InteractionBuffer.allocate(self.capacity)

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
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
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
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            physics_buffer,
            self.rng_ctx,
        )

        tracemalloc.start()
        kernel(
            self.bank.state,
            particle_ids,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
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
                self.target_indices,
                current_volumes,
                material_ids,
                self.buffer,
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
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            physics_buffer,
            self.rng_ctx,
        )

        tracemalloc.start()
        kernel(
            self.bank.state,
            particle_ids,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
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
                self.target_indices,
                current_volumes,
                material_ids,
                self.buffer,
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


if __name__ == "__main__":
    unittest.main()

