"""
Сквозной интеграционный тест полного цикла round-trip и верификации эквивалентности
протоколов StepAndShoot и CustomSweep для конфигурации tb161_nema_protocol.yaml.
"""

from copy import deepcopy
import os
import tempfile
import unittest
from pathlib import Path
import yaml
import hepunits as units
import numpy as np

from core.config.yaml_loader import load_simulation_config, load_raw_config
from core.config.builder import SceneBuilder
from core.config.sweep_compiler import SweepCompiler
from core.config.template_binder import ConfigTemplateBinder
from core.config.models import (
    SimulationConfig,
    StepAndShootProtocolConfig,
    CustomSweepProtocolConfig,
)
from core.data.metadata_collector import ProcedureMetadataCollector
from core.scene.nodes import SpatialNode
from core.source.sources import DiscreteSpectrum, Source


class TestRoundtripTB161Protocol(unittest.TestCase):
    """Тестирование сквозного цикла чтения, экспорта и эквивалентности StepAndShoot vs CustomSweep."""

    def setUp(self) -> None:
        self.yaml_path = Path("examples/tb161_nema_protocol.yaml").resolve()
        self.assertTrue(self.yaml_path.exists(), f"Файл {self.yaml_path} должен существовать")

    def test_tb161_full_roundtrip_integrity(self) -> None:
        # 1. Извлечение биндингов из шаблона конфигурации через сервис ConfigTemplateBinder
        raw_dict = load_raw_config(self.yaml_path)
        bindings = ConfigTemplateBinder.extract_bindings(raw_dict)
        self.assertIn("Gantry", bindings)
        self.assertEqual(bindings["Gantry"]["rotation.alpha"], "gantry_angle")
        self.assertIn("data_manager", bindings)
        self.assertEqual(bindings["data_manager"]["filename"], "gantry_angle")
        self.assertIn("simulation_manager", bindings)
        self.assertEqual(bindings["simulation_manager"]["start_time"], "start_time")
        self.assertEqual(bindings["simulation_manager"]["stop_time"], "stop_time")

        # 2. Загрузка валидированной конфигурации с подстановкой начальных значений протокола
        sim_cfg = load_simulation_config(self.yaml_path, resolve_protocol=True)
        self.assertEqual(sim_cfg.pool_size, 24, "pool_size обязан быть 24")
        self.assertIsInstance(
            sim_cfg.protocol,
            StepAndShootProtocolConfig,
            "Протокол должен быть StepAndShootProtocolConfig",
        )
        assert isinstance(sim_cfg.protocol, StepAndShootProtocolConfig)
        self.assertEqual(sim_cfg.protocol.steps, 60)
        self.assertAlmostEqual(float(sim_cfg.protocol.start_angle), 0.0)
        self.assertAlmostEqual(float(sim_cfg.protocol.end_angle), float(180.0 * units.deg))
        self.assertFalse(sim_cfg.protocol.endpoint)
        self.assertEqual(sim_cfg.protocol.rotation_direction, "CW")
        self.assertAlmostEqual(float(sim_cfg.protocol.time_per_step), float(30.0 * units.s), places=4)
        self.assertAlmostEqual(float(sim_cfg.protocol.initial_time), 0.0)
        self.assertTrue(sim_cfg.protocol.sequential_time)
        self.assertIsNone(sim_cfg.protocol.radius)

        # Построение дерева сцены через чистый детерминированный SceneBuilder
        builder = SceneBuilder()
        root_node = builder.build_scene(sim_cfg.scene)

        # 3. Проверка узлов сцены
        phantom_node = None
        source_node = None
        gantry_node = None

        for node in root_node.childs:
            if node.name == "Phantom":
                phantom_node = node
                for child in phantom_node.childs:
                    if isinstance(child, Source):
                        source_node = child
            elif node.name == "Gantry":
                gantry_node = node

        self.assertIsNotNone(phantom_node, "Узел Phantom должен присутствовать в сцене")
        self.assertIsNotNone(source_node, "Узел Source должен присутствовать внутри Phantom")
        self.assertIsNotNone(gantry_node, "Узел Gantry должен присутствовать в сцене")

        # Проверка спектра источника: должно быть ровно 21 линия (Juget et al. 2021/2023)
        assert source_node is not None
        self.assertGreaterEqual(len(source_node.channels), 1)
        spec_obj = source_node.channels[0].spectrum
        assert isinstance(spec_obj, DiscreteSpectrum)
        self.assertEqual(len(spec_obj.lines), 21, "Спектр 161Tb обязан содержать 21 линию")

        # 4. Экспорт сцены обратно во временный YAML-файл с наложением шаблонов макросов
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_yaml_path = os.path.join(tmp_dir, "roundtrip_exported.yaml")
            ConfigTemplateBinder.export_templated_yaml(
                root_node=root_node,
                filepath=out_yaml_path,
                bindings_registry=bindings,
                simulation_manager_cfg=sim_cfg.simulation_manager,
                data_manager_cfg=sim_cfg.data_manager,
                protocol=sim_cfg.protocol,
                pool_size=sim_cfg.pool_size,
                distribution_registry=builder.distribution_registry,
                slots_registry=builder.slots_registry,
            )

            self.assertTrue(os.path.exists(out_yaml_path))

            # Проверка сырого содержимого экспортированного YAML
            with open(out_yaml_path, "r", encoding="utf-8") as file_handle:
                exported_raw = yaml.safe_load(file_handle)

            self.assertEqual(exported_raw["pool_size"], 24, "Экспортированный pool_size обязан быть 24")
            self.assertIn("protocol", exported_raw)
            self.assertEqual(exported_raw["protocol"]["type"], "StepAndShoot")
            self.assertEqual(exported_raw["protocol"]["steps"], 60)
            self.assertFalse(exported_raw["protocol"]["endpoint"])

            # Проверка фантома и mapping
            scene_dict = exported_raw["scene"]
            phantom_dict = None
            gantry_dict = None
            for child_entry in scene_dict.get("children", []):
                if child_entry.get("name") == "Phantom":
                    phantom_dict = child_entry
                elif child_entry.get("name") == "Gantry":
                    gantry_dict = child_entry

            self.assertIsNotNone(phantom_dict)
            self.assertIsNotNone(gantry_dict)
            assert phantom_dict is not None
            assert gantry_dict is not None

            # Проверка, что оригинальный mapping не деградировал до 0.0/1.0
            dist_map = phantom_dict["distribution"]["mapping"]
            self.assertIn(0.04, dist_map)
            self.assertIn(0.15, dist_map)
            self.assertNotIn(0.0, dist_map, "Фантом не должен содержать фиктивный Vacuum 0.0")

            # Проверка источника: активность со строковыми единицами и 21 линия спектра
            source_dict = phantom_dict["children"][0]
            self.assertEqual(source_dict["name"], "Source")
            self.assertIn("MBq", str(source_dict["activity"]), "Активность должна содержать единицу MBq")
            energy_list = source_dict["channels"][0]["spectrum"]["lines"]
            self.assertIsInstance(energy_list, list)
            self.assertEqual(len(energy_list), 21, "Экспортированный спектр обязан содержать 21 линию")
            self.assertIn("keV", str(energy_list[0][0]))

            # Проверка биндинга поворота гантри и времени экспозиции в hepunits
            gantry_rot = gantry_dict["transformations"][0]
            self.assertEqual(gantry_rot["type"], "rotate")
            self.assertEqual(gantry_rot["alpha"], "${gantry_angle}")
            self.assertEqual(exported_raw["simulation_manager"]["start_time"], "${start_time}")
            self.assertEqual(exported_raw["simulation_manager"]["stop_time"], "${stop_time}")
            self.assertEqual(
                exported_raw["data_manager"]["filename"],
                "output_tb161_dual_cam/${gantry_angle deg}_deg.hdf",
            )

            # 5. Повторный импорт через SceneBuilder (Round-trip)
            raw_dict2 = load_raw_config(out_yaml_path)
            bindings2 = ConfigTemplateBinder.extract_bindings(raw_dict2)
            self.assertIn("Gantry", bindings2)
            self.assertEqual(bindings2["Gantry"]["rotation.alpha"], "gantry_angle")

            sim_cfg2 = load_simulation_config(out_yaml_path, resolve_protocol=True)
            builder2 = SceneBuilder()
            root_node2 = builder2.build_scene(sim_cfg2.scene)
            self.assertIsNotNone(root_node2)
            self.assertEqual(sim_cfg2.pool_size, 24)
            self.assertIsInstance(sim_cfg2.protocol, StepAndShootProtocolConfig)
            assert isinstance(sim_cfg2.protocol, StepAndShootProtocolConfig)
            self.assertEqual(sim_cfg2.protocol.steps, 60)

    def test_step_and_shoot_vs_custom_sweep_full_equivalence(self) -> None:
        """
        Верификация 100%-ной идентичности конфигураций StepAndShoot и CustomSweep
        для tb161_nema_protocol.yaml на всех 60 ракурсах
        с автоматической подстановкой ${start_time} и ${stop_time} (30 минут суммарного сканирования).
        """
        legacy_angles = [float(step_idx * 3.0) for step_idx in range(60)]
        legacy_start_times = [float(step_idx * 30.0) for step_idx in range(60)]
        legacy_stop_times = [float((step_idx + 1) * 30.0) for step_idx in range(60)]

        def _collect_matrices(node: SpatialNode) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            result: dict[str, tuple[np.ndarray, np.ndarray]] = {
                node.name: (node.local_matrix.copy(), node.global_matrix.copy())
            }
            for child_node in node.childs:
                result.update(_collect_matrices(child_node))
            return result

        for config_path in (self.yaml_path,):
            with self.subTest(config_file=config_path.name):
                step_and_shoot_raw = load_raw_config(config_path)
                # В исходном YAML-файле start_time и stop_time опущены в simulation_manager
                self.assertNotIn("start_time", step_and_shoot_raw["simulation_manager"])
                self.assertNotIn("stop_time", step_and_shoot_raw["simulation_manager"])

                # Применяем автоподстановку протокола StepAndShoot
                SweepCompiler.apply_protocol_defaults(step_and_shoot_raw)
                self.assertEqual(step_and_shoot_raw["simulation_manager"]["start_time"], "${start_time}")
                self.assertEqual(step_and_shoot_raw["simulation_manager"]["stop_time"], "${stop_time}")

                step_and_shoot_cfg = load_simulation_config(config_path, resolve_protocol=False)
                self.assertIsInstance(step_and_shoot_cfg.protocol, StepAndShootProtocolConfig)

                # Формируем эталонную CustomSweep-версию той же конфигурации
                custom_sweep_raw = deepcopy(step_and_shoot_raw)
                custom_sweep_raw["protocol"] = {
                    "type": "CustomSweep",
                    "zipped_variables": {
                        "gantry_angle": legacy_angles,
                        "start_time": legacy_start_times,
                        "stop_time": legacy_stop_times,
                    },
                }
                for child_dict in custom_sweep_raw["scene"]["children"]:
                    if child_dict.get("name") == "Gantry":
                        child_dict["transformations"][0]["alpha"] = "${gantry_angle} deg"
                custom_sweep_raw["data_manager"]["filename"] = "output_tb161_dual_cam/${gantry_angle}_deg.hdf"
                custom_sweep_raw["simulation_manager"]["start_time"] = "${start_time} s"
                custom_sweep_raw["simulation_manager"]["stop_time"] = "${stop_time} s"

                custom_sweep_cfg = SimulationConfig.model_validate(custom_sweep_raw)
                self.assertIsInstance(custom_sweep_cfg.protocol, CustomSweepProtocolConfig)

                # Компиляция матриц задач
                compiled_sas = SweepCompiler.compile_protocol(step_and_shoot_cfg.protocol)
                compiled_cs = SweepCompiler.compile_protocol(custom_sweep_cfg.protocol)

                sas_jobs = SweepCompiler.generate_job_matrix(compiled_sas)
                cs_jobs = SweepCompiler.generate_job_matrix(compiled_cs)

                self.assertEqual(len(sas_jobs), 60)
                self.assertEqual(len(cs_jobs), 60)

                # Пошаговая сверка всех 60 состояний симуляции и первого/промежуточного/последнего графа сцены
                for step_index in range(60):
                    sas_injected = SweepCompiler.inject_variables(step_and_shoot_raw, sas_jobs[step_index])
                    cs_injected = SweepCompiler.inject_variables(custom_sweep_raw, cs_jobs[step_index])

                    sas_step_cfg = SimulationConfig.model_validate(sas_injected)
                    cs_step_cfg = SimulationConfig.model_validate(cs_injected)

                    assert sas_step_cfg.data_manager is not None
                    assert cs_step_cfg.data_manager is not None
                    self.assertEqual(
                        sas_step_cfg.data_manager.filename,
                        cs_step_cfg.data_manager.filename,
                        f"Несовпадение имени файла на шаге {step_index}",
                    )
                    self.assertEqual(
                        sas_step_cfg.data_manager.filename,
                        f"output_tb161_dual_cam/{legacy_angles[step_index]}_deg.hdf",
                    )

                    # Сверка параметров SimulationManager (последовательная шкала 0..1800 с)
                    self.assertAlmostEqual(
                        float(sas_step_cfg.simulation_manager.start_time),
                        float(step_index * 30.0 * units.s),
                        places=3,
                    )
                    self.assertAlmostEqual(
                        float(sas_step_cfg.simulation_manager.stop_time),
                        float((step_index + 1) * 30.0 * units.s),
                        places=3,
                    )
                    self.assertAlmostEqual(
                        float(sas_step_cfg.simulation_manager.start_time),
                        float(cs_step_cfg.simulation_manager.start_time),
                        places=3,
                    )
                    self.assertAlmostEqual(
                        float(sas_step_cfg.simulation_manager.stop_time),
                        float(cs_step_cfg.simulation_manager.stop_time),
                        places=3,
                    )
                    self.assertEqual(
                        sas_step_cfg.simulation_manager.particles_number,
                        cs_step_cfg.simulation_manager.particles_number,
                    )
                    self.assertAlmostEqual(
                        float(sas_step_cfg.simulation_manager.min_energy),
                        float(cs_step_cfg.simulation_manager.min_energy),
                    )

                    # Сверка скомпилированных трансформаций Gantry в валидированной модели сцены
                    sas_gantry_cfg = next(
                        ch for ch in sas_step_cfg.scene.children if ch.name == "Gantry"
                    )
                    cs_gantry_cfg = next(
                        ch for ch in cs_step_cfg.scene.children if ch.name == "Gantry"
                    )
                    self.assertAlmostEqual(
                        float(sas_gantry_cfg.transformations[0].alpha),
                        float(cs_gantry_cfg.transformations[0].alpha),
                        places=12,
                    )

                    # На репрезентативных ракурсах (0°, 3°, 90°, 177°) строим полное дерево сцены и сверяем матрицы всех узлов
                    if step_index in (0, 1, 30, 59):
                        sas_root = SceneBuilder().build_scene(sas_step_cfg.scene)
                        cs_root = SceneBuilder().build_scene(cs_step_cfg.scene)

                        sas_matrices = _collect_matrices(sas_root)
                        cs_matrices = _collect_matrices(cs_root)

                        self.assertEqual(set(sas_matrices.keys()), set(cs_matrices.keys()))
                        for node_name, (sas_local, sas_global) in sas_matrices.items():
                            cs_local, cs_global = cs_matrices[node_name]
                            np.testing.assert_allclose(
                                sas_local,
                                cs_local,
                                rtol=1e-12,
                                atol=1e-12,
                                err_msg=f"Несовпадение local_matrix узла {node_name} на шаге {step_index}",
                            )
                            np.testing.assert_allclose(
                                sas_global,
                                cs_global,
                                rtol=1e-12,
                                atol=1e-12,
                                err_msg=f"Несовпадение global_matrix узла {node_name} на шаге {step_index}",
                            )

                        # Сверка метаданных ProcedureMetadataCollector
                        collector = ProcedureMetadataCollector()
                        sas_meta = collector.collect(
                            root_scene=sas_root,
                            protocol=step_and_shoot_cfg.protocol,
                            sim_config=sas_step_cfg.simulation_manager,
                            scene_config=sas_step_cfg.scene,
                            context=sas_jobs[step_index],
                            task_id=step_index,
                        )
                        cs_meta = collector.collect(
                            root_scene=cs_root,
                            protocol=custom_sweep_cfg.protocol,
                            sim_config=cs_step_cfg.simulation_manager,
                            scene_config=cs_step_cfg.scene,
                            context=cs_jobs[step_index],
                            task_id=step_index,
                        )

                        self.assertEqual(sas_meta["protocol"]["modality"], "StepAndShoot")
                        self.assertEqual(sas_meta["protocol"]["views_total"], 60)
                        self.assertAlmostEqual(float(sas_meta["protocol"]["exposure_time"]), 30.0)
                        self.assertNotIn("orbit_radius", sas_meta["protocol"])
                        self.assertNotIn("orbit_radius", sas_meta["gantry"])
                        self.assertEqual(sas_meta["gantry"]["rotation_direction"], "CW")
                        self.assertAlmostEqual(
                            float(sas_meta["gantry"]["angular_step"]),
                            3.0,
                            places=12,
                        )
                        self.assertAlmostEqual(
                            float(sas_meta["gantry"]["gantry_angle"]),
                            float(legacy_angles[step_index] * units.deg),
                            places=12,
                        )
                        np.testing.assert_allclose(
                            sas_meta["gantry"]["pose_matrix_detector_1"],
                            cs_meta["gantry"]["pose_matrix_detector_1"],
                            rtol=1e-12,
                            atol=1e-12,
                        )
                        np.testing.assert_allclose(
                            sas_meta["gantry"]["pose_matrix_detector_2"],
                            cs_meta["gantry"]["pose_matrix_detector_2"],
                            rtol=1e-12,
                            atol=1e-12,
                        )

    def test_tc99m_full_roundtrip_integrity(self) -> None:
        """Проверка загрузки, построения сцены и полного цикла Roundtrip для examples/tc99m_nema_protocol.yaml."""
        tc99m_path = Path("examples/tc99m_nema_protocol.yaml").resolve()
        self.assertTrue(tc99m_path.exists(), f"Файл {tc99m_path} должен существовать")

        raw_dict = load_raw_config(tc99m_path)
        bindings = ConfigTemplateBinder.extract_bindings(raw_dict)
        self.assertIn("Gantry", bindings)
        self.assertEqual(bindings["Gantry"]["rotation.alpha"], "gantry_angle")
        self.assertIn("data_manager", bindings)
        self.assertEqual(bindings["data_manager"]["filename"], "gantry_angle")
        self.assertIn("simulation_manager", bindings)
        self.assertEqual(bindings["simulation_manager"]["start_time"], "start_time")
        self.assertEqual(bindings["simulation_manager"]["stop_time"], "stop_time")

        sim_cfg = load_simulation_config(tc99m_path, resolve_protocol=True)
        self.assertEqual(sim_cfg.pool_size, 24)
        self.assertIsInstance(sim_cfg.protocol, StepAndShootProtocolConfig)
        assert isinstance(sim_cfg.protocol, StepAndShootProtocolConfig)
        self.assertEqual(sim_cfg.protocol.steps, 60)
        self.assertAlmostEqual(float(sim_cfg.protocol.radius), float(270.0 * units.mm), places=5)

        builder = SceneBuilder()
        root_node = builder.build_scene(sim_cfg.scene)

        phantom_node = next((node for node in root_node.childs if node.name == "Phantom"), None)
        self.assertIsNotNone(phantom_node)
        assert phantom_node is not None
        source_node = next((ch for ch in phantom_node.childs if isinstance(ch, Source)), None)
        self.assertIsNotNone(source_node)
        assert source_node is not None
        self.assertEqual(len(source_node.channels), 1)
        spec_obj = source_node.channels[0].spectrum
        assert isinstance(spec_obj, DiscreteSpectrum)
        self.assertEqual(len(spec_obj.lines), 1)
        self.assertAlmostEqual(float(spec_obj.line_energy[0]), float(140.511 * units.keV), places=6)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_yaml_path = os.path.join(tmp_dir, "tc99m_roundtrip_exported.yaml")
            ConfigTemplateBinder.export_templated_yaml(
                root_node=root_node,
                filepath=out_yaml_path,
                bindings_registry=bindings,
                simulation_manager_cfg=sim_cfg.simulation_manager,
                data_manager_cfg=sim_cfg.data_manager,
                protocol=sim_cfg.protocol,
                pool_size=sim_cfg.pool_size,
                distribution_registry=builder.distribution_registry,
                slots_registry=builder.slots_registry,
            )

            with open(out_yaml_path, "r", encoding="utf-8") as file_handle:
                exported_raw = yaml.safe_load(file_handle)

            phantom_dict = next(
                ch for ch in exported_raw["scene"]["children"] if ch.get("name") == "Phantom"
            )
            source_dict = phantom_dict["children"][0]
            self.assertEqual(source_dict["spatial"]["type"], "voxel")
            self.assertEqual(len(source_dict["channels"]), 1)
            self.assertEqual(source_dict["channels"][0]["species"], "photon")
            self.assertEqual(source_dict["channels"][0]["spectrum"]["type"], "discrete")
            self.assertEqual(len(source_dict["channels"][0]["spectrum"]["lines"]), 1)

            sim_cfg2 = load_simulation_config(out_yaml_path, resolve_protocol=True)
            root_node2 = SceneBuilder().build_scene(sim_cfg2.scene)
            self.assertIsNotNone(root_node2)
            self.assertEqual(sim_cfg2.pool_size, 24)

    def test_nema_1_cam_full_roundtrip_integrity(self) -> None:
        """Проверка загрузки, построения сцены и полного цикла Roundtrip для examples/nema_1_cam.yaml."""
        nema1_path = Path("examples/nema_1_cam.yaml").resolve()
        self.assertTrue(nema1_path.exists(), f"Файл {nema1_path} должен существовать")

        raw_dict = load_raw_config(nema1_path)
        bindings = ConfigTemplateBinder.extract_bindings(raw_dict)
        self.assertIn("Gamma_camera", bindings)
        self.assertEqual(bindings["Gamma_camera"]["rotation.alpha"], "current_angle")
        self.assertIn("data_manager", bindings)
        self.assertEqual(bindings["data_manager"]["filename"], "current_angle")
        self.assertIn("simulation_manager", bindings)
        self.assertEqual(bindings["simulation_manager"]["start_time"], "start_time")
        self.assertEqual(bindings["simulation_manager"]["stop_time"], "stop_time")

        sim_cfg = load_simulation_config(nema1_path, resolve_protocol=True)
        self.assertEqual(sim_cfg.pool_size, 24)
        self.assertIsInstance(sim_cfg.protocol, CustomSweepProtocolConfig)
        assert isinstance(sim_cfg.protocol, CustomSweepProtocolConfig)
        compiled_sweep = SweepCompiler.compile_protocol(sim_cfg.protocol)
        job_matrix = SweepCompiler.generate_job_matrix(compiled_sweep)
        self.assertEqual(len(job_matrix), 20)

        builder = SceneBuilder()
        root_node = builder.build_scene(sim_cfg.scene)

        phantom_node = next((node for node in root_node.childs if node.name == "Phantom"), None)
        self.assertIsNotNone(phantom_node)
        assert phantom_node is not None
        source_node = next((ch for ch in phantom_node.childs if isinstance(ch, Source)), None)
        self.assertIsNotNone(source_node)
        assert source_node is not None
        self.assertEqual(len(source_node.channels), 1)
        spec_obj = source_node.channels[0].spectrum
        assert isinstance(spec_obj, DiscreteSpectrum)
        self.assertEqual(len(spec_obj.lines), 1)
        self.assertAlmostEqual(float(spec_obj.line_energy[0]), float(159.0 * units.keV), places=6)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_yaml_path = os.path.join(tmp_dir, "nema_1_cam_roundtrip_exported.yaml")
            ConfigTemplateBinder.export_templated_yaml(
                root_node=root_node,
                filepath=out_yaml_path,
                bindings_registry=bindings,
                simulation_manager_cfg=sim_cfg.simulation_manager,
                data_manager_cfg=sim_cfg.data_manager,
                protocol=sim_cfg.protocol,
                pool_size=sim_cfg.pool_size,
                distribution_registry=builder.distribution_registry,
                slots_registry=builder.slots_registry,
            )

            with open(out_yaml_path, "r", encoding="utf-8") as file_handle:
                exported_raw = yaml.safe_load(file_handle)

            phantom_dict = next(
                ch for ch in exported_raw["scene"]["children"] if ch.get("name") == "Phantom"
            )
            source_dict = phantom_dict["children"][0]
            self.assertEqual(source_dict["spatial"]["type"], "voxel")
            self.assertEqual(len(source_dict["channels"]), 1)
            self.assertEqual(source_dict["channels"][0]["species"], "photon")
            self.assertEqual(source_dict["channels"][0]["spectrum"]["type"], "discrete")
            self.assertEqual(len(source_dict["channels"][0]["spectrum"]["lines"]), 1)

            sim_cfg2 = load_simulation_config(out_yaml_path, resolve_protocol=True)
            root_node2 = SceneBuilder().build_scene(sim_cfg2.scene)
            self.assertIsNotNone(root_node2)
            self.assertEqual(sim_cfg2.pool_size, 24)


if __name__ == "__main__":
    unittest.main()

