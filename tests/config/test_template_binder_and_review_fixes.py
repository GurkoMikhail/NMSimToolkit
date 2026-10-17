"""
Тесты проверки исправлений по результатам архитектурного ревью:
- Усиление валидаторов единиц измерения и счетчиков (units.py);
- Физическая консистентность параметров кинематики SPECT (spect_kinematics.py);
- Надежность и in-place семантика шаблонизатора (template_binder.py);
- Fail-Fast LBYL проверки воркера моделирования (simulation_worker.py);
- Безопасная обработка строковых макросов в MainWindow.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import hepunits as units

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.geometry.spect_kinematics import compute_spect_poses, compute_orbit_matrix
from core.config.units import (
    unit_validator_factory,
    integer_count_validator,
    CountConfig,
    LengthConfig,
)
from core.config.models import (
    SimulationConfig,
    SimulationManagerConfig,
    DataManagerConfig,
    VolumeConfig,
    BoxConfig,
    BaseCompositeNodeConfig,
    StepAndShootProtocolConfig,
)
from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter
from core.config.template_binder import ConfigTemplateBinder
from core.config.yaml_dumper import dump_simulation_config
from core.config.yaml_loader import load_raw_config, load_simulation_config
from core.config.orchestrator import Orchestrator
from core.config.sweep_compiler import SweepCompiler
from core.config.simulation_worker import simulation_worker_task
from core.scene.nodes import SpatialNode
from gui.viewmodels.data_handler_viewmodel import DataManagerViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.procedure_viewmodel import StepAndShootProcedureViewModel


class TestReviewFixesUnitsAndKinematics(unittest.TestCase):
    """Тестирование валидаторов физических единиц и кинематики."""

    def test_unit_validator_rejects_bool(self):
        """Проверка, что валидатор физических величин строго отклоняет значения bool."""
        validator = unit_validator_factory('mm', float(units.mm))
        with self.assertRaises(ValueError):
            validator(True)
        with self.assertRaises(ValueError):
            validator(False)

    def test_integer_count_validator_valid_inputs(self):
        """Проверка корректной валидации целых чисел и научной нотации."""
        self.assertEqual(integer_count_validator(100), 100)
        self.assertEqual(integer_count_validator(100.0), 100)
        self.assertEqual(integer_count_validator("500"), 500)
        self.assertEqual(integer_count_validator("1e6"), 1000000)
        self.assertEqual(integer_count_validator("2.5e3"), 2500)
        self.assertEqual(integer_count_validator("${my_particles}"), "${my_particles}")

    def test_integer_count_validator_rejects_invalid_inputs(self):
        """Проверка, что нецелые, отрицательные значения и bool строго отклоняются."""
        with self.assertRaises(ValueError):
            integer_count_validator(True)
        with self.assertRaises(ValueError):
            integer_count_validator(False)
        with self.assertRaises(ValueError):
            integer_count_validator(0)
        with self.assertRaises(ValueError):
            integer_count_validator(-10)
        with self.assertRaises(ValueError):
            integer_count_validator("1.5")
        with self.assertRaises(ValueError):
            integer_count_validator("abc")

    def test_spect_kinematics_physical_parameters(self):
        """Проверка compute_orbit_matrix и compute_spect_poses с новыми именами параметров."""
        mat = compute_orbit_matrix(radius=250.0, angle=90.0, z=10.0, half_thickness=5.0, roll=0.0)
        self.assertIsInstance(mat, np.ndarray)
        self.assertEqual(mat.shape, (4, 4))
        # Проверка координаты Y при angle=90 deg: center_radius = 255.0
        self.assertAlmostEqual(mat[1, 3], 255.0, places=5)
        self.assertAlmostEqual(mat[0, 3], 0.0, places=5)

        # Проверка compute_spect_poses с start_angle и end_angle
        poses = compute_spect_poses(views_or_protocol=4, gamma_cameras=1, start_angle=0.0, end_angle=180.0)
        self.assertEqual(len(poses), 4)
        self.assertEqual(len(poses[0]), 1)


class TestReviewFixesTemplateBinder(unittest.TestCase):
    """Тестирование надежности и in-place применения шаблонизатора ConfigTemplateBinder."""

    def test_apply_bindings_preserves_multiple_rotations_and_order(self):
        """Проверка сохранения нескольких вращений и порядка трансформаций in-place."""
        scene_dict = {
            "name": "HeadNode",
            "type": "GammaCamera",
            "transformations": [
                {"type": "rotate", "alpha": 10.0, "beta": 0.0, "gamma": 0.0},
                {"type": "translate", "x": 0.0, "y": 200.0, "z": 0.0},
                {"type": "rotate", "alpha": 0.0, "beta": 45.0, "gamma": 0.0},  # Дополнительный поворот
            ]
        }
        bindings = {
            "HeadNode": {
                "rotation.alpha": "gantry_rot",
                "translation.y": "camera_radius",
            }
        }
        ConfigTemplateBinder.apply_bindings(scene_dict, bindings)

        transforms = scene_dict["transformations"]
        self.assertEqual(len(transforms), 3)
        # Первый поворот модифицирован привязкой
        self.assertEqual(transforms[0]["type"], "rotate")
        self.assertEqual(transforms[0]["alpha"], "${gantry_rot} deg")
        # Трансляция модифицирована на своем месте
        self.assertEqual(transforms[1]["type"], "translate")
        self.assertEqual(transforms[1]["y"], "${camera_radius} mm")
        # Второй поворот остался нетронутым и сохранил свои параметры
        self.assertEqual(transforms[2]["type"], "rotate")
        self.assertEqual(transforms[2]["beta"], 45.0)

    def test_extract_bindings_filters_false_aliases_for_regular_nodes(self):
        """Проверка, что radius и gantry_angle извлекаются только для гантри и детекторов."""
        config_dict = {
            "scene": {
                "name": "World",
                "children": [
                    {
                        "name": "PhantomBox",
                        "type": "Box",
                        "transformations": [
                            {"type": "translate", "x": 0.0, "y": "${phantom_y}", "z": 0.0},
                            {"type": "rotate", "alpha": "${box_angle}", "beta": 0.0, "gamma": 0.0},
                        ]
                    },
                    {
                        "name": "DetectorHead",
                        "type": "GammaCamera",
                        "transformations": [
                            {"type": "translate", "x": 0.0, "y": "${camera_r}", "z": 0.0},
                            {"type": "rotate", "alpha": "${camera_angle}", "beta": 0.0, "gamma": 0.0},
                        ]
                    }
                ]
            }
        }
        bindings = ConfigTemplateBinder.extract_bindings(config_dict)

        # Для PhantomBox не должно быть псевдонимов radius и gantry_angle
        phantom_b = bindings.get("PhantomBox", {})
        self.assertEqual(phantom_b.get("y"), "phantom_y")
        self.assertNotIn("radius", phantom_b)
        self.assertEqual(phantom_b.get("alpha"), "box_angle")
        self.assertNotIn("gantry_angle", phantom_b)

        # Для DetectorHead псевдонимы должны присутствовать
        detector_b = bindings.get("DetectorHead", {})
        self.assertEqual(detector_b.get("radius"), "camera_r")
        self.assertEqual(detector_b.get("gantry_angle"), "camera_angle")

    def test_apply_bindings_raises_type_error_on_invalid_key(self):
        """Проверка DbC: возбуждение TypeError при недопустимом типе ключа реестра привязок."""
        with self.assertRaises(TypeError):
            ConfigTemplateBinder.apply_bindings({}, {12345: {"radius": "val"}})

    def test_export_templated_yaml_with_dump_simulation_config(self):
        """Проверка экспорта через dump_simulation_config с формированием якорей материалов."""
        material = Material(name="Lead", density=11.34)
        box_geom = Box(x=100.0 * units.mm, y=100.0 * units.mm, z=100.0 * units.mm)
        root = Volume(geometry=box_geom, material=material, name="World")

        sim_mgr = SimulationManagerConfig(particles_number=1000)
        data_mgr = DataManagerConfig(filename="output.root")

        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "test_templated.yaml"
            ConfigTemplateBinder.export_templated_yaml(
                root_node=root,
                filepath=out_file,
                bindings_registry={"World": {"x": "root_x"}},
                simulation_manager_cfg=sim_mgr,
                data_manager_cfg=data_mgr,
            )
            self.assertTrue(out_file.exists())
            content = out_file.read_text(encoding="utf-8")
            self.assertIn("${root_x} mm", content)
            self.assertIn("Materials:", content)

    def test_tb161_protocol_filename_template_preserved_through_full_pipeline(self):
        """
        Регрессионный тест: проверка сохранения полного шаблона пути data_manager.filename
        для tb161_nema_protocol_direct.yaml в цепочке:
        extract_bindings -> apply_bindings -> Orchestrator -> inject_variables -> SimulationConfig.model_validate.
        Гарантирует, что filename для угла 3.0 равен 'output_tb161_dual_cam/3.0_deg.hdf', а не 3.0 (float).
        """
        yaml_path = Path(__file__).resolve().parents[2] / "tb161_nema_protocol_direct.yaml"
        raw_dict = load_raw_config(yaml_path)
        bindings = ConfigTemplateBinder.extract_bindings(raw_dict)

        self.assertIn("data_manager", bindings)
        self.assertEqual(bindings["data_manager"].get("filename"), "gantry_angle")

        # 1. Проверка сценария GUI: конфиг загружен с resolve_protocol=True, проходит через DataManagerViewModel
        resolved_cfg = load_simulation_config(yaml_path, resolve_protocol=True)
        self.assertIsNotNone(resolved_cfg.data_manager)
        assert resolved_cfg.data_manager is not None
        dm_vm = DataManagerViewModel()
        dm_vm.load_from_config(resolved_cfg.data_manager)
        dm_vm.filename = raw_dict["data_manager"]["filename"]
        roundtripped_dm = dm_vm.to_config()
        self.assertEqual(roundtripped_dm.filename, "output_tb161_dual_cam/${gantry_angle deg}_deg.hdf")
        self.assertEqual(roundtripped_dm.buffer_capacity, 100000)
        self.assertEqual(roundtripped_dm.floating_point_precision, "float32")
        self.assertEqual(roundtripped_dm.compression, "gzip")
        self.assertEqual(roundtripped_dm.compression_opts, 4)
        resolved_cfg.data_manager = roundtripped_dm

        # Применение привязок (как в OrchestratorSession.generate_jobs / start), где в реестре лежит только имя переменной
        gui_style_bindings = {
            target_key: ({"filename": "gantry_angle"} if target_key == "data_manager" else prop_map)
            for target_key, prop_map in bindings.items()
        }
        bound_cfg = ConfigTemplateBinder.apply_bindings(resolved_cfg, gui_style_bindings)
        self.assertIsInstance(bound_cfg, SimulationConfig)
        assert isinstance(bound_cfg, SimulationConfig)
        assert bound_cfg.data_manager is not None
        self.assertEqual(bound_cfg.data_manager.filename, "output_tb161_dual_cam/${gantry_angle deg}_deg.hdf")

        orchestrator = Orchestrator(bound_cfg)
        sweep_cfg = orchestrator.compile_protocol()
        jobs = orchestrator._generate_job_list(sweep_cfg)
        self.assertEqual(len(jobs), 60)

        for idx, job_context in enumerate(jobs):
            injected_dict = SweepCompiler.inject_variables(orchestrator.raw_config_dict, job_context)
            validated_cfg = SimulationConfig.model_validate(injected_dict)
            assert validated_cfg.data_manager is not None
            expected_angle = round(float(job_context["gantry_angle"]) / float(units.deg), 10)
            self.assertIsInstance(validated_cfg.data_manager.filename, str)
            self.assertEqual(
                validated_cfg.data_manager.filename,
                f"output_tb161_dual_cam/{expected_angle}_deg.hdf",
            )
            if idx == 1:
                self.assertEqual(
                    validated_cfg.data_manager.filename,
                    "output_tb161_dual_cam/3.0_deg.hdf",
                )

        # 2. Проверка привязки переменной к простому имени файла без шаблона (защита от голого макроса ${var})
        plain_dict = {
            "data_manager": {"filename": "results/scan.hdf"},
            "scene": {"name": "World", "type": "Volume"},
        }
        ConfigTemplateBinder.apply_bindings(plain_dict, {"data_manager": {"filename": "gantry_angle"}})
        self.assertEqual(plain_dict["data_manager"]["filename"], "results/scan_${gantry_angle}.hdf")

    def test_decompose_matrix_and_builder_roundtrip_preserves_y_translation(self) -> None:
        """
        Регрессионный тест: SceneExporter.decompose_matrix(mat, as_matrix=False) + SceneBuilder
        восстанавливают идентичную 4x4 матрицу mat для комбинации поворота (gamma=90 deg)
        и сдвига (y=329.05 mm) без превращения сдвига по Y в сдвиг по Z.
        """
        source_node = SpatialNode(name="CamHead")
        source_node.rotate(alpha=0.0, beta=0.0, gamma=float(np.radians(90.0)), in_local=False)
        source_node.translate(0.0, 329.05, 0.0, in_local=False)
        original_matrix = source_node.local_matrix.copy()
        np.testing.assert_allclose(original_matrix[0:3, 3], [0.0, 329.05, 0.0], atol=1e-6)

        decomposed_transforms = SceneExporter.decompose_matrix(original_matrix, as_matrix=False)
        self.assertEqual(len(decomposed_transforms), 2)
        self.assertEqual(decomposed_transforms[0].type, "rotate")
        self.assertEqual(decomposed_transforms[1].type, "translate")

        node_cfg = BaseCompositeNodeConfig(name="CamHead", transformations=decomposed_transforms)
        rebuilt_node = SceneBuilder().build_scene(node_cfg)
        np.testing.assert_allclose(rebuilt_node.local_matrix, original_matrix, atol=1e-6)
        np.testing.assert_allclose(rebuilt_node.local_matrix[0:3, 3], [0.0, 329.05, 0.0], atol=1e-6)

    def test_tb161_gui_pipeline_preserves_camera_positions_and_head_angles(self) -> None:
        """
        Регрессионный тест полного цикла GUI для tb161_nema_protocol_direct.yaml:
        загрузка SceneViewModel + StepAndShootProcedureViewModel.sync_with_scene +
        SceneExporter.export_to_config + ConfigTemplateBinder.apply_bindings +
        SweepCompiler.inject_variables для gantry_angle = 0 сохраняет
        GammaCamera_1 в (0, +329.05, 0) и GammaCamera_2 в (0, -329.05, 0),
        а proc_vm.head_angles равны [90.0, 270.0].
        """
        yaml_path = Path(__file__).resolve().parents[2] / "tb161_nema_protocol_direct.yaml"
        raw_dict = load_raw_config(yaml_path)
        bindings = ConfigTemplateBinder.extract_bindings(raw_dict)
        resolved_cfg = load_simulation_config(yaml_path, resolve_protocol=True)

        builder = SceneBuilder(base_dir=yaml_path.parent)
        root_node = builder.build_scene(resolved_cfg.scene)

        scene_vm = SceneViewModel()
        scene_vm.load_scene(
            root_node,
            distribution_registry=builder.distribution_registry,
            slots_registry=builder.slots_registry,
            bindings_registry=bindings,
            base_dir=yaml_path.parent,
        )

        self.assertIsInstance(resolved_cfg.protocol, StepAndShootProtocolConfig)
        assert isinstance(resolved_cfg.protocol, StepAndShootProtocolConfig)
        proc_vm = StepAndShootProcedureViewModel.from_config(resolved_cfg.protocol)
        proc_vm.sync_with_scene(scene_vm)

        # Проверяем, что локальные углы камер считаны из сцены как [90.0, 270.0]
        np.testing.assert_allclose(proc_vm.head_angles, [90.0, 270.0], atol=1e-4)

        # Экспорт сцены и наложение привязок как в GUI перед передачей в воркер
        exported_cfg = SceneExporter.export_to_config(
            root_node,
            simulation_manager_cfg=resolved_cfg.simulation_manager,
            data_manager_cfg=resolved_cfg.data_manager,
            protocol=proc_vm.to_config(),
            distribution_registry=scene_vm.distribution_registry,
            slots_registry=scene_vm.slots_registry,
        )
        merged_bindings = {**scene_vm.bindings_registry, **proc_vm.bindings_registry}
        bound_cfg = ConfigTemplateBinder.apply_bindings(exported_cfg, merged_bindings)
        assert isinstance(bound_cfg, SimulationConfig)

        orchestrator = Orchestrator(bound_cfg)
        sweep_cfg = orchestrator.compile_protocol()
        jobs = orchestrator._generate_job_list(sweep_cfg)
        self.assertGreater(len(jobs), 0)

        # Первый шаг: gantry_angle = 0
        step0_dict = SweepCompiler.inject_variables(orchestrator.raw_config_dict, jobs[0])
        step0_cfg = SimulationConfig.model_validate(step0_dict)
        worker_root = SceneBuilder(base_dir=yaml_path.parent).build_scene(step0_cfg.scene)

        def _find_node(node: SpatialNode, target_name: str) -> SpatialNode | None:
            if node.name == target_name:
                return node
            for child in node.childs:
                found = _find_node(child, target_name)
                if found is not None:
                    return found
            return None

        cam1 = _find_node(worker_root, "GammaCamera_1")
        cam2 = _find_node(worker_root, "GammaCamera_2")
        self.assertIsNotNone(cam1)
        self.assertIsNotNone(cam2)
        assert cam1 is not None and cam2 is not None

        np.testing.assert_allclose(cam1.global_matrix[0:3, 3], [0.0, 329.05, 0.0], atol=1e-4)
        np.testing.assert_allclose(cam2.global_matrix[0:3, 3], [0.0, -329.05, 0.0], atol=1e-4)


class TestReviewFixesSimulationWorker(unittest.TestCase):
    """Тестирование Fail-Fast LBYL проверок в воркере симуляции."""

    def test_worker_fails_fast_on_unresolved_particles_macro(self):
        """Проверка немедленного падения воркера при наличии неразрешенного макроса в particles_number."""
        task_dict = {
            "task_id": 1,
            "random_seed": 42,
            "config": {
                "simulation_manager": {
                    "particles_number": "${unresolved_particles}",
                    "min_energy": 0.05 * units.keV,
                    "stop_time": 10.0 * units.s,
                },
                "data_manager": {
                    "filename": "output.root",
                    "buffer_capacity": 5000,
                },
                "scene": {
                    "type": "Volume",
                    "name": "World",
                    "geometry": {"type": "Box", "x": 100.0, "y": 100.0, "z": 100.0},
                    "material": "Vacuum",
                }
            }
        }
        with self.assertRaises(ValueError) as ctx:
            simulation_worker_task((task_dict["config"], task_dict["random_seed"], None))
        self.assertIn("Неразрешенный шаблон интерполяции в particles_number", str(ctx.exception))
