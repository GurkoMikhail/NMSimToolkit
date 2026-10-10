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
)
from core.config.template_binder import ConfigTemplateBinder
from core.config.yaml_dumper import dump_simulation_config
from core.config.simulation_worker import simulation_worker_task


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

        # Проверка обратной совместимости с angle_deg
        mat_legacy = compute_orbit_matrix(radius=250.0, angle_deg=90.0, z=10.0, half_thickness=5.0)
        np.testing.assert_array_almost_equal(mat, mat_legacy)

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
