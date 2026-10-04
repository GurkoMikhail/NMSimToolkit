"""
Тесты функционала единого реестра чувствительных объемов сцены (SceneViewModel.sensitive_volumes).
Проверяют отсутствие локальных флагов в узлах, реактивную синхронизацию сигналов,
корректную очистку при удалении узлов и загрузку из конфигурации симуляции.
"""

import unittest
import numpy as np

from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.materials.materials import Material
from core.scene.nodes import CompositeNode
from core.config.models import (
    SimulationConfig,
    DataManagerConfig,
    SensitiveVolumeHandlerConfig,
    HistoryAssemblerHandlerConfig,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from core.geometry.direct_collimators import DirectParallelCollimator


class TestSceneSensitiveVolumes(unittest.TestCase):
    """
    Набор тестов для единого реестра чувствительных объемов сцены.
    """

    def setUp(self) -> None:
        self.world_material = Material(name="Air")
        self.world_geometry = Box(1000.0, 1000.0, 1000.0)
        self.root_volume = Volume(geometry=self.world_geometry, material=self.world_material, name="World")
        self.scene_viewmodel = SceneViewModel(root_core_node=self.root_volume)

    def test_sensitive_volumes_property_and_setter(self) -> None:
        """Проверка геттера, сеттера, дедупликации и сигналов списка чувствительных объемов."""
        signal_emission_counter = [0]
        self.scene_viewmodel.sensitive_volumes_changed.connect(lambda: signal_emission_counter.append(1))

        # Установка списка объемов с повторениями и пробелами
        self.scene_viewmodel.sensitive_volumes = ["crystal_1", " crystal_2 ", "crystal_1", ""]
        self.assertEqual(self.scene_viewmodel.sensitive_volumes, ["crystal_1", "crystal_2"])
        self.assertEqual(len(signal_emission_counter), 2)

        # Повторная установка идентичного списка не должна вызывать лишний сигнал
        self.scene_viewmodel.sensitive_volumes = ["crystal_1", "crystal_2"]
        self.assertEqual(len(signal_emission_counter), 2)

        # Полная очистка списка
        self.scene_viewmodel.clear_sensitive_volumes()
        self.assertEqual(len(self.scene_viewmodel.sensitive_volumes), 0)
        self.assertEqual(len(signal_emission_counter), 3)

    def test_is_sensitive_volume_checks(self) -> None:
        """Проверка верификации объема по имени, экземпляру NodeViewModel и Volume ядра."""
        crystal_material = Material(name="NaI")
        crystal_geometry = Box(400.0, 400.0, 10.0)
        crystal_volume = Volume(geometry=crystal_geometry, material=crystal_material, name="Crystal_Detector")
        crystal_vm = VolumeViewModel(crystal_volume)

        self.scene_viewmodel.sensitive_volumes = ["Crystal_Detector"]

        # Проверка по строковому имени
        self.assertTrue(self.scene_viewmodel.is_sensitive_volume("Crystal_Detector"))
        self.assertFalse(self.scene_viewmodel.is_sensitive_volume("Unknown_Volume"))

        # Проверка по ViewModel
        self.assertTrue(self.scene_viewmodel.is_sensitive_volume(crystal_vm))

        # Проверка по Volume ядра
        self.assertTrue(self.scene_viewmodel.is_sensitive_volume(crystal_volume))

    def test_add_and_remove_sensitive_volume(self) -> None:
        """Проверка динамического добавления и удаления детекторов через методы сцены."""
        crystal_volume = Volume(geometry=Box(100.0, 100.0, 10.0), material=Material(name="Water"), name="Target_Vol")
        crystal_vm = VolumeViewModel(crystal_volume)

        changed_events = []
        self.scene_viewmodel.sensitive_volumes_changed.connect(lambda: changed_events.append(True))

        # Добавление
        self.scene_viewmodel.add_sensitive_volume(crystal_vm)
        self.assertTrue(self.scene_viewmodel.is_sensitive_volume("Target_Vol"))
        self.assertEqual(len(changed_events), 1)

        # Повторное добавление не дублирует объем и не вызывает сигнал
        self.scene_viewmodel.add_sensitive_volume("Target_Vol")
        self.assertEqual(len(self.scene_viewmodel.sensitive_volumes), 1)
        self.assertEqual(len(changed_events), 1)

        # Удаление
        self.scene_viewmodel.remove_sensitive_volume(crystal_vm)
        self.assertFalse(self.scene_viewmodel.is_sensitive_volume("Target_Vol"))
        self.assertEqual(len(changed_events), 2)

    def test_node_removal_cleans_sensitive_volumes(self) -> None:
        """Проверка автоматического исключения удаленного узла из реестра чувствительных объемов."""
        crystal_volume = Volume(geometry=Box(100.0, 100.0, 10.0), material=Material(name="Water"), name="DynamicCrystal")
        crystal_vm = VolumeViewModel(crystal_volume)

        root_viewmodel = self.scene_viewmodel.root_vm
        self.assertIsNotNone(root_viewmodel)
        self.scene_viewmodel.add_node(root_viewmodel, crystal_vm)
        self.scene_viewmodel.add_sensitive_volume(crystal_vm)

        self.assertTrue(self.scene_viewmodel.is_sensitive_volume("DynamicCrystal"))

        # Удаляем узел из сцены
        self.scene_viewmodel.remove_node(crystal_vm)
        self.assertFalse(self.scene_viewmodel.is_sensitive_volume("DynamicCrystal"))
        self.assertNotIn("DynamicCrystal", self.scene_viewmodel.sensitive_volumes)

    def test_replace_node_updates_sensitive_volume_name(self) -> None:
        """Проверка замещения узла с переносом статуса детектора со старого имени на новое."""
        old_volume = Volume(geometry=Box(100.0, 100.0, 10.0), material=Material(name="Water"), name="OldCrystal")
        old_vm = VolumeViewModel(old_volume)

        root_viewmodel = self.scene_viewmodel.root_vm
        self.assertIsNotNone(root_viewmodel)
        self.scene_viewmodel.add_node(root_viewmodel, old_vm)
        self.scene_viewmodel.add_sensitive_volume("OldCrystal")

        new_volume = Volume(geometry=Box(100.0, 100.0, 15.0), material=Material(name="Water"), name="NewCrystal")
        new_vm = VolumeViewModel(new_volume)

        # Замещаем узел
        self.scene_viewmodel.replace_node(old_vm, new_vm)

        self.assertFalse(self.scene_viewmodel.is_sensitive_volume("OldCrystal"))
        self.assertTrue(self.scene_viewmodel.is_sensitive_volume("NewCrystal"))
        self.assertIn("NewCrystal", self.scene_viewmodel.sensitive_volumes)

    def test_apply_simulation_config_populates_sensitive_volumes(self) -> None:
        """Проверка синхронизации реестра чувствительных объемов из SimulationConfig."""
        data_manager_config = DataManagerConfig(
            filename="output.h5",
            handlers=[
                SensitiveVolumeHandlerConfig(sensitive_volumes=["detector_front", "detector_back"]),
                HistoryAssemblerHandlerConfig(sensitive_volumes=["detector_back", "detector_aux"]),
            ]
        )
        sim_config = SimulationConfig.model_construct(data_manager=data_manager_config)

        self.scene_viewmodel.apply_simulation_config(sim_config)

        expected_volumes = ["detector_front", "detector_back", "detector_aux"]
        self.assertEqual(self.scene_viewmodel.sensitive_volumes, expected_volumes)
        self.assertTrue(self.scene_viewmodel.is_sensitive_volume("detector_front"))
        self.assertTrue(self.scene_viewmodel.is_sensitive_volume("detector_back"))
        self.assertTrue(self.scene_viewmodel.is_sensitive_volume("detector_aux"))

    def test_node_viewmodels_do_not_have_sensitive_detector_flag(self) -> None:
        """Проверка полного отсутствия устаревшего флага is_sensitive_detector в VolumeViewModel и CollimatorViewModel."""
        test_volume = Volume(geometry=Box(10.0, 10.0, 10.0), material=Material(name="Water"), name="TestVol")
        volume_viewmodel = VolumeViewModel(test_volume)

        # Проверка отсутствия свойства в объекте и классе
        self.assertFalse(hasattr(volume_viewmodel, 'is_sensitive_detector'))
        self.assertFalse(hasattr(VolumeViewModel, 'is_sensitive_detector'))
        self.assertFalse(hasattr(VolumeViewModel, 'get_sensitive_volumes'))
        self.assertFalse(hasattr(VolumeViewModel, 'clear_sensitive_volumes'))

        # Проверка CollimatorViewModel
        direct_collimator = DirectParallelCollimator(size=(200.0, 200.0, 20.0), hole_diameter=1.5, septa=0.2)
        collimator_viewmodel = CollimatorViewModel(direct_collimator)
        self.assertFalse(hasattr(collimator_viewmodel, 'is_sensitive_detector'))
        self.assertFalse(hasattr(CollimatorViewModel, 'is_sensitive_detector'))


if __name__ == '__main__':
    unittest.main()
