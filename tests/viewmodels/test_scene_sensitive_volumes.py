"""
Тесты архитектуры реестра чувствительных объемов в DataHandler / DataManager.
Проверяют управление чувствительными объемами через SensitiveVolumeHandlerViewModel,
иерархическую верификацию в DataManagerViewModel, реактивную синхронизацию сигналов переименования
и удаления узлов со сценой, а также отсутствие устаревших API в SceneViewModel и NodeViewModel.
"""

import unittest
from typing import List

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
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.viewmodels.data_handler_viewmodel import (
    DataManagerViewModel,
    SensitiveVolumeHandlerViewModel,
    HistoryAssemblerHandlerViewModel,
)
from core.geometry.direct_collimators import DirectParallelCollimator


class TestDataHandlerSensitiveVolumes(unittest.TestCase):
    """
    Набор тестов для управления чувствительными объемами через DataHandler и DataManager.
    """

    def setUp(self) -> None:
        self.world_material = Material(name="Air")
        self.world_geometry = Box(1000.0, 1000.0, 1000.0)
        self.root_volume = CompositeNode(name="World")
        self.scene_viewmodel = SceneViewModel(root_core_node=self.root_volume)
        self.data_manager_vm = DataManagerViewModel()

    def test_sensitive_volumes_property_and_setter(self) -> None:
        """Проверка геттера, сеттера, дедупликации и сигналов списка чувствительных объемов в HandlerViewModel."""
        handler_vm = SensitiveVolumeHandlerViewModel()
        signal_emission_counter: List[int] = []
        handler_vm.volumes_changed.connect(lambda: signal_emission_counter.append(1))

        # Установка списка объемов с повторениями, пробелами и пустыми строками
        handler_vm.sensitive_volumes = ["crystal_1", " crystal_2 ", "crystal_1", ""]
        self.assertEqual(handler_vm.sensitive_volumes, ["crystal_1", "crystal_2"])
        self.assertEqual(len(signal_emission_counter), 1)

        # Повторная установка идентичного списка не должна вызывать лишний сигнал
        handler_vm.sensitive_volumes = ["crystal_1", "crystal_2"]
        self.assertEqual(len(signal_emission_counter), 1)

        # Полная очистка списка
        handler_vm.clear_volumes()
        self.assertEqual(len(handler_vm.sensitive_volumes), 0)
        self.assertEqual(len(signal_emission_counter), 2)

    def test_is_sensitive_volume_checks_and_hierarchy(self) -> None:
        """Проверка верификации объема по имени, экземпляру NodeViewModel и иерархическому предку."""
        gantry_core = CompositeNode(name="SPECT_Gantry")
        crystal_material = Material(name="NaI")
        crystal_geometry = Box(400.0, 400.0, 10.0)
        crystal_volume = Volume(geometry=crystal_geometry, material=crystal_material, name="Crystal_Detector")
        gantry_core.add_child(crystal_volume)

        self.scene_viewmodel.add_node(self.scene_viewmodel.root_vm, NodeViewModel(gantry_core))
        gantry_vm = self.scene_viewmodel.find_by_name("SPECT_Gantry")
        self.assertIsNotNone(gantry_vm)
        crystal_vm = self.scene_viewmodel.find_by_name("Crystal_Detector")
        self.assertIsNotNone(crystal_vm)

        handler_vm = SensitiveVolumeHandlerViewModel(sensitive_volumes=["Crystal_Detector"])
        self.data_manager_vm.add_handler(handler_vm)

        # Проверка по строковому имени
        self.assertTrue(self.data_manager_vm.is_sensitive_volume("Crystal_Detector"))
        self.assertFalse(self.data_manager_vm.is_sensitive_volume("Unknown_Volume"))

        # Проверка по ViewModel
        self.assertTrue(self.data_manager_vm.is_sensitive_volume(crystal_vm))

        # Иерархическая проверка: если в чувствительные объемы добавлен CompositeNode, его дочерний узел также чувствителен
        handler_gantry = SensitiveVolumeHandlerViewModel(sensitive_volumes=["SPECT_Gantry"])
        dm_gantry = DataManagerViewModel()
        dm_gantry.add_handler(handler_gantry)
        self.assertTrue(dm_gantry.is_sensitive_volume(crystal_vm))

        # Если обработчик выключен, объемы не считаются чувствительными
        handler_vm.enabled = False
        self.assertFalse(self.data_manager_vm.is_sensitive_volume("Crystal_Detector"))

    def test_add_and_remove_sensitive_volume(self) -> None:
        """Проверка динамического добавления и удаления детекторов через методы HandlerViewModel."""
        crystal_volume = Volume(geometry=Box(100.0, 100.0, 10.0), material=Material(name="Water"), name="Target_Vol")
        crystal_vm = VolumeViewModel(crystal_volume)

        handler_vm = SensitiveVolumeHandlerViewModel()
        changed_events: List[bool] = []
        handler_vm.volumes_changed.connect(lambda: changed_events.append(True))

        # Добавление через NodeViewModel
        res = handler_vm.add_volume(crystal_vm)
        self.assertTrue(res)
        self.assertIn("Target_Vol", handler_vm.sensitive_volumes)
        self.assertEqual(len(changed_events), 1)

        # Повторное добавление не дублирует объем и возвращает False
        res_dup = handler_vm.add_volume("Target_Vol")
        self.assertFalse(res_dup)
        self.assertEqual(len(handler_vm.sensitive_volumes), 1)
        self.assertEqual(len(changed_events), 1)

        # Ошибочный тип вызывает TypeError
        with self.assertRaises(TypeError):
            handler_vm.add_volume(123)  # type: ignore

        # Удаление
        res_del = handler_vm.remove_volume("Target_Vol")
        self.assertTrue(res_del)
        self.assertNotIn("Target_Vol", handler_vm.sensitive_volumes)
        self.assertEqual(len(changed_events), 2)

    def test_node_removal_cleans_sensitive_volumes(self) -> None:
        """Проверка автоматического исключения удаленного узла из реестра обработчиков."""
        crystal_volume = Volume(geometry=Box(100.0, 100.0, 10.0), material=Material(name="Water"), name="DynamicCrystal")
        crystal_vm = VolumeViewModel(crystal_volume)

        root_viewmodel = self.scene_viewmodel.root_vm
        self.assertIsNotNone(root_viewmodel)
        self.scene_viewmodel.add_node(root_viewmodel, crystal_vm)

        handler_vm = SensitiveVolumeHandlerViewModel(sensitive_volumes=["DynamicCrystal"])
        self.data_manager_vm.add_handler(handler_vm)

        # Связываем сигналы сцены с DataManagerViewModel, как в MainWindow
        self.scene_viewmodel.node_removed.connect(lambda vm: self.data_manager_vm.on_node_removed(vm.name))

        self.assertTrue(self.data_manager_vm.is_sensitive_volume("DynamicCrystal"))

        # Удаляем узел из сцены
        self.scene_viewmodel.remove_node(crystal_vm)
        self.assertFalse(self.data_manager_vm.is_sensitive_volume("DynamicCrystal"))
        self.assertNotIn("DynamicCrystal", handler_vm.sensitive_volumes)

    def test_rename_and_replace_node_updates_sensitive_volume_name(self) -> None:
        """Проверка переименования узла с атомарным сохранением настроек чувствительного объема."""
        old_volume = Volume(geometry=Box(100.0, 100.0, 10.0), material=Material(name="Water"), name="OldCrystal")
        old_vm = VolumeViewModel(old_volume)

        root_viewmodel = self.scene_viewmodel.root_vm
        self.assertIsNotNone(root_viewmodel)
        self.scene_viewmodel.add_node(root_viewmodel, old_vm)

        handler_vm = SensitiveVolumeHandlerViewModel(sensitive_volumes=["OldCrystal"])
        self.data_manager_vm.add_handler(handler_vm)

        self.scene_viewmodel.node_renamed.connect(self.data_manager_vm.on_node_renamed)

        # Переименование через scene_vm.rename_node
        self.scene_viewmodel.rename_node(old_vm, "RenamedCrystal")
        self.assertNotIn("OldCrystal", handler_vm.sensitive_volumes)
        self.assertIn("RenamedCrystal", handler_vm.sensitive_volumes)
        self.assertTrue(self.data_manager_vm.is_sensitive_volume("RenamedCrystal"))

        # Замещение узла через scene_vm.replace_node
        new_volume = Volume(geometry=Box(100.0, 100.0, 15.0), material=Material(name="Water"), name="NewCrystal")
        new_vm = VolumeViewModel(new_volume)
        self.scene_viewmodel.replace_node(old_vm, new_vm)

        self.assertNotIn("RenamedCrystal", handler_vm.sensitive_volumes)
        self.assertIn("NewCrystal", handler_vm.sensitive_volumes)
        self.assertTrue(self.data_manager_vm.is_sensitive_volume("NewCrystal"))

    def test_load_from_config_populates_sensitive_volumes(self) -> None:
        """Проверка десериализации чувствительных объемов из конфигурации ядра DataManagerConfig."""
        data_manager_config = DataManagerConfig(
            filename="output.h5",
            handlers=[
                SensitiveVolumeHandlerConfig(sensitive_volumes=["detector_front", "detector_back"]),
                HistoryAssemblerHandlerConfig(sensitive_volumes=["detector_back", "detector_aux"]),
            ]
        )

        self.data_manager_vm.load_from_config(data_manager_config)

        self.assertTrue(self.data_manager_vm.is_sensitive_volume("detector_front"))
        self.assertTrue(self.data_manager_vm.is_sensitive_volume("detector_back"))
        self.assertTrue(self.data_manager_vm.is_sensitive_volume("detector_aux"))
        self.assertFalse(self.data_manager_vm.is_sensitive_volume("unknown_det"))

        # Проверка HistoryAssemblerHandlerViewModel
        handlers = self.data_manager_vm.handlers
        self.assertEqual(len(handlers), 2)
        self.assertIsInstance(handlers[0], SensitiveVolumeHandlerViewModel)
        self.assertIsInstance(handlers[1], HistoryAssemblerHandlerViewModel)
        self.assertTrue(issubclass(HistoryAssemblerHandlerViewModel, SensitiveVolumeHandlerViewModel))

    def test_scene_viewmodel_no_sensitive_volumes_api(self) -> None:
        """Проверка полного отсутствия устаревших API чувствительных объемов в SceneViewModel."""
        self.assertFalse(hasattr(self.scene_viewmodel, "sensitive_volumes"))
        self.assertFalse(hasattr(self.scene_viewmodel, "is_sensitive_volume"))
        self.assertFalse(hasattr(self.scene_viewmodel, "set_volume_sensitive"))
        self.assertFalse(hasattr(self.scene_viewmodel, "add_sensitive_volume"))
        self.assertFalse(hasattr(self.scene_viewmodel, "remove_sensitive_volume"))
        self.assertFalse(hasattr(self.scene_viewmodel, "clear_sensitive_volumes"))

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
