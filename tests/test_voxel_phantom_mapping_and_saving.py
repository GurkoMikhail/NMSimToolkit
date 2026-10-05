import unittest
import os
import tempfile
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication

from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import Material, MaterialArray
from core.source.sources import Source
from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter
from core.config.yaml_loader import load_simulation_config
from core.config.models import NumpyDistributionConfig, RawDistributionConfig
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.views.property_inspector import PropertyInspector
from gui.viewport_3d.vtk_viewport import VTKViewport
from gui.controllers.viewport_controller import SceneViewportController


class TestVoxelPhantomMappingAndSaving(unittest.TestCase):
    """
    Модульные и интеграционные тесты для настройки Mapping материалов,
    сохранения путей распределений и интеграции с PropertyInspector.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if QApplication.instance() is None:
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def test_material_mapping_viewmodel(self) -> None:
        """
        Проверка изменения соответствия материалов через VoxelVolumeViewModel.
        """
        material_array = MaterialArray((5, 5, 5))
        core_node = WoodcockVoxelVolume(
            voxel_size=2.0,
            material_distribution=material_array,
            name="TestPhantom"
        )
        voxel_vm = VoxelVolumeViewModel(core_node)

        changed_events: list[tuple[str, object]] = []
        voxel_vm.property_changed.connect(
            lambda prop_name, prop_val: changed_events.append((prop_name, prop_val))
        )

        # Проверка начального состояния
        self.assertGreaterEqual(len(voxel_vm.material_list), 1)

        # Назначение существующего материала базы NIST
        voxel_vm.set_material_mapping(0, "Water, Liquid")
        self.assertEqual(voxel_vm.material_list[0].name, "Water, Liquid")

        # Проверка эмиссии сигнала 'material_distribution'
        emitted_names = [event[0] for event in changed_events]
        self.assertIn('material_distribution', emitted_names)

        # Назначение материала с выходом за текущий размер (авто-расширение списка)
        voxel_vm.set_material_mapping(2, "Bone, Cortical (ICRU-44)")
        self.assertEqual(len(voxel_vm.material_list), 3)
        self.assertEqual(voxel_vm.material_list[2].name, "Bone, Cortical (ICRU-44)")

        # Проверка установки Vacuum
        voxel_vm.set_material_mapping(1, "Vacuum")
        self.assertEqual(voxel_vm.material_list[1].name, "Vacuum")

        # Проверка валидации ID < 0
        with self.assertRaises(ValueError):
            voxel_vm.set_material_mapping(-1, "Water, Liquid")

        # Проверка валидации несуществующего материала
        with self.assertRaises(ValueError):
            voxel_vm.set_material_mapping(0, "NonExistentMaterialXYZ")

    def test_scene_vm_distribution_registry(self) -> None:
        """
        Проверка регистрации и обновления путей и mapping в SceneViewModel.
        """
        scene_vm = SceneViewModel()
        material_array = MaterialArray((4, 4, 4))
        voxel_node = WoodcockVoxelVolume(voxel_size=1.0, material_distribution=material_array, name="VoxelNode")
        source_node = Source(distribution=np.ones((4, 4, 4)))
        source_node.name = "SourceNode"

        # Регистрация пути фантома (.npy)
        scene_vm.update_distribution_path(voxel_node, "phantoms/brain.npy")
        dist_cfg = scene_vm.distribution_registry.get(voxel_node)
        self.assertIsNotNone(dist_cfg)
        self.assertIsInstance(dist_cfg, NumpyDistributionConfig)
        self.assertEqual(dist_cfg.path, "phantoms/brain.npy")

        # Обновление словаря mapping
        new_mapping = {0.0: "Air, Dry (near sea level)", 1.0: "Water, Liquid"}
        scene_vm.update_distribution_mapping(voxel_node, new_mapping)
        self.assertEqual(dist_cfg.mapping, new_mapping)

        # Регистрация пути источника (.raw)
        scene_vm.update_distribution_path(source_node, "sources/activity.raw")
        source_dist_cfg = scene_vm.distribution_registry.get(source_node)
        self.assertIsNotNone(source_dist_cfg)
        self.assertIsInstance(source_dist_cfg, RawDistributionConfig)
        self.assertEqual(source_dist_cfg.path, "sources/activity.raw")
        self.assertEqual(source_dist_cfg.shape, (4, 4, 4))

    def test_load_scene_restores_file_paths(self) -> None:
        """
        Проверка восстановления непустых путей воксельного фантома и источника
        при загрузке YAML-конфигурации.
        """
        yaml_path = Path("nema_1_cam.yaml")
        self.assertTrue(yaml_path.is_file(), "Файл nema_1_cam.yaml должен присутствовать в корне проекта.")

        cfg = load_simulation_config(yaml_path, resolve_protocol=True)
        builder = SceneBuilder(base_dir=yaml_path.parent)
        root_node = builder.build_scene(cfg.scene)

        scene_vm = SceneViewModel()
        scene_vm.load_scene(
            root_node,
            distribution_registry=builder.distribution_registry,
            slots_registry=builder.slots_registry,
            base_dir=yaml_path.parent,
        )

        all_nodes = scene_vm.all_nodes()
        voxel_vms = [node_vm for node_vm in all_nodes if isinstance(node_vm, VoxelVolumeViewModel)]
        source_vms = [node_vm for node_vm in all_nodes if isinstance(node_vm, SourceViewModel)]

        self.assertGreaterEqual(len(voxel_vms), 1, "В сцене должен присутствовать VoxelVolumeViewModel")
        self.assertGreaterEqual(len(source_vms), 1, "В сцене должен присутствовать SourceViewModel")

        phantom_vm = voxel_vms[0]
        source_vm = source_vms[0]

        # Пути не должны быть пустыми строками
        self.assertTrue(phantom_vm.file_path, "Путь к файлу воксельного фантома не должен быть пустым")
        self.assertTrue(source_vm.file_path, "Путь к файлу распределения источника не должен быть пустым")

        # Пути должны указывать на существующие файлы
        self.assertTrue(
            Path(phantom_vm.file_path).is_file(),
            f"Файл фантома {phantom_vm.file_path} должен существовать"
        )
        self.assertTrue(
            Path(source_vm.file_path).is_file(),
            f"Файл источника {source_vm.file_path} должен существовать"
        )

    def test_export_yaml_roundtrip_with_mapping_and_relative_paths(self) -> None:
        """
        Интеграционный тест полного цикла (round-trip):
        Загрузка ➔ смена материала в mapping ➔ сохранение в YAML с относительными путями ➔
        повторная сборка через SceneBuilder и валидация.
        """
        yaml_path = Path("nema_1_cam.yaml")
        cfg = load_simulation_config(yaml_path, resolve_protocol=True)
        builder = SceneBuilder(base_dir=yaml_path.parent)
        root_node = builder.build_scene(cfg.scene)

        scene_vm = SceneViewModel()
        scene_vm.load_scene(
            root_node,
            distribution_registry=builder.distribution_registry,
            slots_registry=builder.slots_registry,
            base_dir=yaml_path.parent,
        )

        voxel_vms = [node_vm for node_vm in scene_vm.all_nodes() if isinstance(node_vm, VoxelVolumeViewModel)]
        self.assertGreaterEqual(len(voxel_vms), 1)
        phantom_vm = voxel_vms[0]

        # Изменяем mapping материала для фантома
        phantom_vm.set_material_mapping(0, "Bone, Cortical (ICRU-44)")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_yaml_path = Path(temp_dir) / "exported_test_scene.yaml"

            # Экспортируем граф сцены
            SceneExporter.export_to_yaml(
                root_node=scene_vm.root_vm.core_node,
                filepath=temp_yaml_path,
                distribution_registry=scene_vm.distribution_registry,
                slots_registry=scene_vm.slots_registry,
            )

            self.assertTrue(temp_yaml_path.is_file(), "YAML файл должен быть успешно создан на диске.")

            # Загружаем экспортированную конфигурацию
            loaded_cfg = load_simulation_config(temp_yaml_path, resolve_protocol=True)

            # Строим сцену через SceneBuilder
            new_builder = SceneBuilder(base_dir=Path("."))
            rebuilt_root = new_builder.build_scene(loaded_cfg.scene)
            self.assertIsNotNone(rebuilt_root, "Сцена ядра должна быть успешно собрана из экспортированного YAML.")

    def test_property_inspector_material_mapping_table(self) -> None:
        """
        Проверка таблицы материалов и реактивной смены mapping в PropertyInspector.
        """
        material_array = MaterialArray((4, 4, 4))
        core_node = WoodcockVoxelVolume(voxel_size=2.0, material_distribution=material_array, name="Phantom")
        scene_vm = SceneViewModel(core_node)
        inspector = PropertyInspector(scene_vm=scene_vm)

        voxel_vm = scene_vm.root_vm
        self.assertIsInstance(voxel_vm, VoxelVolumeViewModel)

        inspector.set_target_viewmodel(voxel_vm)

        # Проверка заполнения строк таблицы
        row_count = inspector.tbl_material_mapping.rowCount()
        self.assertEqual(row_count, len(voxel_vm.material_list))

        # Вызов смены материала через инспектор
        inspector._on_mapping_material_changed(0, "Bone, Cortical (ICRU-44)")
        self.assertEqual(voxel_vm.material_list[0].name, "Bone, Cortical (ICRU-44)")

        # Проверка обновления реестра в SceneViewModel
        reg_dist = scene_vm.distribution_registry.get(voxel_vm.core_node)
        self.assertIsNotNone(reg_dist)
        self.assertEqual(reg_dist.mapping[0.0], "Bone, Cortical (ICRU-44)")

        inspector.close()

    def test_voxel_size_change_updates_3d_viewport(self) -> None:
        """
        Проверка того, что при смене размера вокселя в PropertyInspector
        актор объема и рамка выделения в 3D вьюпорте корректно перестраиваются.
        """
        grid_dimensions = (8, 8, 8)
        material_array = MaterialArray(grid_dimensions)
        initial_voxel_step = 1.0
        core_node = WoodcockVoxelVolume(
            voxel_size=initial_voxel_step,
            material_distribution=material_array,
            name="VoxelPhantom3DTest"
        )
        scene_vm = SceneViewModel(root_core_node=core_node)
        viewport_widget = VTKViewport()
        viewport_controller = SceneViewportController(viewport=viewport_widget, scene_vm=scene_vm)
        inspector_widget = PropertyInspector(scene_vm=scene_vm)

        voxel_vm = scene_vm.root_vm
        self.assertIsInstance(voxel_vm, VoxelVolumeViewModel)

        inspector_widget.set_target_viewmodel(voxel_vm)
        viewport_controller.on_node_selected(voxel_vm)

        # Проверка начальных границ актора воксельного рендерера
        initial_bounds = viewport_controller.voxel_renderer.volume_actor.GetBounds()
        expected_half_span_initial = 0.5 * 8 * initial_voxel_step
        self.assertAlmostEqual(initial_bounds[0], -expected_half_span_initial, places=4)
        self.assertAlmostEqual(initial_bounds[1], expected_half_span_initial - initial_voxel_step, places=4)

        # Проверка начальных границ рамки выделения
        selection_box_actor = viewport_widget.get_actor(f"selection_box_{id(voxel_vm)}")
        self.assertIsNotNone(selection_box_actor)
        initial_box_bounds = selection_box_actor.GetBounds()
        self.assertAlmostEqual(initial_box_bounds[0], initial_bounds[0], places=4)

        # Изменяем размер вокселей вдвое через спинбоксы инспектора свойств
        new_voxel_step = 2.5
        inspector_widget.spin_voxel_size_x.setValue(new_voxel_step)
        inspector_widget.spin_voxel_size_y.setValue(new_voxel_step)
        inspector_widget.spin_voxel_size_z.setValue(new_voxel_step)

        # Проверяем обновление модели ядра и ViewModel
        np.testing.assert_allclose(voxel_vm.voxel_size, [new_voxel_step, new_voxel_step, new_voxel_step])
        np.testing.assert_allclose(core_node.voxel_size, [new_voxel_step, new_voxel_step, new_voxel_step])

        # Проверяем, что актор объема в VTK обновил свои границы в соответствии с новым шагом
        updated_bounds = viewport_controller.voxel_renderer.volume_actor.GetBounds()
        expected_half_span_updated = 0.5 * 8 * new_voxel_step
        self.assertAlmostEqual(updated_bounds[0], -expected_half_span_updated, places=4)
        self.assertAlmostEqual(updated_bounds[1], expected_half_span_updated - new_voxel_step, places=4)

        # Проверяем, что рамка выделения также адаптировалась под новые границы
        updated_box_actor = viewport_widget.get_actor(f"selection_box_{id(voxel_vm)}")
        self.assertIsNotNone(updated_box_actor)
        updated_box_bounds = updated_box_actor.GetBounds()
        self.assertAlmostEqual(updated_box_bounds[0], updated_bounds[0], places=4)
        self.assertAlmostEqual(updated_box_bounds[1], updated_bounds[1], places=4)

        inspector_widget.close()
