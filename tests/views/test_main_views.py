"""
Модульные тесты графических представлений (Views): дерево сцены, инспектор свойств, панель результатов и главное окно.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hepunits as units
import numpy as np

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QSplitter, QTabWidget
    app = QApplication.instance()
    if app is None:
        app = QApplication(['-platform', 'offscreen'])
except ImportError:
    app = None

from core.config.models import BaseCompositeNodeConfig, LoggingConfig, SimulationConfig, SimulationManagerConfig
from core.config.builder import SceneBuilder
from core.config.yaml_loader import load_simulation_config
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.other.typing_definitions import Float
from core.scene.nodes import CompositeNode
from core.source.sources import PointDistribution, Source
from gui.viewmodels.data_handler_viewmodel import SensitiveVolumeHandlerViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.views.main_window import MainWindow
from gui.views.property_inspector import PropertyInspector
from gui.views.results_viewer import ResultsViewer
from gui.views.scene_tree_widget import SceneTreeWidget


class TestStage4Views(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if app is None:
            raise unittest.SkipTest("PySide6 не установлен в окружении")

    def test_scene_tree_widget(self) -> None:
        """Проверка инициализации и построения дерева сцены."""
        root = CompositeNode(name="RootNode")
        child = Volume(geometry=Box(10.0, 10.0, 10.0), material=Material(name="Water"), name="Target")
        root.add_child(child)

        scene_vm = SceneViewModel(root)
        tree_widget = SceneTreeWidget(scene_vm)
        self.assertEqual(tree_widget.tree.topLevelItemCount(), 1)
        root_item = tree_widget.tree.topLevelItem(0)
        assert root_item is not None
        self.assertEqual(root_item.text(0), "RootNode")
        self.assertEqual(root_item.childCount(), 1)
        child_item = root_item.child(0)
        assert child_item is not None
        self.assertEqual(child_item.text(0), "Target")

    def test_property_inspector_binding(self) -> None:
        """Проверка инспектора свойств и двусторонней реактивности."""
        geo = Box(50.0, 60.0, 70.0)
        vol = Volume(geometry=geo, material=Material(name="Water"), name="WaterBox")
        vm = VolumeViewModel(vol)

        inspector = PropertyInspector()
        inspector.set_target_viewmodel(vm)

        self.assertEqual(inspector.general_section.txt_name.text(), "WaterBox")
        self.assertAlmostEqual(inspector.volume_section.spin_size_x.value(), 50.0)

        # Модификация через UI
        inspector.general_section.txt_name.setText("RenamedBox")
        inspector.general_section._on_name_changed()
        self.assertEqual(vm.name, "RenamedBox")
        self.assertEqual(vol.name, "RenamedBox")

    def test_results_viewer(self) -> None:
        """Проверка панели результатов: обновление 2D проекции и спектра."""
        viewer = ResultsViewer()
        dummy_proj = np.ones((64, 64), dtype=np.float32) * 5.0
        viewer.set_projection_data(dummy_proj)

        self.assertEqual(viewer.lbl_stats.text(), f"Всего отсчетов: {int(np.sum(dummy_proj)):,}")

        rng = np.random.default_rng(42)
        dummy_energies = rng.normal(loc=140.5, scale=5.0, size=1000)
        viewer.set_spectrum_data(dummy_energies)

        viewer.clear_results()
        self.assertEqual(viewer.lbl_stats.text(), "Всего отсчетов: 0")

    def test_results_viewer_unified_splitter_layout(self) -> None:
        """
        Проверка размещения 2D-проекции и энергетического спектра одновременно в одном окне
        через горизонтальный QSplitter без переключения вкладок QTabWidget.
        """
        viewer = ResultsViewer()
        self.assertIsNone(viewer.findChild(QTabWidget))
        assert viewer.splitter is not None
        self.assertIsInstance(viewer.splitter, QSplitter)
        self.assertEqual(viewer.splitter.orientation(), Qt.Orientation.Horizontal)
        self.assertFalse(viewer.splitter.childrenCollapsible())
        self.assertEqual(viewer.splitter.count(), 2)
        self.assertIs(viewer.splitter.widget(0), viewer.projection_panel)
        self.assertIs(viewer.splitter.widget(1), viewer.spectrum_panel)

        # Проверяем, что и 2D-проекция, и спектр находятся в соответствующих панелях сплиттера и не скрыты вкладками
        assert viewer.projection_panel is not None and viewer.spectrum_panel is not None
        self.assertTrue(viewer.projection_panel.isAncestorOf(viewer.image_view))
        self.assertTrue(viewer.spectrum_panel.isAncestorOf(viewer.spectrum_plot))
        self.assertFalse(viewer.image_view.isHidden())
        self.assertFalse(viewer.spectrum_plot.isHidden())

    def test_results_viewer_imageview_orientation(self) -> None:
        """
        Проверка ориентации 2D-матрицы (Rows, Cols) в pg.ImageView:
        col -> ось X (горизонталь), row -> ось Y (вертикаль сверху вниз), proj[0, 0] в верхнем левом углу.
        """
        viewer = ResultsViewer()
        proj_2d = np.zeros((8, 12), dtype=np.float32)
        proj_2d[0, 0] = 100.0
        proj_2d[2, 7] = 50.0
        viewer.set_projection_data(proj_2d)

        img_item = viewer.image_view.getImageItem()
        self.assertEqual(img_item.image.shape, (12, 8))
        self.assertAlmostEqual(float(img_item.image[0, 0]), 100.0)
        self.assertAlmostEqual(float(img_item.image[7, 2]), 50.0)
        self.assertTrue(viewer.image_view.getView().yInverted())

    def test_results_viewer_4d_cube_and_interactive_energy_window(self) -> None:
        """
        Проверка работы ResultsViewer с 4D гиперспектральным кубом (N_det, E_bins, Rows, Cols),
        переключения детекторов и двусторонней синхронизации энергетического окна.
        """
        viewer = ResultsViewer()
        viewer.set_detector_metadata(["Detector_1", "Detector_2"], max_energy=300.0)

        cube = np.zeros((2, 300, 16, 16), dtype=np.float32)
        # Detector_1: 10 отсчетов в фотопике 74.5 кэВ (бин 74) в (2, 5) и 5 отсчетов фона 20.5 кэВ (бин 20) в (10, 12)
        cube[0, 74, 2, 5] = 10.0
        cube[0, 20, 10, 12] = 5.0
        # Detector_2: 20 отсчетов в фотопике 74.5 кэВ (бин 74) в (14, 3)
        cube[1, 74, 14, 3] = 20.0

        viewer.set_projection_data(cube)
        assert viewer._current_projection is not None
        self.assertAlmostEqual(float(np.sum(viewer._current_projection)), 15.0)
        self.assertEqual(viewer.lbl_stats.text(), "Отсчетов в окне: 15 (Всего: 15)")

        # Проверка спектра для Detector_1
        _, y_spec = viewer._spectrum_curve.getData()
        self.assertAlmostEqual(float(np.sum(y_spec)), 15.0)
        self.assertAlmostEqual(float(y_spec[74]), 10.0)
        self.assertAlmostEqual(float(y_spec[20]), 5.0)

        # Сужение энергетического окна до [60.0, 90.0] кэВ через спинбоксы
        assert viewer.spin_emin is not None and viewer.spin_emax is not None
        viewer.spin_emin.setValue(60.0)
        viewer.spin_emax.setValue(90.0)

        r_min, r_max = viewer.energy_region.getRegion()
        self.assertAlmostEqual(float(r_min), 60.0)
        self.assertAlmostEqual(float(r_max), 90.0)
        self.assertAlmostEqual(float(viewer._current_projection[2, 5]), 10.0)
        self.assertAlmostEqual(float(viewer._current_projection[10, 12]), 0.0)
        self.assertEqual(viewer.lbl_stats.text(), "Отсчетов в окне: 10 (Всего: 15)")

        # Переключение на Detector_2
        assert viewer.combo_detector is not None
        viewer.combo_detector.setCurrentIndex(1)
        self.assertAlmostEqual(float(viewer._current_projection[14, 3]), 20.0)
        self.assertEqual(viewer.lbl_stats.text(), "Отсчетов в окне: 20 (Всего: 20)")

        # Переключение на сумму всех детекторов
        viewer.combo_detector.setCurrentIndex(2)
        self.assertAlmostEqual(float(viewer._current_projection[2, 5]), 10.0)
        self.assertAlmostEqual(float(viewer._current_projection[14, 3]), 20.0)
        self.assertEqual(viewer.lbl_stats.text(), "Отсчетов в окне: 30 (Всего: 35)")

        # Перемещение LinearRegionItem на фоновую линию [10.0, 30.0] кэВ
        viewer.energy_region.setRegion((10.0, 30.0))
        self.assertAlmostEqual(float(viewer.spin_emin.value()), 10.0)
        self.assertAlmostEqual(float(viewer.spin_emax.value()), 30.0)
        self.assertAlmostEqual(float(viewer._current_projection[10, 12]), 5.0)
        self.assertAlmostEqual(float(viewer._current_projection[2, 5]), 0.0)
        self.assertEqual(viewer.lbl_stats.text(), "Отсчетов в окне: 5 (Всего: 35)")

        # Сброс окна кнопкой «Весь спектр»
        assert viewer.btn_reset_window is not None
        viewer.btn_reset_window.click()
        self.assertAlmostEqual(float(viewer.spin_emin.value()), 0.0)
        self.assertAlmostEqual(float(viewer.spin_emax.value()), 300.0)
        self.assertEqual(viewer.lbl_stats.text(), "Отсчетов в окне: 35 (Всего: 35)")

        # Убеждаемся, что при активном 4D-кубе вызов set_spectrum_data с сырым 1D-массивом из кольца
        # НЕ перезаписывает точный спектр из 4D-куба и сохраняет LinearRegionItem
        viewer.set_spectrum_data(np.array([50.0, 75.0, 140.0], dtype=np.float32))
        _, y_spec_after = viewer._spectrum_curve.getData()
        self.assertEqual(len(y_spec_after), 300)
        self.assertAlmostEqual(float(np.sum(y_spec_after)), 35.0)
        self.assertIn(viewer.energy_region, viewer.spectrum_plot.items())

        # Проверка сброса пользовательских имен детекторов при передаче пустого списка
        viewer.set_detector_metadata(["Custom_A", "Custom_B"])
        self.assertEqual(viewer.combo_detector.itemText(0), "Custom_A")
        viewer.set_detector_metadata([])
        self.assertEqual(viewer.combo_detector.itemText(0), "Detector_1")
        self.assertEqual(viewer.combo_detector.itemText(1), "Detector_2")

        viewer.clear_results()
        self.assertIn(viewer.energy_region, viewer.spectrum_plot.items())

        # Проверка 5D-стека ОФЭКТ (N_views, N_det, E_bins, Rows, Cols) с одновременной фильтрацией по энергетическому окну
        stack_5d = np.zeros((2, 2, 300, 8, 8), dtype=np.float32)
        stack_5d[0, 0, 74, 1, 1] = 12.0  # View 0, Det 0, 74.5 кэВ
        stack_5d[0, 0, 20, 2, 2] = 4.0   # View 0, Det 0, 20.5 кэВ
        stack_5d[1, 0, 74, 3, 3] = 25.0  # View 1, Det 0, 74.5 кэВ
        stack_5d[1, 0, 20, 4, 4] = 8.0   # View 1, Det 0, 20.5 кэВ

        viewer.set_detector_metadata(["Head_1", "Head_2"], max_energy=300.0)
        viewer.combo_detector.setCurrentIndex(0)
        viewer.set_energy_window(60.0, 90.0)
        viewer.set_projection_stack_data(stack_5d, current_idx=0, total_views=2, angle=0.0)
        self.assertEqual(viewer.lbl_stats.text(), "Отсчетов в окне: 12 (Всего: 16)")
        self.assertAlmostEqual(float(viewer._current_projection[1, 1]), 12.0)

        # Переключение слайдера проекции на ракурс 2 сохраняет фильтрацию по энергетическому окну [60, 90] кэВ
        viewer.slider_proj.setValue(2)
        self.assertEqual(viewer.lbl_stats.text(), "Отсчетов в окне: 25 (Всего: 33)")
        self.assertAlmostEqual(float(viewer._current_projection[3, 3]), 25.0)
        self.assertAlmostEqual(float(viewer._current_projection[4, 4]), 0.0)

    def test_main_window_instantiation(self) -> None:
        """Проверка корректной сборки главного окна, наличия док-панелей и отсутствия фейковых детекторов в дефолтной сцене."""
        main_win = MainWindow()
        self.assertIsNotNone(main_win.viewport)
        self.assertIsNotNone(main_win.dock_tree)
        self.assertIsNotNone(main_win.dock_inspector)
        assert main_win.results_viewer.combo_detector is not None
        default_items = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        # В дефолтной сцене (WorldScene -> WaterPhantom) детекторов нет, поэтому Detector_1 и Detector_2 не должны появляться
        self.assertEqual(default_items, ["Все детекторы (Сумма)"])

        # При добавлении одного детектора в сцену список реактивно обновляется и показывает ровно этот 1 детектор
        det_vol = Volume(geometry=Box(100.0, 100.0, 10.0), material=Material(name="NaI"), name="Detector")
        assert main_win.scene_vm.root_vm is not None
        det_vm = VolumeViewModel(det_vol)
        main_win.scene_vm.add_node(main_win.scene_vm.root_vm, det_vm)
        items_single = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(items_single, ["Detector"])

        # При переименовании детектора имя в комбобоксе реактивно обновляется
        main_win.scene_vm.rename_node(det_vm, "Crystal_Main")
        items_renamed = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(items_renamed, ["Crystal_Main"])

        # При удалении детектора фейковые Detector_1 / Detector_2 не появляются
        main_win.scene_vm.remove_node(det_vm)
        items_after_remove = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(items_after_remove, ["Все детекторы (Сумма)"])

        # При добавлении чувствительного объема через SensitiveVolumeHandlerViewModel список также обновляется
        sv_handlers = [
            handler_vm for handler_vm in main_win.data_manager_vm.handlers
            if isinstance(handler_vm, SensitiveVolumeHandlerViewModel)
        ]
        assert sv_handlers
        sv_handlers[0].add_volume("CustomSensitiveCrystal")
        items_from_handler = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(items_from_handler, ["CustomSensitiveCrystal"])
        sv_handlers[0].clear_volumes()

        # Проверка переименования кристалла внутри GammaCamera и удаления всей камеры целиком
        main_win.procedure_vm.gamma_cameras = 1  # type: ignore[attr-defined]
        main_win.scene_tree._add_gamma_camera(main_win.scene_vm.root_vm)
        items_with_cam = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(items_with_cam, ["Detector_GammaCamera_1"])

        crystal_node_vm = main_win.scene_vm.find_by_name("Detector_GammaCamera_1")
        assert crystal_node_vm is not None
        main_win.scene_vm.rename_node(crystal_node_vm, "NaI_Renamed_Crystal")
        items_after_cam_crystal_rename = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(items_after_cam_crystal_rename, ["NaI_Renamed_Crystal"])

        # Даже если кристалл зарегистрирован в SensitiveVolumeHandler, удаление родительской камеры очищает его
        sv_handlers[0].add_volume("NaI_Renamed_Crystal")
        cam_vm = main_win.scene_vm.find_by_name("GammaCamera_1")
        assert cam_vm is not None
        main_win.scene_vm.remove_node(cam_vm)
        items_after_cam_remove = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(items_after_cam_remove, ["Все детекторы (Сумма)"])

        # Проверка перехода ResultsViewer от 1 детектора к 4D-кубу с 1 и с 2 детекторами
        main_win.results_viewer.set_detector_metadata(["SingleDet"])
        self.assertEqual(main_win.results_viewer.combo_detector.count(), 1)
        cube_1det = np.zeros((1, 64, 8, 8), dtype=np.float32)
        main_win.results_viewer.set_projection_data(cube_1det)
        self.assertEqual(
            [
                main_win.results_viewer.combo_detector.itemText(item_idx)
                for item_idx in range(main_win.results_viewer.combo_detector.count())
            ],
            ["SingleDet"],
        )
        cube_2det = np.zeros((2, 64, 8, 8), dtype=np.float32)
        main_win.results_viewer.set_projection_data(cube_2det)
        self.assertEqual(
            [
                main_win.results_viewer.combo_detector.itemText(item_idx)
                for item_idx in range(main_win.results_viewer.combo_detector.count())
            ],
            ["SingleDet", "Detector_2", "Все детекторы (Сумма)"],
        )
        self.assertEqual(main_win.results_viewer.combo_detector.currentIndex(), 0)
        main_win.close()

    def test_main_window_save_yaml_logic(self) -> None:
        """Проверка логики сохранения конфигурации симуляции из MainWindow в YAML."""
        main_win = MainWindow()
        root = CompositeNode(name="World")
        main_win.scene_vm.load_scene(root)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_saved_config.yaml"
            with patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "YAML files (*.yaml *.yml)"),
            ):
                with patch("PySide6.QtWidgets.QMessageBox.information"), patch(
                    "PySide6.QtWidgets.QMessageBox.critical"
                ) as mock_crit:
                    main_win._on_save_yaml()
                    self.assertFalse(mock_crit.called, f"QMessageBox.critical был вызван: {mock_crit.call_args}")

            self.assertTrue(save_path.exists())
            loaded_cfg = load_simulation_config(str(save_path))
            self.assertIsNotNone(loaded_cfg.simulation_manager)
            self.assertEqual(loaded_cfg.simulation_manager.particles_number, main_win.sim_settings.particles_number)
        main_win.close()

    def test_main_window_load_yaml_logic(self) -> None:
        """Проверка логики загрузки конфигурации симуляции из YAML в MainWindow."""
        main_win = MainWindow()
        config_path = "examples/tb161_nema_protocol.yaml"

        with patch(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(config_path, "YAML files (*.yaml *.yml)"),
        ):
            with patch("PySide6.QtWidgets.QMessageBox.critical"):
                main_win._on_open_yaml()

        assert main_win.scene_vm.root_vm is not None
        self.assertEqual(main_win.scene_vm.root_vm.name, "Simulation_volume")
        assert main_win.results_viewer.combo_detector is not None
        combo_items = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(combo_items, ["Detector_1", "Detector_2", "Все детекторы (Сумма)"])

        # Для Tb-161 максимальная энергия реального спектра источника составляет 550.25 кэВ,
        # поэтому с запасом +20% и округлением до шага 25 кэВ предел равен 675.0 кэВ (окно 0 - 675 кэВ)
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 675.0)
        assert main_win.results_viewer.spin_max_energy is not None
        assert main_win.results_viewer.spin_emin is not None
        assert main_win.results_viewer.spin_emax is not None
        self.assertAlmostEqual(main_win.results_viewer.spin_max_energy.value(), 675.0)
        self.assertAlmostEqual(main_win.results_viewer.spin_emin.value(), 0.0)
        self.assertAlmostEqual(main_win.results_viewer.spin_emax.value(), 675.0)

        # Проверка связи сигнала detector_metadata_updated от OrchestratorSession к ResultsViewer
        main_win.orchestrator_session.detector_metadata_updated.emit(["Head_1", "Head_2"], 500.0)
        updated_items = [
            main_win.results_viewer.combo_detector.itemText(item_idx)
            for item_idx in range(main_win.results_viewer.combo_detector.count())
        ]
        self.assertEqual(updated_items, ["Head_1", "Head_2", "Все детекторы (Сумма)"])
        self.assertAlmostEqual(main_win.results_viewer._max_energy, 500.0)
        main_win.close()

    def test_results_viewer_aspect_ratio_splitter_and_spectrum_controls(self) -> None:
        """
        Проверка привязки ширины панели проекции к соотношению сторон матрицы (Cols / Rows)
        с учетом боковой шкалы Lookup Table (ui.histogram), а также интерактивного управления
        max_energy, energy_bins и автоподбором по источнику.
        """
        viewer = ResultsViewer()
        viewer.resize(1200, 320)
        viewer.show()
        if app is not None:
            app.processEvents()
        assert viewer.splitter is not None
        sizes_square = viewer.splitter.sizes()
        # Для квадратной проекции (aspect=1.0) при ширине 1200 и высоте 320 левая панель занимает ~365 пикселей
        # (холст ~242px + шкала LUT ~115px + отступы 8px, т.е. <45% ширины), а спектр получает большую часть ширины
        self.assertLess(sizes_square[0], sizes_square[1])
        self.assertLessEqual(sizes_square[0], int(1200 * 0.45))

        # Проверяем, что сам холст 2D-проекции (image_view.view) квадратный и не сжат шкалой Lookup Table справа
        canvas_w = float(viewer.image_view.view.width())
        canvas_h = float(viewer.image_view.view.height())
        lut_w = viewer._get_lut_histogram_width()
        self.assertGreaterEqual(lut_w, 110)
        self.assertAlmostEqual(canvas_w / canvas_h, 1.0, delta=0.08)

        # Если скрыть боковую шкалу Lookup Table (ui.histogram), ширина projection_panel уменьшается ровно на lut_w,
        # а ширина самого холста 2D-проекции (image_view.view) остается неизменной
        viewer.image_view.ui.histogram.hide()
        viewer._update_splitter_proportions()
        if app is not None:
            app.processEvents()
        sizes_without_lut = viewer.splitter.sizes()
        self.assertEqual(sizes_square[0] - sizes_without_lut[0], lut_w)

        viewer.image_view.ui.histogram.show()
        viewer._update_splitter_proportions()
        if app is not None:
            app.processEvents()
        self.assertEqual(viewer.splitter.sizes()[0], sizes_square[0])

        # При передаче узкой вертикальной матрицы (Rows=128, Cols=64 -> aspect=0.5) панель проекции становится уже
        viewer.set_projection_data(np.zeros((128, 64), dtype=np.float32))
        if app is not None:
            app.processEvents()
        self.assertAlmostEqual(viewer.projection_aspect, 0.5)
        sizes_narrow = viewer.splitter.sizes()
        self.assertLess(sizes_narrow[0], sizes_square[0])
        self.assertAlmostEqual(
            float(viewer.image_view.view.width()) / float(viewer.image_view.view.height()),
            0.5,
            delta=0.08,
        )

        # При передаче широкой матрицы (Rows=64, Cols=128 -> aspect=2.0) панель проекции расширяется (до лимита 45%)
        viewer.set_projection_data(np.zeros((64, 128), dtype=np.float32))
        if app is not None:
            app.processEvents()
        self.assertAlmostEqual(viewer.projection_aspect, 2.0)
        sizes_wide = viewer.splitter.sizes()
        self.assertGreater(sizes_wide[0], sizes_narrow[0])
        viewer.close()

        # Проверка управления max_energy, energy_bins и режима Авто в связке с MainWindow и OrchestratorSession
        main_win = MainWindow()
        # В сцене без источников дефолтный авто-предел равен 300 кэВ
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 300.0)
        self.assertAlmostEqual(float(main_win.orchestrator_session.max_energy / units.keV), 300.0)

        # Добавляем точечный источник Tc-99m (140.5 кэВ) -> авто-предел 170 кэВ (ceil(140.5 * 1.2 / 10) * 10)
        assert main_win.scene_vm.root_vm is not None
        tc_src = Source.from_point(
            activity=Float(10.0 * units.MBq),
            energy=Float(140.5 * units.keV),
            name="Tc99m_Source",
        )
        tc_vm = SourceViewModel(tc_src)
        main_win.scene_vm.add_node(main_win.scene_vm.root_vm, tc_vm)
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 170.0)
        self.assertAlmostEqual(float(main_win.orchestrator_session.max_energy / units.keV), 170.0)

        # Меняем энергию источника на F-18 (511 кэВ) -> авто-предел реактивно пересчитывается в 625 кэВ
        tc_vm.energy = 511.0
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 625.0)
        self.assertAlmostEqual(float(main_win.orchestrator_session.max_energy / units.keV), 625.0)

        # Ручное изменение max_energy и energy_bins через спинбоксы ResultsViewer отключает авто-режим
        assert main_win.results_viewer.spin_max_energy is not None
        assert main_win.results_viewer.spin_energy_bins is not None
        assert main_win.results_viewer.btn_auto_max_energy is not None
        assert main_win.results_viewer.spin_emin is not None
        assert main_win.results_viewer.spin_emax is not None
        main_win.results_viewer.spin_max_energy.setValue(400.0)
        main_win.results_viewer.spin_energy_bins.setValue(512)
        self.assertFalse(main_win.results_viewer.auto_max_energy)
        self.assertFalse(main_win.orchestrator_session.auto_max_energy)
        self.assertAlmostEqual(float(main_win.orchestrator_session.max_energy / units.keV), 400.0)
        self.assertEqual(main_win.orchestrator_session.energy_bins, 512)

        # При выключенном авто-режиме смена энергии источника на 74.6 кэВ не сбрасывает пользовательский предел 400 кэВ
        tc_vm.energy = 74.6
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 400.0)

        # Нажатие кнопки «Авто» возвращает автоматический подбор по источнику (74.6 кэВ -> 90 кэВ)
        main_win.results_viewer.btn_auto_max_energy.click()
        self.assertTrue(main_win.results_viewer.auto_max_energy)
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 90.0)
        self.assertAlmostEqual(float(main_win.orchestrator_session.max_energy / units.keV), 90.0)

        # Добавляем второй источник с многолинейчатым спектром Tb-161 (до 550.25 кэВ при малом выходе 0.04258%):
        # compute_auto_max_energy учитывает максимум по всем источникам сцены без отсечки слабых линий -> 675.0 кэВ
        tb_src = Source.from_point(
            activity=Float(20.0 * units.MBq),
            energy=[
                [25.65 * units.keV, 23.154],
                [74.57 * units.keV, 10.28],
                [550.25 * units.keV, 0.04258],
            ],
            name="Tb161_Source",
        )
        tb_vm = SourceViewModel(tb_src)
        main_win.scene_vm.add_node(main_win.scene_vm.root_vm, tb_vm)
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 675.0)
        self.assertAlmostEqual(float(main_win.orchestrator_session.max_energy / units.keV), 675.0)
        self.assertAlmostEqual(main_win.results_viewer.spin_emax.value(), 675.0)

        # При удалении высокоэнергетического источника Tb161_Source авто-предел реактивно возвращается к 90.0 кэВ (от первого источника 74.6 кэВ)
        main_win.scene_vm.remove_node(tb_vm)
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 90.0)
        self.assertAlmostEqual(float(main_win.orchestrator_session.max_energy / units.keV), 90.0)
        self.assertAlmostEqual(main_win.results_viewer.spin_emax.value(), 90.0)

        # При удалении последнего источника авто-предел возвращается к дефолтным 300.0 кэВ
        main_win.scene_vm.remove_node(tc_vm)
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 300.0)
        self.assertAlmostEqual(float(main_win.orchestrator_session.max_energy / units.keV), 300.0)

        # Даже если пользователь вручную сузил окно (например, до 0–90 кэВ) и отключил Авто,
        # открытие нового YAML-файла сбрасывает авто-режим и устанавливает полное окно новой сцены (0–675 кэВ)
        main_win.results_viewer.spin_max_energy.setValue(200.0)
        main_win.results_viewer.set_energy_window(0.0, 90.0)
        self.assertFalse(main_win.results_viewer.auto_max_energy)
        with patch(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            return_value=("examples/tb161_nema_protocol.yaml", "YAML files (*.yaml *.yml)"),
        ), patch("PySide6.QtWidgets.QMessageBox.critical"):
            main_win._on_open_yaml()
        self.assertTrue(main_win.results_viewer.auto_max_energy)
        self.assertAlmostEqual(main_win.results_viewer.max_energy, 675.0)
        self.assertAlmostEqual(main_win.results_viewer.spin_emin.value(), 0.0)
        self.assertAlmostEqual(main_win.results_viewer.spin_emax.value(), 675.0)
        main_win.close()

    def test_main_window_save_preserves_manager_fields_and_created_nodes(self) -> None:
        """
        Проверка сохранения seed, batch_capacity, enable_atomic_relaxation, photon_production_cut,
        start_time, logging и созданных с нуля узлов (PointSource, Source, WoodcockVoxelVolume)
        с последующей сборкой через SceneBuilder.
        """
        main_win = MainWindow()
        root = CompositeNode(name="World")
        main_win.scene_vm.load_scene(root)

        # Добавляем созданные с нуля узлы через SceneTreeWidget
        main_win.scene_tree._add_point_source(main_win.scene_vm.root_vm)
        main_win.scene_tree._add_voxel_source(main_win.scene_vm.root_vm)
        main_win.scene_tree._add_voxel_volume(main_win.scene_vm.root_vm)

        # Задаем начальную конфигурацию с seed, batch_capacity, релаксацией, start_time и logging
        main_win.current_config = SimulationConfig(
            simulation_manager=SimulationManagerConfig(
                start_time=5.0,
                stop_time=1e9,
                particles_number=1000,
                batch_capacity=250,
                min_energy=0.01,
                enable_atomic_relaxation=True,
                photon_production_cut=2.5 * units.keV,
                seed=42,
            ),
            data_manager=main_win.data_manager_vm.to_config(),
            scene=BaseCompositeNodeConfig(name="World"),
            logging=LoggingConfig(output_dir="custom_logs", file_level="DEBUG", console_level="INFO"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "saved_full.yaml"
            with patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "YAML files (*.yaml *.yml)"),
            ):
                with patch("PySide6.QtWidgets.QMessageBox.information"), patch(
                    "PySide6.QtWidgets.QMessageBox.critical"
                ) as mock_crit:
                    main_win._on_save_yaml()
                    self.assertFalse(mock_crit.called, f"QMessageBox.critical был вызван: {mock_crit.call_args}")

            loaded_cfg = load_simulation_config(str(save_path))
            self.assertEqual(loaded_cfg.simulation_manager.seed, 42)
            self.assertEqual(loaded_cfg.simulation_manager.batch_capacity, 250)
            self.assertTrue(loaded_cfg.simulation_manager.enable_atomic_relaxation)
            self.assertAlmostEqual(float(loaded_cfg.simulation_manager.photon_production_cut), float(2.5 * units.keV))
            self.assertAlmostEqual(float(loaded_cfg.simulation_manager.start_time), 5.0)
            self.assertIsNotNone(loaded_cfg.logging)
            assert loaded_cfg.logging is not None
            self.assertEqual(loaded_cfg.logging.output_dir, "custom_logs")
            self.assertEqual(loaded_cfg.logging.file_level, "DEBUG")
            self.assertEqual(loaded_cfg.logging.console_level, "INFO")

            # Проверяем, что SceneBuilder успешно собирает сцену без FileNotFoundError
            builder = SceneBuilder(base_dir=Path(tmp_dir))
            rebuilt_root = builder.build_scene(loaded_cfg.scene)
            assert isinstance(rebuilt_root, CompositeNode)
            self.assertEqual(len(rebuilt_root.childs), 3)
            first_child = rebuilt_root.childs[0]
            assert isinstance(first_child, Source)
            self.assertIsInstance(first_child.spatial, PointDistribution)

        main_win.close()


if __name__ == '__main__':
    unittest.main()
