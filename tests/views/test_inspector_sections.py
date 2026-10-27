"""
Модульные тесты для изолированных секций инспектора свойств (gui.views.inspectors)
и компонентного реестра InspectorRegistry.
"""
import unittest
from unittest.mock import patch
from typing import Any, Dict, List
import numpy as np
import hepunits as units
from core.other.typing_definitions import Float

try:
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QTableWidgetItem,
    )
    app = QApplication.instance()
    if app is None:
        app = QApplication(['-platform', 'offscreen'])
except ImportError:
    app = None

from core.config.exporter import SceneExporter
from core.config.orchestrator import Orchestrator
from core.config.template_binder import ConfigTemplateBinder
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material, MaterialArray
from core.scene.nodes import CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from core.geometry.direct_collimators import CollimatorHoleShape, DirectParallelCollimator
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.source.sources import Source
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.gantry_node import GantryNode

from gui.controllers.orchestrator_session import OrchestratorSession
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.procedure_viewmodel import (
    StepAndShootProcedureViewModel,
    CustomSweepProcedureViewModel,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.data_handler_viewmodel import (
    DirectStreamHandlerViewModel,
    DataManagerViewModel,
)
from gui.views.procedure_selector_widget import ProcedureSelectorWidget
from gui.views.property_inspector import PropertyInspector
from gui.views.simulation_jobs_widget import SimulationJobsWidget
from gui.views.inspectors import (
    BaseInspectorSection,
    CollimatorInspectorSection,
    DataHandlerInspectorSection,
    DataManagerInspectorSection,
    DoseGridInspectorSection,
    GammaCameraInspectorSection,
    GantryInspectorSection,
    GeneralInspectorSection,
    InspectorRegistry,
    ProcedureInspectorSection,
    SourceInspectorSection,
    TransformInspectorSection,
    VolumeInspectorSection,
    VoxelVolumeInspectorSection,
)


class TestInspectorSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if app is None:
            raise unittest.SkipTest("PySide6 не установлен в окружении")

    def setUp(self) -> None:
        self.water_mat = Material(name="Water")
        self.lead_mat = Material(name="Lead")

    def test_registry_registration_and_active_sections(self) -> None:
        """Проверка регистрации секций и корректности сопоставления с ViewModels."""
        registry = InspectorRegistry()
        gen = GeneralInspectorSection()
        trans = TransformInspectorSection()
        vol = VolumeInspectorSection()
        dose = DoseGridInspectorSection()
        col = CollimatorInspectorSection()
        vox = VoxelVolumeInspectorSection()
        src = SourceInspectorSection()
        cam = GammaCameraInspectorSection()
        gantry = GantryInspectorSection()
        proc = ProcedureInspectorSection()
        dh = DataHandlerInspectorSection()
        dm = DataManagerInspectorSection()

        all_secs: List[BaseInspectorSection] = [
            gen, trans, vol, dose, col, vox, src, cam, gantry, proc, dh, dm
        ]
        registry.register_sections(all_secs)
        self.assertEqual(len(registry.get_all_sections()), 12)

        # None -> пустой список
        self.assertEqual(registry.get_active_sections(None), [])

        # VolumeViewModel -> [General, Transform, Volume]
        vol_node = Volume(geometry=Box(20.0, 20.0, 20.0), material=self.water_mat, name="BoxNode")
        vol_vm = VolumeViewModel(vol_node)
        active_vol = registry.get_active_sections(vol_vm)
        self.assertIn(gen, active_vol)
        self.assertIn(trans, active_vol)
        self.assertIn(vol, active_vol)
        self.assertNotIn(col, active_vol)
        self.assertNotIn(dose, active_vol)

        # DoseGridViewModel -> [General, Transform, DoseGrid]
        dose_node = DoseGridNode(size=[30.0, 30.0, 30.0], dose_voxel_size=2.0, name="DoseScorer")
        dose_vm = DoseGridViewModel(dose_node)
        active_dose = registry.get_active_sections(dose_vm)
        self.assertIn(gen, active_dose)
        self.assertIn(trans, active_dose)
        self.assertIn(dose, active_dose)
        self.assertNotIn(vol, active_dose)

        # StepAndShootProcedureViewModel -> [Procedure]
        proc_vm = StepAndShootProcedureViewModel(steps=8)
        active_proc = registry.get_active_sections(proc_vm)
        self.assertEqual(active_proc, [proc])

        # DataManagerViewModel -> [DataManager]
        dm_vm = DataManagerViewModel()
        active_dm = registry.get_active_sections(dm_vm)
        self.assertEqual(active_dm, [dm])

    def test_can_handle_all_sections(self) -> None:
        """Проверка строгости предикатов can_handle для каждой из 12 секций."""
        gen = GeneralInspectorSection()
        trans = TransformInspectorSection()
        vol = VolumeInspectorSection()
        dose = DoseGridInspectorSection()
        col = CollimatorInspectorSection()
        vox = VoxelVolumeInspectorSection()
        src = SourceInspectorSection()
        cam = GammaCameraInspectorSection()
        gantry = GantryInspectorSection()
        proc = ProcedureInspectorSection()
        dh = DataHandlerInspectorSection()
        dm = DataManagerInspectorSection()

        node_vm = NodeViewModel(CompositeNode(name="Group"))
        self.assertTrue(gen.can_handle(node_vm))
        self.assertTrue(trans.can_handle(node_vm))
        self.assertFalse(vol.can_handle(node_vm))
        self.assertFalse(dose.can_handle(node_vm))
        self.assertFalse(proc.can_handle(node_vm))

        gantry_node = GantryNode(name="GantryNode")
        gantry_vm = GantryViewModel(gantry_node)
        self.assertTrue(gantry.can_handle(gantry_vm))
        self.assertFalse(vol.can_handle(gantry_vm))

        cam_node = GammaCameraNode(name="CameraNode")
        cam_vm = GammaCameraViewModel(cam_node)
        self.assertTrue(cam.can_handle(cam_vm))
        self.assertFalse(vol.can_handle(cam_vm))

        stream_vm = DirectStreamHandlerViewModel()
        self.assertTrue(dh.can_handle(stream_vm))
        self.assertFalse(proc.can_handle(stream_vm))
        self.assertFalse(dm.can_handle(stream_vm))

    def test_dose_grid_voxel_size_signal_forwarding(self) -> None:
        """Проверка генерации сигнала dose_voxel_size_changed и ретрансляции хостом PropertyInspector."""
        dose_node = DoseGridNode(size=[50.0, 50.0, 50.0], dose_voxel_size=5.0, name="DoseScorer")
        dose_vm = DoseGridViewModel(dose_node)

        inspector = PropertyInspector()
        emitted_values: List[float] = []
        inspector.dose_voxel_size_changed.connect(emitted_values.append)

        inspector.set_target_viewmodel(dose_vm)
        self.assertFalse(inspector.dose_grid_section.isHidden())

        # Меняем шаг вокселя через секцию
        inspector.dose_grid_section.spin_dose_grid_voxel.setValue(2.5)
        self.assertIn(2.5, emitted_values)
        self.assertAlmostEqual(dose_vm.dose_voxel_size, 2.5)

    def test_bind_and_unbind_lifecycle(self) -> None:
        """Проверка связывания, очистки и переключения состояния секций."""
        geo = Box(40.0, 50.0, 60.0)
        vol_node = Volume(geometry=geo, material=self.water_mat, name="TestBox")
        vol_vm = VolumeViewModel(vol_node)

        gen = GeneralInspectorSection()
        trans = TransformInspectorSection()
        vol = VolumeInspectorSection()

        gen.bind_viewmodel(vol_vm)
        self.assertEqual(gen.txt_name.text(), "TestBox")

        trans.bind_viewmodel(vol_vm)
        self.assertAlmostEqual(trans.spin_x.value(), 0.0)

        vol.bind_viewmodel(vol_vm)
        self.assertAlmostEqual(vol.spin_size_x.value(), 40.0)
        self.assertAlmostEqual(vol.spin_size_y.value(), 50.0)
        self.assertAlmostEqual(vol.spin_size_z.value(), 60.0)

        # Unbind очищает состояние
        gen.unbind()
        self.assertEqual(gen.txt_name.text(), "")
        self.assertIsNone(gen._current_vm)

        trans.unbind()
        self.assertIsNone(trans._current_vm)

        vol.unbind()
        self.assertIsNone(vol._current_vm)

    def test_property_inspector_switching_and_clearing(self) -> None:
        """Проверка динамического переключения и сброса выделения в PropertyInspector."""
        inspector = PropertyInspector()

        # 1. Выбор Volume
        vol_node = Volume(geometry=Box(20.0, 20.0, 20.0), material=self.water_mat, name="Box1")
        vol_vm = VolumeViewModel(vol_node)
        inspector.set_target_viewmodel(vol_vm)
        self.assertFalse(inspector.general_section.isHidden())
        self.assertFalse(inspector.transform_section.isHidden())
        self.assertFalse(inspector.volume_section.isHidden())
        self.assertTrue(inspector.gantry_section.isHidden())
        self.assertTrue(inspector.procedure_section.isHidden())

        # 2. Переключение на Gantry
        gantry_node = GantryNode(name="Gantry1")
        gantry_vm = GantryViewModel(gantry_node)
        inspector.set_target_viewmodel(gantry_vm)
        self.assertFalse(inspector.general_section.isHidden())
        self.assertFalse(inspector.transform_section.isHidden())
        self.assertTrue(inspector.volume_section.isHidden())
        self.assertFalse(inspector.gantry_section.isHidden())

        # 3. Переключение на Protocol
        proc_vm = StepAndShootProcedureViewModel(steps=16)
        inspector.set_target_viewmodel(proc_vm)
        self.assertTrue(inspector.general_section.isHidden())
        self.assertTrue(inspector.transform_section.isHidden())
        self.assertFalse(inspector.procedure_section.isHidden())

        # 4. Сброс (None)
        inspector.set_target_viewmodel(None)
        for section in inspector._registry.get_all_sections():
            self.assertTrue(section.isHidden())
            self.assertIsNone(section._current_vm)

    def test_source_section_spectrum_table_and_lbyl_validation(self) -> None:
        """Верификация операций с таблицей спектра источника и LBYL защиты от некорректного ввода."""
        source_core = Source.from_voxels(
            distribution=np.ones((2, 2, 2)),
            activity=Float(10.0 * units.MBq),
            energy=Float(140.5 * units.keV),
        )
        source_vm = SourceViewModel(source_core)
        section = SourceInspectorSection()
        section.bind_viewmodel(source_vm)

        # 1. Начальное заполнение таблицы
        self.assertEqual(section.tbl_source_spectrum.rowCount(), 1)
        cell_00 = section.tbl_source_spectrum.item(0, 0)
        cell_01 = section.tbl_source_spectrum.item(0, 1)
        assert cell_00 is not None and cell_01 is not None
        self.assertEqual(cell_00.text(), "140.5000")
        self.assertEqual(cell_01.text(), "1.0000")

        # 2. Добавление строки
        section._on_add_spectrum_row()
        self.assertEqual(section.tbl_source_spectrum.rowCount(), 2)
        self.assertEqual(len(source_vm.spectrum), 2)

        # 3. Валидный ввод с запятой и экспоненциальной записью
        item_energy = section.tbl_source_spectrum.item(0, 0)
        assert item_energy is not None
        item_energy.setText("364,2")
        section._on_spectrum_cell_changed(0, 0)
        self.assertAlmostEqual(source_vm.spectrum[0][0], 364.2, places=3)

        item_energy = section.tbl_source_spectrum.item(0, 0)
        assert item_energy is not None
        item_energy.setText("5.11e2")
        section._on_spectrum_cell_changed(0, 0)
        self.assertAlmostEqual(source_vm.spectrum[0][0], 511.0, places=3)

        # 4. LBYL: ввод неположительной энергии (<= 0) не выбрасывает исключение и откатывает к валидному
        item_energy = section.tbl_source_spectrum.item(0, 0)
        assert item_energy is not None
        item_energy.setText("-100.0")
        section._on_spectrum_cell_changed(0, 0)
        self.assertGreater(source_vm.spectrum[0][0], 0.0)

        item_energy = section.tbl_source_spectrum.item(0, 0)
        assert item_energy is not None
        item_energy.setText("0")
        section._on_spectrum_cell_changed(0, 0)
        self.assertGreater(source_vm.spectrum[0][0], 0.0)

        # 5. LBYL: ввод отрицательного веса (< 0) не приводит к исключению
        item_weight = section.tbl_source_spectrum.item(0, 1)
        assert item_weight is not None
        item_weight.setText("-0.5")
        section._on_spectrum_cell_changed(0, 1)
        self.assertGreaterEqual(source_vm.spectrum[0][1], 0.0)

        # 6. LBYL: нечисловой ввод откатывает значение
        item_energy = section.tbl_source_spectrum.item(0, 0)
        assert item_energy is not None
        item_energy.setText("bad_text")
        section._on_spectrum_cell_changed(0, 0)
        self.assertGreater(source_vm.spectrum[0][0], 0.0)

        # 7. Удаление строки
        section.tbl_source_spectrum.selectRow(1)
        section._on_remove_spectrum_row()
        self.assertEqual(section.tbl_source_spectrum.rowCount(), 1)

        # 8. Удаление последней строки оставляет дефолтный пик
        section._on_remove_spectrum_row()
        self.assertEqual(section.tbl_source_spectrum.rowCount(), 1)
        self.assertEqual(len(source_vm.spectrum), 1)

    def test_procedure_section_head_angles_parsing_and_lbyl(self) -> None:
        """Верификация LBYL парсинга углов детекторных головок в различных форматах ввода."""
        proc_vm = StepAndShootProcedureViewModel(steps=8, gamma_cameras=2)
        section = ProcedureInspectorSection()
        section.bind_viewmodel(proc_vm)

        # 1. Разделитель - запятая
        section.txt_proc_head_angles.setText("0.0, 90.0")
        section._on_proc_head_angles_edited()
        self.assertEqual(proc_vm.head_angles, [0.0, 90.0])

        # 2. Разделитель - точка с запятой и запятая в качестве десятичной точки
        section.txt_proc_head_angles.setText("0,0; 90,5; 180,0")
        section._on_proc_head_angles_edited()
        self.assertEqual(proc_vm.head_angles, [0.0, 90.5, 180.0])

        # 3. Разделитель - пробелы
        section.txt_proc_head_angles.setText("0.0 120.0 240.0")
        section._on_proc_head_angles_edited()
        self.assertEqual(proc_vm.head_angles, [0.0, 120.0, 240.0])

        # 4. Невалидный ввод откатывает текст в поле ввода без исключений
        section.txt_proc_head_angles.setText("invalid_text")
        section._on_proc_head_angles_edited()
        self.assertEqual(proc_vm.head_angles, [0.0, 120.0, 240.0])
        self.assertIn("120.0", section.txt_proc_head_angles.text())

    def test_procedure_section_zipped_variables_lbyl(self) -> None:
        """Верификация LBYL проверок в таблице связанных переменных (Zipped Variables)."""
        proc_vm = StepAndShootProcedureViewModel(steps=4, radius=250.0)
        section = ProcedureInspectorSection()
        section.bind_viewmodel(proc_vm)

        tbl = section.tbl_zipped_variables
        self.assertGreater(tbl.columnCount(), 0)
        self.assertEqual(tbl.rowCount(), 4)

        # Находим столбец radius
        radius_col = -1
        for col_idx in range(tbl.columnCount()):
            item = tbl.horizontalHeaderItem(col_idx)
            if item and "radius" in item.text():
                radius_col = col_idx
                break
        self.assertNotEqual(radius_col, -1)

        # 1. Редактирование с запятой
        cell = tbl.item(0, radius_col)
        assert cell is not None
        cell.setText("275,5")
        section._on_zipped_cell_changed(0, radius_col)
        self.assertAlmostEqual(proc_vm.zipped_variables["radius"][0], 275.5)

        # 2. Ограничение радиуса по LBYL (не менее 10.0)
        cell = tbl.item(0, radius_col)
        assert cell is not None
        cell.setText("-50.0")
        section._on_zipped_cell_changed(0, radius_col)
        self.assertGreaterEqual(proc_vm.zipped_variables["radius"][0], 10.0)

        # 3. Невалидный ввод восстанавливает таблицу
        cell = tbl.item(0, radius_col)
        assert cell is not None
        cell.setText("bad_val")
        section._on_zipped_cell_changed(0, radius_col)
        self.assertGreaterEqual(proc_vm.zipped_variables["radius"][0], 10.0)

    def test_collimator_shape_direct_lbyl_guard(self) -> None:
        """Верификация LBYL проверки неподдерживаемой формы каналов прямого коллиматора."""
        lead_mat = Material(name="Lead")
        col_node = DirectParallelCollimator(
            size=[Float(50.0), Float(50.0), Float(30.0)],
            hole_diameter=Float(1.5),
            septa=Float(0.2),
            material=lead_mat,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
        )
        col_vm = CollimatorViewModel(col_node)
        section = CollimatorInspectorSection()
        section.bind_viewmodel(col_vm)

        self.assertEqual(col_vm.hole_shape, CollimatorHoleShape.HEXAGONAL)

        # Попытка установить SQUARE в комбобоксе должна быть отклонена LBYL пре-чеком
        square_idx = -1
        for idx in range(section.combo_collimator_shape.count()):
            if section.combo_collimator_shape.itemData(idx) == CollimatorHoleShape.SQUARE:
                square_idx = idx
                break
        self.assertNotEqual(square_idx, -1)

        # Вызываем обработчик смены формы
        section._on_collimator_shape_changed(square_idx)
        # Форма должна остаться HEXAGONAL без возбуждения NotImplementedError
        self.assertEqual(col_vm.hole_shape, CollimatorHoleShape.HEXAGONAL)

    def test_transform_matrix_degeneracy_lbyl(self) -> None:
        """Верификация LBYL устойчивости к вырожденным или неортогональным матрицам."""
        node = CompositeNode(name="TestNode")
        node_vm = NodeViewModel(node)
        section = TransformInspectorSection()
        section.bind_viewmodel(node_vm)

        # Задаем неортогональную матрицу (det = 0)
        bad_matrix = np.zeros((4, 4), dtype=float)
        bad_matrix[3, 3] = 1.0
        node_vm.local_matrix = bad_matrix

        # Метод обновления полей не должен падать с исключением
        section.update_transform_fields()

    def test_voxel_volume_material_mapping_table(self) -> None:
        """Верификация заполнения таблицы материалов воксельного фантома и LBYL защиты."""
        mat_arr = MaterialArray((2, 2, 2))
        voxel_node = WoodcockVoxelVolume(voxel_size=Float(2.0), material_distribution=mat_arr, name="Phantom")
        voxel_vm = VoxelVolumeViewModel(voxel_node)
        section = VoxelVolumeInspectorSection()
        section.bind_viewmodel(voxel_vm)

        # Таблица должна иметь строки по числу материалов
        self.assertGreater(section.tbl_material_mapping.rowCount(), 0)

        # Валидная смена материала на Vacuum
        section._on_mapping_material_changed(0, "Vacuum")
        self.assertEqual(voxel_vm.material_list[0].name, "Vacuum")

        # LBYL: Невалидный материал не выбрасывает исключение и игнорируется
        section._on_mapping_material_changed(0, "NonExistentMaterial123")
        self.assertEqual(voxel_vm.material_list[0].name, "Vacuum")

        # LBYL: Невалидный индекс не приводит к исключению
        section._on_mapping_material_changed(999, "Vacuum")

    def test_gantry_section_angle_binding_and_external_signal(self) -> None:
        """Проверка работы GantryInspectorSection со свойством gantry_angle без суффикса _deg."""
        gantry_node = GantryNode(name="TestGantry")
        gantry_vm = GantryViewModel(gantry_node)
        section = GantryInspectorSection()
        section.bind_viewmodel(gantry_vm)
        gantry_vm.property_changed.connect(section.on_property_changed_externally)

        self.assertAlmostEqual(section.spin_gantry_angle.value(), 0.0)

        # Изменение через UI спинбокс обновляет gantry_vm.gantry_angle
        section.spin_gantry_angle.setValue(60.0)
        self.assertAlmostEqual(gantry_vm.gantry_angle, 60.0)

        # Внешнее изменение свойства gantry_angle во ViewModel обновляет спинбокс
        gantry_vm.gantry_angle = 135.0
        self.assertAlmostEqual(section.spin_gantry_angle.value(), 135.0)

    def test_simulation_jobs_widget_in_place_update_preserves_widgets_and_preview(self) -> None:
        """Проверка in-place обновления SimulationJobsWidget без пересоздания виджетов при неизменном числе задач."""
        widget = SimulationJobsWidget()
        initial_jobs: List[Dict[str, Any]] = [
            {"gantry_angle": 0.0, "radius": 250.0},
            {"gantry_angle": 90.0, "radius": 250.0},
            {"gantry_angle": 180.0, "radius": 250.0},
            {"gantry_angle": 270.0, "radius": 250.0},
        ]
        widget.set_jobs(initial_jobs)
        self.assertEqual(widget.table.rowCount(), 4)

        # Запоминаем ссылки на экземпляры виджетов и ячеек
        id_items = [widget.table.item(row_idx, 0) for row_idx in range(4)]
        radio_widgets = [widget.table.cellWidget(row_idx, 1) for row_idx in range(4)]
        param_items = [widget.table.item(row_idx, 2) for row_idx in range(4)]
        status_items = [widget.table.item(row_idx, 3) for row_idx in range(4)]
        progress_bars = [widget.table.cellWidget(row_idx, 4) for row_idx in range(4)]
        preview_buttons = [widget.table.cellWidget(row_idx, 5) for row_idx in range(4)]

        # Переводим первую задачу в завершенное состояние перед обновлением
        widget.on_job_finished(0)
        assert status_items[0] is not None
        self.assertEqual(status_items[0].text(), "Завершено")
        assert isinstance(progress_bars[0], QProgressBar)
        self.assertEqual(progress_bars[0].value(), 100)

        # Обновляем задачи с тем же количеством (4), но новыми параметрами
        updated_jobs: List[Dict[str, Any]] = [
            {"gantry_angle": 15.0, "radius": 280.0},
            {"gantry_angle": 105.0, "radius": 280.0},
            {"gantry_angle": 195.0, "radius": 280.0},
            {"gantry_angle": 285.0, "radius": 280.0},
        ]
        widget.set_jobs(updated_jobs)
        self.assertEqual(widget.table.rowCount(), 4)

        for row_idx in range(4):
            self.assertIs(widget.table.item(row_idx, 0), id_items[row_idx])
            self.assertIs(widget.table.cellWidget(row_idx, 1), radio_widgets[row_idx])
            self.assertIs(widget.table.item(row_idx, 2), param_items[row_idx])
            self.assertIs(widget.table.item(row_idx, 3), status_items[row_idx])
            self.assertIs(widget.table.cellWidget(row_idx, 4), progress_bars[row_idx])
            self.assertIs(widget.table.cellWidget(row_idx, 5), preview_buttons[row_idx])

        assert param_items[0] is not None
        self.assertIn("gantry_angle=15.0", param_items[0].text())
        self.assertIn("radius=280.0", param_items[0].text())
        self.assertEqual(status_items[0].text(), "В очереди")
        self.assertEqual(progress_bars[0].value(), 0)

        # Проверяем, что кнопка предпросмотра динамически читает актуальный словарь задачи
        previewed_jobs: List[Dict[str, Any]] = []
        widget.job_preview_requested.connect(previewed_jobs.append)
        btn_row_2 = preview_buttons[2]
        assert isinstance(btn_row_2, QPushButton)
        btn_row_2.click()
        self.assertEqual(len(previewed_jobs), 1)
        self.assertEqual(previewed_jobs[0], {"gantry_angle": 195.0, "radius": 280.0})

        # При изменении количества задач таблица корректно меняет число строк
        widget.set_jobs(updated_jobs[:2])
        self.assertEqual(widget.table.rowCount(), 2)
        widget.set_jobs([])
        self.assertEqual(widget.table.rowCount(), 0)

    def test_procedure_section_zipped_table_in_place_update_preserves_items(self) -> None:
        """Проверка in-place обновления ячеек tbl_zipped_variables без пересоздания QTableWidgetItem."""
        proc_vm = StepAndShootProcedureViewModel(steps=4, radius=250.0, start_angle=0.0, end_angle=180.0)
        section = ProcedureInspectorSection()
        section.bind_viewmodel(proc_vm)

        tbl = section.tbl_zipped_variables
        self.assertEqual(tbl.rowCount(), 4)
        self.assertEqual(tbl.columnCount(), 3)

        initial_headers = [tbl.horizontalHeaderItem(col_idx) for col_idx in range(3)]
        initial_cells = [
            [tbl.item(row_idx, col_idx) for col_idx in range(3)]
            for row_idx in range(4)
        ]

        # Меняем радиус и начальный угол при неизменном числе шагов и наборе столбцов
        proc_vm.radius = 310.0
        proc_vm.start_angle = 45.0
        section.update_ui_from_vm()

        for col_idx in range(3):
            self.assertIs(tbl.horizontalHeaderItem(col_idx), initial_headers[col_idx])
            for row_idx in range(4):
                self.assertIs(tbl.item(row_idx, col_idx), initial_cells[row_idx][col_idx])

        cell_angle_0 = tbl.item(0, 0)
        cell_radius_0 = tbl.item(0, 1)
        assert cell_angle_0 is not None and cell_radius_0 is not None
        self.assertEqual(cell_angle_0.text(), "45.0000")
        self.assertEqual(cell_radius_0.text(), "310.0000")

        # При изменении числа шагов таблица корректно перестраивается под новую размерность
        proc_vm.steps = 6
        section.update_ui_from_vm()
        self.assertEqual(tbl.rowCount(), 6)

    def test_transform_section_update_transform_fields_skips_rebind(self) -> None:
        """Проверка, что update_transform_fields обновляет значения спинбоксов без повторного вызова bind_variable/unbind."""
        vol_node = Volume(geometry=Box(20.0, 20.0, 20.0), material=self.water_mat, name="MovingBox")
        vol_vm = VolumeViewModel(vol_node)
        vol_vm.bindings["translation.x"] = "table_x"

        section = TransformInspectorSection()
        section.bind_viewmodel(vol_vm)
        self.assertEqual(section.spin_x.bound_variable, "table_x")

        with (
            patch.object(section.spin_x, "bind_variable") as mock_bind_x,
            patch.object(section.spin_x, "unbind") as mock_unbind_x,
            patch.object(section.spin_y, "bind_variable") as mock_bind_y,
            patch.object(section.spin_y, "unbind") as mock_unbind_y,
        ):
            new_mat = np.eye(4, dtype=float)
            new_mat[0, 3] = 42.5
            new_mat[1, 3] = -18.0
            new_mat[2, 3] = 95.0
            vol_vm.local_matrix = new_mat

            self.assertAlmostEqual(section.spin_x.value(), 42.5)
            self.assertAlmostEqual(section.spin_y.value(), -18.0)
            self.assertAlmostEqual(section.spin_z.value(), 95.0)
            mock_bind_x.assert_not_called()
            mock_unbind_x.assert_not_called()
            mock_bind_y.assert_not_called()
            mock_unbind_y.assert_not_called()

        # При каскадном сигнале transform_changed без изменения самой local_matrix
        # повторного вызова setValue не происходит
        with patch.object(section.spin_x, "setValue") as mock_set_val_x:
            vol_vm.transform_changed.emit()
            mock_set_val_x.assert_not_called()

        # Внешнее изменение реестра привязок синхронизирует привязки полей
        vol_vm.bindings["translation.y"] = "table_y"
        section.on_property_changed_externally("bindings", None)
        self.assertEqual(section.spin_y.bound_variable, "table_y")

    def test_orchestrator_session_generate_jobs_fast_without_scene_export(self) -> None:
        """Проверка быстрой генерации задач в OrchestratorSession без холостого экспорта всей 3D-сцены."""
        # 1. Изначально сцена пуста (без камер), но после первого generate_jobs
        # добавление первой камеры в сцену автоматически подхватывается и монтирует гантри
        root = CompositeNode(name="World")
        scene_vm = SceneViewModel(root)
        proc_vm = StepAndShootProcedureViewModel(steps=4, gamma_cameras=1, radius=250.0, start_angle=0.0, end_angle=180.0)
        selector = ProcedureSelectorWidget(scene_vm=scene_vm)
        selector.set_procedure(proc_vm)
        session = OrchestratorSession(scene_vm=scene_vm, procedure_vm=proc_vm)

        try:
            initial_jobs = session.generate_jobs()
            self.assertEqual(len(initial_jobs), 4)

            # Добавляем первую камеру в сцену после того, как _observed_scene уже установлен
            assert scene_vm.root_vm is not None
            cam_node = GammaCameraNode(name="GammaCamera_1")
            cam_vm = GammaCameraViewModel(cam_node)
            scene_vm.add_node(scene_vm.root_vm, cam_vm)

            gantry_nodes = [n for n in scene_vm.all_nodes() if isinstance(n, GantryViewModel)]
            self.assertEqual(len(gantry_nodes), 1)
            self.assertIs(cam_vm.parent_vm, gantry_nodes[0])

            # 2. Изменение proc_vm.gamma_cameras динамически досоздает и удаляет камеры в сцене
            proc_vm.gamma_cameras = 2
            cameras_after_inc = [n for n in scene_vm.all_nodes() if isinstance(n, GammaCameraViewModel)]
            self.assertEqual(len(cameras_after_inc), 2)

            proc_vm.gamma_cameras = 1
            cameras_after_dec = [n for n in scene_vm.all_nodes() if isinstance(n, GammaCameraViewModel)]
            self.assertEqual(len(cameras_after_dec), 1)

            # 3. При изменении параметров ракурса (start_angle, radius) ни OrchestratorSession,
            # ни ProcedureSelectorWidget не выполняют экспорт сцены или повторный sync_with_scene
            with (
                patch.object(SceneExporter, "export_to_config") as mock_export,
                patch.object(ConfigTemplateBinder, "apply_bindings") as mock_binder,
                patch.object(Orchestrator, "__init__", return_value=None) as mock_orch_init,
                patch.object(proc_vm, "sync_with_scene") as mock_sync,
            ):
                proc_vm.start_angle = 30.0
                proc_vm.radius = 290.0
                updated_jobs = session.generate_jobs()

                mock_export.assert_not_called()
                mock_binder.assert_not_called()
                mock_orch_init.assert_not_called()
                mock_sync.assert_not_called()

                self.assertEqual(len(updated_jobs), 4)
                self.assertAlmostEqual(updated_jobs[0]["gantry_angle"], float(np.radians(30.0)), places=5)
                self.assertAlmostEqual(updated_jobs[0]["radius"], 290.0, places=4)

            # 4. Массив индивидуальных радиусов по шагам (Body Contour) сохраняется при sync_with_scene
            proc_vm.set_variable("radius", [290.0, 310.0, 330.0, 350.0], sweep_type="zipped")
            proc_vm.sync_with_scene(scene_vm)
            self.assertEqual(proc_vm.zipped_variables["radius"], [290.0, 310.0, 330.0, 350.0])
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()


