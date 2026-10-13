"""
Модульные тесты для изолированных секций инспектора свойств (gui.views.inspectors)
и компонентного реестра InspectorRegistry.
"""
import unittest
from typing import Any, List
import numpy as np
import hepunits as units
from core.other.typing_definitions import Float

try:
    from PySide6.QtWidgets import QApplication, QTableWidgetItem, QComboBox
    app = QApplication.instance()
    if app is None:
        app = QApplication(['-platform', 'offscreen'])
except ImportError:
    app = None

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.nodes import CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from core.geometry.direct_collimators import CollimatorHoleShape, DirectParallelCollimator
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.source.sources import Source
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.gantry_node import GantryNode

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
from gui.viewmodels.data_handler_viewmodel import (
    DirectStreamHandlerViewModel,
    DataManagerViewModel,
)
from gui.views.property_inspector import PropertyInspector
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
        inspector.dose_voxel_size_changed.connect(lambda v: emitted_values.append(v))

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
        source_core = Source(
            distribution=np.ones((2, 2, 2)),
            activity=Float(10.0 * units.MBq),
            energy=Float(140.5 * units.keV),
        )
        source_vm = SourceViewModel(source_core)
        section = SourceInspectorSection()
        section.bind_viewmodel(source_vm)

        # 1. Начальное заполнение таблицы
        self.assertEqual(section.tbl_source_spectrum.rowCount(), 1)
        self.assertEqual(section.tbl_source_spectrum.item(0, 0).text(), "140.5000")
        self.assertEqual(section.tbl_source_spectrum.item(0, 1).text(), "1.0000")

        # 2. Добавление строки
        section._on_add_spectrum_row()
        self.assertEqual(section.tbl_source_spectrum.rowCount(), 2)
        self.assertEqual(len(source_vm.spectrum), 2)

        # 3. Валидный ввод с запятой и экспоненциальной записью
        section.tbl_source_spectrum.item(0, 0).setText("364,2")
        section._on_spectrum_cell_changed(0, 0)
        self.assertAlmostEqual(source_vm.spectrum[0][0], 364.2, places=3)

        section.tbl_source_spectrum.item(0, 0).setText("5.11e2")
        section._on_spectrum_cell_changed(0, 0)
        self.assertAlmostEqual(source_vm.spectrum[0][0], 511.0, places=3)

        # 4. LBYL: ввод неположительной энергии (<= 0) не выбрасывает исключение и откатывает к валидному
        section.tbl_source_spectrum.item(0, 0).setText("-100.0")
        section._on_spectrum_cell_changed(0, 0)
        self.assertGreater(source_vm.spectrum[0][0], 0.0)

        section.tbl_source_spectrum.item(0, 0).setText("0")
        section._on_spectrum_cell_changed(0, 0)
        self.assertGreater(source_vm.spectrum[0][0], 0.0)

        # 5. LBYL: ввод отрицательного веса (< 0) не приводит к исключению
        section.tbl_source_spectrum.item(0, 1).setText("-0.5")
        section._on_spectrum_cell_changed(0, 1)
        self.assertGreaterEqual(source_vm.spectrum[0][1], 0.0)

        # 6. LBYL: нечисловой ввод откатывает значение
        section.tbl_source_spectrum.item(0, 0).setText("bad_text")
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
        cell.setText("275,5")
        section._on_zipped_cell_changed(0, radius_col)
        self.assertAlmostEqual(proc_vm.zipped_variables["radius"][0], 275.5)

        # 2. Ограничение радиуса по LBYL (не менее 10.0)
        cell.setText("-50.0")
        section._on_zipped_cell_changed(0, radius_col)
        self.assertGreaterEqual(proc_vm.zipped_variables["radius"][0], 10.0)

        # 3. Невалидный ввод восстанавливает таблицу
        cell.setText("bad_val")
        section._on_zipped_cell_changed(0, radius_col)
        self.assertGreaterEqual(proc_vm.zipped_variables["radius"][0], 10.0)

    def test_collimator_shape_direct_lbyl_guard(self) -> None:
        """Верификация LBYL проверки неподдерживаемой формы каналов прямого коллиматора."""
        lead_mat = Material(name="Lead")
        col_node = DirectParallelCollimator(
            size=[50.0, 50.0, 30.0],
            hole_diameter=1.5,
            septa=0.2,
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
        from core.materials.materials import MaterialArray
        mat_arr = MaterialArray((2, 2, 2))
        voxel_node = WoodcockVoxelVolume(voxel_size=2.0, material_distribution=mat_arr, name="Phantom")
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
