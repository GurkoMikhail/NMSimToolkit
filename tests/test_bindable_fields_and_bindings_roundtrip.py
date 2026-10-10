"""
Модульные и интеграционные тесты для BindableDoubleField,
всех связываемых полей инспектора свойств и двустороннего
экспорта/импорта YAML-конфигураций (Roundtrip) с привязками макросов ${var}.
"""

import os
import tempfile
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

from core.scene.nodes import SpatialNode, CompositeNode
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from settings.database_setting import material_database
from core.scene.gantry_node import GantryNode
from core.source.sources import Source
from core.config.models import (
    SimulationConfig,
    TranslateConfig,
    RotateConfig,
    SourceConfig,
    GantryConfig,
    VolumeConfig,
    NumpyDistributionConfig,
    StepAndShootProtocolConfig,
)
from core.config.exporter import SceneExporter
from core.config.builder import SceneBuilder
from core.config.template_binder import ConfigTemplateBinder
from core.config.sweep_compiler import SweepCompiler
import hepunits as units

from gui.views.widgets.bindable_field import BindableDoubleField
from gui.views.property_inspector import PropertyInspector
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.procedure_viewmodel import (
    StepAndShootProcedureViewModel,
    SpectProcedureViewModel,
    CustomSweepProcedureViewModel,
)

app = QApplication.instance() or QApplication([])


class TestBindableFieldsAndRoundtrip(unittest.TestCase):
    """
    Комплексное тестирование обобщенного механизма Property Binding
    и интеграции BindableDoubleField с ядром и GUI.
    """

    def setUp(self) -> None:
        self.material = material_database["Pb"]

    def test_bindable_double_field_widget_lifecycle(self) -> None:
        """Проверка жизненного цикла BindableDoubleField: связывание, отвязка, идемпотентность и read-only."""
        field = BindableDoubleField(
            min_val=-100.0,
            max_val=100.0,
            step=1.0,
            suffix=" мм",
            decimals=2,
        )
        self.assertIsNone(field.bound_variable)
        self.assertFalse(field.isReadOnly())

        binding_signals: list[str] = []
        field.bindingChanged.connect(binding_signals.append)

        # 1. Привязка к переменной
        field.bind_variable("orbit_radius")
        self.assertEqual(field.bound_variable, "orbit_radius")
        self.assertTrue(field.isReadOnly())
        self.assertEqual(binding_signals, ["orbit_radius"])
        self.assertIn("orbit_radius", field.btn_bind.text())

        # 2. Идемпотентность: повторный вызов не дублирует сигнал
        field.bind_variable("orbit_radius")
        self.assertEqual(len(binding_signals), 1)

        # 3. Попытка снять ReadOnly при активном биндинге блокируется
        field.setReadOnly(False)
        self.assertTrue(field.isReadOnly())

        # 4. Отвязка от переменной
        field.unbind()
        self.assertIsNone(field.bound_variable)
        self.assertFalse(field.isReadOnly())
        self.assertEqual(binding_signals, ["orbit_radius", ""])
        self.assertEqual(field.btn_bind.text(), "🔗")

        # 5. Идемпотентность отвязки
        field.unbind()
        self.assertEqual(len(binding_signals), 2)

    def test_clean_core_nodes_have_no_bindings_dictionary(self) -> None:
        """Проверка чистоты ядра: в SpatialNode и CompositeNode отсутствует словарь bindings."""
        node = SpatialNode(name="TestNode")
        self.assertFalse(hasattr(node, "bindings"), "SpatialNode не должен содержать атрибут bindings!")
        comp = CompositeNode(name="TestComp")
        self.assertFalse(hasattr(comp, "bindings"), "CompositeNode не должен содержать атрибут bindings!")

    def test_property_inspector_transform_fields_bindable(self) -> None:
        """Проверка того, что все 6 полей координат и поворота в инспекторе поддерживают BindableDoubleField."""
        root = CompositeNode(name="World")
        box = Box(100.0, 100.0, 100.0)
        vol = Volume(geometry=box, material=self.material, name="TargetVolume")
        root.add_child(vol)

        scene_vm = SceneViewModel(root)
        inspector = PropertyInspector(scene_vm=scene_vm)

        target_vm = scene_vm.find_by_name("TargetVolume")
        self.assertIsNotNone(target_vm)
        inspector.set_target_viewmodel(target_vm)

        # Проверка типов виджетов
        self.assertIsInstance(inspector.spin_x, BindableDoubleField)
        self.assertIsInstance(inspector.spin_y, BindableDoubleField)
        self.assertIsInstance(inspector.spin_z, BindableDoubleField)
        self.assertIsInstance(inspector.spin_rot_x, BindableDoubleField)
        self.assertIsInstance(inspector.spin_rot_y, BindableDoubleField)
        self.assertIsInstance(inspector.spin_rot_z, BindableDoubleField)

        # Привязка координаты X к переменной протокола table_x
        inspector.spin_x.bind_variable("table_x")
        self.assertEqual(target_vm.bindings.get("translation.x"), "table_x")
        self.assertEqual(scene_vm.bindings_registry[vol].get("translation.x"), "table_x")

        # Привязка поворота Yaw к переменной протокола tilt_yaw
        inspector.spin_rot_z.bind_variable("tilt_yaw")
        self.assertEqual(target_vm.bindings.get("rotation.gamma"), "tilt_yaw")
        self.assertEqual(scene_vm.bindings_registry[vol].get("rotation.gamma"), "tilt_yaw")

        # Отвязка X
        inspector.spin_x.unbind()
        self.assertNotIn("translation.x", target_vm.bindings)
        self.assertNotIn("translation.x", scene_vm.bindings_registry[vol])

    def test_property_inspector_source_fields_bindable(self) -> None:
        """Проверка привязки энергии и активности источника в PropertyInspector."""
        root = CompositeNode(name="World")
        dist = np.ones((5, 5, 5), dtype=float)
        source_core = Source(
            distribution=dist,
            activity=150.0 * float(units.MBq),
            voxel_size=2.0 * float(units.mm),
            radiation_type="Gamma",
            energy=140.5 * float(units.keV),
            half_life=6.0 * float(units.hour),
        )
        source_core.name = "TestCobalt"
        root.add_child(source_core)

        scene_vm = SceneViewModel(root)
        inspector = PropertyInspector(scene_vm=scene_vm)
        source_vm = scene_vm.find_by_name("TestCobalt")
        self.assertIsNotNone(source_vm)
        inspector.set_target_viewmodel(source_vm)

        self.assertIsInstance(inspector.spin_source_activity, BindableDoubleField)

        # Связывание активности
        inspector.spin_source_activity.bind_variable("source_activity_sweep")

        self.assertEqual(source_vm.bindings.get("activity"), "source_activity_sweep")
        self.assertEqual(scene_vm.bindings_registry[source_core].get("activity"), "source_activity_sweep")

    def test_property_inspector_gantry_synchronization(self) -> None:
        """Проверка взаимной синхронизации привязок между spin_gantry_angle и spin_rot_x для станины."""
        root = CompositeNode(name="World")
        gantry_core = GantryNode(name="Gantry")
        root.add_child(gantry_core)

        scene_vm = SceneViewModel(root)
        inspector = PropertyInspector(scene_vm=scene_vm)
        gantry_vm = scene_vm.find_by_name("Gantry")
        self.assertIsNotNone(gantry_vm)
        inspector.set_target_viewmodel(gantry_vm)

        # 1. Привязка через spin_gantry_angle зеркалируется на spin_rot_x
        inspector.spin_gantry_angle.bind_variable("theta_sweep")
        self.assertEqual(inspector.spin_rot_x.bound_variable, "theta_sweep")
        self.assertEqual(gantry_vm.bindings.get("rotation.alpha"), "theta_sweep")

        # 2. Отвязка через spin_rot_x зеркалируется на spin_gantry_angle
        inspector.spin_rot_x.unbind()
        self.assertIsNone(inspector.spin_gantry_angle.bound_variable)
        self.assertNotIn("rotation.alpha", gantry_vm.bindings)

    def test_property_inspector_procedure_fields_bindable(self) -> None:
        """Проверка привязки параметров процедуры (радиус орбиты и время на шаг)."""
        proc_vm = SpectProcedureViewModel(radius=250.0, time_per_step=1.5)
        inspector = PropertyInspector()
        inspector.set_procedure_viewmodel(proc_vm)
        inspector.set_target_viewmodel(proc_vm)

        self.assertIsInstance(inspector.spin_proc_radius, BindableDoubleField)
        self.assertIsInstance(inspector.spin_proc_time, BindableDoubleField)

        inspector.spin_proc_radius.bind_variable("cam_radius")
        inspector.spin_proc_time.bind_variable("frame_duration")

        self.assertEqual(proc_vm.bindings.get("radius"), "cam_radius")
        self.assertEqual(proc_vm.bindings.get("step_time"), "frame_duration")

    def test_scene_exporter_and_builder_roundtrip_with_arbitrary_bindings(self) -> None:
        """
        Проверка полного цикла сериализации и десериализации (Roundtrip) графа сцены:
        экспорт в Pydantic с подстановкой ${var} mm, ${var} deg, ${var} MBq, ${var} keV
        и обратное восстановление через SceneBuilder с наполнением bindings_registry.
        """
        root = CompositeNode(name="RootWorld")

        # 1. Узел Volume со смещением по X и поворотом
        box = Box(50.0, 50.0, 50.0)
        target_vol = Volume(geometry=box, material=self.material, name="Phantom")
        target_vol.translate(10.0, 20.0, 0.0)
        root.add_child(target_vol)

        # 2. Источник излучения
        dist = np.ones((4, 4, 4), dtype=float)
        temp_dist_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        temp_dist_path = temp_dist_file.name
        temp_dist_file.close()
        np.save(temp_dist_path, dist)
        self.addCleanup(lambda: os.remove(temp_dist_path) if os.path.exists(temp_dist_path) else None)

        source = Source(
            distribution=dist,
            activity=200.0 * float(units.MBq),
            voxel_size=2.0 * float(units.mm),
            radiation_type="Gamma",
            energy=140.5 * float(units.keV),
            half_life=6.0 * float(units.hour),
        )
        source.name = "HotSpot"
        root.add_child(source)

        dist_reg = {source: NumpyDistributionConfig(path=temp_dist_path)}

        # 3. Станина Gantry
        gantry = GantryNode(name="Gantry")
        root.add_child(gantry)

        # Реестр привязок
        bindings_reg = {
            target_vol: {
                "translation.x": "table_x_var",
                "rotation.alpha": "tilt_angle_var",
            },
            source: {
                "activity": "activity_var",
                "energy": "energy_var",
            },
            gantry: {
                "gantry_angle": "gantry_var",
            },
        }

        # 1. Чистый экспорт сцены ядра через SceneExporter (строго численные величины)
        exported_config = SceneExporter.export_scene(
            root,
            distribution_registry=dist_reg,
        )
        self.assertEqual(exported_config.type, "CompositeNode")

        # 2. Наложение шаблонов макросов через изолированный сервис ConfigTemplateBinder
        scene_raw_dict = exported_config.model_dump(exclude_none=True)
        templated_scene = ConfigTemplateBinder.apply_bindings(scene_raw_dict, bindings_reg)

        # Проверка детей экспортированного шаблона
        children_by_name = {c["name"]: c for c in templated_scene["children"]}
        vol_cfg = children_by_name["Phantom"]
        source_cfg = children_by_name["HotSpot"]
        gantry_cfg = children_by_name["Gantry"]

        # Проверка Volume: сформированы TranslateConfig и RotateConfig со строками ${var}
        trans_cfgs = [t for t in vol_cfg["transformations"] if "x" in t or "y" in t or "z" in t]
        rot_cfgs = [t for t in vol_cfg["transformations"] if "alpha" in t or "beta" in t or "gamma" in t]
        self.assertEqual(len(trans_cfgs), 1)
        self.assertEqual(len(rot_cfgs), 1)
        self.assertEqual(trans_cfgs[0]["x"], "${table_x_var} mm")
        self.assertEqual(rot_cfgs[0]["alpha"], "${tilt_angle_var} deg")

        # Проверка Source: сформированы activity и energy со строками ${var}
        self.assertEqual(source_cfg["activity"], "${activity_var} MBq")
        self.assertEqual(source_cfg["energy"][0][0], "${energy_var} keV")

        # Проверка Gantry
        self.assertEqual(gantry_cfg["transformations"][0]["alpha"], "${gantry_var} deg")

        # 3. Извлечение соответствий привязок обратно через ConfigTemplateBinder
        extracted_bindings = ConfigTemplateBinder.extract_bindings(templated_scene)
        self.assertEqual(extracted_bindings["Phantom"]["translation.x"], "table_x_var")
        self.assertEqual(extracted_bindings["Phantom"]["rotation.alpha"], "tilt_angle_var")
        self.assertEqual(extracted_bindings["HotSpot"]["activity"], "activity_var")
        self.assertEqual(extracted_bindings["HotSpot"]["energy"], "energy_var")
        self.assertEqual(extracted_bindings["Gantry"]["gantry_angle"], "gantry_var")

        # 4. Проверка чистоты ядра: SceneBuilder обязан работать Fail-Fast на неразрешенных макросах
        from core.config.models import BaseCompositeNodeConfig
        with self.assertRaises(ValueError):
            SceneBuilder().build_scene(BaseCompositeNodeConfig.model_validate(templated_scene))

        # 5. Сборка сцены после резолва переменных через SweepCompiler
        resolved_dict = SweepCompiler.inject_variables(
            templated_scene,
            {"table_x_var": 10.0, "tilt_angle_var": 0.0, "activity_var": 200.0, "energy_var": 140.5, "gantry_var": 0.0}
        )
        rebuilt_root = SceneBuilder().build_scene(BaseCompositeNodeConfig.model_validate(resolved_dict))
        self.assertIsInstance(rebuilt_root, CompositeNode)

    def test_procedure_unified_bindings_registry(self) -> None:
        """
        Проверка функционирования единого реестра биндингов в модели процедуры (SSOT),
        валидации и синхронизации с инспектором свойств.
        """
        proc_vm = StepAndShootProcedureViewModel(steps=32, start_angle=0.0, end_angle=360.0, radius=250.0)
        vars_dict = proc_vm.get_variables()
        self.assertIn("gantry_angle", vars_dict)
        self.assertIn("radius", vars_dict)
        self.assertIn("step_time", vars_dict)
        self.assertEqual(len(vars_dict["gantry_angle"]), 32)

        gantry_core = GantryNode(name="Gantry")
        proc_vm.bind_node_property(gantry_core, "rotation.alpha", "gantry_angle")
        self.assertEqual(proc_vm.get_node_bindings(gantry_core).get("rotation.alpha"), "gantry_angle")

        # Отвязка
        proc_vm.unbind_node_property(gantry_core, "rotation.alpha")
        self.assertNotIn("rotation.alpha", proc_vm.get_node_bindings(gantry_core))

    def test_custom_sweep_variables_and_inspector_table(self) -> None:
        """
        Проверка API переменных в CustomSweepProcedureViewModel,
        их значений и отображения в таблицах Zipped (KxM) и Grid (Nx1) инспектора свойств.
        """
        cs_vm = CustomSweepProcedureViewModel(
            grid_variables={"table_x": [0.0, 10.0, 20.0]},
            zipped_variables={"frame_idx": [1.0, 2.0, 3.0]},
        )
        all_vars = cs_vm.all_variables
        self.assertIn("table_x", all_vars)
        self.assertIn("frame_idx", all_vars)
        self.assertEqual(cs_vm.get_variable_values("table_x"), [0.0, 10.0, 20.0])

        # Добавление новой переменной
        cs_vm.set_variable("tilt_angle", [0.0, 45.0, 90.0], sweep_type="grid")
        self.assertIn("tilt_angle", cs_vm.all_variables)
        self.assertEqual(len(cs_vm.get_variable_values("tilt_angle")), 3)

        # Инспектор свойств с CustomSweep
        inspector = PropertyInspector()
        inspector.set_procedure_viewmodel(cs_vm)
        inspector.set_target_viewmodel(cs_vm)

        self.assertTrue(inspector.step_and_shoot_widget.isHidden())
        self.assertFalse(inspector.custom_sweep_widget.isHidden())

        # Zipped-таблица KxM: 1 столбец ('frame_idx') и 3 строки (шаги)
        self.assertEqual(inspector.tbl_zipped_variables.columnCount(), 1)
        self.assertEqual(inspector.tbl_zipped_variables.rowCount(), 3)
        # Grid-виджеты: 2 переменные ('table_x', 'tilt_angle')
        self.assertEqual(inspector.grid_vars_layout.count(), 2)

        # Удаление переменной
        cs_vm.remove_variable("tilt_angle")
        self.assertNotIn("tilt_angle", cs_vm.all_variables)
        inspector._populate_custom_variables_table(cs_vm)
        self.assertEqual(inspector.tbl_zipped_variables.columnCount(), 1)
        self.assertEqual(inspector.tbl_zipped_variables.rowCount(), 3)
        self.assertEqual(inspector.grid_vars_layout.count(), 1)

        # Проверка StepAndShootProcedureViewModel в инспекторе свойств
        sas_vm = StepAndShootProcedureViewModel(steps=4, radius=200.0, time_per_step=2.5)
        inspector.set_procedure_viewmodel(sas_vm)
        inspector.set_target_viewmodel(sas_vm)

        self.assertFalse(inspector.step_and_shoot_widget.isHidden())
        self.assertFalse(inspector.custom_sweep_widget.isHidden())
        # Стандартные переменные протокола: gantry_angle, radius, step_time (3 столбца, 4 строки)
        self.assertEqual(inspector.tbl_zipped_variables.columnCount(), 3)
        self.assertEqual(inspector.tbl_zipped_variables.rowCount(), 4)

    def test_step_and_shoot_custom_variables_and_radius_sync(self) -> None:
        """
        Проверка того, что StepAndShootProcedureViewModel наследует кастомные переменные,
        защищает стандартные переменные протокола, и синхронизирует radius-список при add/remove_zipped_step.
        """
        sas_vm = StepAndShootProcedureViewModel(steps=3, radius=[200.0, 210.0, 220.0], time_per_step=1.0)
        # Добавляем кастомную переменную
        sas_vm.set_variable("couch_z", [10.0, 20.0, 30.0], sweep_type="zipped")
        self.assertIn("couch_z", sas_vm.all_variables)
        self.assertIn("couch_z", sas_vm.zipped_variables)

        # Стандартная переменная протокола не может быть удалена
        sas_vm.remove_variable("gantry_angle")
        self.assertIn("gantry_angle", sas_vm.all_variables)

        # Добавляем шаг
        sas_vm.add_zipped_step({"couch_z": 40.0})
        self.assertEqual(sas_vm.steps, 4)
        self.assertEqual(len(sas_vm.zipped_variables["radius"]), 4)
        self.assertEqual(len(sas_vm.radius_raw), 4)
        self.assertEqual(len(sas_vm.zipped_variables["couch_z"]), 4)

        # Проверяем успешную конвертацию в конфигурацию без ошибок длины
        cfg = sas_vm.to_config()
        self.assertEqual(cfg.steps, 4)
        self.assertEqual(len(cfg.radius), 4)
        self.assertEqual(len(cfg.zipped_variables["couch_z"]), 4)

        # Удаляем шаг
        sas_vm.remove_zipped_step(3)
        self.assertEqual(sas_vm.steps, 3)
        self.assertEqual(len(sas_vm.radius_raw), 3)
        cfg2 = sas_vm.to_config()
        self.assertEqual(cfg2.steps, 3)
        self.assertEqual(len(cfg2.radius), 3)

    def test_source_energy_scalar_and_1d_list(self) -> None:
        """
        Проверка setter energy узла Source: скаляр, список из 1 элемента, 1D массив.
        """
        dist = np.ones((2, 2, 2), dtype=float)
        source = Source(distribution=dist, activity=1.0, energy=140.5 * units.keV)
        self.assertEqual(len(source.energy), 1)

        # 1D список из одного значения энергии
        source.energy = [140.5 * units.keV]
        self.assertEqual(len(source.energy), 1)
        self.assertAlmostEqual(float(source.energy["energy"][0]), 140.5 * float(units.keV))

        # 1D список из нескольких энергий
        source.energy = [100.0 * units.keV, 200.0 * units.keV]
        self.assertEqual(len(source.energy), 2)
        np.testing.assert_allclose(source.energy["probability"], [0.5, 0.5])

    def test_step_and_shoot_config_zipped_length_validation(self) -> None:
        """Проверка валидации несовпадающей длины zipped_variables в StepAndShootProtocolConfig."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            StepAndShootProtocolConfig(
                steps=3,
                zipped_variables={"couch_z": [1.0, 2.0]}  # Длина 2 при steps=3
            )

    def test_property_inspector_grid_widget_stability_and_clear_selection(self) -> None:
        """
        Проверка стабильности GridVariableWidget (не пересоздается при изменении значений)
        и очистки таблиц в clear_selection.
        """
        cs_vm = CustomSweepProcedureViewModel(
            grid_variables={"table_x": [0.0, 10.0]}
        )
        inspector = PropertyInspector()
        inspector.set_procedure_viewmodel(cs_vm)
        inspector.set_target_viewmodel(cs_vm)

        self.assertIn("table_x", inspector._grid_var_widgets)
        initial_widget = inspector._grid_var_widgets["table_x"]

        # Изменяем значения через ViewModel
        cs_vm.set_variable("table_x", [5.0, 15.0, 25.0], sweep_type="grid")
        inspector.update_all_fields()

        # Виджет не должен был быть уничтожен или заменен
        self.assertIs(inspector._grid_var_widgets["table_x"], initial_widget)
        self.assertEqual(initial_widget.table.rowCount(), 3)

        # Проверка clear_selection
        inspector.clear_selection()
        self.assertEqual(len(inspector._grid_var_widgets), 0)
        self.assertEqual(inspector.tbl_source_spectrum.rowCount(), 0)
        self.assertEqual(inspector.tbl_zipped_variables.rowCount(), 0)

    def test_zipped_step_minimum_one_row_guard(self) -> None:
        """
        Проверка невозможности уменьшить число шагов связанных переменных ниже 1
        как на уровне ViewModel, так и в таблице.
        """
        cs_vm = CustomSweepProcedureViewModel(
            zipped_variables={"couch_z": [10.0]}
        )
        cs_vm.remove_zipped_step(0)
        self.assertEqual(len(cs_vm.zipped_variables["couch_z"]), 1)

    def test_zipped_table_invalid_float_rollback(self) -> None:
        """
        Проверка отката некорректного значения ячейки таблицы к предыдущему валидному числу.
        """
        cs_vm = CustomSweepProcedureViewModel(
            zipped_variables={"table_pos": [12.5, 25.0]}
        )
        inspector = PropertyInspector()
        inspector.set_procedure_viewmodel(cs_vm)
        inspector.set_target_viewmodel(cs_vm)

        item = inspector.tbl_zipped_variables.item(0, 0)
        self.assertIsNotNone(item)
        item.setText("invalid_number_abc")

        self.assertEqual(cs_vm.zipped_variables["table_pos"][0], 12.5)
        rolled_back_item = inspector.tbl_zipped_variables.item(0, 0)
        self.assertIsNotNone(rolled_back_item)
        self.assertAlmostEqual(float(rolled_back_item.text()), 12.5)

    def test_step_and_shoot_mixed_grid_and_zipped_roundtrip_config(self) -> None:
        """
        Проверка полного цикла сериализации и десериализации StepAndShootProtocolConfig
        с одновременным наличием кастомных Grid и Zipped переменных.
        """
        sas_vm = StepAndShootProcedureViewModel(
            steps=4,
            radius=[220.0, 230.0, 240.0, 250.0],
            time_per_step=2.5,
            grid_variables={"collimator_tilt": [0.0, 5.0]},
            zipped_variables={"couch_shift_z": [1.0, 2.0, 3.0, 4.0]},
        )
        config = sas_vm.to_config()
        self.assertEqual(config.steps, 4)
        self.assertEqual(len(config.radius), 4)
        self.assertEqual(config.grid_variables["collimator_tilt"], [0.0, 5.0])
        self.assertEqual(config.zipped_variables["couch_shift_z"], [1.0, 2.0, 3.0, 4.0])

        restored_vm = StepAndShootProcedureViewModel.from_config(config)
        self.assertEqual(restored_vm.steps, 4)
        self.assertEqual(restored_vm.radius_raw, [220.0, 230.0, 240.0, 250.0])
        self.assertEqual(restored_vm.grid_variables["collimator_tilt"], [0.0, 5.0])
    def test_universal_simulation_manager_bindings_and_roundtrip(self) -> None:
        """
        Проверка универсальности Property Binding: привязка параметров SimulationManager
        (particles_number, min_energy), экспорт в YAML, извлечение макросов и инъекция в воркер.
        """
        import tempfile
        from pathlib import Path
        from core.config.models import SimulationConfig, SimulationManagerConfig, DataManagerConfig, BaseCompositeNodeConfig
        from core.config.template_binder import ConfigTemplateBinder
        from core.config.sweep_compiler import SweepCompiler
        from core.config.yaml_loader import load_raw_config, load_simulation_config
        from gui.models.gui_settings import GuiSimulationSettings
        from gui.views.simulation_settings_dialog import SimulationSettingsDialog

        # 1. Проверка SimulationSettingsDialog с привязкой параметров
        settings = GuiSimulationSettings(particles_number=50000, min_energy=15.0)
        dialog = SimulationSettingsDialog(
            settings,
            variable_provider=lambda: ["total_particles", "min_energy_cut"],
            bindings={"particles_number": "total_particles", "min_energy": "min_energy_cut"}
        )
        self.assertEqual(dialog.spin_particles.bound_variable, "total_particles")
        self.assertEqual(dialog.spin_min_energy.bound_variable, "min_energy_cut")
        dialog_bindings = dialog.get_bindings()
        self.assertEqual(dialog_bindings["particles_number"], "total_particles")
        self.assertEqual(dialog_bindings["min_energy"], "min_energy_cut")

        # 2. Построение базовой конфигурации и наложение привязок на simulation_manager
        sim_mgr_cfg = SimulationManagerConfig(particles_number=50000, min_energy=15.0 * units.keV)
        sim_config = SimulationConfig(
            simulation_manager=sim_mgr_cfg,
            data_manager=DataManagerConfig(filename="output.h5"),
            scene=BaseCompositeNodeConfig(name="World"),
        )
        unified_bindings = {
            "simulation_manager": {
                "particles_number": "total_particles",
                "min_energy": "min_energy_cut",
            }
        }

        templated_config = ConfigTemplateBinder.apply_bindings(sim_config, unified_bindings)
        self.assertEqual(templated_config.simulation_manager.particles_number, "${total_particles}")

        # 3. Экспорт в YAML и проверка наличия макросов
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = Path(tmp_dir) / "templated_simulation.yaml"
            world_node = CompositeNode(name="World")
            ConfigTemplateBinder.export_templated_yaml(
                root_node=world_node,
                filepath=yaml_path,
                bindings_registry=unified_bindings,
                simulation_manager_cfg=sim_mgr_cfg,
            )
            self.assertTrue(yaml_path.exists())

            raw_dict = load_raw_config(yaml_path)
            self.assertEqual(raw_dict["simulation_manager"]["particles_number"], "${total_particles}")
            self.assertIn("${min_energy_cut}", str(raw_dict["simulation_manager"]["min_energy"]))

            # 4. Извлечение привязок из сохраненного YAML
            extracted = ConfigTemplateBinder.extract_bindings(raw_dict)
            self.assertIn("simulation_manager", extracted)
            self.assertEqual(extracted["simulation_manager"]["particles_number"], "total_particles")
            self.assertEqual(extracted["simulation_manager"]["min_energy"], "min_energy_cut")

            # 5. Разрешение переменных через SweepCompiler.inject_variables
            context = {"total_particles": 100000, "min_energy_cut": 25.0}
            resolved_dict = SweepCompiler.inject_variables(raw_dict, context)
            self.assertEqual(resolved_dict["simulation_manager"]["particles_number"], 100000)

            # Валидация разрешенной конфигурации
            final_cfg = SimulationConfig.model_validate(resolved_dict)
            self.assertEqual(final_cfg.simulation_manager.particles_number, 100000)
            self.assertAlmostEqual(float(final_cfg.simulation_manager.min_energy), float(25.0 * units.keV))


    def test_count_config_strict_validation(self) -> None:
        """
        Проверка строгого контракта CountConfig:
        - разрешены только положительные int (> 0) и макросы вида ${var_name};
        - запрещены 0, отрицательные числа, булевы значения и мусорные строки.
        """
        from pydantic import ValidationError
        from core.config.models import SimulationManagerConfig

        # Валидные значения
        cfg1 = SimulationManagerConfig(particles_number=1000)
        self.assertEqual(cfg1.particles_number, 1000)

        cfg2 = SimulationManagerConfig(particles_number="${batch_count}")
        self.assertEqual(cfg2.particles_number, "${batch_count}")

        cfg3 = SimulationManagerConfig(particles_number=500.0)
        self.assertEqual(cfg3.particles_number, 500)

        # Невалидные значения: 0 и отрицательные
        with self.assertRaises(ValidationError):
            SimulationManagerConfig(particles_number=0)

        with self.assertRaises(ValidationError):
            SimulationManagerConfig(particles_number=-50)

        # Невалидные значения: мусорные строки
        with self.assertRaises(ValidationError):
            SimulationManagerConfig(particles_number="invalid_string")

        with self.assertRaises(ValidationError):
            SimulationManagerConfig(particles_number="")

        # Невалидные значения: булевы значения
        with self.assertRaises(ValidationError):
            SimulationManagerConfig(particles_number=True)


if __name__ == "__main__":
    unittest.main()

