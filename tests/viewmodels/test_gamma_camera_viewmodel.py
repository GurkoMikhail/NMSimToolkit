"""
Модульные тесты для GammaCameraViewModel и ее интеграции с CollimatorViewModel:
- Типизированный доступ к слоту collimator_vm (CollimatorViewModel);
- Управление габаритами детектора и толщиной коллиматора;
- Замена детерминированного и параметрического коллиматоров в слоте сцены;
- Реактивное уведомление об изменении геометрических параметров.
"""

import sys
import unittest
import hepunits as units
import numpy as np
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication(sys.argv)

import settings.database_setting as database_setting
from core.geometry.direct_collimators import (
    CollimatorHoleShape,
    DirectParallelCollimator,
)
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.other.typing_definitions import Float
from core.scene.gamma_camera_node import GammaCameraNode
from core.source.sources import Source
from gui.factories.gamma_camera_factory import create_default_gamma_camera
from gui.viewmodels import create_default_gamma_camera_vm
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel


class TestGammaCameraViewModel(unittest.TestCase):
    """
    Тестирование взаимодействия GammaCameraViewModel с универсальной CollimatorViewModel
    и строгой конвертации единиц в SourceViewModel.
    """

    def setUp(self) -> None:
        self.lead_material = database_setting.material_database["Pb"]
        self.vacuum_material = database_setting.material_database["Vacuum"]
        self.air_material = database_setting.material_database.get(
            "Air, Dry (near sea level)", database_setting.material_database["Vacuum"]
        )

    def test_default_camera_has_collimator_viewmodel(self) -> None:
        """Проверка, что коллиматор стандартной гамма-камеры является CollimatorViewModel."""
        camera_vm = create_default_gamma_camera_vm(name="SpectCameraDefault")
        col_vm = camera_vm.collimator_vm
        assert isinstance(col_vm, CollimatorViewModel)
        self.assertEqual(col_vm.collimator_kind, "parametric")
        self.assertEqual(col_vm.hole_shape, CollimatorHoleShape.HEXAGONAL)

    def test_camera_dimensions_propagation_to_collimator(self) -> None:
        """Проверка распространения размеров гамма-камеры на узел коллиматора."""
        camera_vm = create_default_gamma_camera_vm(name="SpectCameraResize")
        collimator_vm = camera_vm.collimator_vm
        assert isinstance(collimator_vm, CollimatorViewModel)

        # Изменение активного поля детектора
        camera_vm.detector_size = (500.0, 450.0)
        self.assertAlmostEqual(collimator_vm.size[0], 500.0)
        self.assertAlmostEqual(collimator_vm.size[1], 450.0)

        # Изменение толщины коллиматора
        camera_vm.collimator_thickness = 45.0
        self.assertAlmostEqual(collimator_vm.size[2], 45.0)

    def test_collimator_swap_in_scene(self) -> None:
        """Проверка замены параметрического коллиматора на детерминированный через SceneViewModel."""
        camera_vm = create_default_gamma_camera_vm(name="SpectCameraSwap")
        world_volume = Volume(name="World", geometry=Box(1000.0, 1000.0, 1000.0), material=self.lead_material)
        scene_vm = SceneViewModel(root_core_node=world_volume)
        assert scene_vm.root_vm is not None
        scene_vm.add_node(scene_vm.root_vm, camera_vm)

        old_collimator_vm = camera_vm.collimator_vm
        assert isinstance(old_collimator_vm, CollimatorViewModel)

        new_direct_core = DirectParallelCollimator(
            size=(
                Float(camera_vm.detector_size[0]),
                Float(camera_vm.detector_size[1]),
                Float(camera_vm.collimator_thickness),
            ),
            hole_diameter=Float(2.0),
            septa=Float(0.25),
            material=self.lead_material,
            hole_material=None,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            name="DirectCollimatorReplacement",
        )
        new_collimator_vm = create_node_viewmodel(new_direct_core)
        assert isinstance(new_collimator_vm, CollimatorViewModel)

        scene_vm.replace_node(old_collimator_vm, new_collimator_vm)

        # Камера видит новый коллиматор
        self.assertIs(camera_vm.collimator_vm, new_collimator_vm)
        self.assertEqual(camera_vm.slots.collimator, "DirectCollimatorReplacement")
        self.assertEqual(new_collimator_vm.collimator_kind, "direct")

    def test_source_viewmodel_strict_hepunits_conversion(self) -> None:
        """
        Регрессионный тест SourceViewModel:
        1. Для ядра с half_life=6.0 * units.hour геттер half_life возвращает 6.0 (часов).
        2. Для ядра с activity=500.0 * units.Bq выполняется строгая конвертация через units.MBq
           (500 Бк == 0.0005 МБк) без ложной эвристики и потери данных при обратной синхронизации.
        """
        core_source = Source.from_voxels(
            distribution=np.ones((2, 2, 2), dtype=Float),
            activity=Float(500.0 * units.Bq),
            half_life=Float(6.0 * units.hour),
        )
        source_vm = SourceViewModel(core_source)

        self.assertAlmostEqual(source_vm.half_life, 6.0, places=7)
        self.assertAlmostEqual(source_vm.activity, 500.0 * float(units.Bq) / float(units.MBq), places=9)
        self.assertAlmostEqual(source_vm.activity, 0.0005, places=9)
        self.assertAlmostEqual(float(core_source.initial_activity), float(500.0 * units.Bq), places=12)

        # Проверка обратной записи из ViewModel в ядро (МБк -> Бк в hepunits, часы -> нс в hepunits)
        source_vm.half_life = 12.0
        self.assertAlmostEqual(float(core_source.half_life), float(12.0 * units.hour), places=3)

        source_vm.activity = 2.5
        self.assertAlmostEqual(float(core_source.initial_activity), float(2.5 * units.MBq), places=9)


if __name__ == "__main__":
    unittest.main()

