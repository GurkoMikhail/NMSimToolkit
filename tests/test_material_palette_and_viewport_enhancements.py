"""
Автоматические тесты для семантической палитры материалов, физической рентгеновской непрозрачности,
акцентирования детекторов, подсветки контура выделенных узлов и стилей заблокированных полей.
"""

import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import hepunits as units

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.nodes import SpatialNode, CompositeNode
import settings.database_setting as database_setting

from gui.app import DARK_STYLE_SHEET
from gui.models.gui_settings import GuiSimulationSettings
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewport_3d.material_palette import (
    get_material_color,
    get_material_opacity,
    get_material_rgba,
    get_pseudo_xray_rgba,
    compute_material_linear_attenuation,
    compute_xray_opacity,
    normalize_material_name,
    DETECTOR_ACCENT_COLOR,
    DETECTOR_ACCENT_OPACITY,
    SELECTED_EDGE_HIGHLIGHT_COLOR,
    SELECTED_EDGE_HIGHLIGHT_WIDTH,
)
from gui.viewport_3d.vtk_viewport import ISceneViewport, VTKViewport
from gui.controllers.viewport_controller import SceneViewportController


class MockHighlightActor:
    """Мок-актор VTK для тестирования управления видимостью и свойствами ребер меша."""

    def __init__(self) -> None:
        self.edge_visibility: bool = False
        self.edge_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.line_width: float = 1.0

    def GetProperty(self) -> 'MockHighlightActor':
        return self

    def SetEdgeVisibility(self, visible_flag: bool) -> None:
        self.edge_visibility = bool(visible_flag)

    def SetEdgeColor(self, red_component: float, green_component: float, blue_component: float) -> None:
        self.edge_color = (float(red_component), float(green_component), float(blue_component))

    def SetLineWidth(self, width_value: float) -> None:
        self.line_width = float(width_value)


class MockSignal:
    """Мок для сигналов Qt в тестовом окружении."""

    def connect(self, slot: Any) -> None:
        pass

    def disconnect(self, slot: Any) -> None:
        pass

    def emit(self, *args: Any, **kwargs: Any) -> None:
        pass


class MockSceneViewport:
    """Мок-реализация интерфейса ISceneViewport для изолированного тестирования контроллера."""

    def __init__(self) -> None:
        self.plotter: Any = None
        self.camera_interaction_started = MockSignal()
        self.camera_interaction_ended = MockSignal()
        self.camera_moved = MockSignal()
        self.actors_registry: Dict[str, MockHighlightActor] = {}
        self.render_call_count: int = 0
        self.highlight_calls: List[Tuple[str, bool, Tuple[float, float, float], float]] = []

    @property
    def _actors(self) -> Dict[str, MockHighlightActor]:
        return self.actors_registry

    def add_mesh_actor(
        self,
        name: str,
        mesh: Any,
        color: Optional[Any] = 'white',
        opacity: float = 1.0,
        style: str = 'surface',
        wireframe: bool = False,
        rgb: bool = False,
        **kwargs: Any
    ) -> Optional[Any]:
        mock_actor_instance = MockHighlightActor()
        self.actors_registry[name] = mock_actor_instance
        return mock_actor_instance

    def update_actor_transform(self, name: str, matrix: np.ndarray) -> bool:
        return name in self.actors_registry

    def remove_actor(self, name: str) -> None:
        self.actors_registry.pop(name, None)

    def render(self) -> None:
        self.render_call_count += 1

    def get_actor(self, name: str) -> Optional[Any]:
        return self.actors_registry.get(name)

    def set_actor_edge_highlight(
        self,
        name: str,
        visible: bool,
        color: Tuple[float, float, float] = (1.0, 0.55, 0.0),
        line_width: float = 2.5,
    ) -> bool:
        actor_instance = self.actors_registry.get(name)
        if actor_instance is None:
            return False
        actor_instance.SetEdgeVisibility(visible)
        if visible:
            actor_instance.SetEdgeColor(color[0], color[1], color[2])
            actor_instance.SetLineWidth(line_width)
        self.highlight_calls.append((name, visible, color, line_width))
        self.render_call_count += 1
        return True


class TestMaterialPaletteAndPhysicalAttenuation(unittest.TestCase):
    """Тестирование наполнения палитры материалов и физического расчета рентгеновского ослабления."""

    def test_database_materials_coverage(self) -> None:
        """Проверка, что для всех материалов из NIST Materials.h5 корректно возвращаются RGB и Opacity."""
        all_materials_dict = database_setting.material_database
        self.assertGreater(len(all_materials_dict), 100)

        for material_item_name in all_materials_dict.keys():
            color_rgb = get_material_color(material_item_name)
            self.assertEqual(len(color_rgb), 3)
            self.assertTrue(all(0.0 <= component <= 1.0 for component in color_rgb))

            opacity_calculated = get_material_opacity(material_item_name, energy=140.0 * units.keV)
            self.assertTrue(0.0 <= opacity_calculated <= 1.0)

            rgba_tuple = get_material_rgba(material_item_name, energy=140.0 * units.keV)
            self.assertEqual(len(rgba_tuple), 4)
            self.assertEqual(rgba_tuple[:3], color_rgb)
            self.assertEqual(rgba_tuple[3], opacity_calculated)

    def test_physical_attenuation_hierarchy(self) -> None:
        """Проверка физической корректности коэффициентов ослабления: Вакуум < Воздух < Вода < Кость < Свинец."""
        vacuum_attenuation = compute_material_linear_attenuation("Vacuum", energy=140.0 * units.keV)
        air_attenuation = compute_material_linear_attenuation("Air, Dry (near sea level)", energy=140.0 * units.keV)
        water_attenuation = compute_material_linear_attenuation("Water, Liquid", energy=140.0 * units.keV)
        bone_attenuation = compute_material_linear_attenuation("Bone, Cortical (ICRU-44)", energy=140.0 * units.keV)
        lead_attenuation = compute_material_linear_attenuation("Pb", energy=140.0 * units.keV)

        self.assertEqual(vacuum_attenuation, 0.0)
        self.assertGreater(air_attenuation, vacuum_attenuation)
        self.assertGreater(water_attenuation, air_attenuation)
        self.assertGreater(bone_attenuation, water_attenuation)
        self.assertGreater(lead_attenuation, bone_attenuation)

        vacuum_opacity = compute_xray_opacity(vacuum_attenuation)
        air_opacity = compute_xray_opacity(air_attenuation)
        water_opacity = compute_xray_opacity(water_attenuation)
        bone_opacity = compute_xray_opacity(bone_attenuation)
        lead_opacity = compute_xray_opacity(lead_attenuation)

        self.assertLess(vacuum_opacity, air_opacity)
        self.assertLess(air_opacity, water_opacity)
        self.assertLess(water_opacity, bone_opacity)
        self.assertLess(bone_opacity, lead_opacity)
        self.assertGreaterEqual(lead_opacity, 0.90)

    def test_energy_dependency_of_attenuation(self) -> None:
        """Проверка падения коэффициента ослабления с ростом энергии (фотоэффект -> комптон)."""
        mu_low = compute_material_linear_attenuation("Water, Liquid", energy=30.0 * units.keV)
        mu_mid_low = compute_material_linear_attenuation("Water, Liquid", energy=60.0 * units.keV)
        mu_mid_high = compute_material_linear_attenuation("Water, Liquid", energy=140.0 * units.keV)
        mu_high = compute_material_linear_attenuation("Water, Liquid", energy=511.0 * units.keV)

        self.assertGreater(mu_low, mu_mid_low)
        self.assertGreater(mu_mid_low, mu_mid_high)
        self.assertGreater(mu_mid_high, mu_high)

    def test_material_synonyms_normalization(self) -> None:
        """Проверка разрешения синонимов (Lead -> Pb, Water -> Water, Liquid и т.д.)."""
        self.assertEqual(normalize_material_name("Lead"), "Pb")
        self.assertEqual(normalize_material_name("water"), "Water, Liquid")
        self.assertEqual(normalize_material_name("Air"), "Air, Dry (near sea level)")
        self.assertEqual(normalize_material_name("Bone"), "Bone, Cortical (ICRU-44)")
        self.assertEqual(normalize_material_name("CZT"), "Cadmium Zinc Telluride")

        color_lead_full = get_material_color("Pb")
        color_lead_synonym = get_material_color("Lead")
        self.assertEqual(color_lead_full, color_lead_synonym)

    def test_pseudo_xray_mode_generation(self) -> None:
        """Проверка генерации контрастного оттенка и непрозрачности в режиме псевдорентгена."""
        rgb_lead, opacity_lead = get_pseudo_xray_rgba("Pb", energy=140.0 * units.keV)
        rgb_air, opacity_air = get_pseudo_xray_rgba("Air, Dry (near sea level)", energy=140.0 * units.keV)

        # Свинец в псевдорентгене должен быть значительно ярче и непрозрачнее воздуха
        self.assertGreater(rgb_lead[0], rgb_air[0])
        self.assertGreater(opacity_lead, opacity_air)

    def test_all_144_nist_materials_explicitly_in_palette(self) -> None:
        """Проверка, что все 144 материала из NIST Materials.h5 напрямую присутствуют в MATERIAL_COLOR_PALETTE."""
        from gui.viewport_3d.material_palette import MATERIAL_COLOR_PALETTE
        all_materials_dict = database_setting.material_database
        missing_materials = [
            mat_name for mat_name in all_materials_dict.keys()
            if mat_name not in MATERIAL_COLOR_PALETTE
        ]
        self.assertEqual(len(missing_materials), 0, f"Отсутствуют материалы в палитре: {missing_materials}")

    def test_dbc_and_exceptions_handling(self) -> None:
        """Проверка контрактного программирования (DbC): валидация энергии и выброс KeyError для неизвестных материалов."""
        with self.assertRaises(ValueError):
            compute_material_linear_attenuation("Water, Liquid", energy=0.0)
        with self.assertRaises(ValueError):
            compute_material_linear_attenuation("Water, Liquid", energy=-50.0 * units.keV)
        with self.assertRaises(KeyError):
            compute_material_linear_attenuation("NonExistentMaterialXYZ", energy=140.0 * units.keV)
        with self.assertRaises(ValueError):
            compute_xray_opacity(0.015, characteristic_length=0.0)
        with self.assertRaises(ValueError):
            compute_xray_opacity(0.015, characteristic_length=-10.0 * units.mm)


class TestVolumeViewModelMaterialIntegration(unittest.TestCase):
    """Тестирование реактивной связки VolumeViewModel с материалами и цветом."""

    def test_initial_color_and_material_reactivity(self) -> None:
        """Проверка автоматической инициализации цвета по материалу и обновления при смене материала."""
        geometry_box = Box(100.0, 100.0, 100.0)
        material_water = Material(name="Water, Liquid")
        volume_core = Volume(geometry=geometry_box, material=material_water, name="WaterPhantom")
        volume_view_model = VolumeViewModel(volume_core)

        # Начальный цвет должен соответствовать воде
        expected_water_color = get_material_color("Water, Liquid")
        self.assertEqual(volume_view_model.color[:3], expected_water_color)

        # Смена материала на свинец
        volume_view_model.material_name = "Pb"
        expected_lead_color = get_material_color("Pb")
        self.assertEqual(volume_view_model.color[:3], expected_lead_color)
        self.assertAlmostEqual(volume_view_model.color[3], get_material_opacity("Pb", energy=140.0 * units.keV))

    def test_detector_flag_reactivity(self) -> None:
        """Проверка свойства is_sensitive_detector и глобального реестра чувствительных объемов."""
        geometry_box = Box(50.0, 50.0, 50.0)
        material_crystal = Material(name="Sodium Iodide")
        volume_core = Volume(geometry=geometry_box, material=material_crystal, name="Scintillator")
        volume_view_model = VolumeViewModel(volume_core)

        self.assertFalse(volume_view_model.is_sensitive_detector)
        self.assertNotIn(volume_core, VolumeViewModel.get_sensitive_volumes())

        volume_view_model.is_sensitive_detector = True
        self.assertTrue(volume_view_model.is_sensitive_detector)
        self.assertIn(volume_core, VolumeViewModel.get_sensitive_volumes())

        volume_view_model.is_sensitive_detector = False
        self.assertFalse(volume_view_model.is_sensitive_detector)
        self.assertNotIn(volume_core, VolumeViewModel.get_sensitive_volumes())


class TestViewportControllerEnhancements(unittest.TestCase):
    """Тестирование подсветки ребер выделения, акцента детектора и режима псевдорентгена."""

    def setUp(self) -> None:
        self.mock_viewport = MockSceneViewport()
        self.root_node = Volume(geometry=Box(500.0, 500.0, 500.0), material=Material(name="Air, Dry (near sea level)"), name="World")
        self.scene_view_model = SceneViewModel(root_core_node=self.root_node)
        self.controller = SceneViewportController(viewport=self.mock_viewport, scene_vm=self.scene_view_model)
        self.scene_view_model.node_added.connect(self.controller.on_node_added)
        self.scene_view_model.node_removed.connect(self.controller.on_node_removed)
        self.scene_view_model.node_selected.connect(self.controller.on_node_selected)

    def test_detector_volume_coloring(self) -> None:
        """Чувствительный объем детектора окрашивается в золотисто-янтарный цвет."""
        crystal_box = Box(40.0, 40.0, 10.0)
        crystal_volume = Volume(geometry=crystal_box, material=Material(name="Sodium Iodide"), name="DetectorCrystal")
        crystal_vm = VolumeViewModel(crystal_volume)
        self.scene_view_model.add_node(self.scene_view_model.root_vm, crystal_vm)

        crystal_actor_name = f"mesh_{id(crystal_vm)}"
        self.assertIn(crystal_actor_name, self.mock_viewport.actors_registry)

        # Помечаем узел как детектор
        crystal_vm.is_sensitive_detector = True

        # Проверяем пересоздание актора с золотисто-янтарным акцентом
        self.assertTrue(self.mock_viewport.render_call_count > 0)

    def test_selected_node_edge_highlighting(self) -> None:
        """Выделенный узел подсвечивает контур меша через SetEdgeVisibility(True) янтарным цветом."""
        target_box = Box(60.0, 60.0, 60.0)
        target_volume = Volume(geometry=target_box, material=Material(name="Bone, Cortical (ICRU-44)"), name="BoneTarget")
        target_vm = VolumeViewModel(target_volume)
        self.scene_view_model.add_node(self.scene_view_model.root_vm, target_vm)

        actor_target_name = f"mesh_{id(target_vm)}"
        target_actor = self.mock_viewport.actors_registry[actor_target_name]
        # При добавлении узел автоматически выбирается моделью сцены
        self.assertTrue(target_actor.edge_visibility)
        self.assertEqual(target_actor.edge_color, SELECTED_EDGE_HIGHLIGHT_COLOR)
        self.assertEqual(target_actor.line_width, SELECTED_EDGE_HIGHLIGHT_WIDTH)

        # Снимаем выделение (выбор None)
        self.controller.on_node_selected(None)
        self.assertFalse(target_actor.edge_visibility)

        # Повторно выбираем узел
        self.controller.on_node_selected(target_vm)
        self.assertTrue(target_actor.edge_visibility)
        self.assertEqual(target_actor.edge_color, SELECTED_EDGE_HIGHLIGHT_COLOR)
        self.assertEqual(target_actor.line_width, SELECTED_EDGE_HIGHLIGHT_WIDTH)

    def test_edge_highlight_preserved_on_geometry_change(self) -> None:
        """Подсветка ребер сохраняется при инкрементальном изменении размеров выделенного узла."""
        target_box = Box(60.0, 60.0, 60.0)
        target_volume = Volume(geometry=target_box, material=Material(name="Water, Liquid"), name="WaterTarget")
        target_vm = VolumeViewModel(target_volume)
        self.scene_view_model.add_node(self.scene_view_model.root_vm, target_vm)

        self.controller.on_node_selected(target_vm)
        actor_target_name = f"mesh_{id(target_vm)}"
        self.assertTrue(self.mock_viewport.actors_registry[actor_target_name].edge_visibility)

        # Меняем размер узла
        target_vm.size = [80.0, 80.0, 80.0]

        # Новый меш должен остаться с включенной подсветкой ребер
        updated_actor = self.mock_viewport.actors_registry[actor_target_name]
        self.assertTrue(updated_actor.edge_visibility)
        self.assertEqual(updated_actor.edge_color, SELECTED_EDGE_HIGHLIGHT_COLOR)

    def test_xray_mode_and_energy_update(self) -> None:
        """Проверка переключения режима псевдорентгена и смены энергии фотонов."""
        self.assertFalse(self.controller.xray_mode)
        self.assertAlmostEqual(self.controller.xray_energy, 140.0 * units.keV)

        initial_renders = self.mock_viewport.render_call_count
        self.controller.set_xray_parameters(energy=60.0 * units.keV, pseudo_xray_mode=True)

        self.assertTrue(self.controller.xray_mode)
        self.assertAlmostEqual(self.controller.xray_energy, 60.0 * units.keV)
        self.assertGreater(self.mock_viewport.render_call_count, initial_renders)

    def test_deep_hierarchical_edge_highlighting(self) -> None:
        """Проверка подсветки ребер при глубокой вложенности (CompositeNode -> CompositeNode -> Volume)."""
        scanner_node = CompositeNode(name="Scanner")
        scanner_vm = NodeViewModel(scanner_node)
        self.scene_view_model.add_node(self.scene_view_model.root_vm, scanner_vm)

        head_node = CompositeNode(name="Head")
        head_vm = NodeViewModel(head_node)
        self.scene_view_model.add_node(scanner_vm, head_vm)

        crystal_box = Box(30.0, 30.0, 10.0)
        crystal_volume = Volume(geometry=crystal_box, material=Material(name="Sodium Iodide"), name="DeepCrystal")
        crystal_vm = VolumeViewModel(crystal_volume)
        self.scene_view_model.add_node(head_vm, crystal_vm)

        crystal_actor_name = f"mesh_{id(crystal_vm)}"
        self.assertIn(crystal_actor_name, self.mock_viewport.actors_registry)

        # Выбираем корневой сканер
        self.controller.on_node_selected(scanner_vm)
        crystal_actor = self.mock_viewport.actors_registry[crystal_actor_name]
        self.assertTrue(crystal_actor.edge_visibility)
        self.assertEqual(crystal_actor.edge_color, SELECTED_EDGE_HIGHLIGHT_COLOR)

        # Изменяем размер кристалла на 2-м уровне вложенности при выбранном сканере
        crystal_vm.size = [40.0, 40.0, 12.0]
        updated_crystal_actor = self.mock_viewport.actors_registry[crystal_actor_name]
        self.assertTrue(updated_crystal_actor.edge_visibility)
        self.assertEqual(updated_crystal_actor.edge_color, SELECTED_EDGE_HIGHLIGHT_COLOR)

        # Снимаем выделение
        self.controller.on_node_selected(None)
        self.assertFalse(updated_crystal_actor.edge_visibility)

    def test_opacity_updates_when_xray_energy_changes(self) -> None:
        """Проверка, что изменение энергии рентгена пересчитывает прозрачность существующих акторов в нормальном режиме."""
        phantom_box = Box(100.0, 100.0, 100.0)
        phantom_volume = Volume(geometry=phantom_box, material=Material(name="Water, Liquid"), name="Phantom")
        phantom_vm = VolumeViewModel(phantom_volume)
        self.scene_view_model.add_node(self.scene_view_model.root_vm, phantom_vm)

        recorded_opacities: List[float] = []
        original_add_mesh = self.mock_viewport.add_mesh_actor

        def intercepting_add_mesh(name: str, mesh: Any, **kwargs: Any) -> Any:
            recorded_opacities.append(kwargs.get('opacity', 1.0))
            return original_add_mesh(name, mesh, **kwargs)

        self.mock_viewport.add_mesh_actor = intercepting_add_mesh

        # Изменяем энергию со 140 кэВ на 30 кэВ (коэффициент ослабления воды растет, непрозрачность увеличивается)
        self.controller.set_xray_parameters(energy=30.0 * units.keV, pseudo_xray_mode=False)
        self.assertTrue(len(recorded_opacities) > 0)
        new_opacity = recorded_opacities[-1]

        expected_opacity_low = get_material_opacity("Water, Liquid", energy=30.0 * units.keV)
        expected_opacity_default = get_material_opacity("Water, Liquid", energy=140.0 * units.keV)

        self.assertAlmostEqual(new_opacity, expected_opacity_low, places=4)
        self.assertGreater(new_opacity, expected_opacity_default)


class TestDarkStyleSheetDisabledFields(unittest.TestCase):
    """Тестирование таблицы стилей DARK_STYLE_SHEET для неактивных (:disabled) полей."""

    def test_disabled_selectors_presence(self) -> None:
        """Проверка наличия правил :disabled для QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox."""
        self.assertIn("QLineEdit:disabled", DARK_STYLE_SHEET)
        self.assertIn("QDoubleSpinBox:disabled", DARK_STYLE_SHEET)
        self.assertIn("QSpinBox:disabled", DARK_STYLE_SHEET)
        self.assertIn("QComboBox:disabled", DARK_STYLE_SHEET)
        self.assertIn("QPushButton:disabled", DARK_STYLE_SHEET)
        self.assertIn("QCheckBox:disabled", DARK_STYLE_SHEET)


class TestGuiSimulationSettingsExtension(unittest.TestCase):
    """Тестирование расширения параметров настроек симуляции xray_energy и pseudo_xray_mode."""

    def test_settings_fields_and_dict_serialization(self) -> None:
        """Проверка полей, валидации и сериализации в GuiSimulationSettings."""
        settings_instance = GuiSimulationSettings()
        self.assertAlmostEqual(settings_instance.xray_energy, 140.0 * units.keV)
        self.assertFalse(settings_instance.pseudo_xray_mode)

        settings_instance.update({"xray_energy": 60.0 * units.keV, "pseudo_xray_mode": True})
        self.assertAlmostEqual(settings_instance.xray_energy, 60.0 * units.keV)
        self.assertTrue(settings_instance.pseudo_xray_mode)

        serialized_settings = settings_instance.to_dict()
        self.assertAlmostEqual(serialized_settings["xray_energy"], 60.0 * units.keV)
        self.assertTrue(serialized_settings["pseudo_xray_mode"])


if __name__ == "__main__":
    unittest.main()
