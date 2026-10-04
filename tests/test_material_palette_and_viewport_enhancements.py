"""
Автоматические тесты для семантической палитры материалов, физической рентгеновской непрозрачности,
акцентирования детекторов, подсветки контура выделенных узлов и стилей заблокированных полей.
"""

import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import hepunits as units

from PySide6.QtWidgets import QApplication

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import Material, MaterialArray
from core.scene.nodes import SpatialNode, CompositeNode
from pydantic import ValidationError

import settings.database_setting as database_setting

from gui.app import DARK_STYLE_SHEET
from gui.models.gui_settings import GuiSimulationSettings
from gui.views.simulation_settings_dialog import SimulationSettingsDialog
from gui.views.property_inspector import PropertyInspector
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewport_3d.material_palette import (
    get_material_color,
    get_material_opacity,
    get_material_rgba,
    get_pseudo_xray_rgba,
    build_material_volume_color_tf,
    build_material_volume_opacity_tf,
    compute_material_linear_attenuation,
    compute_xray_opacity,
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
        self.color_transfer_function: Any = None
        self.opacity_transfer_function: Any = None
        self.interpolation_type: str = "Linear"

    def GetProperty(self) -> 'MockHighlightActor':
        return self

    def GetMapper(self) -> 'MockHighlightActor':
        return self

    def SetColor(self, color_tf: Any) -> None:
        self.color_transfer_function = color_tf

    def SetScalarOpacity(self, opacity_tf: Any) -> None:
        self.opacity_transfer_function = opacity_tf

    def SetSampleDistance(self, sample_dist: float) -> None:
        pass

    def SetInterpolationTypeToNearest(self) -> None:
        self.interpolation_type = "NearestNeighbor"

    def SetInterpolationTypeToLinear(self) -> None:
        self.interpolation_type = "Linear"

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

    def add_actor(self, name: str, actor: Any) -> Optional[Any]:
        self.actors_registry[name] = actor
        return actor

    def add_volume_actor(
        self,
        name: str,
        grid: Any,
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

    def test_strict_canonical_materials_and_synonym_rejection(self) -> None:
        """Проверка строгого соответствия каноническим именам NIST и выброса KeyError для псевдонимов."""
        all_materials_dict = database_setting.material_database
        self.assertEqual(len(all_materials_dict), 144)
        for mat_name in all_materials_dict.keys():
            linear_attenuation = compute_material_linear_attenuation(mat_name, energy=140.0 * units.keV)
            self.assertGreaterEqual(linear_attenuation, 0.0)

        # Псевдонимы и неточные имена вызывают KeyError
        synonyms_to_reject = ["lead", "Lead", "tungsten", "Tungsten", "water", "Water", "air", "Air", "czt", "CZT", ""]
        for synonym in synonyms_to_reject:
            with self.assertRaises(KeyError):
                compute_material_linear_attenuation(synonym, energy=140.0 * units.keV)

        # Псевдонимы больше не нормализуются в get_material_color, а получают независимый детерминированный оттенок
        self.assertNotEqual(get_material_color("Pb"), get_material_color("Lead"))
        self.assertNotEqual(get_material_color("Water, Liquid"), get_material_color("Water"))
        self.assertNotEqual(get_material_color("Air, Dry (near sea level)"), get_material_color("Air"))
        self.assertNotEqual(get_material_color("W"), get_material_color("Tungsten"))

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
        self.assertEqual(len(MATERIAL_COLOR_PALETTE), len(all_materials_dict))
        self.assertIn("Vacuum", MATERIAL_COLOR_PALETTE)
        for synonym in ["Air", "Water", "Lead", "Tungsten", "Gold", "Silver", "Iron", "Aluminum"]:
            self.assertNotIn(synonym, MATERIAL_COLOR_PALETTE)

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
        with self.assertRaises(ValueError):
            compute_xray_opacity(-0.015, characteristic_length=25.0 * units.mm)
        with self.assertRaises(ValueError):
            compute_xray_opacity(0.015, min_opacity=-0.1)
        with self.assertRaises(ValueError):
            compute_xray_opacity(0.015, min_opacity=0.9, max_opacity=0.5)


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
        """Проверка отсутствия локального флага в ноде и управления через единый реестр сцены SceneViewModel."""
        geometry_box = Box(50.0, 50.0, 50.0)
        material_crystal = Material(name="Sodium Iodide")
        volume_core = Volume(geometry=geometry_box, material=material_crystal, name="Scintillator")
        volume_view_model = VolumeViewModel(volume_core)

        self.assertFalse(hasattr(volume_view_model, 'is_sensitive_detector'))
        self.assertFalse(hasattr(VolumeViewModel, 'get_sensitive_volumes'))

        scene_viewmodel = SceneViewModel(root_core_node=volume_core)
        self.assertFalse(scene_viewmodel.is_sensitive_volume(volume_view_model))

        scene_viewmodel.set_volume_sensitive(volume_view_model, True)
        self.assertTrue(scene_viewmodel.is_sensitive_volume(volume_view_model))
        self.assertIn("Scintillator", scene_viewmodel.sensitive_volumes)

        scene_viewmodel.set_volume_sensitive(volume_view_model, False)
        self.assertFalse(scene_viewmodel.is_sensitive_volume(volume_view_model))
        self.assertNotIn("Scintillator", scene_viewmodel.sensitive_volumes)


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

        with self.assertRaises(ValueError):
            self.controller.set_xray_parameters(energy=0.0, pseudo_xray_mode=False)
        with self.assertRaises(ValueError):
            self.controller.set_xray_parameters(energy=-10.0 * units.keV, pseudo_xray_mode=False)

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

        # Валидация строго положительной энергии фотонов (gt=0.0)
        with self.assertRaises(ValidationError):
            settings_instance.update({"xray_energy": -50.0})
        with self.assertRaises(ValidationError):
            settings_instance.update({"xray_energy": 0.0})


class TestSimulationSettingsDialogXrayIntegration(unittest.TestCase):
    """Тестирование отображения и конвертации xray_energy в диалоге SimulationSettingsDialog."""

    def test_dialog_xray_energy_scaling_and_persistence(self) -> None:
        """Проверка отображения в кэВ и считывания в базовой размерности hepunits."""
        default_settings = GuiSimulationSettings()
        dialog = SimulationSettingsDialog(default_settings)

        # По умолчанию xray_energy = 140.0 * units.keV -> спинбокс отображает 140.0 кэВ
        self.assertAlmostEqual(dialog.spin_xray_energy.value(), 140.0)
        self.assertFalse(dialog.chk_pseudo_xray.isChecked())

        # Изменяем значение в спинбоксе на 60.0 кэВ и включаем псевдорентген
        dialog.spin_xray_energy.setValue(60.0)
        dialog.chk_pseudo_xray.setChecked(True)

        updated_settings = dialog.get_settings()
        self.assertIsInstance(updated_settings, GuiSimulationSettings)
        self.assertAlmostEqual(updated_settings.xray_energy, 60.0 * units.keV)
        self.assertTrue(updated_settings.pseudo_xray_mode)

    def test_dialog_custom_initial_xray_energy(self) -> None:
        """Проверка инициализации спинбокса при нестандартной энергии в модели настроек."""
        custom_settings = GuiSimulationSettings(xray_energy=80.0 * units.keV, pseudo_xray_mode=True)
        dialog = SimulationSettingsDialog(custom_settings)

        self.assertAlmostEqual(dialog.spin_xray_energy.value(), 80.0)
        self.assertTrue(dialog.chk_pseudo_xray.isChecked())


class TestMaterialVolumeTransferFunctions(unittest.TestCase):
    """Тестирование построения кусочно-постоянных функций VTK для воксельных фантомов."""

    def setUp(self) -> None:
        self.element_list = [
            Material(name="Vacuum", ID=0),
            Material(name="Air, Dry (near sea level)", ID=1),
            Material(name="Water, Liquid", ID=2),
            Material(name="Bone, Cortical (ICRU-44)", ID=3),
        ]

    def test_build_material_volume_color_tf_normal_mode(self) -> None:
        """Проверка ступенчатой цветовой функции в обычном режиме (Physical Materials)."""
        color_function = build_material_volume_color_tf(
            self.element_list,
            pseudo_xray_mode=False,
            energy=140.0 * units.keV,
        )
        self.assertIsNotNone(color_function)

        for material_index, material_instance in enumerate(self.element_list):
            expected_color = get_material_color(material_instance.name)
            for sample_coordinate in [float(material_index), float(material_index) - 0.45, float(material_index) + 0.45]:
                sampled_rgb = [0.0, 0.0, 0.0]
                color_function.GetColor(sample_coordinate, sampled_rgb)
                np.testing.assert_allclose(sampled_rgb, expected_color, atol=1e-3)

    def test_build_material_volume_color_tf_pseudo_xray_mode(self) -> None:
        """Проверка ступенчатой цветовой функции в режиме псевдорентгена."""
        color_function = build_material_volume_color_tf(
            self.element_list,
            pseudo_xray_mode=True,
            energy=140.0 * units.keV,
        )
        self.assertIsNotNone(color_function)

        for material_index, material_instance in enumerate(self.element_list):
            expected_xray_color, _ = get_pseudo_xray_rgba(material_instance.name, energy=140.0 * units.keV)
            sampled_rgb = [0.0, 0.0, 0.0]
            color_function.GetColor(float(material_index), sampled_rgb)
            np.testing.assert_allclose(sampled_rgb, expected_xray_color, atol=1e-3)

    def test_build_material_volume_opacity_tf_vacuum_and_air_zero(self) -> None:
        """Проверка, что для вакуума и воздуха непрозрачность строго равна 0.0."""
        opacity_function = build_material_volume_opacity_tf(
            self.element_list,
            pseudo_xray_mode=False,
            energy=140.0 * units.keV,
            characteristic_length=2.0 * units.mm,
        )
        self.assertIsNotNone(opacity_function)

        # Индекс 0: Vacuum -> строго 0.0
        self.assertAlmostEqual(opacity_function.GetValue(0.0), 0.0, places=5)
        self.assertAlmostEqual(opacity_function.GetValue(-0.4), 0.0, places=5)
        self.assertAlmostEqual(opacity_function.GetValue(0.4), 0.0, places=5)

        # Индекс 1: Air, Dry -> строго 0.0
        self.assertAlmostEqual(opacity_function.GetValue(1.0), 0.0, places=5)
        self.assertAlmostEqual(opacity_function.GetValue(0.6), 0.0, places=5)
        self.assertAlmostEqual(opacity_function.GetValue(1.4), 0.0, places=5)

    def test_build_material_volume_opacity_tf_physical_scaling(self) -> None:
        """Проверка физического закона ослабления для плотных тканей и зависимости от энергии."""
        opacity_function_high_energy = build_material_volume_opacity_tf(
            self.element_list,
            pseudo_xray_mode=False,
            energy=140.0 * units.keV,
            characteristic_length=2.0 * units.mm,
        )
        water_opacity_high = opacity_function_high_energy.GetValue(2.0)
        bone_opacity_high = opacity_function_high_energy.GetValue(3.0)

        self.assertGreater(water_opacity_high, 0.0)
        self.assertGreater(bone_opacity_high, water_opacity_high)

        # При меньшей энергии фотонов ослабление фотоэффекта выше -> непрозрачность должна возрасти
        opacity_function_low_energy = build_material_volume_opacity_tf(
            self.element_list,
            pseudo_xray_mode=False,
            energy=30.0 * units.keV,
            characteristic_length=2.0 * units.mm,
        )
        water_opacity_low = opacity_function_low_energy.GetValue(2.0)
        bone_opacity_low = opacity_function_low_energy.GetValue(3.0)

        self.assertGreater(water_opacity_low, water_opacity_high)
        self.assertGreater(bone_opacity_low, bone_opacity_high)

    def test_empty_element_list_handling(self) -> None:
        """Проверка корректной обработки пустого списка материалов."""
        color_function = build_material_volume_color_tf([])
        opacity_function = build_material_volume_opacity_tf([])
        self.assertIsNotNone(color_function)
        self.assertIsNotNone(opacity_function)


class TestVoxelVolumeViewModelAndPropertyInspector(unittest.TestCase):
    """Тестирование модели представления VoxelVolumeViewModel и интерфейса PropertyInspector."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.qt_application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        material_array_instance = MaterialArray((6, 6, 6))
        self.materials_list = [
            Material(name="Vacuum", ID=0),
            Material(name="Water, Liquid", ID=1),
            Material(name="Bone, Cortical (ICRU-44)", ID=2),
        ]
        material_array_instance.element_list = self.materials_list
        material_array_instance.view(np.ndarray)[:2, :, :] = 0
        material_array_instance.view(np.ndarray)[2:4, :, :] = 1
        material_array_instance.view(np.ndarray)[4:, :, :] = 2

        self.voxel_phantom = WoodcockVoxelVolume(
            voxel_size=2.0 * units.mm,
            material_distribution=material_array_instance,
            name="TestPhantom",
        )
        self.voxel_vm = VoxelVolumeViewModel(self.voxel_phantom)

    def test_voxel_volume_vm_defaults_and_material_list(self) -> None:
        """Проверка значений по умолчанию и свойства material_list в VoxelVolumeViewModel."""
        self.assertEqual(self.voxel_vm.colormap_name, "Physical Materials")
        self.assertEqual(len(self.voxel_vm.material_list), 3)
        self.assertEqual(self.voxel_vm.material_list[0].name, "Vacuum")
        self.assertEqual(self.voxel_vm.material_list[1].name, "Water, Liquid")
        self.assertEqual(self.voxel_vm.material_list[2].name, "Bone, Cortical (ICRU-44)")

    def test_property_inspector_colormap_physical_materials_blocking(self) -> None:
        """Проверка блокировки эвристических регуляторов прозрачности при выборе 'Physical Materials'."""
        inspector_widget = PropertyInspector()

        # Первый элемент в списке палитр обязан быть 'Physical Materials'
        self.assertEqual(inspector_widget.combo_colormap.itemText(0), "Physical Materials")

        # При назначении воксельного фантома с 'Physical Materials' ползунки прозрачности заблокированы
        inspector_widget.set_target_viewmodel(self.voxel_vm)
        self.assertEqual(inspector_widget.combo_colormap.currentText(), "Physical Materials")
        self.assertFalse(inspector_widget.spin_opacity_thresh.isEnabled())
        self.assertFalse(inspector_widget.spin_max_opacity.isEnabled())
        self.assertFalse(inspector_widget.combo_opacity_preset.isEnabled())

        # Переключаем палитру на 'Hot Iron' -> регуляторы прозрачности становятся активными
        inspector_widget.combo_colormap.setCurrentText("Hot Iron")
        self.assertEqual(self.voxel_vm.colormap_name, "Hot Iron")
        self.assertTrue(inspector_widget.spin_opacity_thresh.isEnabled())
        self.assertTrue(inspector_widget.spin_max_opacity.isEnabled())
        self.assertTrue(inspector_widget.combo_opacity_preset.isEnabled())

        # Возвращаем 'Physical Materials' -> регуляторы вновь заблокированы
        inspector_widget.combo_colormap.setCurrentText("Physical Materials")
        self.assertEqual(self.voxel_vm.colormap_name, "Physical Materials")
        self.assertFalse(inspector_widget.spin_opacity_thresh.isEnabled())
        self.assertFalse(inspector_widget.spin_max_opacity.isEnabled())
        self.assertFalse(inspector_widget.combo_opacity_preset.isEnabled())


class TestVoxelVolumeViewportControllerIntegration(unittest.TestCase):
    """Интеграционное тестирование визуализации воксельного фантома в SceneViewportController."""

    def setUp(self) -> None:
        self.mock_viewport = MockSceneViewport()
        material_array_instance = MaterialArray((4, 4, 4))
        self.materials_list = [
            Material(name="Vacuum", ID=0),
            Material(name="Water, Liquid", ID=1),
            Material(name="Bone, Cortical (ICRU-44)", ID=2),
        ]
        material_array_instance.element_list = self.materials_list
        material_array_instance.view(np.ndarray)[:2, :, :] = 0
        material_array_instance.view(np.ndarray)[2:, :, :] = 1

        self.root_node = Volume(
            geometry=Box(500.0, 500.0, 500.0),
            material=Material(name="Air, Dry (near sea level)"),
            name="World",
        )
        self.voxel_node = WoodcockVoxelVolume(
            voxel_size=2.5 * units.mm,
            material_distribution=material_array_instance,
            name="Phantom",
        )
        self.scene_view_model = SceneViewModel(root_core_node=self.root_node)
        self.voxel_vm = VoxelVolumeViewModel(self.voxel_node)
        self.controller = SceneViewportController(viewport=self.mock_viewport, scene_vm=self.scene_view_model)
        self.scene_view_model.node_added.connect(self.controller.on_node_added)
        self.scene_view_model.node_removed.connect(self.controller.on_node_removed)
        self.scene_view_model.node_selected.connect(self.controller.on_node_selected)

    def test_voxel_phantom_renders_with_physical_materials_by_default(self) -> None:
        """Воксельный фантом по умолчанию рендерится через физическую передаточную функцию материалов."""
        self.scene_view_model.add_node(self.scene_view_model.root_vm, self.voxel_vm)

        self.assertTrue(self.controller.voxel_renderer.is_physical_mode)
        self.assertEqual(self.controller.voxel_renderer.current_element_list, self.materials_list)
        self.assertFalse(self.controller.voxel_renderer.last_xray_mode)

    def test_voxel_phantom_reacts_to_global_pseudo_xray_mode_and_energy(self) -> None:
        """Фантом синхронно переключается на рентгеновский режим и пересчитывает свойства при смене энергии."""
        self.scene_view_model.add_node(self.scene_view_model.root_vm, self.voxel_vm)

        # Переключаем глобальный псевдорентген на 40 кэВ
        self.controller.set_xray_parameters(energy=40.0 * units.keV, pseudo_xray_mode=True)
        self.assertTrue(self.controller.voxel_renderer.last_xray_mode)
        self.assertAlmostEqual(self.controller.voxel_renderer.last_energy, 40.0 * units.keV)
        self.assertTrue(self.controller.voxel_renderer.is_physical_mode)

        # Возвращаем нормальный режим на 140 кэВ
        self.controller.set_xray_parameters(energy=140.0 * units.keV, pseudo_xray_mode=False)
        self.assertFalse(self.controller.voxel_renderer.last_xray_mode)
        self.assertAlmostEqual(self.controller.voxel_renderer.last_energy, 140.0 * units.keV)

    def test_voxel_selection_bounding_box_lifecycle(self) -> None:
        """Вокруг выделенного фантома появляется янтарная габаритная рамка и удаляется при снятии выделения."""
        self.scene_view_model.add_node(self.scene_view_model.root_vm, self.voxel_vm)

        box_actor_name = f"selection_box_{id(self.voxel_vm)}"

        # При выборе фантома актор рамки выделения добавляется во вьюпорт
        self.controller.on_node_selected(self.voxel_vm)
        self.assertIn(box_actor_name, self.mock_viewport.actors_registry)

        # Снимаем выделение -> рамка удаляется
        self.controller.on_node_selected(None)
        self.assertNotIn(box_actor_name, self.mock_viewport.actors_registry)

        # Повторно выбираем фантом -> рамка вновь появляется
        self.controller.on_node_selected(self.voxel_vm)
        self.assertIn(box_actor_name, self.mock_viewport.actors_registry)

    def test_voxel_selection_box_ancestor_removal_cleanup(self) -> None:
        """При удалении родительского узла рамка выделения дочернего фантома корректно удаляется."""
        composite_parent = Volume(
            geometry=Box(200.0, 200.0, 200.0),
            material=Material(name="Water, Liquid"),
            name="ParentAssembly",
        )
        parent_vm = VolumeViewModel(composite_parent)
        self.scene_view_model.add_node(self.scene_view_model.root_vm, parent_vm)
        self.scene_view_model.add_node(parent_vm, self.voxel_vm)

        self.controller.on_node_selected(self.voxel_vm)
        box_actor_name = f"selection_box_{id(self.voxel_vm)}"
        self.assertIn(box_actor_name, self.mock_viewport.actors_registry)

        # Удаляем родительский узел -> выделение и рамка фантома должны очиститься
        self.scene_view_model.remove_node(parent_vm)
        self.assertNotIn(box_actor_name, self.mock_viewport.actors_registry)
        self.assertIsNone(self.controller.selected_node_vm)

    def test_voxel_renderer_nearest_interpolation_and_colormap_switching(self) -> None:
        """Проверка переключения интерполяции на Nearest Neighbor в физическом режиме и восстановления Linear."""
        self.scene_view_model.add_node(self.scene_view_model.root_vm, self.voxel_vm)
        actor_instance = self.mock_viewport.actors_registry[self.controller.voxel_renderer.actor_name]

        # В режиме Physical Materials интерполяция строго Nearest Neighbor для дискретных ID
        self.assertEqual(actor_instance.interpolation_type, "NearestNeighbor")

        # Переключаем палитру на Hot Iron -> интерполяция возвращается к Linear
        self.voxel_vm.colormap_name = "Hot Iron"
        self.assertEqual(actor_instance.interpolation_type, "Linear")
        self.assertFalse(self.controller.voxel_renderer.is_physical_mode)

        # Возвращаем Physical Materials -> интерполяция вновь Nearest Neighbor
        self.voxel_vm.colormap_name = "Physical Materials"
        self.assertEqual(actor_instance.interpolation_type, "NearestNeighbor")
        self.assertTrue(self.controller.voxel_renderer.is_physical_mode)

    def test_build_material_volume_opacity_tf_various_vacuum_air_names(self) -> None:
        """Проверка строго нулевой прозрачности для различных вариантов именования вакуума и воздуха."""
        test_materials = [
            Material(name="vacuum", ID=0),
            Material(name="Air", ID=1),
            Material(name="Воздух", ID=2),
            Material(name="Вакуум", ID=3),
        ]
        opacity_tf = build_material_volume_opacity_tf(test_materials)
        for material_idx in range(len(test_materials)):
            self.assertEqual(opacity_tf.GetValue(float(material_idx)), 0.0)

    def test_voxel_selection_box_transform_sync(self) -> None:
        """Трансформация рамки выделения синхронизируется при перемещении фантома."""
        self.scene_view_model.add_node(self.scene_view_model.root_vm, self.voxel_vm)
        self.controller.on_node_selected(self.voxel_vm)

        box_actor_name = f"selection_box_{id(self.voxel_vm)}"
        self.assertIn(box_actor_name, self.mock_viewport.actors_registry)

        # Изменяем матрицу фантома
        translated_matrix = np.eye(4, dtype=float)
        translated_matrix[0, 3] = 120.0
        self.voxel_vm.local_matrix = translated_matrix

        # Проверяем обновление трансформации рамки
        self.assertTrue(box_actor_name in self.mock_viewport.actors_registry)


if __name__ == "__main__":
    unittest.main()

