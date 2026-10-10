"""
Сквозной интеграционный тест полного цикла round-trip для конфигурации tb161_nema_protocol direct.yaml.
Проверяет сохранение pool_size, структуры протокола CustomSweep, карты материалов фантома,
10 линий полиэнергетического спектра источника излучения и биндингов гантри ${gantry_angle}.
"""

import os
import tempfile
import unittest
from pathlib import Path
import yaml
import hepunits as units

from core.config.yaml_loader import load_simulation_config
from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter
from core.config.models import SimulationConfig, CustomSweepProtocolConfig
from core.source.sources import Source


class TestRoundtripTB161Protocol(unittest.TestCase):
    """Тестирование сквозного цикла чтения, инспекции и экспорта tb161_nema_protocol direct.yaml."""

    def setUp(self) -> None:
        self.yaml_path = Path("tb161_nema_protocol direct.yaml").resolve()
        self.assertTrue(self.yaml_path.exists(), f"Файл {self.yaml_path} должен существовать")

    def test_tb161_full_roundtrip_integrity(self) -> None:
        # 1. Загрузка исходной конфигурации через load_simulation_config
        sim_cfg = load_simulation_config(self.yaml_path, resolve_protocol=False)
        self.assertEqual(sim_cfg.pool_size, 24, "pool_size обязан быть 24")
        self.assertIsInstance(
            sim_cfg.protocol,
            CustomSweepProtocolConfig,
            "Протокол должен быть CustomSweepProtocolConfig"
        )
        self.assertIn("gantry_angle", sim_cfg.protocol.grid_variables)
        self.assertEqual(len(sim_cfg.protocol.grid_variables["gantry_angle"]), 60)

        # Построение дерева сцены
        builder = SceneBuilder()
        root_node = builder.build_scene(sim_cfg.scene)

        # 2. Проверка узлов сцены
        phantom_node = None
        source_node = None
        gantry_node = None

        for node in root_node.childs:
            if node.name == "Phantom":
                phantom_node = node
                for child in phantom_node.childs:
                    if isinstance(child, Source):
                        source_node = child
            elif node.name == "Gantry":
                gantry_node = node

        self.assertIsNotNone(phantom_node, "Узел Phantom должен присутствовать в сцене")
        self.assertIsNotNone(source_node, "Узел Source должен присутствовать внутри Phantom")
        self.assertIsNotNone(gantry_node, "Узел Gantry должен присутствовать в сцене")

        # Проверка спектра источника: должно быть ровно 10 линий
        self.assertIsNotNone(source_node.energy)
        self.assertEqual(len(source_node.energy["energy"]), 10, "Спектр 161Tb обязан содержать 10 линий")

        # Проверка биндинга гантри в bindings_registry
        self.assertIn(gantry_node, builder.bindings_registry)
        self.assertEqual(builder.bindings_registry[gantry_node]["rotation.alpha"], "gantry_angle")

        # 3. Экспорт сцены обратно во временный YAML-файл
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_yaml_path = os.path.join(tmp_dir, "roundtrip_exported.yaml")
            SceneExporter.export_to_yaml(
                root_node=root_node,
                filepath=out_yaml_path,
                simulation_manager_cfg=sim_cfg.simulation_manager,
                data_manager_cfg=sim_cfg.data_manager,
                protocol=sim_cfg.protocol,
                pool_size=sim_cfg.pool_size,
                distribution_registry=builder.distribution_registry,
                slots_registry=builder.slots_registry,
                bindings_registry=builder.bindings_registry,
            )

            self.assertTrue(os.path.exists(out_yaml_path))

            # 4. Проверка сырого содержимого экспортированного YAML
            with open(out_yaml_path, "r", encoding="utf-8") as f:
                exported_raw = yaml.safe_load(f)

            self.assertEqual(exported_raw["pool_size"], 24, "Экспортированный pool_size обязан быть 24")
            self.assertIn("protocol", exported_raw)
            self.assertEqual(exported_raw["protocol"]["type"], "CustomSweep")
            self.assertEqual(len(exported_raw["protocol"]["grid_variables"]["gantry_angle"]), 60)

            # Проверка фантома и mapping
            scene_dict = exported_raw["scene"]
            phantom_dict = None
            gantry_dict = None
            for ch in scene_dict.get("children", []):
                if ch.get("name") == "Phantom":
                    phantom_dict = ch
                elif ch.get("name") == "Gantry":
                    gantry_dict = ch

            self.assertIsNotNone(phantom_dict)
            self.assertIsNotNone(gantry_dict)

            # Проверка, что оригинальный mapping не деградировал до 0.0/1.0
            dist_map = phantom_dict["distribution"]["mapping"]
            self.assertIn(0.04, dist_map)
            self.assertIn(0.15, dist_map)
            self.assertNotIn(0.0, dist_map, "Фантом не должен содержать фиктивный Vacuum 0.0")

            # Проверка источника: активность со строковыми единицами и 10 линий спектра
            source_dict = phantom_dict["children"][0]
            self.assertEqual(source_dict["name"], "Source")
            self.assertIn("MBq", str(source_dict["activity"]), "Активность должна содержать единицу MBq")
            energy_list = source_dict["energy"]
            self.assertIsInstance(energy_list, list)
            self.assertEqual(len(energy_list), 10, "Экспортированный спектр обязан содержать 10 линий")
            self.assertIn("keV", str(energy_list[0][0]))

            # Проверка биндинга поворота гантри
            gantry_rot = gantry_dict["transformations"][0]
            self.assertEqual(gantry_rot["type"], "rotate")
            self.assertEqual(gantry_rot["alpha"], "${gantry_angle} deg")

            # 5. Повторный импорт через SceneBuilder (Round-trip)
            sim_cfg2 = load_simulation_config(out_yaml_path, resolve_protocol=False)
            builder2 = SceneBuilder()
            root_node2 = builder2.build_scene(sim_cfg2.scene)
            self.assertIsNotNone(root_node2)
            self.assertEqual(sim_cfg2.pool_size, 24)
            self.assertEqual(len(sim_cfg2.protocol.grid_variables["gantry_angle"]), 60)


if __name__ == "__main__":
    unittest.main()
