import unittest
import math
from core.config.preprocessor import expand_repeaters, _smart_cast, _apply_template

class TestPreprocessor(unittest.TestCase):
    def test_smart_cast(self):
        self.assertEqual(_smart_cast("4"), 4)
        self.assertEqual(_smart_cast("-5"), -5)
        self.assertEqual(_smart_cast("4.5"), 4.5)
        self.assertEqual(_smart_cast("-5.5e-3"), -5.5e-3)
        self.assertEqual(_smart_cast("4 mm"), "4 mm")
        self.assertEqual(_smart_cast("hello"), "hello")

    def test_apply_template(self):
        node = {
            "name": "det_${index}",
            "pos": "${val}",
            "text": "value: ${val}",
            "nested": ["${index}", "${val} keV", "plain string"],
            "no_sub": "45" # no $ should not be cast
        }
        variables = {"index": 2, "val": 150.5}

        result = _apply_template(node, variables)
        self.assertEqual(result["name"], "det_2")
        self.assertEqual(result["pos"], 150.5)
        self.assertEqual(result["text"], "value: 150.5")
        self.assertEqual(result["nested"], [2, "150.5 keV", "plain string"])
        self.assertEqual(result["no_sub"], "45")

    def test_zip_repeater(self):
        config = {
            "type": "ZipRepeater",
            "values": {
                "name": ["A", "B", "C"],
                "size": [10, 20, 30]
            },
            "template": {
                "type": "Item",
                "id": "${index}",
                "name": "${name}",
                "size": "${size}"
            }
        }
        result = expand_repeaters(config)
        self.assertEqual(result["type"], "CompositeNode")
        self.assertEqual(len(result["children"]), 3)
        self.assertEqual(result["children"][0]["name"], "A")
        self.assertEqual(result["children"][0]["size"], 10)
        self.assertEqual(result["children"][0]["id"], 0)
        self.assertEqual(result["children"][2]["name"], "C")
        self.assertEqual(result["children"][2]["size"], 30)
        self.assertEqual(result["children"][2]["id"], 2)

    def test_zip_repeater_mismatch_length(self):
        config = {
            "type": "ZipRepeater",
            "values": {
                "a": [1, 2],
                "b": [1, 2, 3]
            },
            "template": {}
        }
        with self.assertRaises(ValueError):
            expand_repeaters(config)

    def test_grid_repeater(self):
        config = {
            "type": "GridRepeater",
            "count_x": 2,
            "count_y": 2,
            "count_z": 1,
            "pitch_x": 10.0,
            "pitch_y": 20.0,
            "pitch_z": 0.0,
            "origin": [1.0, 2.0, 3.0],
            "template": {
                "type": "Box",
                "name": "box_${index}_${ix}_${iy}"
            }
        }

        result = expand_repeaters(config)
        self.assertEqual(result["type"], "CompositeNode")
        self.assertEqual(len(result["children"]), 4)

        # Grid layout:
        # count_x=2, pitch_x=10 => x from -5 to 5
        # count_y=2, pitch_y=20 => y from -10 to 10
        # Expected x,y pairs: (-5, -10), (5, -10), (-5, 10), (5, 10)

        c0 = result["children"][0]
        self.assertEqual(c0["name"], "box_0_0_0")
        t0 = c0["transformations"]
        self.assertEqual(len(t0), 2) # translate local, translate origin
        self.assertEqual(t0[0], {"type": "translate", "x": -5.0, "y": -10.0, "z": 0.0})
        self.assertEqual(t0[1], {"type": "translate", "x": 1.0, "y": 2.0, "z": 3.0})

        c3 = result["children"][3]
        self.assertEqual(c3["name"], "box_3_1_1")
        t3 = c3["transformations"]
        self.assertEqual(t3[0], {"type": "translate", "x": 5.0, "y": 10.0, "z": 0.0})

    def test_ring_repeater(self):
        config = {
            "type": "RingRepeater",
            "num_nodes": 4,
            "radius": 100.0,
            "start_angle": 0.0,
            "angular_span": 360.0,
            "template": {
                "type": "Detector",
                "id": "det_${index}"
            }
        }

        result = expand_repeaters(config)
        self.assertEqual(result["type"], "CompositeNode")
        self.assertEqual(len(result["children"]), 4)

        # Should be at 0, 90, 180, 270 degrees
        angles = [0.0, 90.0, 180.0, 270.0]

        for i, child in enumerate(result["children"]):
            self.assertEqual(child["id"], f"det_{i}")
            t = child["transformations"]
            self.assertEqual(len(t), 3) # translate radius, rotate angle, translate center
            self.assertEqual(t[0], {"type": "translate", "x": 100.0, "y": 0.0, "z": 0.0})
            self.assertEqual(t[1], {"type": "rotate", "axis": [0.0, 0.0, 1.0], "angle": f"{angles[i]} deg"})
            self.assertEqual(t[2], {"type": "translate", "x": 0.0, "y": 0.0, "z": 0.0})

    def test_ring_repeater_preserves_existing_transformations(self):
        config = {
            "type": "RingRepeater",
            "num_nodes": 2,
            "radius": 50.0,
            "template": {
                "type": "Obj",
                "transformations": [
                    {"type": "translate", "x": 1, "y": 2, "z": 3}
                ]
            }
        }

        result = expand_repeaters(config)
        child = result["children"][0]
        t = child["transformations"]
        self.assertEqual(len(t), 4) # existing 1 + 3 new
        self.assertEqual(t[0], {"type": "translate", "x": 1, "y": 2, "z": 3})
        self.assertEqual(t[1], {"type": "translate", "x": 50.0, "y": 0.0, "z": 0.0})

    def test_recursive_expansion(self):
        config = {
            "type": "RingRepeater",
            "num_nodes": 2,
            "radius": 100.0,
            "template": {
                "type": "Group",
                "id": "group_${index}",
                "children": [
                    {
                        "type": "ZipRepeater",
                        "values": {"sub_id": [1, 2]},
                        "template": {
                            "type": "Element",
                            "name": "el_${index}_${sub_id}"
                        }
                    }
                ]
            }
        }

        result = expand_repeaters(config)
        self.assertEqual(len(result["children"]), 2)

        group0 = result["children"][0]
        self.assertEqual(group0["id"], "group_0")

        group0_children = group0["children"]
        self.assertEqual(len(group0_children), 1)
        self.assertEqual(group0_children[0]["type"], "CompositeNode")

        sub_elements = group0_children[0]["children"]
        self.assertEqual(len(sub_elements), 2)
        self.assertEqual(sub_elements[0]["name"], "el_0_1")
        self.assertEqual(sub_elements[1]["name"], "el_0_2")

if __name__ == '__main__':
    unittest.main()
