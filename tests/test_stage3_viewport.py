import os
import tempfile
import unittest
import numpy as np

from gui.viewport_3d.dicom_colormaps import (
    get_available_colormaps,
    get_colormap_lut,
    import_lut_file,
)
from gui.viewport_3d.spect_manipulator import SPECTManipulator
from gui.viewport_3d.pet_manipulator import PETManipulator
from gui.viewport_3d.track_renderer import TrackRenderer


class TestStage3Viewport(unittest.TestCase):
    def test_dicom_colormaps(self):
        """Проверка генерации DICOM-палитр ядерной медицины."""
        cmaps = get_available_colormaps()
        self.assertIn('Hot Iron', cmaps)
        self.assertIn('Rainbow', cmaps)
        self.assertIn('GE Color', cmaps)
        self.assertIn('PET 20 Step', cmaps)

        lut = get_colormap_lut('Hot Iron', num_colors=256)
        self.assertEqual(lut.shape, (256, 4))
        self.assertTrue(np.all(lut >= 0.0))
        self.assertTrue(np.all(lut <= 1.0))

    def test_lut_file_import(self):
        """Проверка импорта стороннего LUT-файла."""
        with tempfile.NamedTemporaryFile(suffix=".lut", delete=False, mode='w') as tmp:
            tmp.write("0.0 0.0 0.0\n0.5 0.5 0.5\n1.0 1.0 1.0\n")
            tmp_path = tmp.name

        try:
            data = import_lut_file(tmp_path)
            self.assertEqual(data.shape, (3, 4))
            self.assertAlmostEqual(data[0, 0], 0.0)
            self.assertAlmostEqual(data[2, 0], 1.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_spect_manipulator_kinematics(self):
        """Проверка кинематических ограничений манипулятора ОФЭКТ."""
        manipulator = SPECTManipulator(viewport=None, initial_radius=200.0, initial_angle=0.0)

        events = []
        manipulator.orbit_changed.connect(lambda r, a, z: events.append((r, a, z)))

        # Поворот на 90 градусов
        manipulator.set_orbit_parameters(radius=250.0, angle_deg=90.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], (250.0, 90.0, 0.0))

        # При угле 90 градусов X=0, Y=250
        x, y, z = manipulator.get_cartesian_position()
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 250.0, places=5)

        mat = manipulator.get_orientation_matrix()
        self.assertEqual(mat.shape, (4, 4))
        self.assertAlmostEqual(mat[1, 3], 250.0)

    def test_pet_manipulator_geometry(self):
        """Проверка манипулятора геометрии кольца ПЭТ."""
        pet = PETManipulator(viewport=None, diameter=600.0, axial_length=200.0)

        events = []
        pet.geometry_changed.connect(lambda d, l, s: events.append((d, l, s)))

        pet.set_parameters(diameter=700.0, axial_length=220.0, num_sectors=32)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], (700.0, 220.0, 32))
        self.assertEqual(pet.diameter, 700.0)

    def test_track_renderer_batch(self):
        """Проверка накопления треков в буфере TrackRenderer."""
        renderer = TrackRenderer(viewport=None, max_points=100)

        batch = {
            'pos_x': np.array([10.0, 20.0, 30.0]),
            'pos_y': np.array([0.0, 0.0, 0.0]),
            'pos_z': np.array([5.0, 5.0, 5.0]),
            'process_id': np.array([1, 1, 2]),
            'particle_id': np.array([1, 1, 2])
        }

        renderer.add_tracks_batch(batch)
        self.assertEqual(len(renderer._point_buffer), 3)

        renderer.clear()
        self.assertEqual(len(renderer._point_buffer), 0)


if __name__ == '__main__':
    unittest.main()
