import os
import tempfile
import unittest
import numpy as np
import h5py

from core.scene.nodes import CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.scene.gamma_camera_node import GammaCameraNode
from gui.factories.gamma_camera_factory import create_default_gamma_camera
from core.geometry.spect_kinematics import compute_spect_poses
from core.materials.materials import Material
from core.config.models import SpectProtocolConfig, StepAndShootProtocolConfig
from core.config.orchestrator import Orchestrator
from PySide6.QtWidgets import QApplication
from core.data.data_manager import DataManager
from core.data.dose_map_handler import DoseMapHandler
from core.data.data_handlers import HistoryAssemblerHandler
from gui.controllers.viewport_controller import SceneViewportController
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.procedure_viewmodel import SpectProcedureViewModel
from gui.viewport_3d.vtk_viewport import VTKViewport
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel

app = QApplication.instance() or QApplication([])


class TestSpectSimulationHDF5(unittest.TestCase):
    """
    Интеграционное тестирование протокола ОФЭКТ (SPECT) с N детекторными головками (N=4),
    предпросмотра геометрии ракурсов и сохранения сырых данных в HDF5.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.temp_dir.name, "test_spect_output.h5")

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except (OSError, RuntimeError):
            pass

    def test_spect_protocol_multi_head_orchestration(self) -> None:
        """
        Проверка компиляции протокола ОФЭКТ с 4 детекторными головками (сдвиг 90 градусов).
        """
        spect_proto = SpectProtocolConfig(
            views=16,
            gamma_cameras=4,
            head_angles=["0 deg", "90 deg", "180 deg", "270 deg"],
            start_angle="0 deg",
            end_angle="360 deg",
            orbit_radius="250 mm",
            time_per_view="1 s"
        )
        self.assertEqual(spect_proto.gamma_cameras, 4)
        self.assertEqual(len(spect_proto.head_angles), 4)

        poses = compute_spect_poses(spect_proto)
        self.assertEqual(len(poses), 4)
        self.assertEqual(len(poses[0]), 4)

        # Первый ракурс (шаг 0): 0, 90, 180, 270 градусов
        np.testing.assert_allclose(poses[0], [0.0, 90.0, 180.0, 270.0])

        # Второй ракурс (шаг 1): базовый угол 90 градусов (при 4 шагах на 360°)
        # Углы: 90, 180, 270, 0(360)
        np.testing.assert_allclose(poses[1], [90.0, 180.0, 270.0, 0.0])

    def test_viewport_controller_multi_head_preview(self) -> None:
        """
        Проверка предварительного кинематического позиционирования N головок через SceneViewportController.preview_view().
        """
        import settings.database_setting as database_setting
        mat_db = database_setting.material_database

        root = CompositeNode(name="WorldScene")
        def _make_cam(name: str) -> GammaCameraNode:
            cam_node, _ = create_default_gamma_camera(name=name)
            return cam_node

        cam1 = _make_cam("Head1")
        cam2 = _make_cam("Head2")
        cam3 = _make_cam("Head3")
        cam4 = _make_cam("Head4")
        root.add_child(cam1)
        root.add_child(cam2)
        root.add_child(cam3)
        root.add_child(cam4)
        scene_vm = SceneViewModel(root)
        viewport = VTKViewport()
        viewport_ctrl = SceneViewportController(viewport=viewport, scene_vm=scene_vm)

        proc_vm = SpectProcedureViewModel()
        proc_vm.steps = 8
        proc_vm.start_angle = 0.0
        proc_vm.end_angle = 360.0
        proc_vm.gamma_cameras = 4
        proc_vm.head_angles = [0.0, 90.0, 180.0, 270.0]

        # Предпросмотр 2-го ракурса (индекс 2 из 8 при 1-based view_number_1based=3: 2/8 * 360 = 90 градусов)
        base_ang = viewport_ctrl.preview_view(3, proc_vm)
        self.assertAlmostEqual(base_ang, 90.0)

        # Камеры должны занять углы:
        # Head1: 90°, Head2: 180°, Head3: 270°, Head4: 0°
        expected_angles = [90.0, 180.0, 270.0, 0.0]
        for cam, exp_deg in zip([cam1, cam2, cam3, cam4], expected_angles):
            # Проверяем положение по local_matrix
            x = float(cam.local_matrix[0, 3])
            y = float(cam.local_matrix[1, 3])
            angle = float(np.degrees(np.arctan2(y, x)) % 360.0)
            self.assertAlmostEqual(angle, exp_deg, delta=1e-3)

        viewport_ctrl.close()

    def test_hdf5_storage_raw_data_only(self) -> None:
        """
        Проверка сохранения результатов моделирования в HDF5:
        1. Наличие сырых данных (/interactions, /initial_states, /dose, /metadata).
        2. КРИТИЧЕСКОЕ ТРЕБОВАНИЕ: группа /projections НЕ должна создаваться в HDF5!
        """
        import settings.database_setting as database_setting
        mat_db = database_setting.material_database

        root = CompositeNode(name="WorldScene")
        water = mat_db['Water, Liquid']
        phantom = Volume(geometry=Box(100.0, 100.0, 100.0), material=water, name="Phantom")
        detector = Volume(geometry=Box(400.0, 400.0, 10.0), material=water, name="NaI_Detector")
        dose_grid = DoseGridNode(
            size=(100.0, 100.0, 100.0),
            dose_voxel_size=10.0,
            name="OrganDoseGrid"
        )
        root.add_child(phantom)
        root.add_child(detector)
        root.add_child(dose_grid)

        # Создаем обработчик истории и дозы
        history_handler = HistoryAssemblerHandler(
            sensitive_volumes=[detector],
            save_initial_states=True
        )
        dose_handler = DoseMapHandler(
            grid_nodes=[dose_grid],
            create_shm=False
        )

        written_ops = []
        def mock_writer(op):
            written_ops.append(op)
        history_handler.set_writer_callback(mock_writer)
        dose_handler.set_writer_callback(mock_writer)

        # Имитируем передачу чанков через process_chunk
        history_handler.process_chunk({
            'type': 'initial_states',
            'data': {
                'particle_ID': np.array([1, 2], dtype=np.uint64),
                'pos_x': np.array([0.0, 1.0], dtype=np.float32),
                'pos_y': np.array([0.0, 1.0], dtype=np.float32),
                'pos_z': np.array([0.0, 1.0], dtype=np.float32),
                'dir_x': np.array([0.0, 0.0], dtype=np.float32),
                'dir_y': np.array([0.0, 0.0], dtype=np.float32),
                'dir_z': np.array([1.0, 1.0], dtype=np.float32),
                'energy': np.array([140.5, 140.5], dtype=np.float32),
                'time': np.array([0.0, 1.0], dtype=np.float64),
            }
        })

        det_id = history_handler.target_volumes[0]
        history_handler.process_chunk({
            'type': 'interactions',
            'data': {
                'particle_ID': np.array([1, 1, 2, 2, 2], dtype=np.uint64),
                'volume_id': np.array([det_id, det_id, det_id, det_id, det_id], dtype=np.uint32),
                'pos_x': np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
                'pos_y': np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
                'pos_z': np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
                'dir_x': np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'dir_y': np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'dir_z': np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                'energy_deposit': np.array([140.0, 138.5, 140.2, 70.0, 140.0], dtype=np.float32),
                'process_id': np.array([1, 1, 2, 2, 1], dtype=np.uint16),
                'scattering_theta': np.zeros(5, dtype=np.float32),
                'scattering_phi': np.zeros(5, dtype=np.float32),
                'material_id': np.zeros(5, dtype=np.uint32),
                'Z': np.ones(5, dtype=np.uint32),
                'species': np.zeros(5, dtype=np.uint8),
                'distance_traveled': np.ones(5, dtype=np.float32),
            }
        })

        history_handler.process_chunk({
            'type': 'dead_particles',
            'data': np.array([1, 2], dtype=np.uint64)
        })

        dose_handler.entries[0]._dose_grid = np.ones((10, 10, 10), dtype=np.float32) * 5.25

        history_handler.finalize()
        dose_handler.finalize()

        # Запись в HDF5
        with h5py.File(self.h5_path, 'a') as h5_file:
            for op in written_ops:
                op(h5_file)
            meta = h5_file.require_group("/metadata")
            meta.attrs["status"] = "Completed"
            meta.attrs["gamma_cameras"] = 4

        # Верификация содержимого HDF5
        with h5py.File(self.h5_path, 'r') as h5_file:
            # 1. Проверяем наличие сырых взаимодействий
            self.assertIn("interactions", h5_file)
            self.assertIn("NaI_Detector", h5_file["interactions"])
            self.assertEqual(len(h5_file["interactions/NaI_Detector/energy_deposit"]), 5)

            # 2. Проверяем наличие начальных состояний
            self.assertIn("initial_states", h5_file)
            self.assertEqual(len(h5_file["initial_states/energy"]), 2)

            # 3. Проверяем наличие карты поглощенной дозы
            self.assertIn("dose", h5_file)
            self.assertIn("OrganDoseGrid", h5_file["dose"])
            dose_arr = np.array(h5_file["dose/OrganDoseGrid"])
            self.assertEqual(dose_arr.shape, (10, 10, 10))
            self.assertAlmostEqual(float(dose_arr[0, 0, 0]), 5.25)

            # 4. Проверяем метаданные
            self.assertIn("metadata", h5_file)
            self.assertEqual(h5_file["metadata"].attrs["gamma_cameras"], 4)

            # 5. КРИТИЧЕСКАЯ ПРОВЕРКА: группа /projections НЕ ДОЛЖНА присутствовать
            self.assertNotIn("projections", h5_file, "Группа /projections не должна сохраняться в HDF5!")

        dose_handler.close()


if __name__ == '__main__':
    unittest.main()
