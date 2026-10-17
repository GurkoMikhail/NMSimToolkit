import os
import sys
import time
import unittest
from multiprocessing import shared_memory
from typing import Any

import numpy as np

# Настройка headless режима для Qt и PyVista
os.environ['QT_API'] = 'pyside6'
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

import pyvista as pv
import hepunits as units
from core.data.dose_map_handler import DoseMapHandler
from core.physics.interaction_buffers import ChunkType
from gui.controllers.stream_handlers import GuiStreamDataHandler
from gui.viewport_3d.vtk_viewport import VTKViewport
from gui.viewport_3d.dose_volume_renderer import DoseVolumeRenderer
from gui.viewport_3d.track_renderer import TrackRenderer
from gui.views.main_window import MainWindow
from gui.controllers.orchestrator_session import OrchestratorSession
from gui.controllers.ipc_receiver import IPCReceiver
from gui.controllers.viewport_controller import SceneViewportController
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.source.sources import PointSource
from core.scene.dose_grid_node import DoseGridNode
from core.scene.nodes import CompositeNode
from settings.database_setting import material_database


class TestDoseAndTrackFixes(unittest.TestCase):
    """
    Комплексный набор тестов для верификации багфиксов:
    1. Отсутствие сброса камеры PyVista при обновлении DoseVisualizer и TrackRenderer.
    2. Устранение утечек памяти (in-place обновление сетки без пересоздания актора).
    3. Корректность накопления дозы в DoseMapHandler и передача через IPC.
    4. Полный сброс карты дозы, проекции и треков при нажатии 'Сбросить накопление'.
    5. Корректное построение и отображение 3D-треков частиц.
    """

    def test_dose_map_handler_binning_and_lifecycle(self):
        """
        Проверка точности 3D-биннинга, накопления энергии в вокселях
        и корректного освобождения сегмента SharedMemory.
        """
        shm_name = f"test_dose_shm_{int(time.time() * 1000)}"
        grid_node = DoseGridNode(name="TestDoseGrid", size=[40.0, 40.0, 40.0], dose_voxel_size=10.0)
        grid_node.translate(20.0, 20.0, 20.0)
        handler = DoseMapHandler(
            grid_nodes=[grid_node],
            shm_name=shm_name,
            create_shm=True,
        )

        try:
            self.assertEqual(len(handler.entries), 1)
            entry = handler.entries[0]
            self.assertIsNotNone(entry._dose_grid)
            self.assertEqual(entry._dose_grid.shape, (4, 4, 4))
            self.assertEqual(np.sum(entry._dose_grid), 0.0)

            # Подаем взаимодействия:
            # Точка 1: (5, 5, 5) -> воксель (0, 0, 0), edep = 0.150
            # Точка 2: (35, 35, 35) -> воксель (3, 3, 3), edep = 0.050
            # Точка 3: (50, 50, 50) -> вне границ сетки [0, 40), должна игнорироваться
            # Точка 4: (5, 5, 5) -> edep = 0.0 (без энерговыделения)
            chunk = {
                'type': ChunkType.INTERACTIONS,
                'data': {
                    'pos_x': np.array([5.0, 35.0, 50.0, 5.0]),
                    'pos_y': np.array([5.0, 35.0, 50.0, 5.0]),
                    'pos_z': np.array([5.0, 35.0, 50.0, 5.0]),
                    'energy_deposit': np.array([0.150, 0.050, 0.200, 0.0]),
                }
            }
            handler.process_chunk(chunk)

            snapshot = handler.get_dose_snapshot()
            self.assertIsNotNone(snapshot)
            self.assertAlmostEqual(snapshot[0, 0, 0], 0.150, places=6)
            self.assertAlmostEqual(snapshot[3, 3, 3], 0.050, places=6)
            self.assertAlmostEqual(np.sum(snapshot), 0.200, places=6)

            # Проверка сброса накопленной дозы
            handler.clear()
            self.assertEqual(np.sum(entry._dose_grid), 0.0)

        finally:
            handler.close()

    def test_dose_volume_renderer_camera_stability_and_inplace_mutation(self):
        """
        Проверка:
        1. Обновление дозы НЕ сбрасывает положение камеры.
        2. Обновление выполняется in-place без пересоздания vtkActor/ImageData.
        3. Метод clear() корректно обнуляет скалярные данные.
        """
        vp = VTKViewport()
        renderer = DoseVolumeRenderer(vp, actor_name="test_dose")

        try:
            # Первичная инициализация сетки
            grid_shape = (8, 8, 8)
            dose_data = np.zeros(grid_shape, dtype=np.float32)
            dose_data[4, 4, 4] = 100.0

            renderer.update_dose_data(dose_data, voxel_size=2.0, origin=(-8.0, -8.0, -8.0))
            self.assertIsNotNone(renderer.volume_actor)
            initial_actor = renderer.volume_actor
            initial_grid = renderer.grid

            # Устанавливаем пользовательское положение камеры
            if vp.plotter is not None and vp.plotter.camera is not None:
                custom_pos = (123.0, 456.0, 789.0)
                vp.plotter.camera.position = custom_pos

                # Выполняем 5 последовательных обновлений с новыми данными
                for step in range(1, 6):
                    dose_data[step, step, step] = 50.0 * step
                    renderer.update_dose_data(dose_data)
                    # Проверяем, что камера НЕ сбросилась
                    curr_pos = vp.plotter.camera.position
                    self.assertAlmostEqual(curr_pos[0], custom_pos[0], places=1)
                    self.assertAlmostEqual(curr_pos[1], custom_pos[1], places=1)
                    self.assertAlmostEqual(curr_pos[2], custom_pos[2], places=1)

                # Проверяем, что актор и сетка НЕ пересоздавались (in-place)
                self.assertIs(renderer.volume_actor, initial_actor)
                self.assertIs(renderer.grid, initial_grid)

            # Проверка сброса
            renderer.clear()
            if renderer.grid is not None:
                self.assertEqual(np.sum(renderer.grid.point_data['dose']), 0.0)

        finally:
            vp.close()

    def test_track_renderer_camera_stability_and_rendering(self):
        """
        Проверка:
        1. Добавление пакетов треков НЕ сбрасывает положение камеры вьюпорта.
        2. Формируются связные траектории частиц и сферические вершины.
        3. Метод clear() очищает буферы и актор сцены.
        """
        vp = VTKViewport()
        renderer = TrackRenderer(vp, actor_name="test_tracks", render_as_lines=True, min_render_interval=0.0)

        try:
            if vp.plotter is not None and vp.plotter.camera is not None:
                custom_pos = (55.0, 66.0, 77.0)
                vp.plotter.camera.position = custom_pos

                # Пакет треков с частицами, имеющими несколько взаимодействий
                batch = {
                    'pos_x': np.array([0.0, 10.0, 20.0, 0.0, 15.0]),
                    'pos_y': np.array([0.0, 5.0, 10.0, 0.0, -10.0]),
                    'pos_z': np.array([0.0, 2.0, 4.0, 0.0, -5.0]),
                    'process_id': np.array([0, 1, 2, 0, 4]),
                    'particle_id': np.array([1, 1, 1, 2, 2]),
                }

                renderer.add_tracks_batch(batch)
                renderer.update_mesh()

                # Проверка сохранения камеры
                curr_pos = vp.plotter.camera.position
                self.assertAlmostEqual(curr_pos[0], custom_pos[0], places=1)
                self.assertAlmostEqual(curr_pos[1], custom_pos[1], places=1)
                self.assertAlmostEqual(curr_pos[2], custom_pos[2], places=1)

                self.assertIn("test_tracks", vp._actors)
                self.assertGreater(len(renderer._lines_buffer), 0)

            # Проверка очистки
            renderer.clear()
            self.assertEqual(len(renderer._point_buffer), 0)
            self.assertEqual(len(renderer._lines_buffer), 0)
            self.assertNotIn("test_tracks", vp._actors)

        finally:
            vp.close()

    def test_clear_accumulation_full_pipeline(self):
        """
        Проверка сквозного сброса накопления:
        ResultsViewer.clear_results() -> MainWindow._on_clear_accumulation() ->
        OrchestratorSession.clear_accumulation() -> обнуление проекции, спектра,
        3D-карты дозы в SharedMemory и рендереров.
        """
        win = MainWindow()

        try:
            # Инициализируем тестовые данные в ResultsViewer
            proj_data = np.ones((128, 128), dtype=np.float32) * 5.0
            win.results_viewer.set_projection_data(proj_data)
            win.results_viewer.set_spectrum_data(np.array([100.0, 120.0, 140.0]))

            # Инициализируем дозу и треки
            test_dose = np.ones((16, 16, 16), dtype=np.float32)
            win.viewport_controller.dose_renderer.update_dose_data(test_dose)

            batch = {
                'pos_x': np.array([1.0, 2.0]),
                'pos_y': np.array([1.0, 2.0]),
                'pos_z': np.array([1.0, 2.0]),
                'process_id': np.array([1, 1]),
                'particle_id': np.array([1, 1]),
            }
            win.viewport_controller.track_renderer.add_tracks_batch(batch)

            # Нажимаем сброс накопления в ResultsViewer
            win.results_viewer.btn_clear.click()
            app.processEvents()

            # Проверяем обнуление всех подсистем
            self.assertIsNone(win.results_viewer._current_projection)
            self.assertEqual(win.results_viewer.lbl_stats.text(), "Всего отсчетов: 0")

            if win.viewport_controller.dose_renderer.grid is not None:
                self.assertEqual(np.sum(win.viewport_controller.dose_renderer.grid.point_data['dose']), 0.0)

            self.assertEqual(len(win.viewport_controller.track_renderer._point_buffer), 0)

        finally:
            win.close()

    def test_orchestrator_session_dose_and_track_signals(self):
        """
        Проверка сигнатур сигналов и интеграции параметров сетки дозы в OrchestratorSession.
        """
        world = Volume(
            geometry=Box(60 * units.cm, 60 * units.cm, 60 * units.cm),
            material=material_database['Water, Liquid'],
            name='WaterBox'
        )
        source = PointSource(activity=50 * units.MBq, energy=140 * units.keV)
        world.add_child(source)
        dose_grid = DoseGridNode(name="DoseScorer", size=[160.0, 160.0, 160.0], dose_voxel_size=10.0)
        world.add_child(dose_grid)

        scene_vm = SceneViewModel(world)
        session = OrchestratorSession(scene_vm=scene_vm)
        try:
            self.assertEqual(session.dose_voxel_size, 10.0)
            self.assertEqual(session.dose_origin, (-80.0, -80.0, -80.0))

            received_tracks = []
            received_doses = []

            session.tracks_received.connect(received_tracks.append)
            session.dose_volume_received.connect(received_doses.append)

            sample_tracks = {'pos_x': np.array([1.0, 2.0]), 'pos_y': np.array([0.0, 0.0]), 'pos_z': np.array([0.0, 0.0])}
            sample_dose = np.ones((16, 16, 16), dtype=np.float32)

            session.tracks_received.emit(sample_tracks)
            session.dose_volume_received.emit(sample_dose)

            self.assertEqual(len(received_tracks), 1)
            self.assertEqual(len(received_doses), 1)
            self.assertEqual(received_doses[0].shape, (16, 16, 16))
        finally:
            session.close()

    def test_track_renderer_interleaved_particles_and_inplace_actor(self):
        """
        Проверка:
        1. Треки частиц с перемежающимися (не последовательными) индексами корректно связываются.
        2. Треки связываются между несколькими пакетами.
        3. Обновление меша выполняется in-place без пересоздания актора VTK.
        """
        vp = VTKViewport()
        renderer = TrackRenderer(vp, actor_name="test_tracks_interleaved", render_as_lines=True, min_render_interval=0.0)

        try:
            # Пакет 1: частицы 10 и 20 перемежаются (не идут подряд)
            # 10 -> (0,0,0), 20 -> (5,5,5), 10 -> (10,0,0), 20 -> (5,15,5)
            batch1 = {
                'pos_x': np.array([0.0, 5.0, 10.0, 5.0]),
                'pos_y': np.array([0.0, 5.0, 0.0, 15.0]),
                'pos_z': np.array([0.0, 5.0, 0.0, 5.0]),
                'process_id': np.array([1, 1, 2, 2]),
                'particle_id': np.array([10, 20, 10, 20]),
            }
            renderer.add_tracks_batch(batch1)
            self.assertEqual(len(renderer._lines_buffer), 2, "Должно быть 2 линии для перемежающихся частиц")

            initial_actor = vp._actors.get("test_tracks_interleaved")
            self.assertIsNotNone(initial_actor)

            # Пакет 2: продолжение частицы 10 в следующем пакете
            batch2 = {
                'pos_x': np.array([20.0]),
                'pos_y': np.array([0.0]),
                'pos_z': np.array([0.0]),
                'process_id': np.array([1]),
                'particle_id': np.array([10]),
            }
            renderer.add_tracks_batch(batch2)
            self.assertEqual(len(renderer._lines_buffer), 3, "Частица 10 должна связаться через границу пакетов")

            # Проверяем, что актор обновился in-place без удаления и создания заново
            curr_actor = vp._actors.get("test_tracks_interleaved")
            self.assertIs(curr_actor, initial_actor)

        finally:
            vp.close()

    def test_ipc_receiver_spectrum_memory_boundedness(self):
        """
        Проверка:
        IPCReceiver не накапливает сырые массивы энергий бесконечно и удерживает объем памяти ограниченным.
        """
        receiver = IPCReceiver(fps=30.0)
        try:
            receiver.max_spectrum_samples = 500  # Снижаем порог для быстрого теста
            # Отправляем 20 пакетов по 100 событий = 2000 событий
            for _ in range(20):
                item = {
                    'type': 'tracks',
                    'detector_energy_deposit': np.ones(100, dtype=np.float32) * 0.140
                }
                receiver._process_track_item(item)

            total_samples = sum(len(arr) for arr in receiver._accumulated_energies)
            self.assertLessEqual(total_samples, receiver.max_spectrum_samples * 2)
            self.assertGreater(total_samples, 0)

            # Проверка очистки накопления
            receiver.clear_accumulation()
            self.assertEqual(len(receiver._accumulated_energies), 0)
        finally:
            receiver.close()

    def test_viewport_interaction_render_guard(self):
        """
        Проверка:
        VTKViewport не вызывает render() во время интерактивного движения камеры мышью,
        предотвращая сброс фокуса манипулятора и рывки.
        """
        vp = VTKViewport()
        try:
            if vp.plotter is not None:
                # Включаем флаг взаимодействия
                vp._is_interacting = True
                # Вызов render не должен падать и должен игнорироваться
                vp.render()
                # Завершаем взаимодействие
                vp._on_interaction_end(None, '')
                self.assertFalse(vp._is_interacting)
        finally:
            vp.close()

    def test_dose_renderer_origin_and_transform_preservation_on_simulation_stop(self):
        """
        Проверка:
        1. DoseVolumeRenderer сохраняет origin и transform_matrix при обновлениях.
        2. MainWindow сохраняет origin и матрицу трансформации сетки дозы после
           остановки сессии (session = None), предотвращая смещение карты дозы
           в угол объема сканирования (-160, -160, -160).
        """
        win = MainWindow()
        try:
            # Задаем тестовые параметры сетки дозы, отличные от захардкоженных (-160, -160, -160)
            custom_origin = (-50.0, -50.0, -50.0)
            custom_voxel_size = 2.5
            custom_matrix = np.eye(4, dtype=np.float64)
            custom_matrix[0, 3] = 10.0
            custom_matrix[1, 3] = 20.0
            custom_matrix[2, 3] = 30.0

            # Устанавливаем параметры в контроллере вьюпорта окна
            win.viewport_controller.active_dose_origin = custom_origin
            win.viewport_controller.active_dose_voxel_size = custom_voxel_size
            win.viewport_controller.active_dose_transform_matrix = custom_matrix

            # Имитируем прием дозы во время симуляции
            dose_data = np.ones((20, 20, 20), dtype=np.float32)
            win._on_dose_volume_received(dose_data)

            self.assertEqual(win.viewport_controller.dose_renderer.origin, custom_origin)
            self.assertEqual(win.viewport_controller.dose_renderer.base_voxel_size, (custom_voxel_size, custom_voxel_size, custom_voxel_size))
            self.assertIsNotNone(win.viewport_controller.dose_renderer.transform_matrix)
            np.testing.assert_array_almost_equal(win.viewport_controller.dose_renderer.transform_matrix, custom_matrix)

            # Имитируем остановку симуляции: session становится None
            win.session = None

            # Приходит финальный сброс дозы от закрывающегося IPCReceiver
            win._on_dose_volume_received(dose_data * 2.0)

            # Проверяем, что origin и transform_matrix НЕ сбросились в фоллбэк (-160, -160, -160)
            self.assertEqual(win.viewport_controller.dose_renderer.origin, custom_origin)
            self.assertEqual(win.viewport_controller.dose_renderer.base_voxel_size, (custom_voxel_size, custom_voxel_size, custom_voxel_size))
            self.assertIsNotNone(win.viewport_controller.dose_renderer.transform_matrix)
            np.testing.assert_array_almost_equal(win.viewport_controller.dose_renderer.transform_matrix, custom_matrix)
            self.assertNotEqual(win.viewport_controller.dose_renderer.origin, (-160.0, -160.0, -160.0))
        finally:
            win.close()

    def test_multi_actor_dose_volume_renderer_and_controller_sync(self) -> None:
        """
        Проверка мульти-акторной визуализации нескольких сеток дозы:
        1. Одновременная отрисовка N именованных сеток из Dict[str, np.ndarray].
        2. Независимое обновление in-place без пересоздания акторов VTK.
        3. Управление индивидуальной видимостью через is_active и dose_visible.
        4. Синхронизация трансформации актора при перемещении DoseGridViewModel.
        5. Удаление актора сетки при удалении узла DoseGridViewModel из сцены.
        """
        vp = VTKViewport()
        try:
            root = CompositeNode(name="World")
            grid_node_a = DoseGridNode(name="OrganA", size=[40.0, 40.0, 40.0], dose_voxel_size=10.0)
            grid_node_b = DoseGridNode(name="OrganB", size=[60.0, 60.0, 60.0], dose_voxel_size=20.0)
            grid_node_b.translate(100.0, 0.0, 0.0)
            root.add_child(grid_node_a)
            root.add_child(grid_node_b)

            scene_vm = SceneViewModel(root)
            controller = SceneViewportController(viewport=vp, scene_vm=scene_vm)

            vm_a = scene_vm.find_by_name("OrganA")
            vm_b = scene_vm.find_by_name("OrganB")
            self.assertIsInstance(vm_a, DoseGridViewModel)
            self.assertIsInstance(vm_b, DoseGridViewModel)

            dose_a = np.zeros((4, 4, 4), dtype=np.float32)
            dose_a[1, 1, 1] = 25.0
            dose_b = np.zeros((3, 3, 3), dtype=np.float32)
            dose_b[2, 2, 2] = 80.0

            controller.on_dose_volume_received({"OrganA": dose_a, "OrganB": dose_b})

            entries = controller.dose_renderer.entries
            self.assertIn("OrganA", entries)
            self.assertIn("OrganB", entries)
            entry_a = entries["OrganA"]
            entry_b = entries["OrganB"]

            self.assertEqual(entry_a.actor_name, "dose_volume_OrganA")
            self.assertEqual(entry_b.actor_name, "dose_volume_OrganB")
            self.assertEqual(entry_a.base_voxel_size, (10.0, 10.0, 10.0))
            self.assertEqual(entry_b.base_voxel_size, (20.0, 20.0, 20.0))
            self.assertAlmostEqual(float(np.max(entry_a.grid.point_data["dose"])), 25.0)
            self.assertAlmostEqual(float(np.max(entry_b.grid.point_data["dose"])), 80.0)

            initial_grid_a = entry_a.grid
            initial_grid_b = entry_b.grid

            # Повторное обновление мутирует сетки in-place
            dose_a[0, 0, 0] = 50.0
            controller.on_dose_volume_received({"OrganA": dose_a, "OrganB": dose_b})
            self.assertIs(entries["OrganA"].grid, initial_grid_a)
            self.assertIs(entries["OrganB"].grid, initial_grid_b)
            self.assertAlmostEqual(float(np.max(entry_a.grid.point_data["dose"])), 50.0)

            # Отключение dose_visible скрывает сетку OrganA, оставляя OrganB видимой
            assert isinstance(vm_a, DoseGridViewModel)
            assert isinstance(vm_b, DoseGridViewModel)
            vm_a.dose_visible = False
            self.assertFalse(entries["OrganA"].visible)
            self.assertTrue(entries["OrganB"].visible)

            vm_a.dose_visible = True
            self.assertTrue(entries["OrganA"].visible)

            # Отключение is_active также скрывает сетку
            vm_b.is_active = False
            self.assertFalse(entries["OrganB"].visible)
            vm_b.is_active = True
            self.assertTrue(entries["OrganB"].visible)

            # Перемещение узла обновляет матрицу трансформации соответствующей сетки дозы
            vm_a.translate(x=45.0, y=-15.0, z=10.0)
            matrix_a = entries["OrganA"].transform_matrix
            assert matrix_a is not None
            np.testing.assert_allclose(matrix_a, vm_a.global_matrix)

            # Удаление узла из сцены удаляет соответствующую запись из рендерера дозы
            scene_vm.node_removed.connect(controller.on_node_removed)
            scene_vm.remove_node(vm_a)
            self.assertNotIn("OrganA", controller.dose_renderer.entries)
            self.assertIn("OrganB", controller.dose_renderer.entries)
        finally:
            vp.close()


if __name__ == '__main__':
    unittest.main()

