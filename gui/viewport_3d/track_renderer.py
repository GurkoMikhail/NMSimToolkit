from collections import deque
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

_logger = logging.getLogger(__name__)

# Цветовая дифференциация физических процессов (RGB [0, 1])
PROCESS_COLORS: Dict[int, Tuple[float, float, float]] = {
    -2: (1.00, 0.85, 0.20),  # Source Emission (Ярко-желтый / рождение)
    -1: (0.45, 0.85, 0.95),  # Escaped Particle (Светло-голубой / вылетевшие)
    0: (0.20, 0.60, 1.00),   # Rayleigh Scattering (Синий)
    1: (0.95, 0.77, 0.06),   # Compton Scattering (Желтый)
    2: (0.91, 0.30, 0.24),   # Photoelectric Absorption (Красный)
    3: (0.61, 0.35, 0.71),   # Pair Production (Фиолетовый)
    4: (0.18, 0.80, 0.44),   # Detection / Scintillation (Зеленый)
}


class TrackRenderer:
    """
    Рендерер 3D-траекторий фотонов в реальном времени.
    Отображает траектории частиц с цветовой индикацией типов физических взаимодействий.
    Использует кольцевой буфер сегментов для сохранения высокого FPS при интенсивном счете.
    
    Оптимизации:
    - Защита от фризов GUI через троттлинг перерисовки PyVista (min_render_interval).
    - Защита от сброса камеры PyVista (reset_camera=False).
    - Надежное отображение: render_points_as_spheres=True и непрозрачность для видимости внутри фантомов.
    - Корректная обработка отсутствующей координаты Z (2D/3D совместимость).
    """

    def __init__(
        self,
        viewport: Any,
        actor_name: str = "particle_tracks",
        max_points: int = 50000,
        render_as_lines: bool = False,
        min_render_interval: float = 0.05,
    ) -> None:
        self.viewport = viewport
        self.actor_name = actor_name
        self.max_points = max_points
        self.render_as_lines = render_as_lines
        self.min_render_interval = float(min_render_interval)

        # Кольцевые буферы для хранения координат и цветов
        self._point_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=max_points)
        self._color_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=max_points)
        self._lines_buffer: Deque[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = deque(
            maxlen=max_points
        )
        self._particle_last_pos: Dict[int, Tuple[float, float, float]] = {}
        self._visible: bool = True
        self._last_update_time: float = 0.0
        self._current_rendered_style: Optional[str] = None

    def add_tracks_batch(self, batch: Dict[str, np.ndarray]) -> None:
        """
        Добавляет пакет треков из очереди телеметрии ядра.
        """
        pos_x = batch.get('pos_x')
        pos_y = batch.get('pos_y')
        pos_z = batch.get('pos_z')
        process_ids = batch.get('process_id')
        particle_ids = batch.get('particle_id')

        if pos_x is None or len(pos_x) == 0:
            return

        if pos_z is None:
            pos_z = np.zeros_like(pos_x)

        n_new = len(pos_x)

        # Добавляем точки и цвета в кольцевые буферы
        for i in range(n_new):
            pt = (float(pos_x[i]), float(pos_y[i]), float(pos_z[i]))
            self._point_buffer.append(pt)

            proc = int(process_ids[i]) if process_ids is not None else 1
            color = PROCESS_COLORS.get(proc, (0.8, 0.8, 0.8))
            self._color_buffer.append(color)

            # Соединяем последовательные взаимодействия одной и той же частицы
            # даже если события разделены другими частицами в SoA-банке
            if self.render_as_lines:
                if particle_ids is not None:
                    pid = int(particle_ids[i])
                    if pid in self._particle_last_pos:
                        prev_pt = self._particle_last_pos[pid]
                        self._lines_buffer.append((prev_pt, pt, color))
                    self._particle_last_pos[pid] = pt
                elif i > 0:
                    prev_pt = (float(pos_x[i - 1]), float(pos_y[i - 1]), float(pos_z[i - 1]))
                    self._lines_buffer.append((prev_pt, pt, color))

        # Ограничение размера карты предыдущих позиций частиц
        if len(self._particle_last_pos) > 10000:
            keys_to_remove = list(self._particle_last_pos.keys())[:5000]
            for k in keys_to_remove:
                del self._particle_last_pos[k]

        # Троттлинг отрисовки во избежание блокировки главного потока GUI
        now = time.time()
        if (now - self._last_update_time) >= self.min_render_interval:
            self.update_mesh()

    def update_mesh(self) -> None:
        """
        Обновляет полигональный меш треков в VTK / PyVista.
        """
        if not self._visible or len(self._point_buffer) < 2:
            return

        if self.viewport is None or self.viewport.plotter is None:
            return

        try:
            if self.render_as_lines and len(self._lines_buffer) > 0:
                n_lines = len(self._lines_buffer)
                pts = np.empty((n_lines * 2, 3), dtype=np.float32)
                colors = np.empty((n_lines * 2, 3), dtype=np.float32)
                lines_arr = np.empty(n_lines * 3, dtype=np.int64)

                for idx, (p1, p2, col) in enumerate(self._lines_buffer):
                    pts[idx * 2] = p1
                    pts[idx * 2 + 1] = p2
                    colors[idx * 2] = col
                    colors[idx * 2 + 1] = col
                    lines_arr[idx * 3] = 2
                    lines_arr[idx * 3 + 1] = idx * 2
                    lines_arr[idx * 3 + 2] = idx * 2 + 1

                poly = pv.PolyData(pts, lines=lines_arr)
                poly.point_data['RGB'] = (colors * 255).astype(np.uint8)

                # Добавляем вершины для визуализации точек взаимодействий
                n_pts = len(pts)
                verts_arr = np.column_stack((np.ones(n_pts, dtype=np.int64), np.arange(n_pts, dtype=np.int64))).ravel()
                poly.verts = verts_arr

                # In-place обновление существующего актора для устранения утечек памяти и компиляции шейдеров
                existing_actor = self.viewport._actors.get(self.actor_name)
                if (existing_actor is not None and existing_actor.GetMapper() is not None and
                        self._current_rendered_style == 'lines'):
                    mapper = existing_actor.GetMapper()
                    mapper.SetInputData(poly)
                    poly.Modified()
                else:
                    self.viewport.add_mesh_actor(
                        self.actor_name,
                        poly,
                        rgb=True,
                        opacity=0.95,
                        style='surface',
                        line_width=2.0,
                        point_size=5.0,
                        render_points_as_spheres=True,
                        reset_camera=False,
                    )
                    self._current_rendered_style = 'lines'
                self.viewport.render()
            else:
                pts = np.array(self._point_buffer, dtype=np.float32)
                colors = np.array(self._color_buffer, dtype=np.float32)

                poly = pv.PolyData(pts)
                poly.point_data['RGB'] = (colors * 255).astype(np.uint8)

                existing_actor = self.viewport._actors.get(self.actor_name)
                if (existing_actor is not None and existing_actor.GetMapper() is not None and
                        self._current_rendered_style == 'points'):
                    mapper = existing_actor.GetMapper()
                    mapper.SetInputData(poly)
                    poly.Modified()
                else:
                    self.viewport.add_mesh_actor(
                        self.actor_name,
                        poly,
                        rgb=True,
                        opacity=0.95,
                        style='points',
                        point_size=6.0,
                        render_points_as_spheres=True,
                        reset_camera=False,
                    )
                    self._current_rendered_style = 'points'
                self.viewport.render()

            self._last_update_time = time.time()

        except Exception as e:
            _logger.debug(f"Ошибка обновления треков: {e}")

    def clear(self) -> None:
        """
        Очистка всех накопленных треков.
        """
        self._point_buffer.clear()
        self._color_buffer.clear()
        self._lines_buffer.clear()
        self._particle_last_pos.clear()
        self._current_rendered_style = None
        self._last_update_time = 0.0

        if self.viewport is not None:
            self.viewport.remove_actor(self.actor_name)
            self.viewport.render()

    def set_visible(self, visible: bool) -> None:
        """
        Включение / выключение видимости треков.
        """
        self._visible = visible
        if not visible:
            if self.viewport is not None:
                self.viewport.remove_actor(self.actor_name)
                self.viewport.render()
            self._current_rendered_style = None
        else:
            self.update_mesh()

    def set_render_as_lines(self, enable: bool) -> None:
        """
        Переключение режима отображения: связные линии траекторий или точечные взаимодействия.
        """
        if self.render_as_lines != enable:
            self.render_as_lines = enable
            self._current_rendered_style = None
            self.update_mesh()

    def set_max_points(self, max_points: int) -> None:
        """
        Изменение максимального лимита точек/сегментов в буфере треков.
        """
        self.max_points = max(100, int(max_points))
        # Переинициализация буферов с новой емкостью
        new_pt = deque(self._point_buffer, maxlen=self.max_points)
        new_col = deque(self._color_buffer, maxlen=self.max_points)
        new_lines = deque(self._lines_buffer, maxlen=self.max_points)
        self._point_buffer = new_pt
        self._color_buffer = new_col
        self._lines_buffer = new_lines
        self.update_mesh()
