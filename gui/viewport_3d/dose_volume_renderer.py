import logging
import time
from typing import Any, Optional, Tuple, Union

import numpy as np
import pyvista as pv

from gui.viewport_3d.dicom_colormaps import (
    get_colormap_lut,
    to_vtk_color_transfer_function,
    to_vtk_piecewise_function,
)

_logger = logging.getLogger(__name__)


class DoseVolumeRenderer:
    """
    Специализированный 3D-рендерер воксельной дозы (накопленной карты энерговыделения)
    на базе PyVista и vtkSmartVolumeMapper.
    
    Особенности и оптимизации:
    1. Инкрементальное обновление данных in-place: скалярный массив мутируется без
       пересоздания структуры сетки ImageData, актора vtkVolume или vtkSmartVolumeMapper.
    2. Полное отсутствие сброса камеры PyVista при обновлениях (reset_camera=False).
    3. Отсутствие утечек оперативной памяти: однократная аллокация структур VTK/GPU.
    4. Нулевая прозрачность (alpha=0) для фоновых и околонулевых вокселей, предотвращающая
       перекрытие геометрических объектов сцены и треков частиц.
    5. Защита от фризов GUI за счет ограничения частоты перерисовки (throttling).
    """

    def __init__(
        self,
        viewport: Any,
        actor_name: str = "dose_volume",
        colormap: str = "Hot Iron",
        threshold_ratio: float = 0.02,
        min_render_interval: float = 0.15,
    ) -> None:
        self.viewport = viewport
        self.actor_name = actor_name
        self.colormap = colormap
        self.threshold_ratio = float(threshold_ratio)
        self.min_render_interval = float(min_render_interval)

        self.grid: Optional[pv.ImageData] = None
        self.volume_actor: Any = None
        self.volume_mapper: Any = None
        self.volume_property: Any = None

        self.base_voxel_size: Tuple[float, float, float] = (5.0, 5.0, 5.0)
        self.origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.transform_matrix: Optional[np.ndarray] = None
        self.scalar_range: Tuple[float, float] = (0.0, 1.0)
        self._visible: bool = True
        self._last_render_time: float = 0.0

    def setup_grid(
        self,
        grid_shape: Tuple[int, int, int],
        voxel_size: Union[float, Tuple[float, float, float]] = 5.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        transform_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Однократная инициализация ImageData и VTK Volume Actor.
        """
        if isinstance(voxel_size, (int, float)):
            spacing = (float(voxel_size), float(voxel_size), float(voxel_size))
        else:
            spacing = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))

        self.base_voxel_size = spacing
        self.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
        if transform_matrix is not None:
            self.transform_matrix = np.asarray(transform_matrix, dtype=np.float64)

        # Создаем регулярную 3D-сетку ImageData
        self.grid = pv.ImageData(
            dimensions=grid_shape,
            spacing=spacing,
            origin=self.origin
        )
        # Инициализируем плоский массив дозы нулями
        zeros = np.zeros(int(np.prod(grid_shape)), dtype=np.float32)
        self.grid.point_data['dose'] = zeros
        self.scalar_range = (0.0, 1.0)

        if self.viewport is not None:
            # Добавляем актор с reset_camera=False
            self.volume_actor = self.viewport.add_volume_actor(
                self.actor_name,
                self.grid,
                cmap=self.colormap,
                opacity='linear',
                mapper='smart',
                scalars='dose',
                reset_camera=False,
            )
            if self.volume_actor is not None:
                self.volume_mapper = self.volume_actor.GetMapper()
                self.volume_property = self.volume_actor.GetProperty()
                self._update_transfer_functions(max_dose=1.0)
                self.volume_actor.SetVisibility(1 if self._visible else 0)
                mat_to_apply = transform_matrix if transform_matrix is not None else self.transform_matrix
                if mat_to_apply is not None:
                    self.viewport.update_actor_transform(self.actor_name, mat_to_apply)
            self.viewport.render()

    def update_dose_data(
        self,
        dose_3d: np.ndarray,
        voxel_size: Optional[Union[float, Tuple[float, float, float]]] = None,
        origin: Optional[Tuple[float, float, float]] = None,
        transform_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Высокопроизводительное обновление дозы in-place без пересоздания актора.
        """
        if not self._visible:
            return

        grid_shape = dose_3d.shape
        v_size = voxel_size if voxel_size is not None else self.base_voxel_size
        v_orig = origin if origin is not None else self.origin

        if isinstance(v_size, (int, float)):
            v_spacing = (float(v_size), float(v_size), float(v_size))
        else:
            v_spacing = (float(v_size[0]), float(v_size[1]), float(v_size[2]))

        if isinstance(v_orig, (tuple, list, np.ndarray)):
            v_origin = (float(v_orig[0]), float(v_orig[1]), float(v_orig[2]))
        else:
            v_origin = self.origin

        if transform_matrix is not None:
            self.transform_matrix = np.asarray(transform_matrix, dtype=np.float64)

        # Если сетка еще не создана или изменились габариты, шаг вокселя или origin, инициализируем ее
        if (self.grid is None or 
            self.grid.dimensions != grid_shape or 
            self.base_voxel_size != v_spacing or 
            self.origin != v_origin):
            self.setup_grid(grid_shape, voxel_size=v_spacing, origin=v_origin, transform_matrix=self.transform_matrix)
        elif self.transform_matrix is not None:
            self.viewport.update_actor_transform(self.actor_name, self.transform_matrix)

        if self.grid is None:
            return

        try:
            # 1. Мутация скаляров in-place без выделения новых VTK-объектов
            flat_dose = dose_3d.flatten(order='F').astype(np.float32)
            self.grid.point_data['dose'][:] = flat_dose

            # 2. Оповещение конвейера VTK об изменении скалярных данных
            scalars = self.grid.GetPointData().GetScalars()
            if scalars is not None:
                scalars.Modified()
            self.grid.Modified()

            # 3. Подстройка динамического диапазона и функции прозрачности
            max_dose = float(np.max(dose_3d))
            if max_dose > 1e-9:
                prev_max = self.scalar_range[1]
                if prev_max <= 1e-9 or abs(max_dose - prev_max) / max(1e-9, prev_max) > 0.03:
                    self.scalar_range = (0.0, max_dose)
                    self._update_transfer_functions(max_dose=max_dose)

            # 4. Троттлинг отрисовки во избежание фризов графического интерфейса
            now = time.time()
            if (now - self._last_render_time) >= self.min_render_interval:
                if self.viewport is not None:
                    self.viewport.render()
                self._last_render_time = now

        except Exception as e:
            _logger.debug(f"Ошибка обновления карты дозы: {e}")

    def _update_transfer_functions(self, max_dose: float) -> None:
        """
        Обновление функций цвета и прозрачности с абсолютным отсечением фоновых вокселей.
        """
        if self.volume_property is None:
            return

        s_range = (0.0, max(1e-6, max_dose))

        # Цветовая шкала
        color_tf = to_vtk_color_transfer_function(self.colormap, scalar_range=s_range)
        if color_tf is not None:
            self.volume_property.SetColor(color_tf)

        # Функция прозрачности: фоновые воксели и воксели ниже порога абсолютно прозрачны
        opacity_tf = to_vtk_piecewise_function(
            scalar_range=s_range,
            min_alpha=0.0,
            max_alpha=0.85,
            ramp_type='step' if self.threshold_ratio > 0.001 else 'linear',
            threshold=self.threshold_ratio
        )
        if opacity_tf is not None:
            self.volume_property.SetScalarOpacity(opacity_tf)

    def clear(self) -> None:
        """
        Очистка накопленной карты дозы.
        """
        if self.grid is not None:
            try:
                self.grid.point_data['dose'][:] = 0.0
                scalars = self.grid.GetPointData().GetScalars()
                if scalars is not None:
                    scalars.Modified()
                self.grid.Modified()
                self.scalar_range = (0.0, 1.0)
                self._update_transfer_functions(max_dose=1.0)
                if self.viewport is not None:
                    self.viewport.render()
            except Exception as e:
                _logger.debug(f"Ошибка очистки карты дозы: {e}")

    def set_colormap(self, colormap_name: str) -> None:
        """
        Динамическое изменение цветовой палитры дозы.
        """
        self.colormap = colormap_name
        self._update_transfer_functions(max_dose=self.scalar_range[1])
        if self.viewport is not None:
            self.viewport.render()

    def set_threshold(self, min_dose_ratio: float) -> None:
        """
        Установка порога отсечения фонового излучения в долях от максимума.
        """
        self.threshold_ratio = float(np.clip(min_dose_ratio, 0.0, 0.99))
        self._update_transfer_functions(max_dose=self.scalar_range[1])
        if self.viewport is not None:
            self.viewport.render()

    def set_visible(self, visible: bool) -> None:
        """
        Включение / выключение отображения объема дозы.
        """
        self._visible = visible
        if self.volume_actor is not None:
            self.volume_actor.SetVisibility(1 if visible else 0)
            if self.viewport is not None:
                self.viewport.render()


# Псевдонимы для обратной совместимости с различными именованиями
DoseVisualizer = DoseVolumeRenderer
DoseViaualizator = DoseVolumeRenderer
