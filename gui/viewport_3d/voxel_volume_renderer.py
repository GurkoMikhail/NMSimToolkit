import logging
from typing import Any, Optional, Tuple, Union

from matplotlib.colors import ListedColormap
import numpy as np
import pyvista as pv

from gui.viewport_3d.dicom_colormaps import (
    get_available_colormaps,
    get_colormap_lut,
    to_vtk_color_transfer_function,
    to_vtk_piecewise_function,
)

_logger = logging.getLogger(__name__)


class VoxelVolumeRenderer:
    """
    Рендерер воксельных фантомов на базе vtkSmartVolumeMapper с поддержкой
    адаптивного уровня детализации (LOD) и медицинских палитр DICOM.
    """

    def __init__(
        self,
        viewport: Any,
        actor_name: str = "voxel_volume",
        colormap: str = "Hot Iron"
    ) -> None:
        self.viewport = viewport
        self.actor_name = actor_name
        self.colormap = colormap
        self.volume_actor: Any = None
        self.volume_mapper: Any = None
        self.volume_property: Any = None
        self.grid: Optional[pv.ImageData] = None
        self.base_voxel_size: float = 1.0
        self.scalar_range: Tuple[float, float] = (0.0, 1.0)
        self.is_interactive_mode: bool = False
        self.lod_factor: float = 1.0
        self.max_opacity: float = 0.4
        self.opacity_threshold: float = 0.05
        self.opacity_preset: str = 'air_cutoff'

        # Подписка на события камеры во вьюпорте для переключения LOD
        if self.viewport is not None:
            self.viewport.camera_interaction_started.connect(self.on_interaction_start)
            self.viewport.camera_interaction_ended.connect(self.on_interaction_end)

    def set_volume_data(
        self,
        data_3d: np.ndarray,
        voxel_size: Union[float, Tuple[float, float, float]] = 1.0,
        origin: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """
        Загружает трехмерный массив данных воксельного фантома и инициализирует объемный рендеринг.
        При совпадении размерностей сетка обновляется in-place без сброса камеры и пересоздания актора.
        """
        if isinstance(voxel_size, (int, float)):
            spacing = (float(voxel_size), float(voxel_size), float(voxel_size))
            self.base_voxel_size = float(voxel_size)
        else:
            spacing = tuple(float(v) for v in voxel_size)
            self.base_voxel_size = float(np.mean(spacing))

        if origin is None:
            origin = tuple(-0.5 * s * sp for s, sp in zip(data_3d.shape, spacing))

        try:
            self.scalar_range = (float(np.min(data_3d)), float(np.max(data_3d)))

            # Если ImageData уже создана и совпадает по размерности, выполняем быстрое in-place обновление
            if (self.grid is not None and self.volume_actor is not None and
                    self.grid.dimensions == data_3d.shape):
                self.grid.origin = origin
                self.grid.spacing = spacing
                self.grid.point_data['values'][:] = data_3d.flatten(order='F').astype(np.float32)
                scalars = self.grid.GetPointData().GetScalars()
                if scalars is not None:
                    scalars.Modified()
                self.grid.Modified()

                color_tf = to_vtk_color_transfer_function(self.colormap, scalar_range=self.scalar_range)
                if color_tf is not None and self.volume_property is not None:
                    self.volume_property.SetColor(color_tf)

                opacity_tf = to_vtk_piecewise_function(
                    scalar_range=self.scalar_range,
                    min_alpha=0.0,
                    max_alpha=self.max_opacity,
                    threshold=self.opacity_threshold,
                    preset=self.opacity_preset
                )
                if opacity_tf is not None and self.volume_property is not None:
                    self.volume_property.SetScalarOpacity(opacity_tf)

                if self.viewport is not None:
                    self.viewport.render()
                return

            # Создание регулярной воксельной сетки ImageData (UniformGrid)
            grid = pv.ImageData(
                dimensions=data_3d.shape,
                spacing=spacing,
                origin=origin
            )
            grid.point_data['values'] = data_3d.flatten(order='F').astype(np.float32)
            self.grid = grid

            # Создание структуры VTK для Smart Volume Mapping
            color_tf = to_vtk_color_transfer_function(
                self.colormap,
                scalar_range=self.scalar_range
            )
            opacity_tf = to_vtk_piecewise_function(
                scalar_range=self.scalar_range,
                min_alpha=0.0,
                max_alpha=self.max_opacity,
                threshold=self.opacity_threshold,
                preset=self.opacity_preset
            )

            # Получение палитры для PyVista
            cmap_arg: Any = self.colormap
            if isinstance(self.colormap, str) and self.colormap in get_available_colormaps():
                lut = get_colormap_lut(self.colormap)
                cmap_arg = ListedColormap(lut)

            # Добавляем в сцену через viewport с reset_camera=False
            if self.viewport is not None:
                self.volume_actor = self.viewport.add_volume_actor(
                    self.actor_name,
                    grid,
                    cmap=cmap_arg,
                    opacity='linear',
                    mapper='smart',
                    scalars='values',
                    reset_camera=False,
                )
                if self.volume_actor is not None:
                    self.volume_mapper = self.volume_actor.GetMapper()
                    self.volume_property = self.volume_actor.GetProperty()
                    if color_tf is not None:
                        self.volume_property.SetColor(color_tf)
                    if opacity_tf is not None:
                        self.volume_property.SetScalarOpacity(opacity_tf)
                    self._apply_lod_sampling()
                self.viewport.render()

        except Exception as e:
            _logger.info(f"VoxelVolumeRenderer: инициализация через fallback без VTK: {e}")

    def set_colormap(self, colormap_name: str) -> None:
        """
        Динамическое переключение цветовой шкалы без пересоздания воксельной сетки.
        """
        self.colormap = colormap_name
        if self.volume_property is not None:
            color_tf = to_vtk_color_transfer_function(colormap_name, scalar_range=self.scalar_range)
            if color_tf is not None:
                self.volume_property.SetColor(color_tf)
                if self.viewport is not None:
                    self.viewport.render()

    def set_lod_factor(self, factor: float) -> None:
        """
        Установка коэффициента уровня детализации LOD (1.0 = норма, выше = детальнее).
        """
        self.lod_factor = max(0.1, float(factor))
        self._apply_lod_sampling()

    def set_opacity_parameters(
        self,
        threshold: Optional[float] = None,
        max_opacity: Optional[float] = None,
        preset: Optional[str] = None
    ) -> None:
        """
        Комплексная настройка карты прозрачности воксельного объема.
        """
        if threshold is not None:
            self.opacity_threshold = float(threshold)
        if max_opacity is not None:
            self.max_opacity = float(max_opacity)
        if preset is not None:
            self.opacity_preset = str(preset)

        if self.volume_property is not None:
            opacity_tf = to_vtk_piecewise_function(
                scalar_range=self.scalar_range,
                min_alpha=0.0,
                max_alpha=self.max_opacity,
                threshold=self.opacity_threshold,
                preset=self.opacity_preset
            )
            if opacity_tf is not None:
                self.volume_property.SetScalarOpacity(opacity_tf)
                if self.viewport is not None:
                    self.viewport.render()

    def set_opacity_threshold(self, threshold: float) -> None:
        """
        Установка порога отсечения прозрачности фона (Air Cutoff).
        """
        self.set_opacity_parameters(threshold=threshold)

    def set_max_opacity(self, opacity: float) -> None:
        """
        Установка максимальной непрозрачности объема (0.0 = прозрачно, 1.0 = плотно).
        """
        self.set_opacity_parameters(max_opacity=opacity)

    def set_opacity_preset(self, preset: str) -> None:
        """
        Переключение пресета передаточной функции прозрачности.
        """
        self.set_opacity_parameters(preset=preset)

    def on_interaction_start(self) -> None:
        """
        Переключение на пониженное разрешение трассировки лучей при интерактивном перемещении.
        """
        self.is_interactive_mode = True
        self._apply_lod_sampling()

    def on_interaction_end(self) -> None:
        """
        Возврат к полному воксельному разрешению при остановке вращения.
        """
        self.is_interactive_mode = False
        self._apply_lod_sampling()

    def _apply_lod_sampling(self) -> None:
        """
        Управление шагом трассировки лучей в vtkSmartVolumeMapper с учетом коэффициента LOD.
        """
        if self.volume_mapper is not None:
            try:
                # Чем выше lod_factor, тем меньше шаг трассировки (выше детальность)
                mult = 1.0 / max(0.1, self.lod_factor)
                if self.is_interactive_mode:
                    # Увеличенный шаг выборки для высокого FPS в динамике (>30 FPS)
                    sample_dist = self.base_voxel_size * 2.5 * mult
                else:
                    # Физический шаг для максимальной четкости в статике
                    sample_dist = self.base_voxel_size * 0.5 * mult

                self.volume_mapper.SetSampleDistance(sample_dist)
                if self.viewport is not None:
                    self.viewport.render()
            except Exception as e:
                _logger.debug(f"Ошибка применения шага трассировки LOD: {e}")

    def clear(self) -> None:
        """
        Очистка и удаление воксельного объема из вьюпорта.
        """
        if self.viewport is not None:
            self.viewport.remove_actor(self.actor_name)
            self.viewport.render()
        self.volume_actor = None
        self.volume_mapper = None
        self.volume_property = None
        self.grid = None


# Псевдонимы для совместимости
VoxelRenderer = VoxelVolumeRenderer
