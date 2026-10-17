from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pyvista as pv

from gui.viewport_3d.dicom_colormaps import (
    to_vtk_color_transfer_function,
    to_vtk_piecewise_function,
)


@dataclass
class DoseActorEntry:
    """Контейнер VTK-состояния отдельной 3D-сетки дозы в вьюпорте."""

    grid_name: str
    actor_name: str
    grid: pv.ImageData
    volume_actor: Any
    volume_mapper: Any
    volume_property: Any
    base_voxel_size: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    transform_matrix: Optional[np.ndarray] = None
    scalar_range: Tuple[float, float] = (0.0, 1.0)
    visible: bool = True


class DoseVolumeRenderer:
    """
    Специализированный 3D-рендерер воксельной дозы (накопленной карты энерговыделения)
    на базе PyVista и vtkSmartVolumeMapper с поддержкой одновременного отображения
    нескольких независимых сеток дозы (Multi-Actor).

    Особенности и оптимизации:
    1. Инкрементальное обновление данных in-place: скалярный массив мутируется без
       пересоздания структуры сетки ImageData, актора vtkVolume или vtkSmartVolumeMapper
       для каждой именованной сетки дозы.
    2. Полное отсутствие сброса камеры PyVista при обновлениях (reset_camera=False).
    3. Отсутствие утечек оперативной памяти: однократная аллокация структур VTK/GPU на каждую сетку.
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

        self._entries: Dict[str, DoseActorEntry] = {}
        self._active_grid_name: str = "default"
        self._default_voxel_size: Tuple[float, float, float] = (5.0, 5.0, 5.0)
        self._default_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._default_transform_matrix: Optional[np.ndarray] = None
        self._default_scalar_range: Tuple[float, float] = (0.0, 1.0)
        self._visible: bool = True
        self._last_render_time: float = 0.0

    @property
    def entries(self) -> Dict[str, DoseActorEntry]:
        """Словарь зарегистрированных акторов сеток дозы по их имени."""
        return self._entries

    def _get_active_entry(self) -> Optional[DoseActorEntry]:
        """Возвращает запись активной (или первой доступной) сетки дозы."""
        if self._active_grid_name in self._entries:
            return self._entries[self._active_grid_name]
        if self._entries:
            return next(iter(self._entries.values()))
        return None

    def _resolve_actor_name(self, grid_name: str) -> str:
        """Формирует уникальное имя VTK-актора для заданного имени сетки дозы."""
        if grid_name == "default" or grid_name == self.actor_name:
            return self.actor_name
        return f"{self.actor_name}_{grid_name}"

    @property
    def grid(self) -> Optional[pv.ImageData]:
        entry = self._get_active_entry()
        return entry.grid if entry is not None else None

    @grid.setter
    def grid(self, value: Optional[pv.ImageData]) -> None:
        entry = self._get_active_entry()
        if entry is not None and value is not None:
            entry.grid = value
        elif value is None:
            self._entries.clear()

    @property
    def volume_actor(self) -> Any:
        entry = self._get_active_entry()
        return entry.volume_actor if entry is not None else None

    @volume_actor.setter
    def volume_actor(self, value: Any) -> None:
        entry = self._get_active_entry()
        if entry is not None:
            entry.volume_actor = value

    @property
    def volume_mapper(self) -> Any:
        entry = self._get_active_entry()
        return entry.volume_mapper if entry is not None else None

    @volume_mapper.setter
    def volume_mapper(self, value: Any) -> None:
        entry = self._get_active_entry()
        if entry is not None:
            entry.volume_mapper = value

    @property
    def volume_property(self) -> Any:
        entry = self._get_active_entry()
        return entry.volume_property if entry is not None else None

    @volume_property.setter
    def volume_property(self, value: Any) -> None:
        entry = self._get_active_entry()
        if entry is not None:
            entry.volume_property = value

    @property
    def base_voxel_size(self) -> Tuple[float, float, float]:
        entry = self._get_active_entry()
        return entry.base_voxel_size if entry is not None else self._default_voxel_size

    @base_voxel_size.setter
    def base_voxel_size(self, value: Tuple[float, float, float]) -> None:
        spacing = (float(value[0]), float(value[1]), float(value[2]))
        self._default_voxel_size = spacing
        entry = self._get_active_entry()
        if entry is not None:
            entry.base_voxel_size = spacing

    @property
    def origin(self) -> Tuple[float, float, float]:
        entry = self._get_active_entry()
        return entry.origin if entry is not None else self._default_origin

    @origin.setter
    def origin(self, value: Tuple[float, float, float]) -> None:
        origin_tuple = (float(value[0]), float(value[1]), float(value[2]))
        self._default_origin = origin_tuple
        entry = self._get_active_entry()
        if entry is not None:
            entry.origin = origin_tuple

    @property
    def transform_matrix(self) -> Optional[np.ndarray]:
        entry = self._get_active_entry()
        return entry.transform_matrix if entry is not None else self._default_transform_matrix

    @transform_matrix.setter
    def transform_matrix(self, value: Optional[np.ndarray]) -> None:
        matrix_arr = np.asarray(value, dtype=np.float64) if value is not None else None
        self._default_transform_matrix = matrix_arr
        entry = self._get_active_entry()
        if entry is not None:
            entry.transform_matrix = matrix_arr

    @property
    def scalar_range(self) -> Tuple[float, float]:
        entry = self._get_active_entry()
        return entry.scalar_range if entry is not None else self._default_scalar_range

    @scalar_range.setter
    def scalar_range(self, value: Tuple[float, float]) -> None:
        range_tuple = (float(value[0]), float(value[1]))
        self._default_scalar_range = range_tuple
        entry = self._get_active_entry()
        if entry is not None:
            entry.scalar_range = range_tuple

    def get_entry(self, grid_name: str = "default") -> Optional[DoseActorEntry]:
        """Возвращает запись актора сетки дозы по имени или None."""
        return self._entries.get(grid_name)

    def setup_grid(
        self,
        grid_shape: Tuple[int, int, int],
        voxel_size: Union[float, Tuple[float, float, float]] = 5.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        transform_matrix: Optional[np.ndarray] = None,
        grid_name: str = "default",
        render: bool = True,
    ) -> DoseActorEntry:
        """
        Однократная инициализация ImageData и VTK Volume Actor для заданной сетки дозы.
        """
        if isinstance(voxel_size, (int, float)):
            spacing = (float(voxel_size), float(voxel_size), float(voxel_size))
        else:
            spacing = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))

        origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))
        resolved_matrix = (
            np.asarray(transform_matrix, dtype=np.float64)
            if transform_matrix is not None
            else self._default_transform_matrix
        )

        self._default_voxel_size = spacing
        self._default_origin = origin_tuple
        if resolved_matrix is not None:
            self._default_transform_matrix = resolved_matrix

        # Создаем регулярную 3D-сетку ImageData
        grid_data = pv.ImageData(
            dimensions=grid_shape,
            spacing=spacing,
            origin=origin_tuple,
        )
        zeros = np.zeros(int(np.prod(grid_shape)), dtype=np.float32)
        grid_data.point_data["dose"] = zeros

        existing_entry = self._entries.get(grid_name)
        entry_visible = existing_entry.visible if existing_entry is not None else True
        resolved_actor_name = self._resolve_actor_name(grid_name)

        volume_actor: Any = None
        volume_mapper: Any = None
        volume_property: Any = None

        if self.viewport is not None:
            volume_actor = self.viewport.add_volume_actor(
                resolved_actor_name,
                grid_data,
                cmap=self.colormap,
                opacity="linear",
                mapper="smart",
                scalars="dose",
                reset_camera=False,
            )
            if volume_actor is not None:
                volume_mapper = volume_actor.GetMapper()
                volume_property = volume_actor.GetProperty()

        entry = DoseActorEntry(
            grid_name=grid_name,
            actor_name=resolved_actor_name,
            grid=grid_data,
            volume_actor=volume_actor,
            volume_mapper=volume_mapper,
            volume_property=volume_property,
            base_voxel_size=spacing,
            origin=origin_tuple,
            transform_matrix=resolved_matrix,
            scalar_range=(0.0, 1.0),
            visible=entry_visible,
        )
        self._entries[grid_name] = entry
        self._active_grid_name = grid_name

        if volume_actor is not None:
            self._update_entry_transfer_functions(entry, max_dose=1.0)
            effective_visible = self._visible and entry.visible
            volume_actor.SetVisibility(1 if effective_visible else 0)
            if resolved_matrix is not None and self.viewport is not None:
                self.viewport.update_actor_transform(resolved_actor_name, resolved_matrix)

        if render and self.viewport is not None:
            self.viewport.render()

        return entry

    def update_dose_data(
        self,
        dose_3d: np.ndarray,
        voxel_size: Optional[Union[float, Tuple[float, float, float]]] = None,
        origin: Optional[Tuple[float, float, float]] = None,
        transform_matrix: Optional[np.ndarray] = None,
        grid_name: str = "default",
        render: bool = True,
    ) -> None:
        """
        Высокопроизводительное обновление дозы in-place для указанной сетки без пересоздания актора.
        """
        if not self._visible:
            return

        existing_entry = self._entries.get(grid_name)
        if existing_entry is not None and not existing_entry.visible:
            return

        grid_shape = dose_3d.shape
        fallback_voxel_size = (
            existing_entry.base_voxel_size if existing_entry is not None else self._default_voxel_size
        )
        fallback_origin = (
            existing_entry.origin if existing_entry is not None else self._default_origin
        )
        fallback_matrix = (
            existing_entry.transform_matrix if existing_entry is not None else self._default_transform_matrix
        )

        raw_size = voxel_size if voxel_size is not None else fallback_voxel_size
        raw_origin = origin if origin is not None else fallback_origin

        if isinstance(raw_size, (int, float)):
            voxel_spacing = (float(raw_size), float(raw_size), float(raw_size))
        else:
            voxel_spacing = (float(raw_size[0]), float(raw_size[1]), float(raw_size[2]))

        grid_origin = (float(raw_origin[0]), float(raw_origin[1]), float(raw_origin[2]))

        resolved_matrix = (
            np.asarray(transform_matrix, dtype=np.float64)
            if transform_matrix is not None
            else fallback_matrix
        )

        if (
            existing_entry is None
            or existing_entry.grid is None
            or existing_entry.grid.dimensions != grid_shape
            or existing_entry.base_voxel_size != voxel_spacing
            or existing_entry.origin != grid_origin
        ):
            entry = self.setup_grid(
                grid_shape,
                voxel_size=voxel_spacing,
                origin=grid_origin,
                transform_matrix=resolved_matrix,
                grid_name=grid_name,
                render=False,
            )
        else:
            entry = existing_entry
            self._active_grid_name = grid_name
            if resolved_matrix is not None:
                entry.transform_matrix = resolved_matrix
                if self.viewport is not None:
                    self.viewport.update_actor_transform(entry.actor_name, resolved_matrix)

        if entry.grid is None or "dose" not in entry.grid.point_data:
            return

        # 1. Мутация скаляров in-place без выделения новых VTK-объектов
        flat_dose = dose_3d.flatten(order="F").astype(np.float32)
        target_scalars = np.asarray(entry.grid.point_data["dose"])
        if target_scalars.shape != flat_dose.shape:
            return
        target_scalars[:] = flat_dose

        # 2. Оповещение конвейера VTK об изменении скалярных данных
        scalars = entry.grid.GetPointData().GetScalars()
        if scalars is not None:
            scalars.Modified()
        entry.grid.Modified()

        # 3. Подстройка динамического диапазона и функции прозрачности
        max_dose = float(np.max(dose_3d))
        if max_dose > 1e-9:
            prev_max = entry.scalar_range[1]
            if prev_max <= 1e-9 or abs(max_dose - prev_max) / max(1e-9, prev_max) > 0.03:
                entry.scalar_range = (0.0, max_dose)
                self._default_scalar_range = entry.scalar_range
                self._update_entry_transfer_functions(entry, max_dose=max_dose)

        # 4. Троттлинг отрисовки во избежание фризов графического интерфейса
        if render:
            self.request_render()

    def request_render(self, force: bool = False) -> None:
        """Выполняет отрисовку вьюпорта с учетом ограничения частоты кадров (throttling)."""
        now = time.time()
        if force or (now - self._last_render_time) >= self.min_render_interval:
            if self.viewport is not None:
                self.viewport.render()
            self._last_render_time = now

    def _update_entry_transfer_functions(self, entry: DoseActorEntry, max_dose: float) -> None:
        """
        Обновление функций цвета и прозрачности для конкретной записи сетки дозы
        с абсолютным отсечением фоновых вокселей.
        """
        if entry.volume_property is None:
            return

        scalar_bounds = (0.0, max(1e-6, max_dose))

        color_tf = to_vtk_color_transfer_function(self.colormap, scalar_range=scalar_bounds)
        if color_tf is not None:
            entry.volume_property.SetColor(color_tf)

        opacity_tf = to_vtk_piecewise_function(
            scalar_range=scalar_bounds,
            min_alpha=0.0,
            max_alpha=0.85,
            ramp_type="step" if self.threshold_ratio > 0.001 else "linear",
            threshold=self.threshold_ratio,
        )
        if opacity_tf is not None:
            entry.volume_property.SetScalarOpacity(opacity_tf)

    def _update_transfer_functions(self, max_dose: float) -> None:
        """
        Обновление функций цвета и прозрачности для активной сетки дозы.
        """
        active_entry = self._get_active_entry()
        if active_entry is not None:
            self._update_entry_transfer_functions(active_entry, max_dose=max_dose)

    def update_grid_transform(
        self,
        grid_name: str,
        transform_matrix: np.ndarray,
        render: bool = True,
    ) -> None:
        """Обновляет матрицу трансформации VTK-актора для указанной сетки дозы."""
        entry = self._entries.get(grid_name)
        if entry is None:
            return
        matrix_arr = np.asarray(transform_matrix, dtype=np.float64)
        entry.transform_matrix = matrix_arr
        if self.viewport is not None:
            self.viewport.update_actor_transform(entry.actor_name, matrix_arr)
            if render:
                self.viewport.render()

    def set_grid_visible(
        self,
        grid_name: str,
        visible: bool,
        render: bool = True,
    ) -> None:
        """Включает или выключает отображение конкретной сетки дозы по имени."""
        entry = self._entries.get(grid_name)
        if entry is None:
            return
        entry.visible = bool(visible)
        if entry.volume_actor is not None:
            effective_visible = self._visible and entry.visible
            entry.volume_actor.SetVisibility(1 if effective_visible else 0)
            if render and self.viewport is not None:
                self.viewport.render()

    def remove_grid(self, grid_name: str, render: bool = True) -> None:
        """Удаляет VTK-актор и запись указанной сетки дозы из вьюпорта."""
        entry = self._entries.pop(grid_name, None)
        if entry is None:
            return
        if self.viewport is not None:
            self.viewport.remove_actor(entry.actor_name)
            if render:
                self.viewport.render()
        if self._active_grid_name == grid_name:
            self._active_grid_name = next(iter(self._entries.keys()), "default")

    def clear(self) -> None:
        """
        Очистка накопленной карты дозы во всех зарегистрированных сетках.
        """
        if not self._entries:
            return
        for entry in self._entries.values():
            if entry.grid is not None and "dose" in entry.grid.point_data:
                np.asarray(entry.grid.point_data["dose"])[:] = 0.0
                scalars = entry.grid.GetPointData().GetScalars()
                if scalars is not None:
                    scalars.Modified()
                entry.grid.Modified()
                entry.scalar_range = (0.0, 1.0)
                self._update_entry_transfer_functions(entry, max_dose=1.0)
        self._default_scalar_range = (0.0, 1.0)
        if self.viewport is not None:
            self.viewport.render()

    def set_colormap(self, colormap_name: str) -> None:
        """
        Динамическое изменение цветовой палитры дозы для всех сеток.
        """
        self.colormap = colormap_name
        for entry in self._entries.values():
            self._update_entry_transfer_functions(entry, max_dose=entry.scalar_range[1])
        if self.viewport is not None:
            self.viewport.render()

    def set_threshold(self, min_dose_ratio: float) -> None:
        """
        Установка порога отсечения фонового излучения в долях от максимума для всех сеток.
        """
        self.threshold_ratio = float(np.clip(min_dose_ratio, 0.0, 0.99))
        for entry in self._entries.values():
            self._update_entry_transfer_functions(entry, max_dose=entry.scalar_range[1])
        if self.viewport is not None:
            self.viewport.render()

    def set_visible(self, visible: bool) -> None:
        """
        Глобальное включение / выключение отображения всех объемов дозы.
        """
        self._visible = bool(visible)
        has_actors = False
        for entry in self._entries.values():
            if entry.volume_actor is not None:
                has_actors = True
                effective_visible = self._visible and entry.visible
                entry.volume_actor.SetVisibility(1 if effective_visible else 0)
        if has_actors and self.viewport is not None:
            self.viewport.render()


__all__ = [
    "DoseActorEntry",
    "DoseVolumeRenderer",
]


