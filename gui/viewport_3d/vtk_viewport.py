import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import vtk
from matplotlib.colors import ListedColormap
from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
import pyvista as pv
from pyvistaqt import QtInteractor

from gui.viewport_3d.dicom_colormaps import get_available_colormaps, get_colormap_lut

_logger = logging.getLogger(__name__)


@runtime_checkable
class ISceneViewport(Protocol):
    """
    Контракт взаимодействия с 3D-вьюпортом сцены.
    Исключает неявную утиную типизацию (hasattr/getattr).
    """

    def add_mesh_actor(
        self,
        name: str,
        mesh: Any,
        color: Optional[str] = 'white',
        opacity: float = 1.0,
        style: str = 'surface',
        wireframe: bool = False,
        rgb: bool = False,
        **kwargs: Any
    ) -> Optional[Any]: ...

    def update_actor_transform(self, name: str, matrix: np.ndarray) -> bool: ...

    def remove_actor(self, name: str) -> None: ...

    def render(self) -> None: ...

    def get_actor(self, name: str) -> Optional[Any]: ...

    def set_actor_edge_highlight(
        self,
        name: str,
        visible: bool,
        color: Tuple[float, float, float] = (1.0, 0.55, 0.0),
        line_width: float = 2.5,
    ) -> bool: ...


class VTKViewport(QWidget):
    """
    Интерактивный 3D-вьюпорт на базе PyVista / QtInteractor.
    Обеспечивает визуализацию геометрии сцены, воксельных объемов,
    траекторий фотонов и интерактивных манипуляторов.
    """

    camera_interaction_started = Signal()
    camera_interaction_ended = Signal()
    actor_picked = Signal(str)

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.plotter: Optional[QtInteractor] = None
        self._actors: Dict[str, Any] = {}
        self._is_interacting: bool = False

        self._init_plotter()

    def _init_plotter(self) -> None:
        """
        Инициализация QtInteractor или информативной заглушки при недоступности графического сервера.
        """
        try:
            self.plotter = QtInteractor(self)
            self.plotter.set_background('#1e1e1e')
            self.plotter.add_axes(interactive=True)
            self.plotter.enable_anti_aliasing('fxaa')
            self._layout.addWidget(self.plotter.interactor)

            # Настройка событий взаимодействия с камерой для адаптивного LOD
            if self.plotter.iren is not None:
                self.plotter.iren.add_observer('StartInteractionEvent', self._on_interaction_start)
                self.plotter.iren.add_observer('EndInteractionEvent', self._on_interaction_end)

        except (RuntimeError, ImportError, TypeError, AttributeError) as init_error:
            _logger.info(f"PyVista QtInteractor не инициализирован (headless/mock режим): {init_error}")
            self.plotter = None
            fallback_label = QLabel(
                "3D Viewport недоступен (QtInteractor не инициализирован).\n"
                "Убедитесь, что установлены пакеты: pyvista, pyvistaqt, vtk, PySide6."
            )
            fallback_label.setWordWrap(True)
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_label.setStyleSheet(
                "color: #aaaaaa; font-size: 13px; padding: 24px; "
                "background-color: #1e1e1e; border: 1px dashed #444444;"
            )
            self._layout.addWidget(fallback_label)
            self.fallback_label = fallback_label

    def _on_interaction_start(self, obj: Any, event: str) -> None:
        if not self._is_interacting:
            self._is_interacting = True
            self.camera_interaction_started.emit()

    def _on_interaction_end(self, obj: Any, event: str) -> None:
        if self._is_interacting:
            self._is_interacting = False
            self.camera_interaction_ended.emit()
            self.render()

    @property
    def interactor(self) -> Optional[Any]:
        """
        Низкоуровневый vtkRenderWindowInteractor для подключения интерактивных манипуляторов и наблюдателей.
        """
        if self.plotter is None:
            return None
        if self.plotter.render_window is not None:
            return self.plotter.render_window.GetInteractor()
        return None

    def add_mesh_actor(
        self,
        name: str,
        mesh: Any,
        color: Optional[str] = 'white',
        opacity: float = 1.0,
        style: str = 'surface',
        wireframe: bool = False,
        rgb: bool = False,
        **kwargs: Any
    ) -> Optional[Any]:
        """
        Добавляет полигональный меш в сцену 3D-вьюпорта.
        """
        if self.plotter is None:
            return None

        self.remove_actor(name)
        try:
            is_rgb = kwargs.pop('rgb', rgb)
            mesh_kwargs: Dict[str, Any] = {
                'opacity': opacity,
                'style': 'wireframe' if wireframe else style,
                'name': name,
                'rgb': is_rgb,
                'reset_camera': kwargs.pop('reset_camera', False),
            }
            if not is_rgb and color is not None:
                mesh_kwargs['color'] = color
            mesh_kwargs.update(kwargs)

            actor = self.plotter.add_mesh(mesh, **mesh_kwargs)
            self._actors[name] = actor
            return actor
        except (RuntimeError, ValueError, TypeError) as mesh_error:
            _logger.error(f"Ошибка добавления меша {name}: {mesh_error}")
            return None

    def update_actor_transform(self, name: str, matrix: np.ndarray) -> bool:
        """
        Инкрементальное обновление матрицы трансформации существующего актора
        без затратного пересоздания полигонального меша.
        """
        if self.plotter is None or name not in self._actors:
            return False
        actor = self._actors[name]
        if actor is None:
            return False
        try:
            if isinstance(actor, pv.Actor):
                actor.user_matrix = matrix
            else:
                mat_vtk = vtk.vtkMatrix4x4()
                for i in range(4):
                    for j in range(4):
                        mat_vtk.SetElement(i, j, float(matrix[i, j]))
                actor.SetUserMatrix(mat_vtk)
            self.render()
            return True
        except (RuntimeError, ValueError, TypeError) as transform_error:
            _logger.error(f"Ошибка обновления трансформации актора {name}: {transform_error}")
            return False

    def add_volume_actor(
        self,
        name: str,
        grid: Any,
        cmap: str = 'Hot Iron',
        opacity: Any = 'linear',
        mapper: str = 'smart',
        **kwargs: Any
    ) -> Optional[Any]:
        """
        Добавляет воксельный объем (Volume Rendering) в сцену.
        """
        if self.plotter is None:
            return None

        self.remove_actor(name)
        try:
            if isinstance(cmap, str) and cmap in get_available_colormaps():
                cmap = ListedColormap(get_colormap_lut(cmap))

            vol_kwargs: Dict[str, Any] = {
                'cmap': cmap,
                'opacity': opacity,
                'mapper': mapper,
                'name': name,
                'reset_camera': kwargs.pop('reset_camera', False),
            }
            vol_kwargs.update(kwargs)
            actor = self.plotter.add_volume(grid, **vol_kwargs)
            self._actors[name] = actor
            return actor
        except (RuntimeError, ValueError, TypeError) as volume_error:
            _logger.error(f"Ошибка добавления объема {name}: {volume_error}")
            return None

    def remove_actor(self, name: str) -> None:
        """
        Удаляет актор из сцены по его уникальному имени.
        """
        if self.plotter is not None and name in self._actors:
            try:
                self.plotter.remove_actor(name)
            except (KeyError, RuntimeError, AttributeError):
                pass
        self._actors.pop(name, None)

    def get_actor(self, name: str) -> Optional[Any]:
        """
        Возвращает существующий VTK/PyVista актор по его имени.
        """
        return self._actors.get(name)

    def set_actor_edge_highlight(
        self,
        name: str,
        visible: bool,
        color: Tuple[float, float, float] = (1.0, 0.55, 0.0),
        line_width: float = 2.5,
    ) -> bool:
        """
        Управляет подсветкой контура (рёбер) полигонального актора через SetEdgeVisibility.
        Используется для выделения выбранных в графе узлов контрастным amber/оранжевым цветом.
        """
        target_actor = self._actors.get(name)
        if target_actor is None:
            return False
        try:
            property_obj = target_actor.GetProperty()
            if property_obj is None:
                return False
            property_obj.SetEdgeVisibility(bool(visible))
            if visible:
                property_obj.SetEdgeColor(float(color[0]), float(color[1]), float(color[2]))
                property_obj.SetLineWidth(float(line_width))
            self.render()
            return True
        except (AttributeError, RuntimeError, TypeError) as highlight_error:
            _logger.error(f"Ошибка настройки подсветки рёбер актора {name}: {highlight_error}")
            return False

    def clear_actors(self) -> None:
        """
        Очищает все добавленные пользовательские объекты из сцены.
        """
        for name in list(self._actors.keys()):
            self.remove_actor(name)

    def reset_camera(self) -> None:
        """
        Сбрасывает положение камеры для охвата всех объектов.
        """
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.render()

    def view_isometric(self) -> None:
        if self.plotter is not None:
            self.plotter.view_isometric()
            self.render()

    def view_xy(self) -> None:
        if self.plotter is not None:
            self.plotter.view_xy()
            self.render()

    def view_xz(self) -> None:
        if self.plotter is not None:
            self.plotter.view_xz()
            self.render()

    def view_yz(self) -> None:
        if self.plotter is not None:
            self.plotter.view_yz()
            self.render()

    def render(self) -> None:
        if self.plotter is not None:
            if self._is_interacting:
                return
            try:
                self.plotter.render()
            except (RuntimeError, AttributeError):
                pass

    def close(self) -> bool:
        """
        Корректное завершение работы и освобождение ресурсов QtInteractor.
        """
        if self.plotter is not None:
            try:
                self.plotter.close()
            except (RuntimeError, AttributeError):
                pass
            self.plotter = None
        return super().close()
