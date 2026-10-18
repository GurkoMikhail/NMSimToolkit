from contextlib import contextmanager
import logging
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

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
        color: Optional[Union[str, Sequence[float]]] = 'white',
        opacity: float = 1.0,
        style: str = 'surface',
        wireframe: bool = False,
        rgb: bool = False,
        **kwargs: Any
    ) -> Optional[Any]: ...

    def add_actor(self, name: str, actor: Any) -> Optional[Any]: ...

    def add_volume_actor(
        self,
        name: str,
        grid: Any,
        cmap: Any = 'Hot Iron',
        opacity: Any = 'linear',
        mapper: str = 'smart',
        **kwargs: Any
    ) -> Optional[Any]: ...

    def update_actor_transform(self, name: str, matrix: np.ndarray) -> bool: ...

    def set_node_local_transform(
        self,
        node_id: int,
        local_matrix: np.ndarray,
        parent_id: Optional[int] = None,
    ) -> Any: ...

    def attach_actor_to_node(self, actor_name: str, node_id: int) -> bool: ...

    def remove_node_transform(self, node_id: int) -> None: ...

    def remove_actor(self, name: str) -> None: ...

    def render(self) -> None: ...

    def request_render(self) -> None: ...

    def begin_update(self) -> None: ...

    def end_update(self) -> None: ...

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

        self.plotter: Optional[Any] = None
        self._actors: Dict[str, Any] = {}
        self._node_transforms: Dict[int, vtk.vtkTransform] = {}
        self._is_interacting: bool = False
        self._update_depth: int = 0
        self._render_pending: bool = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self._flush_pending_render)

        self._init_plotter()

    def _init_plotter(self) -> None:
        """
        Инициализация QtInteractor или информативной заглушки при недоступности графического сервера.
        """
        try:
            plotter_instance: Any = QtInteractor(self)
            self.plotter = plotter_instance
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
            fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

    def begin_update(self) -> None:
        """
        Включает пакетный режим обновления сцены, откладывая выполнение отрисовки
        до парного вызова end_update().
        """
        self._update_depth += 1

    def end_update(self) -> None:
        """
        Завершает пакетный режим обновления сцены и однократно выполняет накопленный рендер.
        """
        if self._update_depth > 0:
            self._update_depth -= 1
        if self._update_depth == 0 and self._render_pending:
            self._flush_pending_render()

    @contextmanager
    def batch_update(self) -> Iterator[None]:
        """
        Контекстный менеджер пакетного обновления акторов и трансформаций с единичным рендером.
        """
        self.begin_update()
        try:
            yield
        finally:
            self.end_update()

    def __enter__(self) -> 'VTKViewport':
        self.begin_update()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_update()

    def request_render(self) -> None:
        """
        Запрашивает коалесцирующую перерисовку сцены через одноразовый QTimer(0).
        Все запросы внутри одного тика событий гарантированно схлопываются в один вызов render().
        """
        if self._is_interacting:
            return
        self._render_pending = True
        if self._update_depth > 0:
            return
        if not self._render_timer.isActive():
            self._render_timer.start(0)

    def _flush_pending_render(self) -> None:
        """
        Выполняет отложенный рендер по срабатыванию коалесцирующего таймера или завершению пакета.
        """
        if self._render_timer.isActive():
            self._render_timer.stop()
        if not self._render_pending:
            return
        self._render_pending = False
        self.render()

    def set_node_local_transform(
        self,
        node_id: int,
        local_matrix: np.ndarray,
        parent_id: Optional[int] = None,
    ) -> vtk.vtkTransform:
        """
        Создает или обновляет C++ объект vtkTransform для узла графа сцены и связывает
        его с родительским vtkTransform через SetInput для автоматического каскадного пересчета в VTK.
        """
        node_transform = self._node_transforms.get(node_id)
        if node_transform is None:
            node_transform = vtk.vtkTransform()
            self._node_transforms[node_id] = node_transform

        mat_vtk = vtk.vtkMatrix4x4()
        mat_vtk.DeepCopy(np.asarray(local_matrix, dtype=float).ravel())
        node_transform.SetMatrix(mat_vtk)

        if parent_id is not None:
            parent_transform = self._node_transforms.get(parent_id)
            if parent_transform is None:
                parent_transform = vtk.vtkTransform()
                self._node_transforms[parent_id] = parent_transform
            if node_transform.GetInput() is not parent_transform:
                node_transform.SetInput(parent_transform)
        else:
            if node_transform.GetInput() is not None:
                node_transform.SetInput(None)

        self.request_render()
        return node_transform

    def attach_actor_to_node(self, actor_name: str, node_id: int) -> bool:
        """
        Привязывает иерархический VTK-актор узла сцены к его C++ объекту vtkTransform через SetUserTransform.
        """
        actor = self._actors.get(actor_name)
        node_transform = self._node_transforms.get(node_id)
        if actor is None or node_transform is None:
            return False
        try:
            if actor.GetUserTransform() is not node_transform:
                actor.SetUserTransform(node_transform)
            self.request_render()
            return True
        except (RuntimeError, ValueError, TypeError, AttributeError) as attach_error:
            _logger.error(f"Ошибка привязки актора {actor_name} к узлу {node_id}: {attach_error}")
            return False

    def get_node_transform(self, node_id: int) -> Optional[vtk.vtkTransform]:
        """
        Возвращает зарегистрированный C++ объект vtkTransform для указанного идентификатора узла.
        """
        return self._node_transforms.get(node_id)

    def remove_node_transform(self, node_id: int) -> None:
        """
        Удаляет C++ объект vtkTransform узла из реестра при удалении узла со сцены
        и отвязывает зависящие от него дочерние трансформации.
        """
        removed_transform = self._node_transforms.pop(node_id, None)
        if removed_transform is not None:
            for child_transform in self._node_transforms.values():
                if child_transform.GetInput() is removed_transform:
                    child_transform.SetInput(None)

    def add_mesh_actor(
        self,
        name: str,
        mesh: Any,
        color: Optional[Union[str, Sequence[float]]] = 'white',
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
                'render': kwargs.pop('render', False),
            }
            if not is_rgb and color is not None:
                mesh_kwargs['color'] = color
            mesh_kwargs.update(kwargs)

            actor = self.plotter.add_mesh(mesh, **mesh_kwargs)
            self._actors[name] = actor
            self.request_render()
            return actor
        except (RuntimeError, ValueError, TypeError) as mesh_error:
            _logger.error(f"Ошибка добавления меша {name}: {mesh_error}")
            return None

    def add_actor(self, name: str, actor: Any) -> Optional[Any]:
        """
        Добавляет готовый VTK-актор (например, с vtkGlyph3DMapper) в 3D-сцену вьюпорта.
        """
        if self.plotter is None:
            return None

        self.remove_actor(name)
        try:
            is_pickable = bool(actor.GetPickable()) if isinstance(actor, vtk.vtkProp) else True
            try:
                self.plotter.add_actor(
                    actor,
                    reset_camera=False,
                    name=name,
                    pickable=is_pickable,
                    render=False,
                )
            except TypeError:
                self.plotter.add_actor(actor, reset_camera=False, name=name)
                if isinstance(actor, vtk.vtkProp):
                    actor.SetPickable(1 if is_pickable else 0)
            self._actors[name] = actor
            self.request_render()
            return actor
        except (RuntimeError, ValueError, TypeError) as actor_error:
            _logger.error(f"Ошибка добавления актора {name}: {actor_error}")
            return None

    def update_actor_transform(self, name: str, matrix: np.ndarray) -> bool:
        """
        Инкрементальное обновление матрицы трансформации сервисного (неиерархического) актора
        через SetUserMatrix без затратного пересоздания полигонального меша.
        """
        if self.plotter is None or name not in self._actors:
            return False
        actor = self._actors[name]
        if actor is None:
            return False
        try:
            mat_vtk = vtk.vtkMatrix4x4()
            mat_vtk.DeepCopy(np.asarray(matrix, dtype=float).ravel())
            actor.SetUserMatrix(mat_vtk)
            self.request_render()
            return True
        except (RuntimeError, ValueError, TypeError, AttributeError) as transform_error:
            _logger.error(f"Ошибка обновления трансформации актора {name}: {transform_error}")
            return False

    def add_volume_actor(
        self,
        name: str,
        grid: Any,
        cmap: Any = 'Hot Iron',
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
            cmap_resolved: Any = cmap
            if isinstance(cmap, str) and cmap in get_available_colormaps():
                cmap_resolved = ListedColormap(get_colormap_lut(cmap))

            vol_kwargs: Dict[str, Any] = {
                'cmap': cmap_resolved,
                'opacity': opacity,
                'mapper': mapper,
                'name': name,
                'reset_camera': kwargs.pop('reset_camera', False),
                'render': kwargs.pop('render', False),
            }
            vol_kwargs.update(kwargs)
            actor = self.plotter.add_volume(grid, **vol_kwargs)
            self._actors[name] = actor
            self.request_render()
            return actor
        except (RuntimeError, ValueError, TypeError) as volume_error:
            _logger.error(f"Ошибка добавления объема {name}: {volume_error}")
            return None

    def remove_actor(self, name: str) -> None:
        """
        Удаляет актор из сцены по его уникальному имени.
        """
        was_present = name in self._actors
        if self.plotter is not None and was_present:
            try:
                self.plotter.remove_actor(name, reset_camera=False, render=False)
            except TypeError:
                try:
                    self.plotter.remove_actor(name)
                except (KeyError, RuntimeError, AttributeError):
                    pass
            except (KeyError, RuntimeError, AttributeError):
                pass
        self._actors.pop(name, None)
        if was_present:
            self.request_render()

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
            self.request_render()
            return True
        except (AttributeError, RuntimeError, TypeError) as highlight_error:
            _logger.error(f"Ошибка настройки подсветки рёбер актора {name}: {highlight_error}")
            return False

    def clear_actors(self) -> None:
        """
        Очищает все добавленные пользовательские объекты и узловые трансформации из сцены.
        """
        for name in list(self._actors.keys()):
            self.remove_actor(name)
        self._node_transforms.clear()

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

    def render(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            super().render(*args, **kwargs)
            return
        if self._update_depth > 0:
            self._render_pending = True
            return
        if self._render_timer.isActive():
            self._render_timer.stop()
        self._render_pending = False
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
        if self._render_timer.isActive():
            self._render_timer.stop()
        self._render_pending = False
        self._node_transforms.clear()
        if self.plotter is not None:
            try:
                self.plotter.close()
            except (RuntimeError, AttributeError):
                pass
            self.plotter = None
        return super().close()


__all__ = [
    "ISceneViewport",
    "VTKViewport",
]

