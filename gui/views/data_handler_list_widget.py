import logging
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QMenu, QDockWidget, QMessageBox
)

from gui.viewmodels.data_handler_viewmodel import (
    BaseDataHandlerViewModel,
    DirectStreamHandlerViewModel,
    SensitiveVolumeHandlerViewModel,
    HistoryAssemblerHandlerViewModel,
    DoseMapHandlerViewModel,
    DataManagerViewModel,
)

_logger = logging.getLogger(__name__)


class DataHandlerListWidget(QDockWidget):
    """
    Док-панель плоского списка обработчиков данных (DataHandlers).
    Обеспечивает визуальное управление обработчиками, их включение/выключение,
    добавление новых и открытие параметров в PropertyInspector.
    """

    handler_selected = Signal(object)  # BaseDataHandlerViewModel
    data_manager_selected = Signal(object)  # DataManagerViewModel

    def __init__(
        self,
        data_manager_vm: Optional[DataManagerViewModel] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("Обработчики данных", parent)
        self.setObjectName("DockDataHandlerList")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.data_manager_vm = data_manager_vm or DataManagerViewModel()
        self._is_updating_ui: bool = False

        self._init_ui()
        self._connect_signals()
        self.rebuild_list()

    def _init_ui(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Плоский список обработчиков данных
        self.list_widget = QListWidget(container)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.currentItemChanged.connect(lambda cur, prev: self._on_item_clicked(cur) if cur else None)
        layout.addWidget(self.list_widget)

        # 2. Кнопки управления списком
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_add = QPushButton("+ Добавить", container)
        self.btn_add.setStyleSheet("padding: 4px;")
        self._setup_add_menu()
        btn_layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton("- Удалить", container)
        self.btn_remove.setStyleSheet("padding: 4px;")
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        btn_layout.addWidget(self.btn_remove)

        layout.addLayout(btn_layout)

        # 3. Кнопка общих настроек DataManager (HDF5, буфер)
        self.btn_manager_settings = QPushButton("⚙ Параметры HDF5 / Буфера", container)
        self.btn_manager_settings.setStyleSheet("padding: 5px; font-weight: bold;")
        self.btn_manager_settings.clicked.connect(self._on_manager_settings_clicked)
        layout.addWidget(self.btn_manager_settings)

        self.setWidget(container)

    def _setup_add_menu(self) -> None:
        menu = QMenu(self.btn_add)
        act_stream = menu.addAction("Потоковый стриминг в GUI (DirectStreamHandler)")
        act_stream.triggered.connect(lambda: self._add_handler_type("DirectStreamHandler"))

        act_history = menu.addAction("История треков детектора (HistoryAssemblerHandler)")
        act_history.triggered.connect(lambda: self._add_handler_type("HistoryAssemblerHandler"))

        act_sens = menu.addAction("Сбор попаданий в объемы (SensitiveVolumeHandler)")
        act_sens.triggered.connect(lambda: self._add_handler_type("SensitiveVolumeHandler"))

        act_dose = menu.addAction("Накопление карты дозы (DoseMapHandler)")
        act_dose.triggered.connect(lambda: self._add_handler_type("DoseMapHandler"))

        self.btn_add.setMenu(menu)

    def _connect_signals(self) -> None:
        self.data_manager_vm.handlers_changed.connect(self.rebuild_list)

    def rebuild_list(self) -> None:
        """Перестроение элементов списка на основе модели DataManagerViewModel."""
        self._is_updating_ui = True
        self.list_widget.clear()

        for handler in self.data_manager_vm.handlers:
            item = QListWidgetItem(self.list_widget)
            item.setText(handler.name)
            item.setData(Qt.UserRole, handler)

        self._is_updating_ui = False

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        pass

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        handler = item.data(Qt.UserRole)
        if isinstance(handler, BaseDataHandlerViewModel):
            self.handler_selected.emit(handler)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        handler = item.data(Qt.UserRole)
        if isinstance(handler, BaseDataHandlerViewModel):
            self.handler_selected.emit(handler)

    def _add_handler_type(self, handler_type: str) -> None:
        if handler_type == "DirectStreamHandler":
            vm = DirectStreamHandlerViewModel()
        elif handler_type == "HistoryAssemblerHandler":
            vm = HistoryAssemblerHandlerViewModel()
        elif handler_type == "SensitiveVolumeHandler":
            vm = SensitiveVolumeHandlerViewModel()
        elif handler_type == "DoseMapHandler":
            vm = DoseMapHandlerViewModel()
        else:
            return

        self.data_manager_vm.add_handler(vm)
        self.handler_selected.emit(vm)

    def _on_remove_clicked(self) -> None:
        current_item = self.list_widget.currentItem()
        if current_item is None:
            return

        handler = current_item.data(Qt.UserRole)
        if isinstance(handler, BaseDataHandlerViewModel):
            self.data_manager_vm.remove_handler(handler)

    def _on_manager_settings_clicked(self) -> None:
        self.data_manager_selected.emit(self.data_manager_vm)

    def set_data_manager_viewmodel(self, vm: DataManagerViewModel) -> None:
        """Смена активной модели DataManagerViewModel."""
        if self.data_manager_vm is not vm:
            try:
                self.data_manager_vm.handlers_changed.disconnect(self.rebuild_list)
            except (RuntimeError, TypeError):
                pass
            self.data_manager_vm = vm
            self._connect_signals()
            self.rebuild_list()
