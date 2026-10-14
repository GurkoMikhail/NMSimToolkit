"""
Секция параметров менеджера данных (DataManagerViewModel).
"""
from typing import Any, Optional

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from gui.viewmodels.data_handler_viewmodel import DataManagerViewModel
from gui.views.inspectors.base_section import BaseInspectorSection


class DataManagerInspectorSection(BaseInspectorSection):
    """Секция настройки файла HDF5 и емкости входного буфера хранения событий."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию настройки диспетчера данных."""
        self._min_buffer_capacity: int = 1
        super().__init__("Параметры диспетчера данных (DataManager)", parent)

    def _init_ui(self) -> None:
        """Создает элементы управления выбором файла HDF5 и емкостью буфера."""
        dm_form = QFormLayout(self)

        fn_layout = QHBoxLayout()
        self.txt_dm_filename = QLineEdit()
        self.txt_dm_filename.editingFinished.connect(self._on_dm_filename_edited)
        btn_browse = QPushButton("Обзор...")
        btn_browse.clicked.connect(self._on_dm_browse_clicked)
        fn_layout.addWidget(self.txt_dm_filename)
        fn_layout.addWidget(btn_browse)
        dm_form.addRow("Файл HDF5:", fn_layout)

        self.spin_dm_buffer = QSpinBox()
        self.spin_dm_buffer.setRange(1, 2_147_483_647)
        self.spin_dm_buffer.setSingleStep(1)
        self.spin_dm_buffer.setValue(1)
        self.spin_dm_buffer.valueChanged.connect(self._on_dm_buffer_changed)
        dm_form.addRow("Емкость буфера (частиц):", self.spin_dm_buffer)

    def set_min_buffer_capacity(self, min_capacity: int) -> None:
        """Устанавливает нижнюю границу емкости буфера данных (не менее числа частиц)."""
        self._min_buffer_capacity = max(1, int(min_capacity))
        self.spin_dm_buffer.setMinimum(self._min_buffer_capacity)
        if self.spin_dm_buffer.value() < self._min_buffer_capacity:
            self.spin_dm_buffer.setValue(self._min_buffer_capacity)

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, является ли модель представления диспетчером данных."""
        return isinstance(view_model, DataManagerViewModel)

    def update_ui_from_vm(self) -> None:
        """Синхронизирует имя файла и емкость буфера с текущей DataManagerViewModel."""
        view_model = self._current_vm
        if not isinstance(view_model, DataManagerViewModel):
            return

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            self.txt_dm_filename.setText(view_model.filename)
            effective_min = max(self._min_buffer_capacity, view_model.min_buffer_capacity)
            self.spin_dm_buffer.setMinimum(effective_min)
            if view_model.buffer_capacity < effective_min:
                view_model.buffer_capacity = effective_min
            self.spin_dm_buffer.setValue(int(view_model.buffer_capacity))
        finally:
            self._is_updating_ui = old_state

    def _on_dm_filename_edited(self) -> None:
        """Сохраняет измененный путь к файлу HDF5 в ViewModel."""
        if not self._is_updating_ui and isinstance(self._current_vm, DataManagerViewModel):
            self._current_vm.filename = self.txt_dm_filename.text().strip()

    def _on_dm_browse_clicked(self) -> None:
        """Открывает диалог выбора выходного файла HDF5."""
        cur_file = self.txt_dm_filename.text() or "simulation_results.h5"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Выбрать HDF5 файл для сохранения результатов",
            cur_file,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)",
        )
        if path:
            self.txt_dm_filename.setText(path)
            if isinstance(self._current_vm, DataManagerViewModel):
                self._current_vm.filename = path

    def _on_dm_buffer_changed(self, new_buffer_size: int) -> None:
        """Обновляет емкость буфера частиц в DataManagerViewModel."""
        if not self._is_updating_ui and isinstance(self._current_vm, DataManagerViewModel):
            effective_val = max(new_buffer_size, self._min_buffer_capacity)
            self._current_vm.buffer_capacity = effective_val


__all__ = ["DataManagerInspectorSection"]
