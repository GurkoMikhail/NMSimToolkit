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
    QToolTip,
    QWidget,
)

from gui.viewmodels.data_handler_viewmodel import DataManagerViewModel
from gui.views.inspectors.base_section import BaseInspectorSection


class DataManagerInspectorSection(BaseInspectorSection):
    """Секция настройки файла HDF5 и объёма выходного буфера записи данных."""

    _BUFFER_SIZE_HINT: str = (
        "Объём выходного буфера на группу датасетов: определяет размер чанка HDF5 и порог сброса данных. "
        "Формат: число и единица (B, KiB, MiB, GiB, kB, MB, GB), например '3 MiB'."
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию настройки диспетчера данных."""
        super().__init__("Параметры диспетчера данных (DataManager)", parent)

    def _init_ui(self) -> None:
        """Создает элементы управления выбором файла HDF5 и объёмом выходного буфера."""
        dm_form = QFormLayout(self)

        fn_layout = QHBoxLayout()
        self.txt_dm_filename = QLineEdit()
        self.txt_dm_filename.editingFinished.connect(self._on_dm_filename_edited)
        btn_browse = QPushButton("Обзор...")
        btn_browse.clicked.connect(self._on_dm_browse_clicked)
        fn_layout.addWidget(self.txt_dm_filename)
        fn_layout.addWidget(btn_browse)
        dm_form.addRow("Файл HDF5:", fn_layout)

        self.txt_dm_buffer_size = QLineEdit()
        self.txt_dm_buffer_size.setToolTip(self._BUFFER_SIZE_HINT)
        self.txt_dm_buffer_size.editingFinished.connect(self._on_dm_buffer_size_edited)
        dm_form.addRow("Объём буфера записи:", self.txt_dm_buffer_size)

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, является ли модель представления диспетчером данных."""
        return isinstance(view_model, DataManagerViewModel)

    def update_ui_from_vm(self) -> None:
        """Синхронизирует имя файла и объём буфера записи с текущей DataManagerViewModel."""
        view_model = self._current_vm
        if not isinstance(view_model, DataManagerViewModel):
            return

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            self.txt_dm_filename.setText(view_model.filename)
            self.txt_dm_buffer_size.setText(view_model.buffer_size_text)
            self.txt_dm_buffer_size.setToolTip(self._BUFFER_SIZE_HINT)
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

    def _on_dm_buffer_size_edited(self) -> None:
        """
        Передаёт введённый объём буфера в DataManagerViewModel.
        При недопустимом значении откатывает текст к текущему состоянию модели и показывает подсказку.
        """
        if self._is_updating_ui or not isinstance(self._current_vm, DataManagerViewModel):
            return
        try:
            self._current_vm.buffer_size = self.txt_dm_buffer_size.text()
        except ValueError as validation_error:
            self.txt_dm_buffer_size.setText(self._current_vm.buffer_size_text)
            error_hint = f"{validation_error}\n\n{self._BUFFER_SIZE_HINT}"
            self.txt_dm_buffer_size.setToolTip(error_hint)
            QToolTip.showText(
                self.txt_dm_buffer_size.mapToGlobal(self.txt_dm_buffer_size.rect().bottomLeft()),
                error_hint,
                self.txt_dm_buffer_size,
            )
            return
        self.txt_dm_buffer_size.setText(self._current_vm.buffer_size_text)
        self.txt_dm_buffer_size.setToolTip(self._BUFFER_SIZE_HINT)


__all__ = ["DataManagerInspectorSection"]
