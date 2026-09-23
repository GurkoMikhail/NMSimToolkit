import logging
import os
from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup,
    QProgressBar, QDockWidget, QMessageBox
)

_logger = logging.getLogger(__name__)


class SimulationJobsWidget(QDockWidget):
    """
    Док-панель управления и мониторинга параллельных задач симуляции (Simulation Jobs).
    Отображает список сгенерированных воркеров, их статус и прогресс,
    позволяет переключать фокус глубокой инспекции (3D-треки, SharedMemory)
    и осуществлять быстрый предпросмотр ракурса в 3D-вьюпорте.
    """

    job_preview_requested = Signal(dict)  # Контекст задачи для кинематического предпросмотра
    focused_job_changed = Signal(int)     # Индекс сфокусированной задачи
    pool_size_changed = Signal(int)       # Изменение размера пула воркеров
    generate_jobs_requested = Signal()    # Запрос на генерацию пула задач

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Задачи симуляции (Jobs)", parent)
        self.setObjectName("DockSimulationJobs")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        self._jobs: List[Dict[str, Any]] = []
        self._focused_index: int = 0
        self._focus_button_group = QButtonGroup(self)
        self._focus_button_group.setExclusive(True)
        self._progress_bars: Dict[int, QProgressBar] = {}

        self._init_ui()

    def _init_ui(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Верхняя панель управления пулом
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        top_layout.addWidget(QLabel("Процессов в пуле:", container))
        self.spin_pool_size = QSpinBox(container)
        cpu_cnt = os.cpu_count() or 4
        self.spin_pool_size.setRange(1, max(1, cpu_cnt))
        self.spin_pool_size.setValue(min(4, max(1, cpu_cnt)))
        self.spin_pool_size.valueChanged.connect(self.pool_size_changed)
        top_layout.addWidget(self.spin_pool_size)

        top_layout.addStretch()
        layout.addLayout(top_layout)

        # Информационная строка состояния
        self.lbl_status = QLabel("Задачи не сгенерированы", container)
        self.lbl_status.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        # 2. Таблица задач
        self.table = QTableWidget(0, 6, container)
        self.table.setHorizontalHeaderLabels(["№", "Фокус", "Параметры ракурса", "Статус", "Прогресс", "Предпросмотр"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self.table)

        self.setWidget(container)

    @property
    def pool_size(self) -> int:
        return self.spin_pool_size.value()

    @property
    def focused_index(self) -> int:
        return self._focused_index

    def set_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        """Обновление таблицы на основе сгенерированного списка задач."""
        self._jobs = list(jobs)
        self._progress_bars.clear()

        # Очистка радиокнопок
        for btn in self._focus_button_group.buttons():
            self._focus_button_group.removeButton(btn)

        self.table.setRowCount(len(jobs))

        for row_idx, job in enumerate(jobs):
            # №
            item_id = QTableWidgetItem(f"#{row_idx + 1}")
            item_id.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 0, item_id)

            # Фокус радиокнопка
            radio_widget = QWidget()
            radio_layout = QHBoxLayout(radio_widget)
            radio_layout.setContentsMargins(0, 0, 0, 0)
            radio_layout.setAlignment(Qt.AlignCenter)
            radio_btn = QRadioButton()
            if row_idx == self._focused_index:
                radio_btn.setChecked(True)
            self._focus_button_group.addButton(radio_btn, row_idx)
            radio_btn.toggled.connect(lambda checked, idx=row_idx: self._on_focus_toggled(idx, checked))
            radio_layout.addWidget(radio_btn)
            self.table.setCellWidget(row_idx, 1, radio_widget)

            # Параметры ракурса
            params_str = ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in job.items() if not k.startswith('_'))
            item_params = QTableWidgetItem(params_str or "Базовая задача")
            self.table.setItem(row_idx, 2, item_params)

            # Статус
            item_status = QTableWidgetItem("В очереди")
            item_status.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 3, item_status)

            # Прогресс бар
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            pbar.setTextVisible(True)
            pbar.setStyleSheet("QProgressBar { max-height: 14px; text-align: center; }")
            self._progress_bars[row_idx] = pbar
            self.table.setCellWidget(row_idx, 4, pbar)

            # Кнопка предпросмотра в 3D
            btn_preview = QPushButton("👁 Предпросмотр")
            btn_preview.setStyleSheet("padding: 2px 6px; font-size: 11px;")
            btn_preview.clicked.connect(lambda _, j=job: self.job_preview_requested.emit(j))
            self.table.setCellWidget(row_idx, 5, btn_preview)

        self._update_status_counts()

    def _on_focus_toggled(self, index: int, checked: bool) -> None:
        if checked:
            self._focused_index = index
            self.focused_job_changed.emit(index)

    def _on_table_selection_changed(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._jobs):
            # Автоматический предпросмотр при выборе строки
            self.job_preview_requested.emit(self._jobs[row])

    def on_job_started(self, task_id: int) -> None:
        """Оповещение о старте выполнения задачи воркером."""
        if 0 <= task_id < self.table.rowCount():
            item = self.table.item(task_id, 3)
            if item is not None:
                item.setText("Выполняется")
                item.setForeground(Qt.yellow)
        self._update_status_counts()

    def on_job_progress(self, task_id: int, progress: float, counts: int, cps: float) -> None:
        """Оповещение о прогрессе задачи."""
        if task_id in self._progress_bars:
            pct = int(min(100.0, max(0.0, progress * 100.0)))
            self._progress_bars[task_id].setValue(pct)

    def on_job_finished(self, task_id: int) -> None:
        """Оповещение о завершении задачи."""
        if 0 <= task_id < self.table.rowCount():
            item = self.table.item(task_id, 3)
            if item is not None:
                item.setText("Завершено")
                item.setForeground(Qt.green)
            if task_id in self._progress_bars:
                self._progress_bars[task_id].setValue(100)
        self._update_status_counts()

    def on_job_error(self, task_id: int, error_text: str) -> None:
        """Оповещение об ошибке задачи."""
        if 0 <= task_id < self.table.rowCount():
            item = self.table.item(task_id, 3)
            if item is not None:
                item.setText("Ошибка")
                item.setForeground(Qt.red)
        self._update_status_counts()

    def _update_status_counts(self) -> None:
        total = self.table.rowCount()
        if total == 0:
            self.lbl_status.setText("Задачи не сгенерированы")
            return

        queued = 0
        running = 0
        done = 0
        err = 0

        for r in range(total):
            item = self.table.item(r, 3)
            text = item.text() if item is not None else ""
            if text == "Выполняется":
                running += 1
            elif text == "Завершено":
                done += 1
            elif text == "Ошибка":
                err += 1
            else:
                queued += 1

        self.lbl_status.setText(f"Всего задач: {total} | В очереди: {queued} | Выполняется: {running} | Завершено: {done}" + (f" | Ошибок: {err}" if err > 0 else ""))
