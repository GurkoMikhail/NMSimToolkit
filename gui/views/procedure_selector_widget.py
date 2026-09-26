import logging
from typing import Any, Dict, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QDockWidget
)

from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    SpectProcedureViewModel,
    PetProcedureViewModel,
    CustomSweepProcedureViewModel,
)

_logger = logging.getLogger(__name__)


class ProcedureSelectorWidget(QDockWidget):
    """
    Док-виджет выбора и управления активным протоколом / процедурой исследования.
    Позволяет переключаться между ОФЭКТ, ПЭТ и пользовательскими сканами,
    а также связывается с PropertyInspector для детальной настройки параметров.
    """

    procedure_changed = Signal(object)  # BaseProcedureViewModel
    procedure_selected = Signal(object)  # BaseProcedureViewModel

    def __init__(self, scene_vm: Optional[Any] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__("Процедуры / Протоколы", parent)
        self.setObjectName("DockProcedureSelector")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.scene_vm = scene_vm

        # Предустановленные модели процедур
        self._procedures: Dict[str, BaseProcedureViewModel] = {
            "SPECT": SpectProcedureViewModel(),
            "PET": PetProcedureViewModel(),
            "CustomSweep": CustomSweepProcedureViewModel(),
        }
        self._active_key: str = "SPECT"

        self._init_ui()
        self._connect_procedure_signals()
        self._update_summary()

    def _init_ui(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # 1. Выбор типа исследования
        group_sel = QGroupBox("Тип протокола", container)
        sel_layout = QVBoxLayout(group_sel)

        self.combo_type = QComboBox(group_sel)
        self.combo_type.addItem("ОФЭКТ (SPECT)", "SPECT")
        self.combo_type.addItem("ПЭТ (PET)", "PET")
        self.combo_type.addItem("Пользовательский скан", "CustomSweep")
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        sel_layout.addWidget(self.combo_type)

        layout.addWidget(group_sel)

        # 2. Краткая сводка параметров
        group_info = QGroupBox("Текущие параметры", container)
        info_layout = QVBoxLayout(group_info)

        self.lbl_summary = QLabel(group_info)
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("color: #dcdcdc; font-size: 11px;")
        info_layout.addWidget(self.lbl_summary)

        self.btn_inspect = QPushButton("⚙ Настроить параметры в инспекторе", group_info)
        self.btn_inspect.setStyleSheet("padding: 6px; font-weight: bold;")
        self.btn_inspect.clicked.connect(self._on_inspect_clicked)
        info_layout.addWidget(self.btn_inspect)

        layout.addWidget(group_info)
        layout.addStretch()

        self.setWidget(container)

    def _connect_procedure_signals(self) -> None:
        for vm in self._procedures.values():
            vm.changed.connect(self._on_procedure_data_changed)

    def _on_procedure_data_changed(self) -> None:
        self._update_summary()
        # Синхронизация с геометрией сцены
        if self.scene_vm is not None:
            self.active_procedure.sync_with_scene(self.scene_vm)

    def _update_summary(self) -> None:
        vm = self.active_procedure
        if isinstance(vm, SpectProcedureViewModel):
            text = (
                f"<b>ОФЭКТ (SPECT):</b><br>"
                f"• Ракурсов (views): {vm.views}<br>"
                f"• Гамма-камер: {vm.gamma_cameras}<br>"
                f"• Конфигурация головок: {vm.head_mode}<br>"
                f"• Радиус орбиты: {vm.radius:.1f} мм<br>"
                f"• Диапазон углов: {vm.start_angle:.1f}° — {vm.end_angle:.1f}°<br>"
                f"• Время на ракурс: {vm.time_per_view:.2f} с"
            )
        elif isinstance(vm, PetProcedureViewModel):
            text = (
                f"<b>ПЭТ (PET):</b><br>"
                f"• Радиус кольца: {vm.ring_radius:.1f} мм<br>"
                f"• Модулей детекторов: {vm.detector_heads}<br>"
                f"• Время кадра: {vm.time_per_frame:.1f} с"
            )
        elif isinstance(vm, CustomSweepProcedureViewModel):
            n_grid = len(vm.grid_variables)
            n_zip = len(vm.zipped_variables)
            text = (
                f"<b>Пользовательский скан:</b><br>"
                f"• Grid-переменных: {n_grid}<br>"
                f"• Zipped-переменных: {n_zip}"
            )
        else:
            text = f"<b>{vm.name}</b>"

        self.lbl_summary.setText(text)

    def _on_type_changed(self, index: int) -> None:
        key = str(self.combo_type.itemData(index))
        if key in self._procedures:
            self._active_key = key
            self._update_summary()
            if self.scene_vm is not None:
                self.active_procedure.sync_with_scene(self.scene_vm)
            self.procedure_changed.emit(self.active_procedure)
            self.procedure_selected.emit(self.active_procedure)

    def _on_inspect_clicked(self) -> None:
        self.procedure_selected.emit(self.active_procedure)

    @property
    def active_procedure(self) -> BaseProcedureViewModel:
        """Активная модель представления процедуры."""
        return self._procedures[self._active_key]

    def set_scene_viewmodel(self, scene_vm: Any) -> None:
        """Привязка модели сцены для синхронизации геометрии."""
        self.scene_vm = scene_vm
        if self.scene_vm is not None:
            self.active_procedure.sync_with_scene(self.scene_vm)

    def set_procedure(self, proc_vm: BaseProcedureViewModel) -> None:
        """Установка внешней модели процедуры (например, при загрузке из YAML)."""
        ptype = proc_vm.procedure_type
        self._procedures[ptype] = proc_vm
        proc_vm.changed.connect(self._on_procedure_data_changed)
        for i in range(self.combo_type.count()):
            if self.combo_type.itemData(i) == ptype:
                self.combo_type.blockSignals(True)
                self.combo_type.setCurrentIndex(i)
                self.combo_type.blockSignals(False)
                break
        self._active_key = ptype
        self._update_summary()
        self.procedure_changed.emit(proc_vm)
        self.procedure_selected.emit(proc_vm)
