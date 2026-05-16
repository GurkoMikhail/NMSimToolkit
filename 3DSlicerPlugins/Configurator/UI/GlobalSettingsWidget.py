import qt
import ctk

class GlobalSettingsWidget(qt.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup()

    def setup(self):
        self.mainLayout = qt.QVBoxLayout(self)

        # 1. Pool Size
        poolLayout = qt.QHBoxLayout()
        poolLayout.addWidget(qt.QLabel("Pool Size:"))
        self.poolSizeSpin = qt.QSpinBox()
        self.poolSizeSpin.setRange(1, 128)
        self.poolSizeSpin.setValue(4)
        poolLayout.addWidget(self.poolSizeSpin)
        self.mainLayout.addLayout(poolLayout)

        # 2. Simulation Manager
        self.simManagerCollapsible = ctk.ctkCollapsibleButton()
        self.simManagerCollapsible.text = "Simulation Manager"
        self.mainLayout.addWidget(self.simManagerCollapsible)
        simLayout = qt.QFormLayout(self.simManagerCollapsible)

        self.startTimeSpin = qt.QDoubleSpinBox()
        self.startTimeSpin.setRange(0, 1e9)
        self.startTimeUnit = qt.QComboBox()
        self.startTimeUnit.addItems(["ns", "us", "ms", "s"])

        startLayout = qt.QHBoxLayout()
        startLayout.addWidget(self.startTimeSpin)
        startLayout.addWidget(self.startTimeUnit)
        simLayout.addRow("Start Time:", startLayout)

        self.stopTimeSpin = qt.QDoubleSpinBox()
        self.stopTimeSpin.setRange(0, 1e9)
        self.stopTimeSpin.setValue(1.0)
        self.stopTimeUnit = qt.QComboBox()
        self.stopTimeUnit.addItems(["s", "ms", "us", "ns"])

        stopLayout = qt.QHBoxLayout()
        stopLayout.addWidget(self.stopTimeSpin)
        stopLayout.addWidget(self.stopTimeUnit)
        simLayout.addRow("Stop Time:", stopLayout)

        self.particlesSpin = qt.QSpinBox()
        self.particlesSpin.setRange(1, int(1e9))
        self.particlesSpin.setValue(1000000)
        simLayout.addRow("Particles Number:", self.particlesSpin)

        self.minEnergySpin = qt.QDoubleSpinBox()
        self.minEnergySpin.setRange(0, 1e6)
        self.minEnergySpin.setValue(10.0)
        self.minEnergyUnit = qt.QComboBox()
        self.minEnergyUnit.addItems(["keV", "MeV", "eV"])

        energyLayout = qt.QHBoxLayout()
        energyLayout.addWidget(self.minEnergySpin)
        energyLayout.addWidget(self.minEnergyUnit)
        simLayout.addRow("Min Energy:", energyLayout)

        # 3. Data Manager
        self.dataManagerCollapsible = ctk.ctkCollapsibleButton()
        self.dataManagerCollapsible.text = "Data Manager"
        self.mainLayout.addWidget(self.dataManagerCollapsible)
        dataLayout = qt.QFormLayout(self.dataManagerCollapsible)

        self.filenameEdit = qt.QLineEdit("simulation_output")
        dataLayout.addRow("Filename Prefix:", self.filenameEdit)

        self.bufferCapacitySpin = qt.QSpinBox()
        self.bufferCapacitySpin.setRange(1000, int(1e9))
        self.bufferCapacitySpin.setValue(100000)
        dataLayout.addRow("Buffer Capacity:", self.bufferCapacitySpin)

        # Handlers List
        dataLayout.addRow(qt.QLabel("Data Handlers:"))
        self.handlersListWidget = qt.QListWidget()
        dataLayout.addRow(self.handlersListWidget)

        addHandlerLayout = qt.QHBoxLayout()
        self.handlerTypeCombo = qt.QComboBox()
        self.handlerTypeCombo.addItems(["DirectStreamHandler", "SensitiveVolumeHandler", "HistoryAssemblerHandler"])
        self.addHandlerBtn = qt.QPushButton("Add Handler")
        self.addHandlerBtn.clicked.connect(self.onAddHandler)
        self.removeHandlerBtn = qt.QPushButton("Remove Selected")
        self.removeHandlerBtn.clicked.connect(self.onRemoveHandler)

        addHandlerLayout.addWidget(self.handlerTypeCombo)
        addHandlerLayout.addWidget(self.addHandlerBtn)
        addHandlerLayout.addWidget(self.removeHandlerBtn)
        dataLayout.addRow(addHandlerLayout)

        # 4. Protocol
        self.protocolCollapsible = ctk.ctkCollapsibleButton()
        self.protocolCollapsible.text = "Protocol"
        self.mainLayout.addWidget(self.protocolCollapsible)
        protoLayout = qt.QFormLayout(self.protocolCollapsible)

        self.protocolTypeCombo = qt.QComboBox()
        self.protocolTypeCombo.addItems(["None", "CustomSweep"])
        protoLayout.addRow("Type:", self.protocolTypeCombo)

        # Currently we only support CustomSweep or None.
        self.protocolTypeCombo.currentTextChanged.connect(self.onProtocolTypeChanged)

        # We'll just put simple text edits for custom sweep dicts as json/yaml string inputs for simplicity
        # or just hardcode if needed. For a proper UI, we might need a complex dynamic table,
        # but for this base implementation, QLineEdit is adequate to input a simple dict.
        self.customSweepWidget = qt.QWidget()
        customSweepLayout = qt.QFormLayout(self.customSweepWidget)

        self.gridVarsEdit = qt.QLineEdit()
        self.gridVarsEdit.setPlaceholderText("e.g. {'angle': [0, 90, 180]}")
        customSweepLayout.addRow("Grid Variables:", self.gridVarsEdit)

        self.zipVarsEdit = qt.QLineEdit()
        self.zipVarsEdit.setPlaceholderText("e.g. {'x': [1,2], 'y': [3,4]}")
        customSweepLayout.addRow("Zipped Variables:", self.zipVarsEdit)

        protoLayout.addRow(self.customSweepWidget)
        self.customSweepWidget.setVisible(False)

        self.mainLayout.addStretch(1)

    def onProtocolTypeChanged(self, text):
        self.customSweepWidget.setVisible(text == "CustomSweep")

    def onAddHandler(self):
        hType = self.handlerTypeCombo.currentText
        item = qt.QListWidgetItem(hType)

        # We need to store parameters for the handler. We can attach a custom widget
        # or store a dict and open a dialog. For simplicity, we create a small widget
        # and insert it into the list using setItemWidget.

        self.handlersListWidget.addItem(item)

        widget = qt.QWidget()
        layout = qt.QHBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0)

        layout.addWidget(qt.QLabel(hType))

        config = {'type': hType}

        if hType in ["SensitiveVolumeHandler", "HistoryAssemblerHandler"]:
            svEdit = qt.QLineEdit()
            svEdit.setPlaceholderText("sens_vol_1, sens_vol_2")
            layout.addWidget(qt.QLabel("Sensitive Volumes:"))
            layout.addWidget(svEdit)
            config['__sv_edit'] = svEdit # temporary reference

        if hType == "HistoryAssemblerHandler":
            saveInitCb = qt.QCheckBox("Save Initial States")
            saveInitCb.setChecked(True)
            layout.addWidget(saveInitCb)
            config['__si_cb'] = saveInitCb

        layout.addStretch(1)
        item.setData(qt.Qt.UserRole, config)
        item.setSizeHint(widget.sizeHint())
        self.handlersListWidget.setItemWidget(item, widget)

    def onRemoveHandler(self):
        row = self.handlersListWidget.currentRow
        if row >= 0:
            self.handlersListWidget.takeItem(row)

    # --- Data Getters ---

    def get_pool_size(self):
        return self.poolSizeSpin.value

    def get_simulation_manager_settings(self):
        return {
            'start_time': f"{self.startTimeSpin.value} {self.startTimeUnit.currentText}",
            'stop_time': f"{self.stopTimeSpin.value} {self.stopTimeUnit.currentText}",
            'particles_number': self.particlesSpin.value,
            'min_energy': f"{self.minEnergySpin.value} {self.minEnergyUnit.currentText}"
        }

    def get_data_manager_settings(self):
        handlers = []
        for i in range(self.handlersListWidget.count):
            item = self.handlersListWidget.item(i)
            config_ref = item.data(qt.Qt.UserRole)

            h_dict = {'type': config_ref['type']}

            if '__sv_edit' in config_ref:
                text = config_ref['__sv_edit'].text.strip()
                h_dict['sensitive_volumes'] = [x.strip() for x in text.split(',')] if text else []

            if '__si_cb' in config_ref:
                h_dict['save_initial_states'] = config_ref['__si_cb'].isChecked()

            handlers.append(h_dict)

        return {
            'filename': self.filenameEdit.text,
            'buffer_capacity': self.bufferCapacitySpin.value,
            'handlers': handlers
        }

    def get_protocol_settings(self):
        pType = self.protocolTypeCombo.currentText
        if pType == "None":
            return None
        elif pType == "CustomSweep":
            import ast
            import slicer

            grid_text = self.gridVarsEdit.text.strip()
            grid_vars = {}
            if grid_text:
                try:
                    grid_vars = ast.literal_eval(grid_text)
                except Exception as e:
                    slicer.util.errorDisplay(f"Failed to parse Grid Variables: {e}")

            zip_text = self.zipVarsEdit.text.strip()
            zip_vars = {}
            if zip_text:
                try:
                    zip_vars = ast.literal_eval(zip_text)
                except Exception as e:
                    slicer.util.errorDisplay(f"Failed to parse Zipped Variables: {e}")

            return {
                'type': 'CustomSweep',
                'grid_variables': grid_vars,
                'zipped_variables': zip_vars
            }
