import qt
import slicer
from Utils.TransformHelpers import TransformHelpers

class BaseInspectorWidget(qt.QWidget):
    """
    Base class for dynamically drawn inspectors.
    """
    def __init__(self, node_data, parent=None):
        super().__init__(parent)
        self.node_data = node_data
        self.mainLayout = qt.QFormLayout(self)
        self.setup()

    def setup(self):
        # Override in subclasses
        pass

    def update_data(self):
        # Read from UI elements and update self.node_data
        pass

class SpatialInspectorWidget(BaseInspectorWidget):
    """
    Adds transformation inspector logic.
    """
    def setup(self):
        super().setup()

        self.transformCombo = slicer.qMRMLNodeComboBox()
        self.transformCombo.nodeTypes = ["vtkMRMLTransformNode"]
        self.transformCombo.selectNodeUponCreation = True
        self.transformCombo.addEnabled = False
        self.transformCombo.removeEnabled = False
        self.transformCombo.noneEnabled = True
        self.transformCombo.showHidden = False
        self.transformCombo.showChildNodeTypes = False
        self.transformCombo.setMRMLScene(slicer.mrmlScene)

        # Restore selection
        transform_id = self.node_data.get('__transform_node_id')
        if transform_id:
            self.transformCombo.setCurrentNodeID(transform_id)

        self.mainLayout.addRow("Transform Node:", self.transformCombo)

        # Connect change
        self.transformCombo.connect("currentNodeChanged(vtkMRMLNode*)", self.on_transform_changed)

    def on_transform_changed(self, node):
        if node:
            self.node_data['__transform_node_id'] = node.GetID()
            matrix_dict = TransformHelpers.extract_matrix(node)
            if matrix_dict:
                self.node_data['transformations'] = [matrix_dict]
        else:
            self.node_data.pop('__transform_node_id', None)
            if 'transformations' in self.node_data:
                del self.node_data['transformations']

class VolumeInspectorWidget(SpatialInspectorWidget):
    def setup(self):
        super().setup()

        self.materialEdit = qt.QLineEdit()
        self.materialEdit.setText(self.node_data.get('material', 'water'))
        self.mainLayout.addRow("Material:", self.materialEdit)

        self.sizeLayout = qt.QHBoxLayout()
        self.sizeSpinX = qt.QDoubleSpinBox()
        self.sizeSpinY = qt.QDoubleSpinBox()
        self.sizeSpinZ = qt.QDoubleSpinBox()

        sizes = self.node_data.get('size', ["10.0 cm", "10.0 cm", "10.0 cm"])
        try:
            valX, unit = sizes[0].split()
            valY, _ = sizes[1].split()
            valZ, _ = sizes[2].split()
        except:
            valX, valY, valZ, unit = 10.0, 10.0, 10.0, "cm"

        for i, spin in enumerate([self.sizeSpinX, self.sizeSpinY, self.sizeSpinZ]):
            spin.setRange(0.001, 10000)
            val = float([valX, valY, valZ][i])
            spin.setValue(val)
            self.sizeLayout.addWidget(spin)

        self.sizeUnit = qt.QComboBox()
        self.sizeUnit.addItems(["cm", "mm", "m"])
        self.sizeUnit.setCurrentText(unit)
        self.sizeLayout.addWidget(self.sizeUnit)

        self.mainLayout.addRow("Size (X,Y,Z):", self.sizeLayout)

        # Connect to update
        self.materialEdit.textChanged.connect(self.update_data)
        self.sizeSpinX.valueChanged.connect(self.update_data)
        self.sizeSpinY.valueChanged.connect(self.update_data)
        self.sizeSpinZ.valueChanged.connect(self.update_data)
        self.sizeUnit.currentTextChanged.connect(self.update_data)

    def update_data(self):
        super().update_data()
        self.node_data['material'] = self.materialEdit.text()
        unit = self.sizeUnit.currentText()
        self.node_data['size'] = [
            f"{self.sizeSpinX.value()} {unit}",
            f"{self.sizeSpinY.value()} {unit}",
            f"{self.sizeSpinZ.value()} {unit}"
        ]

class WoodcockVoxelVolumeInspectorWidget(SpatialInspectorWidget):
    def setup(self):
        super().setup()

        self.materialEdit = qt.QLineEdit()
        self.materialEdit.setText(self.node_data.get('material', 'water'))
        self.mainLayout.addRow("Base Material:", self.materialEdit)

        self.volumeCombo = slicer.qMRMLNodeComboBox()
        self.volumeCombo.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"]
        self.volumeCombo.noneEnabled = True
        self.volumeCombo.setMRMLScene(slicer.mrmlScene)

        # Restore selection
        volume_id = self.node_data.get('__volume_node_id')
        if volume_id:
            self.volumeCombo.setCurrentNodeID(volume_id)

        self.mainLayout.addRow("Phantom Volume:", self.volumeCombo)

        self.materialEdit.textChanged.connect(self.update_data)
        self.volumeCombo.connect("currentNodeChanged(vtkMRMLNode*)", self.update_data)

    def update_data(self, *args):
        super().update_data()
        self.node_data['material'] = self.materialEdit.text()

        node = self.volumeCombo.currentNode()
        if node:
            self.node_data['__volume_node_id'] = node.GetID()
        else:
            self.node_data.pop('__volume_node_id', None)

class SourceInspectorWidget(SpatialInspectorWidget):
    def setup(self):
        super().setup()

        # Parse existing
        en = self.node_data.get('energy', "140.5 keV")
        try:
            en_val, en_unit = en.split()
        except:
            en_val, en_unit = 140.5, "keV"

        act = self.node_data.get('activity', "100.0 MBq")
        try:
            act_val, act_unit = act.split()
        except:
            act_val, act_unit = 100.0, "MBq"

        hl = self.node_data.get('half_life', "6.0 h")
        try:
            hl_val, hl_unit = hl.split()
        except:
            hl_val, hl_unit = 6.0, "h"

        # Energy
        energyLayout = qt.QHBoxLayout()
        self.energySpin = qt.QDoubleSpinBox()
        self.energySpin.setRange(0, 1e6)
        self.energySpin.setValue(float(en_val))
        self.energyUnit = qt.QComboBox()
        self.energyUnit.addItems(["keV", "MeV", "eV"])
        self.energyUnit.setCurrentText(en_unit)
        energyLayout.addWidget(self.energySpin)
        energyLayout.addWidget(self.energyUnit)
        self.mainLayout.addRow("Energy:", energyLayout)

        # Activity
        activityLayout = qt.QHBoxLayout()
        self.activitySpin = qt.QDoubleSpinBox()
        self.activitySpin.setRange(0, 1e12)
        self.activitySpin.setValue(float(act_val))
        self.activityUnit = qt.QComboBox()
        self.activityUnit.addItems(["MBq", "kBq", "Bq"])
        self.activityUnit.setCurrentText(act_unit)
        activityLayout.addWidget(self.activitySpin)
        activityLayout.addWidget(self.activityUnit)
        self.mainLayout.addRow("Activity:", activityLayout)

        # Half Life
        hlLayout = qt.QHBoxLayout()
        self.hlSpin = qt.QDoubleSpinBox()
        self.hlSpin.setRange(0, 1e12)
        self.hlSpin.setValue(float(hl_val))
        self.hlUnit = qt.QComboBox()
        self.hlUnit.addItems(["h", "m", "s", "d", "y"])
        self.hlUnit.setCurrentText(hl_unit)
        hlLayout.addWidget(self.hlSpin)
        hlLayout.addWidget(self.hlUnit)
        self.mainLayout.addRow("Half Life:", hlLayout)

        # Activity Volume
        self.volumeCombo = slicer.qMRMLNodeComboBox()
        self.volumeCombo.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeCombo.noneEnabled = True
        self.volumeCombo.setMRMLScene(slicer.mrmlScene)

        # Restore selection
        activity_id = self.node_data.get('__activity_node_id')
        if activity_id:
            self.volumeCombo.setCurrentNodeID(activity_id)

        self.mainLayout.addRow("Activity Matrix:", self.volumeCombo)

        # Connections
        self.energySpin.valueChanged.connect(self.update_data)
        self.activitySpin.valueChanged.connect(self.update_data)
        self.hlSpin.valueChanged.connect(self.update_data)
        self.volumeCombo.connect("currentNodeChanged(vtkMRMLNode*)", self.update_data)

    def update_data(self, *args):
        super().update_data()
        self.node_data['energy'] = f"{self.energySpin.value()} {self.energyUnit.currentText()}"
        self.node_data['activity'] = f"{self.activitySpin.value()} {self.activityUnit.currentText()}"
        self.node_data['half_life'] = f"{self.hlSpin.value()} {self.hlUnit.currentText()}"

        node = self.volumeCombo.currentNode()
        if node:
            self.node_data['__activity_node_id'] = node.GetID()
        else:
            self.node_data.pop('__activity_node_id', None)

class GammaCameraInspectorWidget(SpatialInspectorWidget):
    def setup(self):
        super().setup()

        sizes = self.node_data.get('size', ["50.0 cm", "50.0 cm", "1.0 cm"])
        try:
            valX, unit = sizes[0].split()
            valY, _ = sizes[1].split()
        except:
            valX, valY, unit = 50.0, 50.0, "cm"

        pxs = self.node_data.get('pixels', [128, 128])

        self.sizeLayout = qt.QHBoxLayout()
        self.sizeSpinX = qt.QDoubleSpinBox()
        self.sizeSpinY = qt.QDoubleSpinBox()
        for i, spin in enumerate([self.sizeSpinX, self.sizeSpinY]):
            spin.setRange(0.001, 10000)
            spin.setValue(float([valX, valY][i]))
            self.sizeLayout.addWidget(spin)

        self.sizeUnit = qt.QComboBox()
        self.sizeUnit.addItems(["cm", "mm", "m"])
        self.sizeUnit.setCurrentText(unit)
        self.sizeLayout.addWidget(self.sizeUnit)
        self.mainLayout.addRow("Size (X,Y):", self.sizeLayout)

        self.pixelsLayout = qt.QHBoxLayout()
        self.pxSpinX = qt.QSpinBox()
        self.pxSpinY = qt.QSpinBox()
        for i, spin in enumerate([self.pxSpinX, self.pxSpinY]):
            spin.setRange(1, 4096)
            spin.setValue(int(pxs[i]))
            self.pixelsLayout.addWidget(spin)
        self.mainLayout.addRow("Pixels (X,Y):", self.pixelsLayout)

        # Material
        self.materialEdit = qt.QLineEdit()
        self.materialEdit.setText(self.node_data.get('material', 'NaI'))
        self.mainLayout.addRow("Crystal Material:", self.materialEdit)

        # Connect
        self.sizeSpinX.valueChanged.connect(self.update_data)
        self.sizeSpinY.valueChanged.connect(self.update_data)
        self.pxSpinX.valueChanged.connect(self.update_data)
        self.pxSpinY.valueChanged.connect(self.update_data)
        self.materialEdit.textChanged.connect(self.update_data)

    def update_data(self, *args):
        super().update_data()
        unit = self.sizeUnit.currentText()
        self.node_data['size'] = [
            f"{self.sizeSpinX.value()} {unit}",
            f"{self.sizeSpinY.value()} {unit}",
            "1.0 cm" # dummy Z for flat detector
        ]
        self.node_data['pixels'] = [self.pxSpinX.value(), self.pxSpinY.value()]
        self.node_data['material'] = self.materialEdit.text()

# A simple map to get the correct widget
INSPECTOR_MAP = {
    'CompositeNode': SpatialInspectorWidget,
    'SpatialNode': SpatialInspectorWidget,
    'Volume': VolumeInspectorWidget,
    'WoodcockVoxelVolume': WoodcockVoxelVolumeInspectorWidget,
    'Source': SourceInspectorWidget,
    'GammaCamera': GammaCameraInspectorWidget,
    'ParametricParallelCollimator': SpatialInspectorWidget # For simplicity
}
