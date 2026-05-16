import qt
import slicer
from ..Utils.TransformHelpers import TransformHelpers

class BaseInspectorWidget(qt.QWidget):
    """
    Base class for dynamically drawn inspectors.
    """
    def __init__(self, node_data, parent=None):
        super().__init__(parent)
        self.node_data = node_data
        self.layout = qt.QFormLayout(self)
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

        self.layout.addRow("Transform Node:", self.transformCombo)

        # Connect change
        self.transformCombo.connect("currentNodeChanged(vtkMRMLNode*)", self.on_transform_changed)

    def on_transform_changed(self, node):
        if node:
            matrix_dict = TransformHelpers.extract_matrix(node)
            if matrix_dict:
                self.node_data['transformations'] = [matrix_dict]
        else:
            if 'transformations' in self.node_data:
                del self.node_data['transformations']

class VolumeInspectorWidget(SpatialInspectorWidget):
    def setup(self):
        super().setup()

        self.materialEdit = qt.QLineEdit()
        self.materialEdit.setText(self.node_data.get('material', 'water'))
        self.layout.addRow("Material:", self.materialEdit)

        self.sizeLayout = qt.QHBoxLayout()
        self.sizeSpinX = qt.QDoubleSpinBox()
        self.sizeSpinY = qt.QDoubleSpinBox()
        self.sizeSpinZ = qt.QDoubleSpinBox()

        for spin in [self.sizeSpinX, self.sizeSpinY, self.sizeSpinZ]:
            spin.setRange(0.001, 10000)
            spin.setValue(10.0)
            self.sizeLayout.addWidget(spin)

        self.sizeUnit = qt.QComboBox()
        self.sizeUnit.addItems(["cm", "mm", "m"])
        self.sizeLayout.addWidget(self.sizeUnit)

        self.layout.addRow("Size (X,Y,Z):", self.sizeLayout)

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
        self.layout.addRow("Base Material:", self.materialEdit)

        self.volumeCombo = slicer.qMRMLNodeComboBox()
        self.volumeCombo.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"]
        self.volumeCombo.noneEnabled = True
        self.volumeCombo.setMRMLScene(slicer.mrmlScene)
        self.layout.addRow("Phantom Volume:", self.volumeCombo)

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

        # Energy
        energyLayout = qt.QHBoxLayout()
        self.energySpin = qt.QDoubleSpinBox()
        self.energySpin.setRange(0, 1e6)
        self.energySpin.setValue(140.5)
        self.energyUnit = qt.QComboBox()
        self.energyUnit.addItems(["keV", "MeV", "eV"])
        energyLayout.addWidget(self.energySpin)
        energyLayout.addWidget(self.energyUnit)
        self.layout.addRow("Energy:", energyLayout)

        # Activity
        activityLayout = qt.QHBoxLayout()
        self.activitySpin = qt.QDoubleSpinBox()
        self.activitySpin.setRange(0, 1e12)
        self.activitySpin.setValue(100.0)
        self.activityUnit = qt.QComboBox()
        self.activityUnit.addItems(["MBq", "kBq", "Bq"])
        activityLayout.addWidget(self.activitySpin)
        activityLayout.addWidget(self.activityUnit)
        self.layout.addRow("Activity:", activityLayout)

        # Half Life
        hlLayout = qt.QHBoxLayout()
        self.hlSpin = qt.QDoubleSpinBox()
        self.hlSpin.setRange(0, 1e12)
        self.hlSpin.setValue(6.0)
        self.hlUnit = qt.QComboBox()
        self.hlUnit.addItems(["h", "m", "s", "d", "y"])
        hlLayout.addWidget(self.hlSpin)
        hlLayout.addWidget(self.hlUnit)
        self.layout.addRow("Half Life:", hlLayout)

        # Activity Volume
        self.volumeCombo = slicer.qMRMLNodeComboBox()
        self.volumeCombo.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeCombo.noneEnabled = True
        self.volumeCombo.setMRMLScene(slicer.mrmlScene)
        self.layout.addRow("Activity Matrix:", self.volumeCombo)

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

        self.sizeLayout = qt.QHBoxLayout()
        self.sizeSpinX = qt.QDoubleSpinBox()
        self.sizeSpinY = qt.QDoubleSpinBox()
        for spin in [self.sizeSpinX, self.sizeSpinY]:
            spin.setRange(0.001, 10000)
            spin.setValue(50.0)
            self.sizeLayout.addWidget(spin)

        self.sizeUnit = qt.QComboBox()
        self.sizeUnit.addItems(["cm", "mm", "m"])
        self.sizeLayout.addWidget(self.sizeUnit)
        self.layout.addRow("Size (X,Y):", self.sizeLayout)

        self.pixelsLayout = qt.QHBoxLayout()
        self.pxSpinX = qt.QSpinBox()
        self.pxSpinY = qt.QSpinBox()
        for spin in [self.pxSpinX, self.pxSpinY]:
            spin.setRange(1, 4096)
            spin.setValue(128)
            self.pixelsLayout.addWidget(spin)
        self.layout.addRow("Pixels (X,Y):", self.pixelsLayout)

        # Material
        self.materialEdit = qt.QLineEdit("NaI")
        self.layout.addRow("Crystal Material:", self.materialEdit)

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
