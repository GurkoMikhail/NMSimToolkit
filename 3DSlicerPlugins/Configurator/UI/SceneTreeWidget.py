import qt
from UI.InspectorWidgets import INSPECTOR_MAP

class SceneTreeWidget(qt.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup()

    def setup(self):
        self.mainLayout = qt.QVBoxLayout(self)

        self.splitter = qt.QSplitter(qt.Qt.Vertical)
        self.mainLayout.addWidget(self.splitter)

        # Left side: Tree and Toolbar
        leftWidget = qt.QWidget()
        leftLayout = qt.QVBoxLayout(leftWidget)

        # Toolbar for adding nodes
        toolbarLayout = qt.QHBoxLayout()

        self.nodeTypeCombo = qt.QComboBox()
        self.nodeTypeCombo.addItems([
            "CompositeNode", "SpatialNode", "Volume",
            "WoodcockVoxelVolume", "GammaCamera",
            "ParametricParallelCollimator", "Source"
        ])

        self.addNodeBtn = qt.QPushButton("Add Node")
        self.addNodeBtn.clicked.connect(self.onAddNode)

        self.deleteNodeBtn = qt.QPushButton("Delete Node")
        self.deleteNodeBtn.clicked.connect(self.onDeleteNode)

        toolbarLayout.addWidget(self.nodeTypeCombo)
        toolbarLayout.addWidget(self.addNodeBtn)
        toolbarLayout.addWidget(self.deleteNodeBtn)
        leftLayout.addLayout(toolbarLayout)

        # Tree Widget
        self.tree = qt.QTreeWidget()
        self.tree.setHeaderLabels(["Scene Hierarchy"])
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDragDropMode(qt.QAbstractItemView.InternalMove)
        leftLayout.addWidget(self.tree)

        self.splitter.addWidget(leftWidget)

        # Right side: Inspector
        self.inspectorGroup = qt.QGroupBox("Property Inspector")
        self.inspectorLayout = qt.QVBoxLayout(self.inspectorGroup)
        self.currentInspector = None

        self.splitter.addWidget(self.inspectorGroup)

        # Initialize Root
        self.root_item = qt.QTreeWidgetItem(self.tree, ["Main Scene"])
        # Root is a CompositeNode that cannot be deleted
        root_data = {'type': 'CompositeNode'}
        self.root_item.setData(0, qt.Qt.UserRole, root_data)
        # Remove Drag/Drop flags from Root so it stays fixed
        flags = self.root_item.flags() & ~qt.Qt.ItemIsUserCheckable & ~qt.Qt.ItemIsDragEnabled & ~qt.Qt.ItemIsDropEnabled
        self.root_item.setFlags(flags)

        self.tree.addTopLevelItem(self.root_item)
        self.root_item.setExpanded(True)

        self.tree.itemSelectionChanged.connect(self.onSelectionChanged)
        self.tree.setCurrentItem(self.root_item)

    def onAddNode(self):
        node_type = self.nodeTypeCombo.currentText
        selected = self.tree.currentItem()
        if not selected:
            selected = self.root_item

        # Optional: ensure selected can have children (e.g. CompositeNode or WoodcockVoxelVolume)
        # For simplicity, we assume we can add anywhere, or we restrict based on type.
        selected_data = selected.data(0, qt.Qt.UserRole)
        # In reality WoodcockVoxelVolume raises error if another Volume is added, but we skip complex validation here

        new_item = qt.QTreeWidgetItem(selected, [f"New {node_type}"])
        new_item.setFlags(new_item.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsDragEnabled | qt.Qt.ItemIsDropEnabled)
        new_data = {'type': node_type}
        new_item.setData(0, qt.Qt.UserRole, new_data)
        selected.addChild(new_item)
        selected.setExpanded(True)
        self.tree.setCurrentItem(new_item)

    def onDeleteNode(self):
        selected = self.tree.currentItem()
        if not selected or selected == self.root_item:
            return

        parent = selected.parent()
        if parent:
            parent.removeChild(selected)
        else:
            # Fallback for top-level items, though we only have one root.
            idx = self.tree.indexOfTopLevelItem(selected)
            if idx >= 0:
                self.tree.takeTopLevelItem(idx)

    def onSelectionChanged(self):
        selected = self.tree.currentItem()

        # Clear current inspector
        if self.currentInspector:
            self.inspectorLayout.removeWidget(self.currentInspector)
            self.currentInspector.deleteLater()
            self.currentInspector = None

        if not selected:
            return

        node_data = selected.data(0, qt.Qt.UserRole)
        node_type = node_data.get('type')

        WidgetClass = INSPECTOR_MAP.get(node_type)
        if WidgetClass:
            self.currentInspector = WidgetClass(node_data, self.inspectorGroup)
            self.inspectorLayout.addWidget(self.currentInspector)
            # Force update to push initial values from UI to dict
            self.currentInspector.update_data()
        else:
            self.currentInspector = qt.QLabel(f"No inspector available for type: {node_type}")
            self.inspectorLayout.addWidget(self.currentInspector)

    def get_root_item(self):
        return self.root_item
