import os
import sys
import vtk
import qt
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import ctk

# Ensure our local packages (Logic, UI, Utils) can be imported
# when Slicer runs this file.
module_dir = os.path.dirname(__file__)
if module_dir not in sys.path:
    sys.path.append(module_dir)

class NMSimToolkitConfigurator(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "NMSimToolkit Configurator"
        self.parent.categories = ["Simulation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Jules"]
        self.parent.helpText = """
This is a module for generating the simulation configuration YAML file and extracting
associated MRML nodes (like numpy matrices for phantoms/sources) for NMSimToolkit.
"""
        self.parent.acknowledgementText = """"""

class NMSimToolkitConfiguratorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from UI/ if any, or build manually
        # For this we will build programmatically based on the plan.

        # UI layout
        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs)

        # Import our custom UI classes
        from UI.GlobalSettingsWidget import GlobalSettingsWidget
        from UI.SceneTreeWidget import SceneTreeWidget
        from Logic.ConfigExporter import ConfigExporter
        from Logic.ScenePreviewLogic import ScenePreviewLogic

        # 1. Global Settings Tab
        self.globalSettingsTab = GlobalSettingsWidget()
        self.tabs.addTab(self.globalSettingsTab, "Global Settings")

        # 2. Scene Editor Tab
        self.sceneEditorTab = SceneTreeWidget()
        self.tabs.addTab(self.sceneEditorTab, "Scene Editor")

        # 3. Export Tab
        self.exportTab = qt.QWidget()
        self.exportLayout = qt.QVBoxLayout(self.exportTab)
        self.tabs.addTab(self.exportTab, "Export")

        self.exportDirSelector = ctk.ctkPathLineEdit()
        self.exportDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.exportLayout.addWidget(qt.QLabel("Export Directory:"))
        self.exportLayout.addWidget(self.exportDirSelector)

        self.exportFilenameEdit = qt.QLineEdit("simulation_config.yaml")
        self.exportLayout.addWidget(qt.QLabel("Filename:"))
        self.exportLayout.addWidget(self.exportFilenameEdit)

        self.exportButton = qt.QPushButton("Generate Simulation Config")
        self.exportButton.clicked.connect(self.onExportClicked)
        self.exportLayout.addWidget(self.exportButton)
        self.exportLayout.addStretch(1)

        self.layout.addStretch(1)

        # Logic setup
        self.exporter = ConfigExporter(self.globalSettingsTab, self.sceneEditorTab)
        self.previewer = ScenePreviewLogic(self.sceneEditorTab)

        # Add a Preview Button to the Scene Editor layout or Export layout
        # The prompt asked for "под Tree/Inspector в Scene Editor tab" or similar.
        self.previewBtn = qt.QPushButton("Update 3D Preview")
        self.previewBtn.clicked.connect(self.previewer.update_preview)

        # We can add this to the scene editor's main layout
        self.sceneEditorTab.mainLayout.addWidget(self.previewBtn)

    def onExportClicked(self):
        export_dir = self.exportDirSelector.currentPath
        if not export_dir:
            slicer.util.errorDisplay("Please select an export directory.")
            return

        filename = self.exportFilenameEdit.text
        if not filename.endswith('.yaml'):
            filename += '.yaml'

        import os
        filepath = os.path.join(export_dir, filename)

        try:
            self.exporter.export(filepath)
            slicer.util.infoDisplay(f"Configuration successfully exported to:\n{filepath}")
        except Exception as e:
            import traceback
            slicer.util.errorDisplay(f"Failed to export configuration:\n{str(e)}\n\n{traceback.format_exc()}")

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()
