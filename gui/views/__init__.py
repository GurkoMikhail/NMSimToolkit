"""
Пакет представлений (Views) графического интерфейса NMSimToolkit.
Экспортирует главное окно, док-панели и диалоговые окна настройки.
"""
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.views.property_inspector import PropertyInspector
from gui.views.results_viewer import ResultsViewer
from gui.views.procedure_selector_widget import ProcedureSelectorWidget
from gui.views.data_handler_list_widget import DataHandlerListWidget
from gui.views.simulation_jobs_widget import SimulationJobsWidget
from gui.views.simulation_settings_dialog import SimulationSettingsDialog
from gui.views.main_window import MainWindow
from gui.views.distribution_import_dialog import DistributionImportDialog

__all__ = [
    'SceneTreeWidget',
    'PropertyInspector',
    'ResultsViewer',
    'ProcedureSelectorWidget',
    'DataHandlerListWidget',
    'SimulationJobsWidget',
    'SimulationSettingsDialog',
    'MainWindow',
    'DistributionImportDialog',
]


