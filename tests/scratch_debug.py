import sys
import traceback

try:
    print("Step 1: Imports")
    from PySide6.QtWidgets import QApplication, QLineEdit
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from gui.views.main_window import MainWindow
    from gui.viewport_3d.transform_gizmo import GizmoMode, GizmoSpace

    print("Step 2: QApplication")
    app = QApplication.instance() or QApplication(sys.argv)

    print("Step 3: MainWindow")
    mw = MainWindow()

    print("Step 4: show")
    mw.show()
    mw.activateWindow()
    app.processEvents()

    print("Step 5: tree focus")
    tree = mw.scene_tree.tree
    tree.setFocus()
    app.processEvents()
    print("Tree focus:", tree.hasFocus(), mw.focusWidget() is tree)

    print("Step 6: gizmo")
    gizmo = mw.viewport_controller.transform_gizmo
    print("Gizmo mode:", gizmo.mode)

    print("Step 7: Key E")
    QTest.keyClick(tree, Qt.Key.Key_E)
    print("Key E done, mode:", gizmo.mode)

    print("Step 8: close")
    mw.close()
    print("Done successfully")
except Exception as e:
    traceback.print_exc()
