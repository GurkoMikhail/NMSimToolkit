# План оптимизации производительности 3D Viewport, кинематических констрейнтов и иерархии сцены

## 1. Диагностика узких мест (As-Is)

Комплексный аудит конвейера обработки событий мыши (`MouseMoveEvent`) при взаимодействии с гамма-камерами (`GammaCameraNode` / `GammaCameraViewModel`), размещёнными на гантри (`GantryNode` / `GantryViewModel`), выявил комбинацию каскадных циклов обратной связи, избыточных синхронных вызовов OpenGL и пересоздания геометрии/UI-виджетов на каждый пиксель движения курсора.

### Сводка накладных расходов на ОДИН тик движения мыши (`MouseMoveEvent`) для двухдетекторного гантри (17 узлов в поддереве, ~12 VTK-акторов, до 120 000 треугольников коллиматоров):
1. **35–40 синхронных вызовов `self.plotter.render()` за один шаг мыши:**
   - В [`VTKViewport.update_actor_transform`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/vtk_viewport.py#L196-L219) на строке 215 выполняется безусловный синхронный вызов `self.render()`.
   - Флаг `self._is_interacting` ([`vtk_viewport.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/vtk_viewport.py#L342-L343)) остаётся `False`, так как `TransformGizmo._on_left_button_press` ([`transform_gizmo.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py#L1030-L1032)) перехватывает событие VTK (`_abort_vtk_event`), не пропуская его в интерактор камеры.
   - Каждый `update_actor_transform` для каждого дочернего объёма камеры (`casing`, `detector_box`, `lead_body`, `holes_<id>`, `crystal`, `glass_backend`) немедленно запускает полный кадр OpenGL-рендера.
2. **Тройная перезапись матриц (Loopback Storm) между `TransformGizmo`, `IKinematicConstraint` и `StepAndShootProcedureViewModel`:**
   - **При радиальном перемещении (`action == "radial"`):**
     1. `TransformGizmo` записывает `target_node.local_matrix = new_matrix` ([`transform_gizmo.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py#L1374)) $\to$ `StepAndShootProcedureViewModel._on_camera_property_changed` ([`procedure_viewmodel.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/procedure_viewmodel.py#L546-L574)) пересчитывает `local_matrix` второй камеры (**проход 1**, теряя при этом угол `roll` второй камеры!) и испускает `procedure.changed.emit()`.
     2. `CameraMountKinematicConstraint.on_transform_changed` ([`kinematic_constraints.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/kinematic_constraints.py#L722-L743)) повторно обходит `gantry_node_vm.children`, перезаписывает `sibling.local_matrix` (**проход 2**) и вызывает `procedure.radius = float(new_radius)`.
     3. Сеттер `StepAndShootProcedureViewModel.radius` ([`procedure_viewmodel.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/procedure_viewmodel.py#L640-L664)) (при `_is_syncing_with_scene == False` и без проверки на равенство) **в третий раз** перезаписывает `local_matrix` обеих камер (**проход 3**) и снова испускает `procedure.changed.emit()`.
   - **При тангенциальном вращении (`action == "tangential"`):**
     1. `TransformGizmo` безусловно присваивает `target_node.local_matrix = new_matrix` ([`transform_gizmo.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py#L1374)), хотя локальная матрица камеры на рельсах гантри не меняется — вызывая ложный каскад `transform_changed` по 8 узлам камеры.
     2. `CameraMountKinematicConstraint.on_transform_changed` ставит `gantry_node_vm.gantry_angle = float(new_gantry_angle)` (каскад по 17 узлам), а затем ставит `procedure.start_angle = float(new_gantry_angle)` ([`kinematic_constraints.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/kinematic_constraints.py#L719)), сеттер которого ([`procedure_viewmodel.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/procedure_viewmodel.py#L727)) повторно присваивает `self._observed_gantry.gantry_angle = angle_value`, запуская второй полный каскад по всем 17 узлам.
3. **Холостой экспорт всей 3D-сцены в `generate_jobs()` и полное пересоздание ~450 Qt-виджетов таблиц на каждый шаг мыши:**
   - Сигнал `proc_vm.changed` напрямую подключён к [`MainWindow._on_generate_jobs`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py#L696) и [`ProcedureInspectorSection.update_ui_from_vm`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/procedure_section.py#L281-L309).
   - В [`OrchestratorSession.generate_jobs()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/orchestrator_session.py#L535-L592) ради вызова статических методов `SweepCompiler.compile_protocol(protocol_cfg)` и `SweepCompiler.generate_job_matrix(sweep_config)` каждый раз выполняется полный экспорт дерева сцены (`SceneExporter.export_to_config`), `ConfigTemplateBinder.apply_bindings` (`deepcopy` всей сцены) и валидация `Orchestrator(sim_config)`, результат которых тут же отбрасывается.
   - В [`SimulationJobsWidget.set_jobs()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/simulation_jobs_widget.py#L102-L161) при неизменном числе шагов полностью очищается `QTableWidget` и заново создаются для каждого из 32–64 ракурсов виджеты `QProgressBar`, `QPushButton`, `QRadioButton` с парсингом CSS `setStyleSheet`, хотя меняется только строка параметров в колонке 2.
   - В [`ProcedureInspectorSection._populate_zipped_table()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/procedure_section.py#L359-L412) на каждый сигнал заново создаются объекты `QTableWidgetItem` и `QColor` для всех ячеек таблицы вместо обновления текста in-place.
4. **4-кратное удаление и пересоздание мешей `TransformGizmo` и пересоздание `SPECTManipulator`:**
   - [`TransformGizmo.update_visuals()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py#L753-L790) на каждый вызов удаляет все акторы ручек из рендерера (`remove_visuals`), заново генерирует `pv.Arrow`, `pv.Disc`, `pv.Cylinder`, `pv.Cube`, трансформирует вершины в Python и вызывает `add_mesh_actor`.
   - [`SPECTManipulator.update_visuals()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/spect_manipulator.py#L104-L142) на каждый тик удаляет и заново строит `pv.lines_from_points` (120 точек) и `pv.Sphere`.
5. **Безусловный запуск каскада в констрейнтах и гизмо внутри одного дискретного шага снаппинга (шаг 5° / 10 мм):**
   - Констрейнты ([`CameraMountKinematicConstraint`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/kinematic_constraints.py#L639-L703), [`SpectOrbitKinematicConstraint`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/kinematic_constraints.py#L298-L355), [`GantryKinematicConstraint`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/kinematic_constraints.py#L847-L880)) и [`TransformGizmo`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py#L1420-L1475) округляют координаты/углы до сетки (10 мм / 5°), но не проверяют, изменилось ли дискретное значение относительно предыдущего шага. В результате при микросдвиге мыши внутри одной ступеньки снаппинга констрейнт каждый раз заново строит матрицы, а гизмо запускает полный каскад обновления узлов, манипуляторов, UI и рендера вхолостую.
6. **Алгоритмические накладные расходы в иерархии и пикинге:**
   - Ручной обход 17 узлов в Python для пересчёта `global_matrix` вместо использования нативной C++ иерархии `vtkTransform` (`SetInput(parent_transform)`).
   - В [`GammaCameraViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/nodes/gamma_camera_vm.py#L76-L139) свойства слотов (`casing_vm`, `crystal_vm`, `collimator_vm` и др.) не кэшируются и при каждом чтении `camera_vm.half_thickness` выполняют DFS-поиск по поддереву.
   - В [`Volume.invalidate_matrix_cache`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/volumes.py#L98-L110) происходит квадратичный обход $O(N \cdot D)$ (каждый дочерний `Volume` вызывает `invalidate_geometry()`, поднимаясь до корня сцены).
   - В [`TransformGizmo._pick_gizmo_axis`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py#L1090-L1132) при обычном движении курсора (`Hover`) на каждый пиксель создаётся новый `vtkCellPicker()`, выполняющий ray-casting по всем 120 000 треугольников коллиматоров вместо проверки только акторов гизмо. А сами инстансированные акторы отверстий `holes_<id>` в [`CollimatorHoleRenderer.render_holes`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/collimator_hole_renderer.py#L405-L415) имеют `Pickable=True`, нагружая пикинг сцены при наличии внешнего габаритного бокса коллиматора.

---

## 2. Пошаговый архитектурный план оптимизации

### Этап 1: Коалесцирующий рендеринг и нативная C++ иерархия `vtkTransform` в `VTKViewport`
**Затрагиваемые файлы:**
- [`gui/viewport_3d/vtk_viewport.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/vtk_viewport.py)
- [`gui/viewport_3d/collimator_hole_renderer.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/collimator_hole_renderer.py)
- [`gui/controllers/viewport_controller.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/viewport_controller.py)
- [`gui/viewmodels/nodes/base_node_vm.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/nodes/base_node_vm.py)

**Изменения:**
1. **Отложенный (коалесцирующий) рендеринг и пакетный режим в `VTKViewport`:**
   - Добавить метод `request_render()` на базе одноразового `QTimer(0)` (`self._render_timer`, `self._render_pending`) и контекстный менеджер / пару `begin_update()` / `end_update()`.
   - Заменить синхронные вызовы `self.render()` в `update_actor_transform` и `set_actor_edge_highlight` на `self.request_render()`, а поэлементный цикл `SetElement` заменить на атомарный `mat_vtk.DeepCopy(matrix.ravel())`.
   - Все обновления акторов и манипуляторов в рамках одного тика мыши гарантированно схлопываются в **ровно 1 вызов `self.plotter.render()`** в конце обработки события.
2. **Делегирование расчёта мировых матриц иерархии на C++ движок VTK (`vtkTransform.SetInput`):**
   - В `VTKViewport` ввести реестр узловых C++ трансформаций `self._node_transforms: Dict[int, vtk.vtkTransform]`:
     - Метод `set_node_local_transform(node_id: int, local_matrix: np.ndarray, parent_id: Optional[int] = None) -> vtk.vtkTransform` обновляет локальную матрицу `vtkMatrix4x4` через `node_transform.SetMatrix(mat_vtk)` и связывает узел с родителем через `child_transform.SetInput(parent_transform)`.
     - Метод `attach_actor_to_node(actor_name: str, node_id: int)` назначает актору `actor.SetUserTransform(node_transform)`.
     - Строго разделить использование `SetUserTransform` (для всех иерархических акторов узлов сцены: `mesh_<id>`, `holes_<id>`, `selection_box_<id>`, не вызывая для них `SetUserMatrix`) и `update_actor_transform` (`SetUserMatrix` — только для неиерархических сервисных акторов `TransformGizmo`, `SPECTManipulator`, `PETManipulator`), чтобы не разрывать связь с родительским `vtkTransform`.
   - В `NodeViewModel` добавить сигнал `local_transform_changed = Signal()`, который испускается **только на самом изменившемся узле** (без рекурсивного обхода потомков).
   - В `SceneViewportController`:
     - При построении сцены (`add_or_update_node_actor`) регистрировать `set_node_local_transform` для каждого `node_vm` (включая `CompositeNode` без собственного меша, такие как `GantryViewModel` и `GammaCameraViewModel`) и привязывать акторы узла через `attach_actor_to_node`.
     - Подписать контроллер на `node_vm.local_transform_changed`: при повороте `GantryViewModel` или перемещении `GammaCameraViewModel` обновляется **только 1 объект `vtkTransform`** изменившегося узла (1 вызов вместо 17!), а C++ конвейер VTK автоматически по `MTime` пересчитывает мировые позиции всех 12 дочерних акторов камер.
     - Если текущий выбранный узел (`_selected_node_vm`) совпадает с изменившимся узлом или является его потомком, однократно обновлять трансформ `TransformGizmo`.
3. **Защита от просадки FPS и пикинга на каналах коллиматора (`holes_<id>`):**
   - В `SceneViewportController` не включать `set_actor_edge_highlight("holes_<id>", True)` на 5000 гексагональных призмах коллиматора (оставляя янтарную подсветку рёбер только на внешнем боксе коллиматора).
   - В [`CollimatorHoleRenderer.render_holes`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/collimator_hole_renderer.py#L383-L418) выставлять `hole_actor.SetPickable(False)` (клик всегда обрабатывается внешним боксом `mesh_<id>`) и формировать облако точек центров каналов через векторный `vtk.util.numpy_support.numpy_to_vtk` вместо Python-цикла.

---

### Этап 2: Перевод `TransformGizmo` и `SPECTManipulator` на матричные трансформации VTK и оптимизация пикинга
**Затрагиваемые файлы:**
- [`gui/viewport_3d/transform_gizmo.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py)
- [`gui/viewport_3d/spect_manipulator.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/spect_manipulator.py)

**Изменения:**
1. **Трансформация акторов `TransformGizmo` через `SetUserMatrix` без пересоздания мешей:**
   - Строить полигональные меши ручек (`pv.Arrow`, `pv.Disc`, `pv.Cylinder`, `pv.Cube`, `pv.Plane`) в канонических локальных координатах **только при смене набора активных ручек или их геометрии** (смена `GizmoMode`, `allowed_axes`, `is_gantry`, `hovered_axis` или изменение радиуса орбитального кольца гантри).
   - Во время перетаскивания узла или вращения гантри (когда геометрия ручек и дистанция до камеры неизменны) вычислять матрицу трансформации гизмо и применять её к существующим акторам ручек через `viewport.update_actor_transform(actor_name, gizmo_matrix)` без вызова `remove_visuals()` и `add_mesh_actor()`.
   - Убрать дублирующий вызов `self.update_visuals()` в конце `_apply_drag_translation` / `_apply_drag_rotation` / `_apply_drag_scale`, выполняя обновление позиции гизмо строго один раз за тик мыши.
2. **Кэширование пикеров с `PickFromListOn()` в `TransformGizmo`:**
   - Инициализировать `self._prop_picker = vtk.vtkPropPicker()` и `self._cell_picker = vtk.vtkCellPicker()` один раз в `__init__` с включённым `PickFromListOn()`.
   - При регистрации акторов гизмо добавлять в `PickList` пикеров **только акторы ручек гизмо**. Это полностью исключает программный ray-casting по геометрии сцены при движении курсора (`Hover`).
3. **Оптимизация `SPECTManipulator`:**
   - Создавать `spect_detector_indicator` (`pv.Sphere`) один раз в локальном начале координат и перемещать его по орбите через `viewport.update_actor_transform`.
   - Перестраивать полилинию орбиты `spect_orbit_trajectory` только при фактическом изменении `radius` или `z_pos` (при чистом вращении по углу `angle` кольцо орбиты остаётся неизменным, обновляется только матрица сферы-индикатора).

---

### Этап 3: Контроль дискретного шага в констрейнтах, устранение петли `ProcedureViewModel` и быстрое in-place обновление UI в runtime
**Затрагиваемые файлы:**
- [`core/geometry/spect_kinematics.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/spect_kinematics.py)
- [`gui/viewport_3d/kinematic_constraints.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/kinematic_constraints.py)
- [`gui/viewport_3d/transform_gizmo.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py)
- [`gui/viewmodels/procedure_viewmodel.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/procedure_viewmodel.py)
- [`gui/controllers/orchestrator_session.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/orchestrator_session.py)
- [`gui/views/simulation_jobs_widget.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/simulation_jobs_widget.py)
- [`gui/views/inspectors/procedure_section.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/procedure_section.py)
- [`gui/views/inspectors/transform_section.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/transform_section.py)

**Изменения:**
1. **Каноническая декомпозиция орбитальных параметров в `core/geometry/spect_kinematics.py`:**
   - Реализовать чистую функцию `extract_orbit_parameters(matrix: np.ndarray, half_thickness: float = 0.0) -> Tuple[float, float, float, float]` (возвращающую `(radius, angle, z, roll)`), заменив ею 7 дублирующихся ручных расчётов в `kinematic_constraints.py`, `procedure_viewmodel.py` и `viewport_controller.py`.
   - Исправить потерю угла `roll` второй камеры при синхронизации радиуса в `StepAndShootProcedureViewModel` (`_on_camera_property_changed` и `radius.setter`).
2. **Явный контроль дискретного шага снаппинга (Snap Gate) в констрейнтах и `TransformGizmo`:**
   - В `CameraMountKinematicConstraint`, `SpectOrbitKinematicConstraint`, `GantryKinematicConstraint` и методах перетаскивания `TransformGizmo`: после вычисления снэпнутого значения (`snapped_radius`, `snapped_gantry`, `snapped_z`, `snapped_angle_deg`, `snapped_target_pos`) сравнивать его с текущим применённым дискретным значением узла и возвращать в `changed_data` явный флаг `'changed': bool`.
   - Если курсор сместился внутри текущего шага сетки (5° / 10 мм) и дискретное значение не изменилось (`changed == False`), констрейнт не пересчитывает `compute_orbit_matrix`, а `TransformGizmo._on_mouse_move` немедленно завершает обработку тика на входе — без записи в узлы, без вызова `on_transform_changed` и без перерисовки.
3. **Разрыв петли обратной связи `Gizmo ↔ Constraint ↔ ProcedureVM` (однократный проход на изменившийся шаг):**
   - В `TransformGizmo._apply_drag_translation`: если `changed_data.get("action") == "tangential"` (камера на гантри тянется по касательной, вращая только ротор станины), не модифицировать неизменную `self._target_node.local_matrix`.
   - Убрать дублирующую тройную перезапись матриц: когда трансформация инициирована перетаскиванием в сцене (`CameraMountKinematicConstraint` / `GantryKinematicConstraint`), обновление спаренных камер и параметров `StepAndShootProcedureViewModel` (`start_angle`, `radius`) выполняется ровно **один раз** за шаг без обратной перезаписи `local_matrix` тех же камер из сеттеров `procedure.radius` / `procedure.start_angle`.
4. **Мгновенное runtime-обновление таблиц и инспекторов in-place в конце тика мыши (без пересоздания виджетов):**
   - В [`OrchestratorSession.generate_jobs()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/orchestrator_session.py#L535-L592): убрать холостой вызов `SceneExporter.export_to_config`, `ConfigTemplateBinder.apply_bindings` и `Orchestrator(sim_config)` (которые нужны только при запуске симуляции в `start()`); формировать матрицу задач напрямую из конфигурации протокола через `SweepCompiler.compile_protocol(self.procedure_vm.to_config())` и `SweepCompiler.generate_job_matrix(sweep_config)`.
   - В [`SimulationJobsWidget.set_jobs()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/simulation_jobs_widget.py#L102-L161): если количество задач не изменилось (`len(jobs) == self.table.rowCount()`), **не пересоздавать** `QRadioButton`, `QProgressBar`, `QPushButton` и `QTableWidgetItem` — обновлять in-place `self._jobs[row_idx] = job` и текст ячейки параметров `self.table.item(row_idx, 2).setText(params_str)` (привязав обработчик кнопки предпросмотра к актуальному `self._jobs[row_idx]`).
   - В [`ProcedureInspectorSection._populate_zipped_table()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/procedure_section.py#L359-L412): если размерность `(rows, cols)` и колонки таблицы не изменились, обновлять только текст существующих ячеек `tbl.item(row_idx, col_idx).setText(...)` in-place.
   - В [`TransformInspectorSection.update_transform_fields()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/transform_section.py#L140-L185): обновлять числовые значения спинбоксов (`setValue`) без повторного вызова `bind_variable` / `unbind` на каждый шаг трансформации.

---

### Этап 4: Минималистичная бескэшевая модель `core/` и кэширование слотов в `GammaCameraViewModel`
**Затрагиваемые файлы:**
- [`core/scene/nodes.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/scene/nodes.py)
- [`core/geometry/volumes.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/volumes.py)
- [`core/geometry/flattened_scene.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/flattened_scene.py)
- [`gui/viewmodels/nodes/gamma_camera_vm.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/nodes/gamma_camera_vm.py)

**Изменения:**
1. **Отказ от кэширования матриц и `inverse_global_matrix` в `SpatialNode` и `CompositeNode`:**
   - Полностью убрать `_global_matrix_cache`, `_inverse_global_matrix_cache`, свойство `inverse_global_matrix`, а также методы `invalidate_matrix_cache()`, `_invalidate_subtree_caches()` и `invalidate_geometry()` из `SpatialNode` и `CompositeNode`.
   - Свойство `SpatialNode.global_matrix` вычисляет матрицу напрямую (`self.parent.global_matrix @ self.local_matrix` при наличии родителя, иначе `self.local_matrix`), исключая каскады инвалидации кэшей при вращении гантри и перемещении камер в GUI.
   - Обратная матрица вычисляется через `np.linalg.inv(node.global_matrix)` непосредственно в точках использования (`SpatialNode.convert_to_local_position`, `SpatialNode.convert_to_local_direction`, `FlattenedScene._flatten_scene_graph`, `DoseMapHandler`, `StreamHandlers`).
2. **Удаление кэша `_geometry_buffer` и `invalidate_geometry()` из `Volume` и связанных классов:**
   - В `Volume.geometry_buffer` выполнять компиляцию `GeometryCompiler().compile_scene(self)` напрямую без хранения промежуточного кэша `_geometry_buffer`.
   - Очистить `Volume`, `DirectParallelCollimator`, `WoodcockParametricVolume`, `WoodcockVoxelVolume`, `ParametricParallelCollimator`, `GantryNode` и узловые ViewModel от вызовов `invalidate_matrix_cache()` и `invalidate_geometry()`.
3. **Кэширование ссылок на слоты в `GammaCameraViewModel`:**
   - Кэшировать разрешённые ссылки на дочерние ViewModel (`casing_vm`, `detector_box_vm`, `collimator_vm`, `crystal_vm`, `glass_backend_vm`) и инвалидировать кэш только при структурных изменениях (`add_child`, `remove_child`, `sync_children_from_core`), полностью убирая DFS-обход дерева из `half_thickness` и `size`.

---

### Этап 5: Верификация, тестирование и гигиена графа
1. Прогон всех существующих доменных тестов (`tests/geometry/`, `tests/viewmodels/`, `tests/viewport_3d/`, `tests/views/`) через `.venv\Scripts\pytest.exe`.
2. Добавление новых тестов в доменные тестовые модули:
   - проверка отсечения холостых тиков внутри дискретного шага снаппинга (`changed == False`) и однократности рендера/каскада при переходе через шаг сетки (`radial`, `tangential`, `axial`);
   - проверка сохранения угла `roll` второй камеры при радиальном перемещении первой камеры;
   - проверка корректности иерархических `vtkTransform` в `VTKViewport` и отсутствия пересоздания мешей в `TransformGizmo`;
   - проверка быстрого runtime-обновления `OrchestratorSession.generate_jobs` и сохранения экземпляров виджетов (in-place update без пересоздания `QProgressBar` / `QRadioButton` / `QTableWidgetItem`) в `SimulationJobsWidget` и `ProcedureInspectorSection`.
3. Актуализация графа знаний `codebase-memory` через `index_repository`.

