# Архитектурный план: Физическая очистка GammaCamera, Единый кинематический контракт (Gizmo + Property Inspector + Toolbar)

## 1. Контекст и архитектурные предпосылки

В процессе развития подсистемы пространственной геометрии и кинематики проекта `NMSimToolkit` были введены:
- узел станины томографа (`GantryNode` в ядре и `GantryViewModel` в GUI);
- специализированный модуль кинематики ОФЭКТ (`core/geometry/spect_kinematics.py`);
- двухуровневый механизм кинематических ограничений манипуляторов (`IKinematicConstraint` в `NodeViewModel`).

Однако в подсистеме детекторов и пользовательского интерфейса сохранился ряд исторических дефектов и легаси-решений:
1. **Нарушение физической природы гамма-камеры:** Класс `GammaCamera` (ядро) и `GammaCameraViewModel` (GUI) содержат параметры и методы орбитального вращения (`orbit_radius`, `orbit_angle`, `orbit_z`, `compute_orbit_matrix`, `set_orbit_position`), которые не относятся к детекторной головке как физическому объекту.
2. **Отсутствие управления размерами детектора в UI:** В `PropertyInspector` секция общих размеров объема (`volume_group`) искусственно скрывалась для гамма-камеры (`and not is_spect`), а полезный размер активного поля зрения (`detector_size` $X \times Y$) не был вынесен в интерфейс. Для его изменения пользователю приходилось вручную разворачивать технические дочерние узлы в дереве сцены.
3. **Рассинхронизация степеней свободы между 3D Gizmo, Property Inspector и Toolbar:**
   - 3D-манипулятор Gizmo учитывает кинематические ограничения (`IKinematicConstraint`), но в `PropertyInspector` все спинбоксы положения, ориентации и размеров оставались полностью открытыми для произвольного ввода, что позволяло сломать кинематику в обход ограничений;
   - кнопки переключения режимов манипулятора (W — Перемещение, E — Вращение, R — Масштабирование, Q — Мировые/Локальные координаты) в панели инструментов оставались видимыми и активными, даже если соответствующий режим полностью заблокирован для выбранного узла;
   - внутренние компоненты гамма-камеры (коллиматор, сцинтилляционный кристалл, стекло) могли быть случайно смещены манипулятором или через инспектор, нарушая герметичность корпуса и соосность каналов.

Данная спецификация фиксирует глобальное, системное решение указанных проблем.

---

## 2. Архитектурная концепция

### 2.1 Гамма-камера как чистый физический объект
- Гамма-камера (`GammaCamera` / `GammaCameraViewModel`) представляет собой физическую детекторную головку (Detector Head).
- Она освобождается от любых орбитальных понятий.
- Основной размерный параметр камеры — **`detector_size`** (двумерный вектор $[L_x, L_y]$ активной чувствительной области кристалла детектора и коллиматора).
- Внутренние толщины слоев по оси Z: `detector_thickness`, `collimator_thickness`, `gap`, `shielding_thickness`, `glass_backend_thickness`.
- Габариты внешнего корпуса (`housing_size` $[X, Y, Z]$) вычисляются строго автоматически на базе размеров активного поля и толщин слоев.

### 2.2 Единый источник истины: `IKinematicConstraint`
Контракт `IKinematicConstraint` становится единым источником разрешений для любых геометрических манипуляций в системе:
1. **3D Gizmo во вьюпорте:** отображает только те оси/кольца, которые разрешены констрейнтом (`get_allowed_axes()`). Если все степени свободы заблокированы — манипулятор скрывается (`detach()`).
2. **2D Property Inspector:** поля ввода координат ($X, Y, Z$), углов вращения ($\text{Roll}, \text{Pitch}, \text{Yaw}$) и размеров ($X, Y, Z$) динамически становятся активными или `disabled` (только для чтения) в строгом соответствии с осями констрейнта.
3. **Панель инструментов (Toolbar) и горячие клавиши:** кнопки W (Перемещение), E (Вращение), R (Масштабирование), Q (Мировые/Локальные) и комбобоксы привязки динамически скрываются (`setVisible(False)`), если соответствующий режим недоступен. Нажатие горячих клавиш для заблокированных режимов игнорируется.

### 2.3 Защита внутренних компонентов гамма-камеры (`FixedSubcomponentKinematicConstraint`)
- Внутренние узлы камеры (`detector_box`, `collimator`, `detector`, `glass_backend`) остаются видимыми и выбираемыми в графе сцены и во вьюпорте.
- Их геометрические параметры (положение, ориентация, размеры) жестко фиксируются через ограничение `FixedSubcomponentKinematicConstraint`:
  * Gizmo во вьюпорте для них скрыт (`detach()`);
  * в `PropertyInspector` поля положения и размеров отображаются серыми (`disabled`);
  * негеометрические параметры (материал `combo_material`, параметры септ и отверстий коллиматора, цвет) остаются доступными для редактирования.

---

## 3. Детальная спецификация изменений по модулям

### 3.1 Расчетное ядро: `core/geometry/gamma_cameras.py`
1. **Удаление устаревших методов:**
   - Полностью удалить статический метод `compute_orbit_matrix()`;
   - Полностью удалить метод экземпляра `set_orbit_position()`.
2. **Добавление свойств активного поля и слоев:**
   - `@property detector_size(self) -> np.ndarray`: возвращает `[detector.size[0], detector.size[1]]` (тип `np.float64`);
   - `@detector_size.setter def detector_size(self, size_xy: Sequence[float]) -> None`: валидирует `size_xy > 0`, обновляет размеры кристалла детектора и коллиматора по осям X и Y, после чего вызывает `rebuild_camera()`;
   - `@property detector_thickness(self) -> Float`: толщина кристалла детектора по Z;
   - `@detector_thickness.setter def detector_thickness(self, value: Float) -> None`: изменяет толщину кристалла и перестраивает камеру;
   - `@property collimator_thickness(self) -> Float`: толщина коллиматора по Z;
   - `@collimator_thickness.setter def collimator_thickness(self, value: Float) -> None`: изменяет толщину коллиматора и перестраивает камеру.
3. **Метод `rebuild_camera()`:**
   - Корректно синхронизирует размеры `detector_box`, `collimator`, `detector`, `glass_backend` и внешнего корпуса свинцовой защиты `self.size`.

### 3.2 Кинематические ограничения: `gui/viewport_3d/kinematic_constraints.py`
1. **Расширение контракта `IKinematicConstraint`:**
   - Убедиться в наличии методов `get_allowed_axes(mode: GizmoMode) -> Set[GizmoAxis]`, `is_scale_allowed() -> bool`, `get_forced_space() -> Optional[GizmoSpace]`.
2. **Новый класс `FixedSubcomponentKinematicConstraint`:**
   - Реализует `IKinematicConstraint` для абсолютно зафиксированных внутренних компонентов;
   - `get_allowed_axes(mode)` возвращает `set()` (пустое множество для всех режимов);
   - `is_scale_allowed()` возвращает `False`;
   - `get_forced_space()` возвращает `None`;
   - `filter_translation` и `filter_rotation` блокируют любые смещения.
3. **Обновление `CameraMountKinematicConstraint`:**
   - Удалить зависимость от `camera_vm.orbit_radius`, `camera_vm.orbit_angle`, `camera_vm.orbit_z`;
   - Использовать `core.geometry.spect_kinematics.compute_orbit_matrix` и относительные координаты каретки на станине.

### 3.3 Модель представления гамма-камеры: `gui/viewmodels/nodes/gamma_camera_vm.py`
1. **Удаление устаревших свойств:**
   - Удалить `orbit_radius`, `orbit_angle`, `orbit_z`, `_sync_orbit_params_from_matrix()`, перегруженные сеттеры `local_matrix`, `translate`, `rotate` с вызовом синхронизации орбиты, методы `compute_orbit_matrix`, `set_orbit_position`.
2. **Добавление реактивных свойств:**
   - `detector_size: Tuple[float, float]` (геттер/сеттер с вызовом ядра и сигналом `property_changed.emit('detector_size', ...)` и `'size'`);
   - `detector_thickness: float` (геттер/сеттер);
   - `collimator_thickness: float` (геттер/сеттер);
   - `housing_size: Tuple[float, float, float]` (read-only габариты внешнего корпуса);
   - `@property half_thickness(self) -> float`: половина толщины корпуса по Z.
3. **Назначение ограничений дочерним компонентам:**
   - В `__init__` после создания или привязки дочерних узлов регистрировать для них `FixedSubcomponentKinematicConstraint` через `set_child_kinematic_constraint()`.

### 3.4 Инспектор свойств: `gui/views/property_inspector.py`
1. **Динамическая синхронизация геометрических полей с `IKinematicConstraint`:**
   - В методе `_update_transform_fields()` и `update_all_fields()`:
     * Получать `constraint = self.current_vm.get_effective_kinematic_constraint()`;
     * Если `constraint is not None`:
       - `allowed_trans = constraint.get_allowed_axes(GizmoMode.TRANSLATE)`
       - `allowed_rot = constraint.get_allowed_axes(GizmoMode.ROTATE)`
       - `scale_allowed = constraint.is_scale_allowed()`
       - `spin_x.setEnabled(GizmoAxis.X in allowed_trans)`
       - `spin_y.setEnabled(GizmoAxis.Y in allowed_trans)`
       - `spin_z.setEnabled(GizmoAxis.Z in allowed_trans)`
       - `spin_rot_x.setEnabled(GizmoAxis.X in allowed_rot)`
       - `spin_rot_y.setEnabled(GizmoAxis.Y in allowed_rot)`
       - `spin_rot_z.setEnabled(GizmoAxis.Z in allowed_rot)`
       - `spin_size_x.setEnabled(scale_allowed)`
       - `spin_size_y.setEnabled(scale_allowed)`
       - `spin_size_z.setEnabled(scale_allowed)`
     * Иначе: все поля разблокированы (`setEnabled(True)`).
2. **Секция гамма-камеры (`spect_group` $\rightarrow$ переименовать в «Параметры гамма-камеры»):**
   - Удалить спинбоксы орбиты: `spin_orbit_radius`, `spin_orbit_angle`, `spin_orbit_z`;
   - Добавить спинбоксы:
     * `spin_detector_size_x`, `spin_detector_size_y` (размер чувствительной области детектора);
     * `spin_detector_thickness` (толщина кристалла);
     * `spin_collimator_thickness` (толщина коллиматора);
     * сохранить поля `spin_cam_gap`, `spin_cam_shielding`, `spin_cam_glass`;
     * добавить информационную метку `lbl_housing_size` (габариты корпуса $X \times Y \times Z$ мм).
3. **Секция станины (`GantryViewModel`):**
   - Добавить `gantry_group`:
     * угол ротора `spin_gantry_angle` (двустороннее связывание с `gantry_vm.gantry_angle_deg`);
     * чекбокс `chk_wireframe_visible` (двустороннее связывание с `gantry_vm.wireframe_visible`).

### 3.5 Главное окно и панель инструментов: `gui/views/main_window.py`
1. **Динамическое обновление панели 3D-манипулятора (`_update_gizmo_toolbar_state`):**
   - При выборе узла (`_on_node_selected`):
     * Если выбран узел и у него есть `effective_constraint`:
       - `has_trans = len(constraint.get_allowed_axes(GizmoMode.TRANSLATE)) > 0`
       - `has_rot = len(constraint.get_allowed_axes(GizmoMode.ROTATE)) > 0`
       - `has_scale = constraint.is_scale_allowed()`
       - `forced_space = constraint.get_forced_space()`
       - `self.act_gizmo_translate.setVisible(has_trans)`
       - `self.act_gizmo_rotate.setVisible(has_rot)`
       - `self.act_gizmo_scale.setVisible(has_scale)`
       - `self.act_gizmo_space.setVisible(forced_space is None)`
       - `self.combo_grid_snap.setVisible(has_trans)`
       - `self.combo_angle_snap.setVisible(has_rot)`
       - Если текущий активный режим манипулятора оказался скрыт: автоматически переключать режим на первый доступный (например, `ROTATE` для Gantry).
     * Если у узла нет ограничений (`constraint is None`):
       - Все действия видимы (`setVisible(True)`).
     * Если ни один режим не разрешен (`not has_trans and not has_rot and not has_scale`):
       - Скрывать все кнопки манипулятора.
2. **Блокировка горячих клавиш W/E/R/Q:**
   - В `eventFilter` и `keyPressEvent`: проверять доступность запрашиваемого режима. Если `act_gizmo_*.isVisible() == False`, нажатие клавиши игнорируется.

### 3.6 Контроллер вьюпорта: `gui/controllers/viewport_controller.py`
1. **Скрытие Gizmo при отсутствии доступных степеней свободы:**
   - В `on_node_selected(node_vm)`:
     * Если у узла `constraint` блокирует все режимы (`not has_trans and not has_rot and not has_scale`):
       вызывать `self.transform_gizmo.detach()`.
     * Иначе: штатный `attach(node_vm)`.

---

## 4. Пошаговый план реализации

- [x] **Этап 1: Ядро (`core/geometry/gamma_cameras.py`)**
  - Удаление `compute_orbit_matrix`, `set_orbit_position`.
  - Добавление `detector_size`, `detector_thickness`, `collimator_thickness`, синхронизация `rebuild_camera`.
- [x] **Этап 2: Кинематические ограничения (`gui/viewport_3d/kinematic_constraints.py`)**
  - Реализация `FixedSubcomponentKinematicConstraint`.
  - Очистка `CameraMountKinematicConstraint` от `cam.orbit_*`.
- [x] **Этап 3: ViewModel гамма-камеры (`gui/viewmodels/nodes/gamma_camera_vm.py`)**
  - Удаление орбитальных полей.
  - Добавление реактивных свойств `detector_size`, `detector_thickness`, `collimator_thickness`, `housing_size`.
  - Назначение `FixedSubcomponentKinematicConstraint` дочерним узлам.
- [x] **Этап 4: Инспектор свойств (`gui/views/property_inspector.py`)**
  - Динамическая блокировка (`setEnabled`) спинбоксов геометрии по осям `IKinematicConstraint`.
  - Обновление секции параметров гамма-камеры (`detector_size`, толщины, габариты).
  - Добавление секции параметров станины (`GantryViewModel`).
- [x] **Этап 5: Панель инструментов и горячие клавиши (`gui/views/main_window.py`)**
  - Динамическое скрытие кнопок W, E, R, Q и комбобоксов привязки.
  - Фильтрация горячих клавиш для недопустимых режимов.
- [x] **Этап 6: Контроллер вьюпорта (`gui/controllers/viewport_controller.py`)**
  - Скрытие (`detach`) Gizmo при отсутствии степеней свободы.
- [x] **Этап 7: Тестирование и регрессионная верификация**
  - Обновление тестов, завязанных на старые параметры `camera.orbit_*`.
  - Новые модульные тесты для `detector_size`, `FixedSubcomponentKinematicConstraint`, синхронизации инспектора и тулбара.
  - Полный запуск pytest (`.venv\Scripts\pytest.exe`) со 100% успехом (248+ passed).
  - Аудит кода на соответствие `project_rules.md` (отсутствие hasattr/getattr, single-letter переменных, чистота ядра).

---

## 5. Критерии приемки

1. `core/` полностью изолирован от GUI-библиотек и не содержит кинематических понятий орбиты.
2. В `GammaCamera` и `GammaCameraViewModel` отсутствуют поля `orbit_radius`, `orbit_angle`, `orbit_z`.
3. Изменение `detector_size` в инспекторе приводит к корректному пересчету всех слоев камеры и обновлению 3D-меша во вьюпорте.
4. При выборе дочерних узлов камеры (коллиматор, кристалл, стекло) Gizmo во вьюпорте скрывается, кнопки W/E/R/Q в тулбаре скрываются, спинбоксы положения и размеров в инспекторе становятся неактивными (disabled), а выбор материала остается активным.
5. При выборе станины `Gantry` кнопка перемещения W скрыта, активно только вращение E (кольцо Z), в инспекторе спинбоксы позиции заблокированы, а угол Z активен.
6. 100% прохождение всех тестов проекта в виртуальном окружении `.venv`.
