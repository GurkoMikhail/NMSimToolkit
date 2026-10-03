# Архитектурный план: Физическая визуализация воксельных фантомов на базе палитры материалов и глобального псевдорентгена

## 1. Контекст и архитектурные предпосылки

В рамках недавнего рефакторинга в проект была внедрена физическая семантическая система визуализации:
1. Семантическая палитра материалов NIST ([`gui/viewport_3d/material_palette.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/material_palette.py)), обеспечивающая взаимно-однозначное соответствие всем 144 материалам базы данных NIST;
2. Расчет физического коэффициента ослабления $\mu(E)$ и оптической непрозрачности (Opacity) по закону Бугера-Ламберта-Бера в согласованной системе единиц `hepunits`;
3. Глобальный режим отображения в «псевдорентгене» (`pseudo_xray_mode`) с динамической реакцией на рабочую энергию фотонов (`xray_energy`);
4. Контурная янтарная подсветка ребер реального меша для выделенных в графе сцены узлов.

Однако подсистема воксельных фантомов ([`WoodcockVoxelVolume`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/voxel_volumes.py) в ядре и [`VoxelVolumeViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/nodes/voxel_volume_vm.py) в GUI) на текущий момент остается изолированной от новой схемы:
* **Разрыв с физической природой фантома:** В ядре фантом представляет собой строгое дискретное распределение материалов [`MaterialArray`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/materials/materials.py) со списком `element_list` (`Vacuum`, `Water, Liquid`, `Bone, Cortical`, `Tissue, Soft`, `Lung`, `Adipose` и т.д.). При этом [`VoxelVolumeRenderer`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/voxel_volume_renderer.py) накладывает на целочисленные индексы материалов градиентные непрерывные шкалы ядерной медицины (`Hot Iron`, `Rainbow`, `GE Color`), смешивая органы и ткани в неестественные псевдоцвета.
* **Игнорирование глобального режима псевдорентгена:** При активации чекбокса `pseudo_xray_mode` вся полигональная сцена (корпус, коллиматор, детекторы камеры) переходит в физический рентгеновский рендеринг, а фантом внутри продолжает гореть кислотными псевдоцветами `Hot Iron`.
* **Отсутствие реакции на энергию излучения:** При изменении `xray_energy` в настройках симуляции непрозрачность тканей фантома не пересчитывается.
* **Отсутствие визуализации выделения:** Поскольку фантом рендерится через `vtkVolume`, у него нет полигональных ребер (`SetEdgeVisibility`), и при клике на фантом в дереве сцены он визуально никак не выделяется во вьюпорте.

---

## 2. Архитектурные принципы и концепция решения

1. **Единый источник истины (Single Source of Truth):**
   Цвета и физическая непрозрачность органов и тканей фантома берутся строго из модуля [`gui/viewport_3d/material_palette.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/material_palette.py) на основе имен материалов из `dist.element_list`.
2. **Строгая глобальность `pseudo_xray_mode`:**
   Режим псевдорентгена управляется централизованно через `GuiSimulationSettings` / `SceneViewportController`. У фантома **нет собственного обособленного режима рентгена** — при включении глобального `pseudo_xray_mode` фантом синхронно вместе со всей сценой переключается на физический рентгенографический контраст (DRR — Digitally Reconstructed Radiograph).
3. **Исключение непрерывных приближений:**
   Фантом рассматривается строго как дискретное воксельное распределение материалов (`MaterialArray`). Непрерывные шкалы плотности не закладываются — воксели мапятся дискретно через ступенчатые функции VTK (Step Transfer Functions).
4. **Ступенчатая дискретная передаточная функция (Step Transfer Function):**
   Для дискретных целочисленных ID материалов формируются кусочно-постоянные `vtkColorTransferFunction` и `vtkPiecewiseFunction`:
   * Окрестность индекса $[i - 0.499, i + 0.499]$ имеет постоянный цвет и непрозрачность $i$-го материала;
   * Для фонового материала (`Vacuum`, `Air`) непрозрачность строго равна $0.0$ (абсолютная прозрачность);
   * Для биологических тканей непрозрачность рассчитывается по физическому закону ослабления:
     $$\text{Opacity} = 1 - e^{-\mu(E) \cdot L}$$
     где $L$ — характерный размер вокселя (в единицах `hepunits`).
5. **Контурная рамка выделения (Selection Bounding Box):**
   При выборе фантома в графе сцены вокруг него во вьюпорте создается динамический каркасный параллелепипед (`pv.Box`, `wireframe`) фирменного янтарно-оранжевого цвета `(1.0, 0.55, 0.0)` толщиной 2.5 px, повторяющий габариты воксельной сетки с учетом смещения `origin` и глобальной матрицы трансформации.

---

## 3. Детальная спецификация изменений по модулям

### 3.1. Генерация ступенчатых передаточных функций: `gui/viewport_3d/material_palette.py`
Добавить функции построения передаточных функций VTK для дискретных наборов материалов:
1. `build_material_volume_color_tf(element_list: Sequence[Material], pseudo_xray_mode: bool = False, energy: float = 140.0 * units.keV) -> vtk.vtkColorTransferFunction`:
   - При `pseudo_xray_mode == False`: для каждого материала $i$ цвет задается через `get_material_color(material.name)`;
   - При `pseudo_xray_mode == True`: цвет задается через `get_pseudo_xray_rgba(material.name, energy)[0]`;
   - Задает ступенчатый переход вокруг каждого целочисленного индекса (точки $i - 0.499$ и $i + 0.499$).
2. `build_material_volume_opacity_tf(element_list: Sequence[Material], pseudo_xray_mode: bool = False, energy: float = 140.0 * units.keV, characteristic_length: float = 2.0 * units.mm) -> vtk.vtkPiecewiseFunction`:
   - Если `material.name == "Vacuum"` или содержит "Air": непрозрачность строго равна $0.0$;
   - Иначе: непрозрачность рассчитывается через `get_material_opacity(...)` (в обычном режиме) или `get_pseudo_xray_rgba(...)[1]` (в режиме псевдорентгена);
   - Задает ступенчатые точки $i - 0.499$ и $i + 0.499$.

### 3.2. Рендерер фантома: `gui/viewport_3d/voxel_volume_renderer.py`
1. **Интеграция физических передаточных функций:**
   - Добавить метод:
     ```python
     def apply_material_transfer_functions(
         self,
         element_list: Sequence[Material],
         pseudo_xray_mode: bool,
         energy: float,
         characteristic_length: float,
     ) -> None
     ```
   - Метод строит и назначает `self.volume_property.SetColor(...)` и `self.volume_property.SetScalarOpacity(...)` без затратного пересоздания структуры `vtkImageData` или актора `vtkVolume`.
2. **Кэширование и флаг режима:**
   - Сохранять ссылку на текущий `element_list` для быстрого пересчета при смене `energy` или `pseudo_xray_mode`.
   - Если активна классическая палитра (например, `Hot Iron`), сохранять традиционный рендеринг через `set_colormap()`.

### 3.3. Контроллер вьюпорта: `gui/controllers/viewport_controller.py`
1. **Синхронизация в `add_or_update_node_actor`:**
   - При обработке `VoxelVolumeViewModel`:
     * Получать `dist = node_vm.core_node.material_distribution`;
     * Если у узла `node_vm.colormap_name == 'Physical Materials'` или активен `self._xray_mode`:
       - Вызывать `self.voxel_renderer.apply_material_transfer_functions(...)` с передачей `dist.element_list`, `self._xray_mode`, `self._xray_energy` и `float(np.mean(node_vm.voxel_size))`;
     * Иначе: вызывать традиционный `set_colormap(node_vm.colormap_name)`.
2. **Динамическое обновление при смене параметров рентгена (`set_xray_parameters`):**
   - В цикле обновления узлов при смене `energy` или `pseudo_xray_mode` обновлять не только `VolumeViewModel`, но и `VoxelVolumeViewModel`, вызывая пересчет передаточных функций фантома.
3. **Рамка выделения фантома (`voxel_selection_box`):**
   - В методе `on_node_selected`:
     * Если выбран `VoxelVolumeViewModel`:
       - Создавать/обновлять каркасный меш `pv.Box` по размерам фантома со смещением `origin`;
       - Добавлять актор `f"selection_box_{id(node_vm)}"` со стилем `wireframe`, цветом `SELECTED_EDGE_HIGHLIGHT_COLOR` и `line_width=2.5`;
       - Обновлять его матрицу трансформации через `update_actor_transform`.
     * Если узел снят с выделения — удалять актор рамки выделения.

### 3.4. Модель представления и инспектор: `VoxelVolumeViewModel` и `PropertyInspector`
1. В [`gui/viewmodels/nodes/voxel_volume_vm.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/nodes/voxel_volume_vm.py):
   - Установить значение по умолчанию:
     `colormap_name = gui_field(default='Physical Materials')`
   - Добавить свойство `@property material_list(self) -> List[Material]`: возвращает `element_list` из `core_node.material_distribution`.
2. В [`gui/views/property_inspector.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/property_inspector.py):
   - В `combo_colormap` первым пунктом добавить `"Physical Materials"`.
   - При выборе `"Physical Materials"` делать спинбоксы эвристической прозрачности (`spin_opacity_thresh`, `spin_max_opacity`, `combo_opacity_preset`) неактивными (`setEnabled(False)`), так как прозрачность рассчитывается автоматически по физическому закону ослабления.

---

## 4. Пошаговый план реализации

- [ ] **Этап 1: Функции построения ступенчатых передаточных функций материалов**
  - Реализовать `build_material_volume_color_tf` и `build_material_volume_opacity_tf` в `gui/viewport_3d/material_palette.py`.
  - Покрыть модульными тестами корректность ступенчатых координат и нулевую прозрачность вакуума/воздуха.
- [ ] **Этап 2: Обновление VoxelVolumeRenderer**
  - Добавить метод `apply_material_transfer_functions` в `VoxelVolumeRenderer`.
  - Обеспечить бесшовное in-place обновление функций без утечек памяти и без сброса камеры.
- [ ] **Этап 3: Поддержка в VoxelVolumeViewModel и PropertyInspector**
  - Добавить `'Physical Materials'` в качестве дефолтной схемы фантома.
  - Адаптировать поля инспектора свойств при выборе физической палитры.
- [ ] **Этап 4: Интеграция в SceneViewportController**
  - Подключить вызов физических передаточных функций в `add_or_update_node_actor`.
  - Обеспечить реакцию фантома на вызов `set_xray_parameters(energy, pseudo_xray_mode)`.
  - Реализовать динамическую рамку выделения `voxel_selection_box` при выборе фантома в графе сцены.
- [ ] **Этап 5: Тестирование и регрессионная верификация**
  - Добавить тесты переключения энергии рентгена и синхронного обновления непрозрачности фантома.
  - Добавить тесты появления и скрытия рамки выделения фантома.
  - Полный запуск pytest (`.venv\Scripts\pytest.exe`) со 100% успехом (291+ тестов).

---

## 5. Критерии приемки

1. По умолчанию воксельный фантом визуализируется в анатомических цветах своей структуры материалов (кость — слоновая кость, мягкие ткани — розовые, жир — бледно-желтый, вода — синяя, воздух/вакуум — прозрачные).
2. При переключении глобального чекбокса `pseudo_xray_mode` фантом синхронно переходит в режим виртуальной рентгенограммы (DRR) вместе со всей сценой без наличия дублирующих настроек у фантома.
3. Изменение рабочей энергии `xray_energy` приводит к физическому пересчету прозрачности вокселей фантома по закону Бугера-Ламберта-Бера $\mu(E)$.
4. При выборе узла фантома в дереве сцены вокруг него во вьюпорте загорается янтарно-оранжевая рамка выделения.
5. 100% успешное прохождение всех модульных и интеграционных тестов проекта в виртуальном окружении `.venv`.
