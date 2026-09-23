# Комплексное ревью интеграции GUI NMSimToolkit

## Введение и архитектурный контекст

Графический интерфейс NMSimToolkit реализован на базе стека **PySide6 (Qt)**, **PyVista / VTK (3D-визуализация)** и **pyqtgraph (2D-спектры и проекции)** с применением архитектурного паттерна **MVVM (Model-View-ViewModel)** и выделенного фасада управления сессией моделирования (`SimulationSession`).

Данный отчет содержит результаты детального аудита кодовой базы подсистемы GUI на предмет дефектов надежности, утечек памяти, состояния гонки (race conditions), нарушения принципов чистого кода (Clean Code, SOLID, DRY) и скрытых архитектурных проблем.

---

## 1. Критические дефекты (Bugs & Resource Leaks)

### [BUG-1] Утечка памяти и сигналов Qt в механизме подписки на события узлов
* **Файл:** [`gui/views/main_window.py:289-298, 321-328, 433-439`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py#L289-L298)
* **Суть проблемы:** 
  При добавлении актора узла в `_add_or_update_node_actor` создаются анонимные lambda-обработчики сигналов:
  ```python
  node_vm.transform_changed.connect(lambda n=node_vm: self._on_node_transform_changed(n))
  node_vm.property_changed.connect(lambda prop, val, n=node_vm: self._on_node_property_changed(n, prop, val))
  self._connected_node_ids.add(node_id)
  ```
  При удалении узла (`_on_node_removed`) идентификатор просто удаляется из множества `_connected_node_ids.discard(id(node_vm))`, но сами сигналы `disconnect()` **не вызываются**. Замыкание lambda продолжает удерживать сильную ссылку на объект `node_vm` в памяти Qt-рантайма.
  Аналогично в `_on_new_scene` выполняется `self._connected_node_ids.clear()`, оставляя висячие подписки старой сцены.
* **Последствия:** Утечка памяти при длительной работе с редактированием сцены; фантомные вызовы обработчиков для удаленных узлов; деградация производительности.
* **Рекомендация:** Хранить ссылки на фактические методы/слоты или словари подписок `Dict[int, List[QMetaObject.Connection]]` и явно вызывать `disconnect()` при удалении или сбросе сцены.

---

### [BUG-2] Риск рекурсивного зацикливания обновления трансформаций в PropertyInspector
* **Файл:** [`gui/views/property_inspector.py:183-184, 243-286`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/property_inspector.py#L183-L184)
* **Суть проблемы:**
  Слот `_update_transform_fields` подписан на сигнал `current_vm.transform_changed`:
  ```python
  self.current_vm.transform_changed.connect(self._update_transform_fields)
  ```
  Внутри `_update_transform_fields` меняются значения спинбоксов (`self.spin_x.setValue(...)`), однако флаг `self._is_updating_ui = True` внутри этого метода **не выставляется** (в отличие от `update_all_fields`). В результате изменение значения спинбокса вызывает событие `valueChanged` -> вызывается `_on_transform_changed` -> пересчитывается матрица -> вызывается `current_vm.local_matrix = mat` -> генерируется повторный сигнал `transform_changed`.
* **Последствия:** Потенциальный рекурсивный шторм событий Qt, паразитные перерасчеты и зависание интерфейса при установке координат.
* **Рекомендация:** Обернуть вызовы в `_update_transform_fields` в блок защиты флагом:
  ```python
  self._is_updating_ui = True
  try:
      ... # установка значений спинбоксов
  finally:
      self._is_updating_ui = False
  ```
  либо временно блокировать сигналы спинбоксов через `QSignalBlocker`.

---

### [BUG-3] Деструктивная подмена типа геометрии в `VolumeViewModel.size.setter`
* **Файл:** [`gui/viewmodels/node_viewmodel.py:140-145`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py#L140-L145)
* **Суть проблемы:**
  ```python
  @size.setter
  def size(self, new_size: Sequence[float]) -> None:
      from core.geometry.geometries import Box
      new_size_arr = np.asarray(new_size, dtype=float)
      self.core_node.geometry = Box(new_size_arr[0], new_size_arr[1], new_size_arr[2])
      self.property_changed.emit('size', new_size_arr)
  ```
  Если объем `Volume` содержал геометрию цилиндра (`Cylinder`), сферы (`Sphere`) или параметрического коллиматора, любое изменение размера через ViewModel принудительно заменяет геометрию ядра на экземпляр `Box`.
* **Последствия:** Безвозвратная потеря исходного геометрического типа объекта в расчетном ядре.
* **Рекомендация:** Изменять параметры существующей геометрии полиморфно (например, через специализированные сеттеры или проверяя `isinstance(self.core_node.geometry, Box)`).

---

### [BUG-4] Состояние гонки (Race Condition) в контроллерах потоков
* **Файлы:**
  - [`gui/controllers/ipc_receiver.py:44, 73, 126`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/ipc_receiver.py#L44)
  - [`gui/controllers/simulation_runner.py:39, 66, 90`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_runner.py#L39)
* **Суть проблемы:**
  Флаги `_stop_requested` и `_is_running` объявлены как обычные атрибуты Python `bool`. Они модифицируются из главного потока GUI (`stop()`), а считываются и модифицируются в рабочем цикле `QThread.run()`.
* **Последствия:** Недетерминированное поведение завершения потоков, риск зависания фонового цикла опроса очереди IPC.
* **Рекомендация:** Использовать потокобезопасные примитивы синхронизации (`threading.Event` или атомарные флаги `QAtomicInt` / `threading.Lock`).

---

### [BUG-5] Мертвый код и повреждение логики буфера в `TrackRenderer`
* **Файл:** [`gui/viewport_3d/track_renderer.py:38-40, 68-73, 80-102`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/track_renderer.py#L38-L40)
* **Суть проблемы:**
  В `__init__` создается буфер линий:
  ```python
  self._lines_buffer: Deque[Tuple[int, int]] = deque(maxlen=max_points)
  ```
  В методе `add_tracks_batch` этот буфер заполняется:
  ```python
  p1 = (start_idx + i - 1) % self.max_points
  p2 = (start_idx + i) % self.max_points
  self._lines_buffer.append((p1, p2))
  ```
  Однако в методе `update_mesh()` буфер `_lines_buffer` **вообще не используется**: полигональный меш всегда рендерится как облако некоррелированных точек `style='points'`. Более того, формула вычисления индексов некорректна при переполнении `_point_buffer`, так как `start_idx` вычисляется от текущей длины, а не от абсолютного смещения кольцевого буфера.
* **Последствия:** Бесполезный расход CPU и памяти на поддержание `_lines_buffer`, невозможность отображения непрерывных траекторий частиц линиями.
* **Рекомендация:** Либо полноценно сформировать VTK CellArray для линий и передавать их в `PolyData(pts, lines=...)`, либо удалить неиспользуемый буфер.

---

### [BUG-6] Блокировка главного графического потока в `IPCReceiver.stop()`
* **Файл:** [`gui/controllers/ipc_receiver.py:126-133`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/ipc_receiver.py#L126-L133)
* **Суть проблемы:**
  ```python
  def stop(self) -> None:
      self._stop_requested = True
      if self.track_queue is not None:
          try:
              self.track_queue.put_nowait('stop')
          except Exception:
              pass
      self.wait(1000)
  ```
  Вызов `self.wait(1000)` в основном потоке GUI блокирует весь цикл событий Qt на срок до 1 секунды при каждой остановке симуляции или закрытии окна.
* **Последствия:** Ощутимый «фриз» интерфейса при остановке расчета пользователем.
* **Рекомендация:** Использовать неблокирующее завершение потока по сигналу `finished` либо передавать таймаут 0 с отложенной очисткой (`deleteLater`).

---

### [BUG-7] Некорректная передача параметров в `VoxelVolumeRenderer.set_opacity_threshold`
* **Файл:** [`gui/viewport_3d/voxel_volume_renderer.py:122-136`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/voxel_volume_renderer.py#L122-L136)
* **Суть проблемы:**
  ```python
  def set_opacity_threshold(self, threshold: float) -> None:
      if self.volume_property is not None:
          opacity_tf = to_vtk_piecewise_function(
              min_alpha=0.0,
              max_alpha=0.9,
              ramp_type='step' if threshold > 0.01 else 'linear'
          )
  ```
  Параметр `threshold` не передается в `to_vtk_piecewise_function` в качестве границы порога (он лишь выбирает строковый режим `'step'` / `'linear'`), а также не передается диапазон значений `scalar_range`.
* **Последствия:** Изменение порога прозрачности вокселей в инспекторе свойств фактически не работает и сбрасывает шкалу прозрачности к дефолтной.
* **Рекомендация:** Передавать `threshold` и сохраненный `scalar_range` в генератор кусочно-линейной функции VTK.

---

## 2. Серьезные нарушения чистого кода (Clean Code & Architecture)

### [CLEAN-1] Двойной рендеринг при обновлении положения ОФЭКТ-детектора
* **Файлы:**
  - [`gui/viewport_3d/spect_manipulator.py:44-55`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/spect_manipulator.py#L44-L55)
  - [`gui/views/main_window.py:336-342`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py#L336-L342)
* **Суть проблемы:**
  1. `spect_manipulator.set_orbit_parameters()` вызывает `self.update_visuals()`, внутри которого вызывается `self.viewport.render()`.
  2. Сразу после этого манипулятор эмитирует сигнал `self.orbit_changed.emit(...)`.
  3. `MainWindow._on_spect_manipulator_changed()` перехватывает сигнал и вызывает `node_vm.set_orbit_position(...)`.
  4. Это обновляет `local_matrix` -> эмитируется `transform_changed` -> вызывается `_on_node_transform_changed()` -> повторный `self.viewport.render()`.
* **Последствия:** Проседание частоты кадров в два раза при интерактивном перетаскивании детектора.
* **Рекомендация:** Избавиться от дублирующего вызова рендера в манипуляторе или агрегировать рендеринг через очередь кадра Qt.

---

### [CLEAN-2] Нарушение DRY: Дублирование кинематической матрицы ОФЭКТ
* **Файлы:**
  - [`gui/viewport_3d/spect_manipulator.py:77-94`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/spect_manipulator.py#L77-L94)
  - [`gui/viewmodels/node_viewmodel.py:183-195`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py#L183-L195)
* **Суть проблемы:**
  Математический расчет матрицы поворота и положения детектора (`-sin, -cos / cos, -sin`) скопирован в двух местах проекта.
* **Рекомендация:** Вынести расчет кинематической матрицы в общий хелпер (например, метод `GammaCameraViewModel.compute_orbit_matrix(radius, angle_deg)`).

---

### [CLEAN-3] Хрупкая логика определения типа сканера по подстроке в имени
* **Файл:** [`gui/views/main_window.py:254`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py#L254)
* **Суть проблемы:**
  ```python
  has_pet = any('pet' in node_vm.name.lower() for node_vm in all_nodes)
  ```
  Использование поиска подстроки `'pet'` в имени объекта для активации манипулятора ПЭТ ненадежно. Если пользователь назовет узел `"Phantom_carpet"`, условие сработает ложно.
* **Рекомендация:** Использовать типизированную проверку сущности ядра или специализированную `PetScannerViewModel`.

---

### [CLEAN-4] Неэффективный повторный обход дерева в `_sync_viewport_scene`
* **Файл:** [`gui/views/main_window.py:242-256`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py#L242-L256)
* **Суть проблемы:**
  Метод получает плоский список `all_nodes = self.scene_vm.all_nodes()`, затем итерируется по нему, а затем дважды выполняет генераторы `any(...)` по тому же списку.
* **Рекомендация:** Собрать флаги наличия SPECT/PET за один проход по списку `all_nodes`.

---

### [CLEAN-5] Утечка подписок в `SceneTreeWidget.rebuild_tree`
* **Файл:** [`gui/views/scene_tree_widget.py:82-90, 107-113`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/scene_tree_widget.py#L82-L90)
* **Суть проблемы:**
  Метод `rebuild_tree()` вызывает `self._connected_vms.clear()`, но не отключает ранее подключенные слоты:
  ```python
  vm.property_changed.connect(lambda prop, val, n=vm: self._on_node_property_changed(n, prop, val))
  ```
  При повторных вызовах `rebuild_tree()` накапливаются дублирующиеся обработчики на один и тот же `property_changed`.
* **Рекомендация:** Отключать старые подписки перед очисткой или подключать именованный слот без lambda.

---

### [CLEAN-6] Подавление исключений в дескрипторах реактивности
* **Файл:** [`gui/viewmodels/decorators.py:77-87, 124-135, 189-200`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py#L77-L87)
* **Суть проблемы:**
  Во всех дескрипторах (`core_field`, `gui_field`, `observable_field`) любые ошибки в колбэках `on_change` и при эмиссии сигналов Qt перехватываются пустым `except Exception: pass`.
* **Последствия:** Скрытые ошибки логики UI невозможно диагностировать при отладке.
* **Рекомендация:** Логировать ошибки через модуль `logging.error(..., exc_info=True)`.

---

### [CLEAN-7] Избыточность и дублирование типов дескрипторов
* **Файл:** [`gui/viewmodels/decorators.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py)
* **Суть проблемы:**
  Реализованы три дескриптора: `core_field`, `gui_field` и `observable_field`. При этом `observable_field` полностью покрывает функционал первых двух. В то же время в `node_viewmodel.py` импортируется `observable_field`, но не используется ни в одном классе.
* **Рекомендация:** Унифицировать дескрипторы и удалить неиспользуемые концептуальные дубликаты.

---

### [CLEAN-8] Преждевременный старт фонового потока DataManager в конструкторе сессии
* **Файл:** [`gui/controllers/simulation_session.py:94-100`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_session.py#L94-L100)
* **Суть проблемы:**
  Вызов `self.data_manager.start()` происходит прямо в конструкторе `_setup_pipeline()`, еще до того, как пользователь нажал кнопку запуска и был вызван `session.start()`.
* **Последствия:** Поток HDF5 DataManager запускается и потребляет ресурсы даже если запуск моделирования отменен или завершился ошибкой валидации на этапе сборки.
* **Рекомендация:** Перенести `self.data_manager.start()` в метод `SimulationSession.start()`.

---

### [CLEAN-9] Вызов приватного метода расчетного ядра `_run()`
* **Файл:** [`gui/controllers/simulation_runner.py:73`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_runner.py#L73)
* **Суть проблемы:**
  `SimulationRunner` напрямую вызывает приватный метод `self.manager._run()`.
* **Рекомендация:** Предоставить публичный метод `run()` в интерфейсе `SimulationManager`.

---

## 3. Замечания по стилю, оформлению и чистоте кода (Style & Smells)

1. **[STYLE-1] Магические числа в `MainWindow._on_start_simulation`**:
   Значения `particles_number=5000` и `stop_time=1.0` жестко зашиты в код вместо считывания из настроек или текущей загруженной конфигурации `self.current_config`.
2. **[STYLE-2] Скрытый ленивый импорт в сеттере `VolumeViewModel.material_name`**:
   Импорт `import settings.database_setting as database_setting` внутри сеттера нарушает явность зависимостей.
3. **[STYLE-3] Побочные эффекты на уровне модуля в `results_viewer.py`**:
   Глобальные вызовы `pg.setConfigOption(...)` при импорте модуля влияют на всю среду выполнения. Инициализацию стилей следует вынести в `app.py`.
4. **[STYLE-4] Отсутствующие графические ресурсы в QSS**:
   В `DARK_STYLE_SHEET` (`app.py:34-35`) прописаны `url(close.png)` и `url(float.png)`, отсутствующие в репозитории.
5. **[STYLE-5] Чрезмерная гранулярность перерисовки инспектора свойств**:
   `_on_property_changed_externally` при изменении любого скалярного свойства вызывает полный метод `update_all_fields()`, пересчитывающий все поля формы и углы Эйлера.
6. **[STYLE-6] Неиспользуемые свойства-делегаты в `MainWindow`**:
   Геттеры `simulation_runner`, `ipc_receiver`, `stream_handler`, `track_queue` нигде не используются внутри `MainWindow`.
7. **[STYLE-7] Неиспользуемое состояние `current_config`**:
   Атрибут `self.current_config` сохраняется при загрузке YAML, но никак не задействуется при старте симуляции.
8. **[STYLE-8] Недетерминированная генерация имен по умолчанию в SceneTreeWidget**:
   Генерация `Volume_{len(self._item_map) + 1}` при удалении промежуточных узлов приводит к дубликатам имен в сцене.
9. **[STYLE-9] Проблема сингулярности углов Эйлера (Gimbal Lock) в инспекторе**:
   Декомпозиция матрицы через `Rotation.from_matrix(rot_mat).as_euler('xyz', degrees=True)` подвержена скачкам значений при углах тангажа ±90°.
10. **[STYLE-10] Утечка элементов легенды в ResultsViewer**:
    В `_init_ui` вызывается `self.profile_plot.addLegend()`. При последующих вызовах `clear()` и повторном `plot(..., name=...)` в старых версиях pyqtgraph легенда накапливает повторяющиеся элементы.

---

## 4. Сводная матрица выявленных дефектов

| Идентификатор | Категория | Файл | Влияние | Приоритет |
|---|---|---|---|---|
| **BUG-1** | Утечка ресурсов | `gui/views/main_window.py` | Утечка памяти при работе со сценой | 🔴 Критический |
| **BUG-2** | Надежность | `gui/views/property_inspector.py` | Риск рекурсивного зацикливания Qt-сигналов | 🔴 Критический |
| **BUG-3** | Целостность данных | `gui/viewmodels/node_viewmodel.py` | Потеря исходного типа геометрии (Box, Cylinder, Sphere) | 🔴 Критический |
| **BUG-4** | Многопоточность | `gui/controllers/*.py` | Состояние гонки без примитивов синхронизации | 🔴 Критический |
| **BUG-5** | Логика / Производительность | `gui/viewport_3d/track_renderer.py` | Мертвый буфер линий и утечка вычислений | 🟠 Серьезный |
| **BUG-6** | UX / Отзывчивость | `gui/controllers/ipc_receiver.py` | Блокировка GUI-потока на 1000 мс при остановке | 🟠 Серьезный |
| **BUG-7** | Функциональность | `gui/viewport_3d/voxel_volume_renderer.py` | Нерабочий регулятор порога прозрачности вокселей | 🟠 Серьезный |
| **CLEAN-1** | Производительность | `gui/viewport_3d/spect_manipulator.py` | Двойной рендеринг кадра при манипуляциях | 🟠 Серьезный |
| **CLEAN-2** | DRY | `spect_manipulator.py` / `node_viewmodel.py` | Дублирование кинематических формул детектора | 🟠 Серьезный |
| **CLEAN-3** | Надежность | `gui/views/main_window.py` | Определение сканера по подстроке в имени | 🟠 Серьезный |
| **CLEAN-5** | Утечка сигналов | `gui/views/scene_tree_widget.py` | Накопление lambda-подписок при перестройке дерева | 🟠 Серьезный |
| **CLEAN-6** | Отладка | `gui/viewmodels/decorators.py` | Полное подавление ошибок в реактивных полях | 🟠 Серьезный |
| **CLEAN-8** | Жизненный цикл | `gui/controllers/simulation_session.py` | Преждевременный старт DataManager до запуска | 🟡 Средний |
| **CLEAN-9** | Инкапсуляция | `gui/controllers/simulation_runner.py` | Вызов приватного метода ядра `_run()` | 🟡 Средний |
| **STYLE-1..10** | Чистый код | Различные файлы GUI | Читаемость, согласованность и архитектурная чистота | 🟡 Средний |
