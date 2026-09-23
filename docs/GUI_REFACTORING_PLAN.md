# План доработок и архитектурного рефакторинга интеграции GUI (NMSimToolkit)

**Текущий статус плана:** ✅ **ПОЛНОСТЬЮ ВЫПОЛНЕН (Этапы 1–5, Шаги 1–6 завершены, 43/43 тестов пройдены)**

В данном документе зафиксированы результаты архитектурного ревью интеграции GUI, выявленные проблемы, антипаттерны, пошаговый план их устранения, а также отчет об исправлении критических скрытых дефектов и итоговой верификации системы.

---

## 1. Концептуальные принципы рефакторинга

1. **Принцип Fail-Fast для зависимостей и импортов:**
   * Полный отказ от конструкций `try/except ImportError` внутри рабочих модулей GUI. Если обязательная зависимость (PySide6, NumPy, SciPy, PyVista, pyqtgraph) отсутствует, модуль должен завершаться явным исключением на этапе импорта.
   * Единственная точка проверки среды — точка входа [`gui/app.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/app.py). Если среда не готова к запуску GUI, приложение немедленно завершает работу с кодом ошибки и информативным сообщением. Работа в "полуживом" режиме априори недопустима.
2. **Искоренение псевдо-утиной типизации (`hasattr`):**
   * Полное удаление проверок вида `if hasattr(self, 'setCentralWidget')`, `if hasattr(QMessageBox, 'warning')` и подобных.
   * Производственный код не должен адаптироваться под костыльные моки в тестах. Тестовое окружение обязано предоставлять реальный экземпляр `QApplication` (headless-режим / offscreen).
3. **Строгая изоляция слоев (MVVM + Mediator/Session):**
   * Представление ([`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py)) не должно знать деталей IPC, структур разделяемой памяти и конвейера запуска ядра.

---

## 2. Этапы реализации

### Этап 1. Декомпозиция `MainWindow` и создание `SimulationSession` (🔴 Критический приоритет — [ВЫПОЛНЕНО ✅])

**Цель:** Ликвидировать статус God Object у [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py), изолировать сборку вычислительного и телеметрического конвейера.

* **Новый компонент:** [`gui/controllers/simulation_session.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_session.py) ([`SimulationSession`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_session.py#L22))
* **Обязанности:**
  * Создание и владение ресурсами IPC: `multiprocessing.Queue`, блок `multiprocessing.shared_memory.SharedMemory`.
  * Провязка и запуск цепочки:
    ```
    SimulationManager (Core) 
      └──> Queue 
            └──> GuiStreamDataHandler (Core) 
                  ├──> SharedMemory (2D-проекции)
                  └──> track_queue
                        └──> IPCReceiver (GUI QThread) 
                              └──> SimulationSession Signals 
                                    └──> Viewport & ResultsViewer
    ```
  * Взаимодействие с [`SimulationManager`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py), [`GuiStreamDataHandler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/gui_stream_data_handler.py), [`IPCReceiver`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/ipc_receiver.py), [`VTKViewport`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/vtk_viewport.py) и [`ResultsViewer`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/results_viewer.py).
  * Детерминированное освобождение дескрипторов и ресурсов (`close()`, отвязка и `unlink()` shared memory, остановка потоков).
  * Единый фасад управления жизненным циклом: `start()`, `pause()`, `resume()`, `stop()`, `step_once()`.
  * Единый интерфейс Qt-сигналов наружу:
    * `session_started`, `session_paused`, `session_resumed`, `session_stopped`, `session_finished`
    * `session_error(str)`
    * `tracks_received(dict)`
    * `projection_received(object)`
    * `spectrum_received(object)`
    * `stats_updated(int, float)`
* **Результат для [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py):**
  * Сокращение метода `_on_start_simulation()` до инициализации сессии и подписки виджетов на ее сигналы.

---

### Этап 2. Чистка от `hasattr` и `try/except ImportError` (🔴 Высокий приоритет — [ВЫПОЛНЕНО ✅])

**Цель:** Чистая типизация, строгий контракт интерфейсов, Fail-Fast подход.

1. **Модули GUI (`gui/views/*`, `gui/viewmodels/*`, `gui/viewport_3d/*`, `gui/controllers/*`):**
   * Заменить все защитные `try/except ImportError` на прямые импорты на уровне модуля:
     ```python
     from PySide6.QtCore import QObject, Signal, Qt, QTimer, QSize
     from PySide6.QtWidgets import (
         QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
         QDockWidget, QToolBar, QStatusBar, QLabel, QMessageBox, ...
     )
     import pyqtgraph as pg
     import pyvista as pv
     from pyvistaqt import QtInteractor
     ```
   * Удалить все проверки `hasattr(self, ...)` и `hasattr(QMessageBox, ...)` во всех методах:
     * [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py): `__init__`, `_init_components`, `_init_docks`, `_init_menus`, `_init_toolbar`, `_init_statusbar`, `closeEvent`
     * [`SceneTreeWidget`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/scene_tree_widget.py): `__init__`, `_init_ui`, `rebuild_tree`, `_on_node_selected_externally`
     * [`PropertyInspector`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/property_inspector.py): `__init__`, `_init_ui`, `clear_selection`, `update_all_fields`
     * [`ResultsViewer`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/results_viewer.py): `__init__`, `_init_ui`, `set_projection_data`, `clear_results`
     * [`VTKViewport`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/vtk_viewport.py): `__init__`, `_init_plotter`, `update_mesh`
2. **Точка входа ([`gui/app.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/app.py)):**
   * Сохранить строгий Fail-Fast барьер при запуске приложения:
     ```python
     try:
         from PySide6.QtWidgets import QApplication
         from gui.views.main_window import MainWindow
     except ImportError as exc:
         sys.stderr.write(f"Фатальная ошибка: графическое окружение не настроено ({exc}).\n")
         sys.exit(1)
     ```
3. **Тестовая инфраструктура ([`tests/`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests)):**
   * Настроить [`tests/conftest.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/conftest.py) и [`tests/__init__.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/__init__.py) для предоставления реального headless экземпляра `QApplication`:
     ```python
     import os
     os.environ["QT_QPA_PLATFORM"] = "offscreen"
     ```
   * Устранить моки Qt-классов, которые вынуждали использовать `hasattr`.

---

### Этап 3. Оптимизация производительности синхронизации Viewport и Tree (🟡 Высокий приоритет — [ВЫПОЛНЕНО ✅])

**Цель:** Переход от полного сброса и перерисовки $O(N)$ к инкрементальным обновлениям (Dirty Flag / Event-driven).

1. **3D-Сцена ([`MainWindow._sync_viewport_scene`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py) и [`VTKViewport`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/vtk_viewport.py)):**
   * Убрать пересоздание полигональных мешей при изменении координат и ориентации узлов.
   * Реализовать метод `update_actor_transform(actor_name, matrix)` во вьюпорте (обновление матрицы актора через `vtkTransform`).
   * Разделить подписки [`SceneViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/scene_viewmodel.py):
     * `scene_loaded` → полная перестройка акторов.
     * `node_added` → добавление отдельного актора в сцену.
     * `node_removed` → точечное удаление актора по `mesh_{id(node_vm)}`.
     * `transform_changed` узла → точечный вызов `update_actor_transform`.
2. **Дерево сцены ([`SceneTreeWidget`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/scene_tree_widget.py)):**
   * Ввести обратный словарь `self._vm_by_item: Dict[QTreeWidgetItem, NodeViewModel]`.
   * Ликвидировать вложенные поиски $O(N^2)$ в методе `_on_tree_selection_changed`, заменив их на `self._vm_by_item.get(selected_item)`.
   * *(Перспектива)*: Перевод на `QAbstractItemModel` + `QTreeView` для полного исключения `tree.clear()` и сохранения состояния раскрытия ветвей.

---

### Этап 4. Single Source of Truth в `observable_field` (🟡 Средний приоритет — [ВЫПОЛНЕНО ✅])

**Цель:** Устранение дублирования данных и исключение рассинхронизации между ViewModel и ядром.

* **Файл:** [`gui/viewmodels/decorators.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py)
* **Доработки:**
  * Запретить сохранение копии данных в `instance.__dict__` для атрибутов, существующих в `core_node`.
  * Чтение (`__get__`) и запись (`__set__`) свойств модели осуществляются **исключительно** через `core_node`.
  * Разделить поля на:
    * [`core_field(core_attr)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py#L11) — прокси для атрибутов ядра с генерацией Qt-сигналов.
    * [`gui_field(default)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py#L48) — локальные свойства отображения (цвет, статус выделения), отсутствующие в расчетном ядре.
    * [`observable_field(field_name)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py#L86) — реактивный дескриптор, маршрутизирующий изменения через `core_node` при наличии или `gui_field`.

---

### Этап 5. Оптимизация потока `IPCReceiver` (🟡 Средний приоритет — [ВЫПОЛНЕНО ✅])

**Цель:** Исключение холостых циклов опроса (busy-wait) и лишних пробуждений CPU.

* **Файл:** [`gui/controllers/ipc_receiver.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/ipc_receiver.py) ([`IPCReceiver`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/ipc_receiver.py#L26))
* **Доработки:**
  * Заменить неблокирующий опрос `get_nowait()` со `sleep(0.01)` на блокирующее чтение очереди с таймаутом кадра:
     ```python
     item = self.track_queue.get(timeout=self.poll_interval)
     ```
  * Привязать периодическое чтение 2D-проекции из SharedMemory к фактическому таймингу кадров.

---

## 3. График выполнения работ

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ [ВЫПОЛНЕНО ✅] Шаг 1: Настройка headless QApplication в тестах (tests/conftest) │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ [ВЫПОЛНЕНО ✅] Шаг 2: Удаление hasattr и try/except ImportError в GUI-модулях  │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ [ВЫПОЛНЕНО ✅] Шаг 3: Выделение SimulationSession и рефакторинг MainWindow     │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ [ВЫПОЛНЕНО ✅] Шаг 4: Рефакторинг observable_field (Single Source of Truth)    │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ [ВЫПОЛНЕНО ✅] Шаг 5: Инкрементальный VTKViewport и словарь SceneTreeWidget    │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ [ВЫПОЛНЕНО ✅] Шаг 6: Блокирующий IPCReceiver (устранение busy-wait)           │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Исправление критических скрытых дефектов и архитектурных ошибок интеграции

В процессе интеграции компонентов и расширенного приемочного тестирования был выявлен и устранен ряд скрытых архитектурных ошибок, дефектов синхронизации и утечек системных ресурсов:

### 4.1. Иерархия наследования `GammaCameraViewModel` и устранение сброса параметров орбиты
* **Проблема:** Класс [`GammaCameraViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py) наследовался напрямую от базового [`NodeViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py) вместо [`VolumeViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py). Из-за этого гамма-камера не распознавалась полиморфно как объемное тело (отсутствовали свойства `size`, `color`), не отображалась в 3D-вьюпорте как полигональный меш коллиматора/детектора, а [`PropertyInspector`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/property_inspector.py) не имел доступа к геометрии камеры. Кроме того, метод `set_orbit_position` устанавливал значения в локальные атрибуты в обход дескрипторов, что приводило к сбросу значений радиуса и угла на дефолтные при обновлениях.
* **Решение:**
  - Установлено прямое наследование: [`class GammaCameraViewModel(VolumeViewModel)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py#L167).
  - Свойства `orbit_radius` и `orbit_angle` реализованы через реактивные дескрипторы [`gui_field`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py#L48), генерирующие сигналы `property_changed`.
  - Метод [`set_orbit_position(radius, angle_deg)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py#L177) переписан для детерминированного обновления дескрипторов `self.orbit_radius = radius` и `self.orbit_angle = angle_deg` с пересчетом матрицы ориентации узла к центру сцены и записью в `self.local_matrix` без сброса орбитальных параметров.

### 4.2. Корректная остановка `DataManager.stop()`, ликвидация утечки HDF5-потока и стабилизация IPC
* **Проблема:** Поток записи телеметрии [`DataManager`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py) (наследник `threading.Thread`) ожидал порции данных из очереди `queue.get()` в цикле `run()`. В классе отсутствовал метод `stop()`. При закрытии сессии метод [`SimulationSession.close()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_session.py#L187) пытался вызвать `self.data_manager.stop()`, что приводило к `AttributeError: 'DataManager' object has no attribute 'stop'`. Фоновый рабочий поток оставался заблокированным в памяти, вызывая утечку системных ресурсов и дескриптора открытого HDF5-файла на диске. Кроме того, [`IPCReceiver`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/ipc_receiver.py) блокировался на таймауте чтения очереди треков без немедленного пробуждения при остановке, а [`SimulationRunner`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_runner.py) не имел типизированного свойства `is_running`.
* **Решение:**
  - В класс [`DataManager`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py) добавлен метод [`stop(timeout: Optional[float] = 1.0)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py#L54), отправляющий в очередь стоп-маркер `'stop'` и вызывающий `self.join(timeout=timeout)`.
  - При получении маркера цикл `run()` потока завершается; контекстный менеджер `with h5py.File(...) as f` внутри `_write_with_retry` гарантирует корректный сброс буферов и закрытие HDF5-файла.
  - В метод [`IPCReceiver.stop()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/ipc_receiver.py#L122) добавлена отправка маркера `'stop'` через `put_nowait('stop')` в очередь треков, что мгновенно разблокирует ожидающий вызов `track_queue.get(timeout=self.poll_interval)`.
  - В [`SimulationRunner`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_runner.py) добавлено типизированное свойство [`is_running: bool`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_runner.py#L42), исключившее небезопасные проверки через `getattr`.

### 4.3. Устранение скрытого сбоя коннектов сигналов из-за `UniqueConnection` с лямбдами в PySide6
* **Проблема:** В классах [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py) и [`SceneTreeWidget`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/scene_tree_widget.py) при связывании событий модели использовался флаг соединения `Qt.ConnectionType.UniqueConnection` в паре с динамическими лямбда-функциями Python (`lambda node: ...`). В PySide6 каждая лямбда создает новый Python-объект в памяти, из-за чего механизм поиска дубликатов слотов в C++ рантайме Qt не мог корректно сопоставить слот. В результате сигналы молча не подключались, и реактивное обновление интерфейса при манипуляциях со сценой не происходило.
* **Решение:** Флаг `UniqueConnection` удален из вызовов `connect` с анонимными лямбдами. Гарантия отсутствия дублирующих подписок обеспечена архитектурно через детерминированный жизненный цикл компонентов и отслеживание множеств подключений (`_connected_node_ids` в [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py) и `_connected_vms` в [`SceneTreeWidget`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/scene_tree_widget.py)) со сбросом при смене сцены.

### 4.4. Исправление порядка проверки типов в `PropertyInspector`
* **Проблема:** В методе [`PropertyInspector.update_all_fields()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/property_inspector.py#L202) проверка типа `isinstance(self.current_vm, VolumeViewModel)` предшествовала проверке `isinstance(self.current_vm, GammaCameraViewModel)`. Поскольку [`GammaCameraViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py) наследовалась от [`VolumeViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py), первое условие перехватывало обработку в цепочке условий, и специфическая панель настроек ОФЭКТ (`spect_group` с радиусом и углом орбиты) затенялась и не отображалась.
* **Решение:** Проверка `isinstance(self.current_vm, GammaCameraViewModel)` вынесена на первое место перед проверкой базового класса `VolumeViewModel`. Специфические элементы управления гамма-камеры ([`spin_orbit_radius`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/property_inspector.py), [`spin_orbit_angle`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/property_inspector.py)) теперь отображаются корректно вместе с унаследованными свойствами объема.

### 4.5. Связывание сигнала `orbit_changed` 3D-манипулятора ОФЭКТ с `GammaCameraViewModel`
* **Проблема:** 3D-манипулятор ОФЭКТ [`SPECTManipulator`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/spect_manipulator.py) генерировал сигнал `orbit_changed(radius, angle_deg, z_pos)` при интерактивном перемещении камеры пользователем в 3D-вьюпорте, однако в [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py) отсутствовала подписка на данный сигнал. Перемещение манипулятора мышью оставалось локальным эффектом вьюпорта и не транслировалось в саму модель [`GammaCameraViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py), приводя к рассинхронизации числовых полей инспектора свойств и положения камеры в расчетном ядре.
* **Решение:**
  - В [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py) реализован слот [`_on_spect_manipulator_changed(radius, angle_deg, z_pos)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py#L336).
  - В методе `_connect_signals()` установлено соединение: `self.spect_manipulator.orbit_changed.connect(self._on_spect_manipulator_changed)`.
  - При срабатывании манипулятора слот проверяет текущий выделенный узел `self.scene_vm.selected_node` и, если это [`GammaCameraViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py), вызывает `set_orbit_position(radius, angle_deg)`. Это мгновенно обновляет дескрипторы модели, синхронизирует спинбоксы [`PropertyInspector`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/property_inspector.py) и пересчитывает мировую матрицу камеры в ядре.

### 4.6. Подписка на обновление геометрии объема (размеры/материал) и реактивное переименование в дереве сцены
* **Проблема:**
  1. При редактировании пользователем размеров (`size`) или материала/цвета объема в инспекторе свойств актор во вьюпорте не перерисовывался (обновлялась только матрица 4х4 при трансформациях). Пользователь не видел изменений геометрии вплоть до перезагрузки всей сцены.
  2. При редактировании имени узла в инспекторе текстовая метка узла в дереве сцены [`SceneTreeWidget`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/scene_tree_widget.py) не обновлялась без ручного перезапуска или полного сброса дерева.
* **Решение:**
  - В [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py) реализован слот [`_on_node_property_changed(self, node_vm: NodeViewModel, prop_name: str, value: Any)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py#L306): при изменении `prop_name in ('size', 'color')` вызывается `self._add_or_update_node_actor(node_vm)` и `self.viewport.render()`, пересоздавая полигональный меш актора с новой геометрией или цветом без перезагрузки сцены.
  - В [`SceneTreeWidget`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/scene_tree_widget.py) реализован слот [`_on_node_property_changed(self, node_vm: NodeViewModel, prop_name: str, value: Any)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/scene_tree_widget.py#L120): при изменении `prop_name == 'name'` выполняется быстрый $O(1)$ поиск элемента `item = self._item_map.get(id(node_vm))` и вызывается `item.setText(0, str(value))`, мгновенно обновляя текст узла в дереве сцены.

---

## 5. Результаты верификации и статус готовности кодовой базы

### 5.1. Сводные результаты автоматизированного тестирования

Верификация обновленной кодовой базы выполнена полным прогоном модульных, интеграционных и системных тестов:

1. **Полный репозиторный прогон тестов:**
   * Команда: `.venv\Scripts\python -m unittest discover tests`
   * Результат: **43 теста из 43 успешно пройдены (OK)**
   * Время выполнения: ~4.68 с.
2. **Комплексные тесты рефакторинга GUI:**
   * Набор тестов: [`tests/test_gui_refactoring_verification.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/test_gui_refactoring_verification.py) (13 тестов из 13 OK):
     - `test_single_source_of_truth_descriptors` — валидация прямого чтения/записи в `core_node`, исключение дублирования в `__dict__`.
     - `test_simulation_session_lifecycle` — создание конвейера IPC, управление жизненным циклом и детерминированное освобождение ресурсов SharedMemory.
     - `test_scene_tree_widget_reverse_map` — валидация $O(1)$ поиска ViewModel по элементу дерева.
     - `test_vtk_viewport_update_actor_transform` — инкрементальное обновление матрицы трансформации 4х4 через `vtkMatrix4x4` без пересоздания полигонального меша.
     - `test_main_window_incremental_scene_sync` — раздельные обработчики добавления, удаления и перемещения акторов.
     - `test_simulation_session_edge_cases` — идемпотентность `close()`, предотвращение повторного старта, безопасность вызовов до запуска.
     - `test_descriptors_edge_cases` — доступ к дескрипторам через класс, обработка отсутствующих атрибутов ядра.
     - `test_scene_tree_widget_edge_cases` — корректность при пустом дереве и защита корня сцены от удаления.
     - `test_gamma_camera_viewmodel_orbit_sync_and_inheritance` — полиморфизм `VolumeViewModel`, реактивность дескрипторов и отсутствие сброса радиуса/угла орбиты в `set_orbit_position`.
     - `test_data_manager_stop_method` — детерминированная остановка потока `DataManager` маркером `'stop'` без утечек и `AttributeError`.
     - `test_scene_tree_widget_live_rename` — реактивное обновление текста узла в дереве сцены при переименовании ViewModel.
     - `test_main_window_spect_manipulator_coupling` — двусторонняя синхронизация сигнала `orbit_changed` 3D-манипулятора с `GammaCameraViewModel`.
     - `test_main_window_volume_property_changed_sync` — динамическое обновление меша актора вьюпорта при изменении геометрии объема.
3. **Модульные тесты GUI:**
   * Команда: `.venv\Scripts\python -m unittest tests.test_gui_refactoring_verification tests.test_gui_integration_and_fixes tests.test_stage4_views tests.test_stage5_controllers`
   * Результат: **24 теста из 24 успешно пройдены (OK)** (время выполнения ~1.17 с).
4. **Конфигурационные тесты ядра:**
   * Команда: `.venv\Scripts\python -m unittest discover tests/config`
   * Результат: **8 тестов из 8 успешно пройдены (OK)** (время выполнения ~0.57 с).

### 5.2. Статус готовности кодовой базы

Кодовая база подсистемы GUI и архитектурного моста с расчетным ядром NMSimToolkit:
* Полностью переведена на чистую архитектуру MVVM с медиатором [`SimulationSession`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_session.py).
* Очищена от псевдо-утиной типизации (`hasattr`), опасных конструкций подавления импортов `try/except ImportError` и холостых циклов опроса (busy-wait).
* Обеспечивает строгий контракт Single Source of Truth между представлением и расчетным ядром через типизированные дескрипторы [`core_field`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py) и [`gui_field`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py).
* Ликвидированы утечки дескрипторов SharedMemory, зависших потоков HDF5 и блокировок очередей IPC.
* **Итоговый вердикт:** Архитектурный рефакторинг завершен на 100%. Кодовая база **полностью стабилизирована и готова к продакшен-эксплуатации**.

