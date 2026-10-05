# Архитектурное ревью интеграции GUI в NMSimToolkit

**Дата:** 25 сентября 2026 г.  
**Статус подсистемы:** Pre-production / активная разработка  
**Общий вердикт:** Направление верное, архитектурный фундамент (MVVM, изоляция расчетного ядра, Single Source of Truth через дескрипторы) выбран правильно. Сворачивать разработку не требуется. Однако в текущей реализации накопился ряд критических архитектурных нарушений и легаси-решений, требующих обязательного устранения перед расширением функциональности.

---

## 1. Сильные стороны и верные архитектурные решения

1. **Строгое разделение ядра (`core/`) и интерфейса (`gui/`):**
   - Модули расчетного ядра ([`core/scene/nodes.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/scene/nodes.py), [`core/scene/dose_grid_node.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/scene/dose_grid_node.py)) полностью очищены от графических библиотек (`PySide6`, `VTK`, `PyVista`).
   - Свойства отображения (`color`, `visible`, `is_sensitive_detector`, настройки прозрачности) локализованы исключительно в слое моделей представления (`gui/viewmodels/`).
2. **Паттерн MVVM и дескрипторы доступа (Single Source of Truth):**
   - Реализация [`core_field`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py#L53-L102) обеспечивает проксирование изменений напрямую в объекты ядра без дублирования состояния в локальном словаре `__dict__` ViewModel.
   - Реализация [`gui_field`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py#L104-L137) изолирует чисто графические атрибуты.
   - Реактивное оповещение представлений и инспектора свойств через сигнал `property_changed(str, object)`.
3. **Инкрементальная синхронизация с 3D-вьюпортом:**
   - Разделение событий трансформации (`transform_changed`) и глубоких изменений геометрии/свойств.
   - Метод `VTKViewport.update_actor_transform()` обновляет матрицу актора (`user_matrix`) без ресурсоемкого пересоздания полигональной сетки.
4. **Выделенные контроллеры сессий (Mediator):**
   - Перенос логики координации многопроцессного расчета, фонового потока Qt и кольцевого буфера IPC в специализированные классы ([`OrchestratorSession`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/orchestrator_session.py), [`IPCReceiver`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/ipc_receiver.py)).

---

## 2. Критические замечания (Critical)

### 2.1. Антипаттерн God Object: `MainWindow` (979 строк, ~50 КБ)
- **Файл:** [`gui/views/main_window.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py)
- **Проблема:** Класс `MainWindow` перегружен несвойственными представлению обязанностями:
  - Формирование и инициализация графа сцены по умолчанию (`_init_components`, создание `CompositeNode` и `Volume`).
  - Синхронизация акторов 3D-сцены (`_sync_viewport_scene`, `_add_or_update_node_actor`, `_disconnect_node`, `_node_connections`).
  - Расчет кинематики и орбитального позиционирования камер ОФЭКТ (`_on_preview_view_changed`, `_apply_job_angles_to_viewport`).
  - Прямое управление параметрами генерации расчетных задач и сопоставление их со списком задач.
  - Маршрутизация и кэширование геометрии карты дозы (`_active_dose_origin`, `_active_dose_transform_matrix`, `_on_dose_volume_received`).
- **Риски:** Нарушение принципа единственной ответственности (SRP), невозможность изолированного модульного тестирования логики синхронизации вьюпорта без инициализации всего GUI окна.
- **Решение:** Выделить специализированный контроллер синхронизации сцены и вьюпорта — `SceneViewportController` (или `ViewportBridge`), отвечающий за сопоставление `NodeViewModel <-> VTK Actor`, подписки на сигналы узлов и обновление акторов. В `MainWindow` оставить только конфигурацию док-панелей, меню, панелей инструментов и верхнеуровневую маршрутизацию команд.

### 2.2. Нарушение проектных правил типизации: дескриптор `observable_field`
- **Файл:** [`gui/viewmodels/decorators.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/decorators.py#L140-L188)
- **Проблема:** Дескриптор проверяет наличие атрибута ядра через `try: getattr / except AttributeError` и ветвит логику сохранения (в ядро или в `__dict__`). Это скрытая утиная типизация (`hasattr`/`getattr`), прямо запрещенная правилом раздела 3 `project_rules.md`. Размывается понятие Single Source of Truth — вызывающий код не знает, сохраняется ли свойство в расчетное ядро или остается только в UI.
- **Решение:** Полностью ликвидировать класс `observable_field`. Использовать строго либо `core_field` (для атрибутов расчетного ядра), либо `gui_field` (для атрибутов интерфейса).

### 2.3. Дублирование и легаси-код: параллельное сосуществование `SimulationSession` и `OrchestratorSession`
- **Файлы:** [`gui/controllers/simulation_session.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/simulation_session.py) (638 строк) и [`gui/controllers/orchestrator_session.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/orchestrator_session.py) (402 строки)
- **Проблема:** В проекте присутствуют два независимых класса сессий с дублирующимся набором сигналов (`tracks_received`, `projection_received`, `dose_volume_received`, `stats_updated` и т.д.). В `MainWindow` фактически используется `OrchestratorSession`, однако `SimulationSession` продолжает поддерживаться и содержать устаревшую логику сборки конвейера. Это нарушает правило раздела 2 `project_rules.md` («Отсутствие костылей обратной совместимости / No Legacy Crutches»).
- **Решение:** Полностью удалить `SimulationSession` и связанные устаревшие тесты, закрепив единым контрактом сессии `OrchestratorSession`.

---

## 3. Серьезные архитектурные замечания (Major)

### 3.1. Монолитный модуль `node_viewmodel.py` (878 строк)
- **Файл:** [`gui/viewmodels/node_viewmodel.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py)
- **Проблема:** В одном файле сконцентрированы базовый класс `NodeViewModel`, модели объемов (`VolumeViewModel`, `CollimatorViewModel`, `ParametricParallelCollimatorViewModel`, `ParametricParallelSquareCollimatorViewModel`), воксельных фантомов (`VoxelVolumeViewModel`), гамма-камер (`GammaCameraViewModel`), ПЭТ-сканеров (`PetScannerViewModel`), источников излучения (`SourceViewModel`), сеток дозы (`DoseGridViewModel`), фабрика `create_node_viewmodel`, а также внешние процедурные колбэки.
- **Решение:** Декомпозировать пакет `gui/viewmodels/nodes/`:
  - `base_node_vm.py` (`NodeViewModel`)
  - `volume_vm.py` (`VolumeViewModel`, `CollimatorViewModel` и его специализации)
  - `voxel_volume_vm.py` (`VoxelVolumeViewModel`)
  - `gamma_camera_vm.py` (`GammaCameraViewModel`)
  - `source_vm.py` (`SourceViewModel`)
  - `dose_grid_vm.py` (`DoseGridViewModel`)
  - `factory.py` (`create_node_viewmodel`)

### 3.2. Фабрика `create_node_viewmodel` опирается на строковые имена классов
- **Файл:** [`gui/viewmodels/node_viewmodel.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py#L869-L872)
- **Проблема:** Проверка типов сущностей через строковое имя:
  ```python
  if isinstance(core_node, (Source, PointSource)) or core_node.__class__.__name__ in ('Source', 'PointSource', 'I123'):
  if core_node.__class__.__name__ in ('PetScanner', 'PETScanner'):
  ```
  Это признак разрыва в системе типов или циклических импортов в ядре. Нарушает контрактное ООП.
- **Решение:** Ввести в ядре явные базовые абстракции / протоколы (например, базовый класс `PetScanner` в `core/geometry/` и единый `Source` в `core/source/`) и выполнять проверку строго через явный `isinstance`.

### 3.3. Антипаттерн угадывания форматов файлов через перехват исключений
- **Файл:** [`gui/viewmodels/node_viewmodel.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py#L431-L435), строки 776–780
- **Проблема:** Метод `reload_distribution` в `VoxelVolumeViewModel` и `SourceViewModel` дублирует код и пытается загрузить файл через `np.loadtxt`, а при падении откатывается на `np.fromfile`:
  ```python
  try:
      data = np.loadtxt(p).reshape(s, order=order)
  except (ValueError, OSError):
      data = np.fromfile(p, dtype=np.float32).reshape(s, order=order)
  ```
  Прямое нарушение правила раздела 5 `project_rules.md`: «Запрещено применять антипаттерны угадывания форматов через подавление исключений».
- **Решение:** Вынести загрузку матриц распределения в отдельный сервис ядра/утилит `DistributionLoader` со строгой диспетчеризацией по расширению файла (`.npy`, `.raw`, `.dat`) и обязательной LBYL-валидацией объема бинарного буфера байт: `file_size == count * sizeof(dtype)`.

### 3.4. Маскирование ошибок конфигурации и типов через широкие блоки `except Exception: pass`
- **Файл:** [`gui/viewmodels/node_viewmodel.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/node_viewmodel.py#L701-L729)
- **Проблема:** В `SourceViewModel._sync_from_core` содержится четыре подряд блока `except Exception: pass`, подавляющих ошибки распаковки и вычисления активностей, энергий и периодов полураспада. Это нарушает раздел 5 `project_rules.md` («Абсолютный запрет скрытых дефолтов и замалчивания ошибок конфигурации»).
- **Решение:** Четко типизировать структуры данных источника в ядре (`initial_activity`, `energy`) и производить строгую валидацию данных без подавления исключений.

---

## 4. Замечания средней и низкой критичности (Minor / Code Smells)

1. **Неструктурированный словарь параметров симуляции:**
   - В `MainWindow` настройки симуляции хранятся в `Dict[str, Any]` (`self.sim_settings`). Отсутствует статическая типизация и валидация диапазонов.
   - *Рекомендация:* Заменить на строгую модель Pydantic (`GuiSimulationSettings`) или типизированный `dataclass`.
2. **Опечатка в псевдониме визуализатора дозы:**
   - В `MainWindow` (строка 124): `self.dose_viaualizator = self.dose_renderer`. Неиспользуемый alias с ошибкой в имени.
3. **Потеря статической типизации в представлении инспектора:**
   - В `PropertyInspector`: свойство `scene_vm: Optional[Any]` объявлено как `Any`, вместо `Optional[SceneViewModel]`.
4. **Класс-заглушка `PetScannerViewModel`:**
   - В `node_viewmodel.py` класс содержит нетипизированный конструктор `core_node: Any` без привязки к реальным контрактам ПЭТ-геометрии ядра.
5. **Скрытый `hasattr` в сигнальном диспетчере дескрипторов:**
   - В `decorators.py` метод `_emit_property_change` извлекает сигнал через `getattr(instance, 'property_changed', None)` / `try: instance.property_changed except AttributeError`.
   - *Рекомендация:* Описать протокол `IViewModelWithPropertyChanged` (typing Protocol или ABC) с обязательным наличием `property_changed: Signal`.

---

## 5. Дорожная карта исправления архитектуры (Action Plan)

| Этап | Задача | Затрагиваемые модули | Приоритет |
| :--- | :--- | :--- | :--- |
| **Этап 1** | **Ликвидация легаси и нарушений DbC**<br>1. Удалить `SimulationSession` и переключить тесты на `OrchestratorSession`.<br>2. Удалить `observable_field`, заменив на `core_field` / `gui_field`.<br>3. Устранить блоки `try: ... except Exception: pass` в `SourceViewModel`. | `gui/controllers/`<br>`gui/viewmodels/decorators.py`<br>`gui/viewmodels/node_viewmodel.py` | 🔴 Высокий |
| **Этап 2** | **Декомпозиция `node_viewmodel.py`**<br>Разнести специализированные ViewModel узлов в пакет `gui/viewmodels/nodes/`. Создать строгую фабрику на базе явных проверок типов. | `gui/viewmodels/`<br>`gui/viewmodels/nodes/` | 🔴 Высокий |
| **Этап 3** | **Разгрузка `MainWindow` (выделение `SceneViewportController`)**<br>Перенести управление VTK-акторами, подписками на трансформации и ОФЭКТ-манипуляторы из `MainWindow` в выделенный контроллер. | `gui/views/main_window.py`<br>`gui/controllers/viewport_controller.py` | 🔴 Высокий |
| **Этап 4** | **Унификация загрузки воксельных данных**<br>Создать сервис детерминированной загрузки файлов без антипаттерна угадывания форматов (`DistributionLoader`). | `core/data/` или `gui/viewmodels/` | 🟡 Средний |
| **Этап 5** | **Типизация настроек и устранение мелких недочетов**<br>Заменить `sim_settings: dict` на Pydantic-модель, удалить мертвые алиасы, закрыть пробелы в `typing`. | `gui/views/main_window.py`<br>`gui/views/property_inspector.py` | 🟢 Низкий |
