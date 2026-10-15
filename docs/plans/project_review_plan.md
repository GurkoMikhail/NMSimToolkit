# План комплексного архитектурного и физического ревью NMSimToolkit

Данный документ определяет структуру, критерии и пошаговый план полного аудита кодовой базы `NMSimToolkit-Refactor` (222 Python-файла, 287 классов, 1826 методов) на соответствие архитектурным, физическим и контрактным инвариантам проекта.

> [!IMPORTANT]
> **Статус аудита:** Все 8 этапов (Этап 0 + Блоки 1–7) **выполнены в полном объеме**. Полный сводный отчет с классификацией дефектов и дорожной картой исправлений зафиксирован в [`docs/plans/project_review_report.md`](project_review_report.md).

---

## 1. Методология и регламент проведения ревью

### 1.1. Итеративный поблочный подход
Аудит выполняется последовательно по 8 изолированным этапам (Этап 0 + 7 доменных блоков). Каждый блок проходит полный цикл:
1. **Графовый и статический анализ (`codebase-memory-mcp`, `mypy`):** проверка связей, зависимостей, сигнатур и вызовов внутри подсистемы.
2. **Глубокая инспекция кода:** построчная проверка алгоритмов, математических моделей, контрактов LBYL и типизации.
3. **Отчет по блоку:** формирование структурированного перечня замечаний с разбивкой по критичности:
   * 🔴 **Критические дефекты (Critical / Physics & Contract Violations):** ошибки физической модели, рассогласование размерностей `hepunits`, нарушение инварианта `Roundtrip`, утечки памяти/сигналов, нарушение изоляции `core/` и `gui/`, аллокации в `@njit`-ядрах.
   * 🟡 **Архитектурные и контрактные замечания (Major / Code Quality & Typing):** использование `hasattr`/`getattr`, широкие `try...except`, дублирование логики, неполная типизация, однобуквенные идентификаторы, суффиксы единиц измерения.
   * 🔵 **Стратегические улучшения (Backlog Candidates):** масштабные идеи и оптимизации, подлежащие фиксации в [`docs/architecture_backlog.md`](../architecture_backlog.md).
4. **Устранение замечаний и верификация:** внесение согласованных исправлений, прогон профильных тестов через `.venv\Scripts\pytest.exe` и обновление индекса графа (`index_repository`).

### 1.2. Базовая матрица проверяемых инвариантов
Каждый модуль проверяется по чек-листу правил проекта ([`.agents/rules/project_rules.md`](../../.agents/rules/project_rules.md)):
* [x] **Изоляция слоев:** отсутствие импортов `PySide6`, `vtk`, `pyvista`, `pyqtgraph` и UI-атрибутов в `core/`; отсутствие импортов `core.config.*` в `core/geometry/` и `core/physics/`.
* [x] **Физика и `hepunits`:** базовые единицы (`mm == 1.0`, `MeV == 1.0`, `ns == 1.0`, `rad == 1.0`), отсутствие суффиксов единиц в именах (`_kev`, `_mm`, `_deg`, `_sec`), отсутствие физических аппроксимаций (*Zero-Compromise Physics*).
* [x] **Hot-Path & RNG:** отсутствие аллокаций и логирования внутри `@njit`-кернелов (`*_kernels.py`) и шага транспорта; отсутствие вызовов глобального `np.random.*` / `random.*`; отсутствие реэкспорта `*_kernels.py` в `__init__.py`.
* [x] **Контракты (DbC / LBYL):** отсутствие `hasattr`, `getattr`, `setattr`; валидация предусловий до выполнения; отсутствие подавляющих `try...except Exception:` и скрытых дефолтов (Fail-Fast).
* [x] **Roundtrip & Data Purity:** замкнутый цикл `Pydantic Config ➔ SceneBuilder ➔ Core Node ➔ SceneExporter ➔ ViewModel`; отсутствие неразрешенных макросов `${...}` внутри строителей ядра.
* [x] **Стиль, SRP и типизация:** отсутствие псевдо-хелперов `*_helper.py` с передачей `Any`; только Top-Level импорты (и `if TYPE_CHECKING:`); отсутствие однобуквенных имен; полные аннотации типов и русскоязычные docstrings.

---

## 2. Декомпозиция ревью по этапам и блокам

### [x] Этап 0. Глобальный автоматизированный скрининг инвариантов
**Цель:** Быстрое выявление формальных нарушений контрактов по всему репозиторию перед погружением в модули.
* **Задачи:**
  1. Поиск по графу `codebase-memory` всех вхождений `hasattr`, `getattr`, `setattr`, `isinstance`.
  2. Проверка кросс-пакетных ребер `IMPORTS` в графе: отсутствие связей `core -> gui`, `core.geometry -> core.config`, `core.physics -> core.config`.
  3. Аудит всех идентификаторов (функций, методов, параметров, полей) на наличие запрещенных суффиксов единиц (`_kev`, `_mev`, `_ev`, `_mm`, `_cm`, `_m`, `_deg`, `_rad`, `_sec`, `_s`, `_ns`) и однобуквенных имен.
  4. Проверка всех `__init__.py` на наличие `__all__` и отсутствие реэкспорта `*_kernels.py`.
  5. Поиск вызовов глобального ГСЧ (`np.random.rand`, `np.random.choice`, `np.random.seed`, `random.`).
  6. Контрольный запуск `.venv\Scripts\mypy.exe` и `.venv\Scripts\pytest.exe` для фиксации базовой линии (baseline).

---

### [x] Блок 1. Сцена, Геометрия и Пространственная математика
**Охват (~24 файла):** `core/scene/`, `core/geometry/`, `core/other/`
* **Проверяемые модули:**
  * Узлы сцены: `core/scene/nodes.py`, `core/scene/gantry_node.py`, `core/scene/gamma_camera_node.py`, `core/scene/dose_grid_node.py`.
  * Геометрия и компиляция: `core/geometry/geometries.py`, `core/geometry/volumes.py`, `core/geometry/voxel_volumes.py`, `core/geometry/woodcock_volumes.py`, `core/geometry/parametric_collimators.py`, `core/geometry/direct_collimators.py`, `core/geometry/pet_scanners.py`, `core/geometry/spect_kinematics.py`, `core/geometry/flattened_scene.py`, `core/geometry/geometry_compiler.py`, `core/geometry/geometry_kernels.py`, `core/geometry/navigation_state.py`.
  * Математика и утилиты: `core/other/transform.py`, `core/other/vectors.py`, `core/other/utils.py`, `core/other/nonunique_array.py`, `core/other/typing_definitions.py`.
* **Ключевые вопросы аудита:**
  * Строгость иерархии `CompositeNode -> SpatialNode -> Volume` и корректность инвалидации матриц трансформации при мутациях дерева `Parent-Child`.
  * Размещение геометрических свойств (`local_bound`) исключительно в `Volume`, а не в `SpatialNode`.
  * Чистота слотовой архитектуры `GammaCameraNode` и кинематики `GantryNode`.
  * Точность аналитического RayCasting, расчетов пересечений с границами объемов и обработки `PeriodicHexPrism` / `DirectCollimator` в `geometry_kernels.py` (отсутствие пропуска границ и аллокаций).
  * Математическая строгость операций в `transform.py` и `utils.py` (валидация матриц перед декомпозицией Эйлера).

---

### [x] Блок 2. Материалы, Частицы, Источники и Физика транспорта
**Охват (~32 файла):** `core/materials/`, `core/particles/`, `core/source/`, `core/physics/`, `core/transport/`
* **Проверяемые модули:**
  * Материалы: `core/materials/materials.py`, `material_database.py`, `material_bank.py`, `attenuation_database.py`, `attenuation_functions.py`, `atomic_properties.py`.
  * Частицы (SoA): `core/particles/particles.py`, `particles_kernels.py`, `initial_state.py`, `kinematic_state.py`.
  * Источники: `core/source/sources.py`, `source_compiler.py`.
  * Физические процессы: `core/physics/processes.py`, `processes_kernels.py`, `physics_kernels.py`, `physics_compiler.py`, `physics_buffer.py`, `interaction_buffers.py`, `g4compton.py`, `g4coherent.py`, `g4coherent_arrays.py`, `scat_func_fit_param_array.py`.
  * Транспорт: `core/transport/propagator.py`, `simulation_managers.py`, `transport_buffer.py`, `transport_kernels.py`, `ipc_pause_bridge.py`.
* **Ключевые вопросы аудита:**
  * Соблюдение *Zero-Compromise Physics* в кернелах Комптона, Рэлея, фотоэффекта и Woodcock-трекинга.
  * Согласованность размерностей `hepunits` при чтении таблиц NIST, вычислении макроскопических сечений $\mu(E)$ и порогов энергии.
  * Отсутствие аллокаций памяти, создания временных массивов и вызовов логгера на горячем пути (`Propagator._step`, `transport_kernels.py`, `physics_kernels.py`).
  * Корректность работы SoA-буферов (`ParticleBank`, `DeadParticlesBuffer`) и детерминированность ГСЧ во всех источниках и физических процессах.

---

### [x] Блок 3. Конфигурация, Оркестрация, Сбор данных и Инфраструктура
**Охват (~26 файлов):** `core/config/`, `core/data/`, `infra/`, `settings/`, `run_simulation.py`
* **Проверяемые модули:**
  * Конфигурация и оркестрация: `core/config/models.py`, `builder.py`, `exporter.py`, `sweep_compiler.py`, `template_binder.py`, `orchestrator.py`, `simulation_worker.py`, `units.py`, `yaml_loader.py`, `yaml_dumper.py`, `logging_setup.py`.
  * Данные и HDF5: `core/data/data_manager.py`, `data_handlers.py`, `dose_map_handler.py`, `distribution_loader.py`, `metadata_collector.py`.
  * Инфраструктура и настройки: `infra/monitoring/telegram.py`, `settings/database_setting.py`, `processes_settings.py`, `telegram_bot_settings.py`, `run_simulation.py`.
* **Ключевые вопросы аудита:**
  * Полнота и симметрия Roundtrip-контура (`models.py <-> builder.py <-> exporter.py`) для всех типов узлов, источников, коллиматоров, детекторов и процедур.
  * Соблюдение *Data Purity*: гарантия того, что `template_binder.py` и `sweep_compiler.py` полностью разрешают макросы `${...}` до вызова `SceneBuilder`, а неразрешенный макрос вызывает немедленный `ValueError`.
  * Надежность мультипроцессной оркестрации (`orchestrator.py`, `simulation_worker.py`), передача `SeedSequence`, обработка ошибок воркеров и корректность записи в HDF5 (`data_manager.py`, `metadata_collector.py`).
  * Ревизия каталога `settings/` на предмет дублирования или устаревших настроек относительно `core/config/` и `infra/monitoring/`.

---

### [x] Блок 4. Логика GUI: Models, ViewModels, Factories и Controllers
**Охват (~28 файлов):** `gui/models/`, `gui/factories/`, `gui/viewmodels/`, `gui/controllers/`
* **Проверяемые модули:**
  * Модели и фабрики: `gui/models/gui_settings.py`, `distribution_import_params.py`, `gui/factories/gamma_camera_factory.py`.
  * ViewModels: `gui/viewmodels/scene_viewmodel.py`, `procedure_viewmodel.py`, `data_handler_viewmodel.py`, `decorators.py` и все модули в `gui/viewmodels/nodes/` (`base_node_vm.py`, `volume_vm.py`, `voxel_volume_vm.py`, `source_vm.py`, `collimator_vm.py`, `gamma_camera_vm.py`, `gantry_vm.py`, `pet_scanner_vm.py`, `dose_grid_vm.py`, `factory.py`).
  * Контроллеры: `gui/controllers/orchestrator_session.py`, `simulation_runner.py`, `ipc_receiver.py`, `stream_handlers.py`, `viewport_controller.py`.
* **Ключевые вопросы аудита:**
  * Строгость границ MVVM: отсутствие прямых зависимостей `ViewModel` от виджетов `PySide6.QtWidgets` (модальных диалогов и т.д.).
  * Отсутствие паразитных рекурсий и дублирующих сигналов синхронизации между узлами ядра и `NodeViewModel`.
  * Консистентность работы `Property Binding`, стандартных переменных процедур (`SpectProcedureViewModel`, `StepAndShootProcedureViewModel`, `CustomSweepProcedureViewModel`) и сессии оркестратора (`orchestrator_session.py`).
  * Потокобезопасность `ipc_receiver.py` и корректное завершение дочерних процессов в `simulation_runner.py`.

---

### [x] Блок 5. 3D-вьюпорт, Рендереры и Кинематические манипуляторы
**Охват (13 файлов):** `gui/viewport_3d/`
* **Проверяемые модули:**
  * Вьюпорт и рендереры: `vtk_viewport.py`, `voxel_volume_renderer.py`, `dose_volume_renderer.py`, `track_renderer.py`, `collimator_hole_renderer.py`, `material_palette.py`, `dicom_colormaps.py`.
  * Гизмо и кинематика: `transform_gizmo.py`, `gizmo_types.py`, `kinematic_constraints.py`, `spect_manipulator.py`, `pet_manipulator.py`.
* **Ключевые вопросы аудита:**
  * Жизненный цикл VTK-актеров и рендереров при перезагрузке сцены, удалении узлов и переключении режимов отображения (отсутствие висящих актеров и утечек памяти).
  * Корректность работы кинематических ограничений `IKinematicConstraint`, `TransformGizmo`, `SpectManipulator` и `PetManipulator` при интерактивном перемещении узлов.
  * Соблюдение стандартов `hepunits` (например, `xray_energy`) и корректность визуализации воксельных фантомов, дозовых сеток и треков частиц.

---

### [x] Блок 6. Графические представления, Инспекторы свойств и Виджеты
**Охват (33 файла):** `gui/views/`, `gui/views/inspectors/`, `gui/views/widgets/`, `gui/app.py`
* **Проверяемые модули:**
  * Главное окно и панели: `gui/app.py`, `gui/views/main_window.py`, `property_inspector.py`, `scene_tree_widget.py`, `results_viewer.py`, `simulation_jobs_widget.py`, `simulation_settings_dialog.py`, `distribution_import_dialog.py`, `procedure_selector_widget.py`, `data_handler_list_widget.py`.
  * Секции инспектора (`gui/views/inspectors/`): `base_section.py`, `inspector_registry.py`, `general_section.py`, `transform_section.py`, `volume_section.py`, `voxel_volume_section.py`, `source_section.py`, `collimator_section.py`, `gamma_camera_section.py`, `gantry_section.py`, `dose_grid_section.py`, `procedure_section.py`, `data_manager_section.py`, `data_handler_section.py`.
  * Виджеты (`gui/views/widgets/`): `bindable_field.py`, `grid_chips_widget.py`, `grid_variable_widget.py`, `sensitive_volumes_editor.py`.
* **Ключевые вопросы аудита:**
  * Строгое отслеживание подписок Qt-сигналов (`_is_connected`), отсутствие двойных подписок или обращений к удаленным C++ объектам Qt при смене выбранного узла.
  * Корректность перевода физических единиц через константы `hepunits` на границе между `ViewModel` и спинбоксами/`BindableDoubleField`.
  * Целостность классов секций инспектора (SRP, отсутствие утечки `Any`), LBYL-валидация пользовательского ввода.

---

### [x] Блок 7. Тестовый контур, Примеры, Конфигурации и Документация
**Охват (~60+ файлов):** `tests/`, `examples/`, `docs/`, корневые скрипты и `*.yaml`
* **Проверяемые модули:**
  * Тестовый набор `tests/`: доменные папки (`tests/config/`, `tests/data/`, `tests/geometry/`, `tests/other/`, `tests/particles/`, `tests/physics/`, `tests/source/`, `tests/transport/`, `tests/viewmodels/`, `tests/viewport_3d/`, `tests/views/`, `tests/controllers/`) и корневые файлы `tests/test_*.py`, `tests/scratch_debug.py`.
  * Примеры и корневые файлы: `examples/*.py`, `tb161_nema_simulation.py`, `nema_1_cam.yaml`, `simulation_config.yaml`, `tb161_nema_protocol.yaml`, `tb161_nema_protocol direct.yaml`.
  * Документация: `docs/README.md`, `docs/core_architecture.md`, `docs/gui_architecture.md`, `docs/kinematic_system.md`, `docs/architecture_backlog.md`, `docs/plans/`.
* **Ключевые вопросы аудита:**
  * План консолидации и очистки корня `tests/`: перенос актуальных тестов из файлов-однодневок (`test_all_14_gui_fixes.py`, `test_gui_integration_fixes.py`, `test_stage*.py` и др.) в профильные доменные пакеты и удаление `scratch_debug.py`.
  * Проверка работоспособности скриптов в `examples/` и актуальности YAML-конфигураций в корне проекта (отсутствие устаревшего API до рефакторинга).
  * Сверка документации в `docs/` с фактическим состоянием кодовой базы.

---

## 3. Фазы устранения замечаний (Remediation Phases)

* [x] **Фаза 1:** Исправление критических ошибок физики, геометрии и транспорта в `core/` (Блоки 1 и 2).
* [x] **Фаза 2:** Унификация единиц в протоколах, восстановление Roundtrip-инварианта (`SourceMetadataConfig` / `source_metadata_registry`) и надежности HDF5/IPC в `core/config/` и `core/data/` (Блок 3).
* [x] **Фаза 3:** Исправление дефектов `gui/models/`, `gui/viewmodels/`, `gui/controllers/` и `gui/viewport_3d/` (Блоки 4 и 5).
* [x] **Фаза 4:** Исправление дефектов `gui/views/`, секций инспектора, виджетов и приведение к чистому `mypy` на Python 3.12 (Блок 6).
* [x] **Фаза 5:** Консолидация `tests/`, восстановление `examples/*.py`, `tb161_nema_simulation.py`, корневых `*.yaml`, удаление последнего легаси `Tc99m_MIBI` и актуализация `docs/` (Блок 7).

