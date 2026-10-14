# Техническая документация NMSimToolkit

Добро пожаловать в репозиторий технической и архитектурной документации программного комплекса **NMSimToolkit** (Python 3.12+) — высокопроизводительного пакета для физико-математического моделирования методами Монте-Карло приборов ядерной медицины (ОФЭКТ, ПЭТ, дозиметрия).

---

## 📚 Основные разделы документации

Вся действующая архитектурная и инженерная документация поддерживается в актуальном виде и сгруппирована по доменам:

1. **[Архитектура расчетного ядра (`core_architecture.md`)](core_architecture.md)**
   * Data-Oriented Design (DoD) и упаковка частиц в Structure of Arrays (SoA);
   * Разделение горячих (`ParticleState`) и холодных данных (`InitialStateBuffer`);
   * Трассировка Вудкока (Woodcock Tracking) по мажорантному материалу объема (`majorant_material_map`, определяемому максимумом $Z_{\text{eff}} \cdot \rho$) и аналитический RayCasting периодических коллиматоров (`PeriodicHexPrism` в `DirectParallelCollimator`);
   * Компиляция плоской геометрии (`GeometryCompiler`), источников (`SourceCompiler`, `Source`, `PointSource`) и физических таблиц (`PhysicsCompiler`, включая CSR-состав элементов `ElementCSR` с нормированными массовыми долями для выбора атомного номера $Z$);
   * Реестр метаданных источников (`SourceMetadataConfig` / `source_metadata_registry`) и строгий `Roundtrip`-инвариант конфигурации;
   * Асинхронный конвейер записи взаимодействий (`DataManager`) в HDF5.

2. **[Архитектура графического интерфейса (`gui_architecture.md`)](gui_architecture.md)**
   * Паттерн MVVM (Model-View-ViewModel), обобщенный `NodeViewModel[TCoreNode]` и медиаторы сессий (`SceneViewportController`, `OrchestratorSession`);
   * Принцип единого источника истины (Single Source of Truth, SSOT) для параметров расчета и привязки переменных (`Property Binding`);
   * 3D-интерактивный вьюпорт на базе VTK/PyVista и UE-Style манипулятор (`TransformGizmo`);
   * Архитектура межпроцессного взаимодействия (IPC) и потоковая телеметрия.

3. **[Кинематическая система и детекторы (`kinematic_system.md`)](kinematic_system.md)**
   * Декларативная слотовая сборка гамма-камеры (`GammaCameraNode` со слотами `casing`, `detector_box`, `collimator`, `crystal`, `glass_backend`);
   * Кинематика станины томографа (`GantryNode`) и расчет орбитальных траекторий ОФЭКТ/ПЭТ в канонических радианах `hepunits`;
   * Контракт кинематических ограничений (`IKinematicConstraint`);
   * Синхронизация 3D Gizmo, панели инструментов и инспектора свойств (`PropertyInspector`).

4. **[Архитектурный бэклог проекта (`architecture_backlog.md`)](architecture_backlog.md)**
   * Реестр реализованных и отложенных архитектурных решений, планов оптимизации и долгосрочных улучшений (согласно Правилу №11 проекта).

---

## 🗂️ Проектные планы и отчеты (`plans/`)

* **[`plans/project_review_plan.md`](plans/project_review_plan.md)** — План и матрица инвариантов комплексного 8-этапного архитектурного и физического аудита проекта.
* **[`plans/project_review_report.md`](plans/project_review_report.md)** — Сводный отчет результатов аудита, карта консолидации доменных пакетов `tests/` и выполненная 5-фазная дорожная карта исправлений.

