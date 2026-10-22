# Архитектура расчетного ядра NMSimToolkit (Core Architecture)

Данный документ описывает фундаментальную архитектуру расчетного ядра (**Core Engine**, Python 3.12+) программного комплекса **NMSimToolkit**, включая физико-математическую модель, высокопроизводительный Data-Oriented Design (DoD) транспорт частиц, структуру плоских буферов SoA, компилятор геометрии, реестр метаданных источников и принципы абсолютной физической точности.

---

## 1. Базовые архитектурные принципы и границы

1. **Абсолютная автономность и изоляция от интерфейса:**
   * Каталог `core/` полностью изолирован от графических зависимостей (`PySide6`, `pyqtgraph`, `vtk`, `pyvista`).
   * В физических структурах ядра отсутствуют визуальные атрибуты, флаги инспектора или UI-состояния (цвет, прозрачность, статус выделения).
   * Ядро оперирует исключительно физическими концепциями: объемы (`Volume`), геометрии, материалы (`Material`), сечения взаимодействий, источники частиц (`Source`, `PointSource` в `core/source/sources.py`), плоские буферы геометрии (`GeometryCompiler`) и менеджеры транспорта (`ParticlePropagator`, `SimulationManager`).
   * Подпакеты геометрии (`core/geometry/`) и физики (`core/physics/`) не зависят от конфигурационных Pydantic-схем (`core/config/`).

2. **Нулевой компромисс физической точности (Zero-Compromise Physics):**
   * Никаких приближений или искусственных аппроксимаций при моделировании физических процессов (фотоэффект, комптоновское рассеяние, когерентное рэлеевское рассеяние).
   * Отбор атома-мишени $Z$ в сложных веществах и соединениях (например, $\text{NaI}$, костная и мягкая ткань) выполняется по компактной CSR-структуре элементного состава материалов (`ElementCSR`: массивы `element_offsets`, `element_Z`, `element_fraction` с нормированными массовыми долями элементов, формируемые в `PhysicsCompiler._build_element_csr`).
   * Геометрическая навигация и аналитический RayCasting периодических структур (`PeriodicHexPrism` в `DirectParallelCollimator`) гарантируют математически точное определение точек пересечения границ объемов с компенсацией погрешностей машинной точности.

3. **Конвенция физических единиц (`hepunits`):**
   * Все физические расчеты ведутся строго в согласованной Geant4/HEP системе единиц: длина — **мм** (`mm == 1.0`), энергия — **МэВ** (`MeV == 1.0`), время — **нс** (`ns == 1.0`), плоский угол — **рад** (`rad == 1.0`).
   * Запрещены суффиксы размерностей в именах переменных и параметров (только `energy`, `linear_attenuation`, `step_size`, `gantry_angle`).

4. **C-подобная надежность и Design by Contract (DbC):**
   * Строгий принцип LBYL (Look Before You Leap): валидация физических инвариантов на входе (`energy > 0`, `size > 0`, `voxel_size > 0`).
   * Полный запрет утиной типизации (`hasattr`, `getattr`, `setattr`), глобального ГСЧ (`np.random.*`) и подавляющих блоков `try...except Exception:`.

---

## 2. Парадигма Data-Oriented Design (DoD) и SoA-буферы частиц

Для достижения максимальной производительности в Numba-компилируемых ядрах вычислений ядро `NMSimToolkit` построено на плоской компоновке структур массивов (Structure of Arrays, SoA).

### 2.1 Разделение данных частицы на Hot и Cold потоки
Память частицы (`ParticleBank`) разделена на независимые зоны по частоте обращения в цикле трекинга:

```mermaid
flowchart LR
    subgraph HotState ["Горячие данные (ParticleState - Numba JIT)"]
        direction TB
        H1["position (x, y, z): float64"]
        H2["direction (x, y, z): float64"]
        H3["energy: float64 (Кинетическая энергия)"]
        H4["distance_traveled: float64"]
        H5["species: int8"]
        H6["is_active: bool (Флаг активности частицы)"]
    end

    subgraph ColdMetadata ["Холодные метаданные (InitialState)"]
        direction TB
        C1["ID: int64"]
        C2["emission_position (x, y, z): float64"]
        C3["emission_direction (x, y, z): float64"]
        C4["emission_energy: float64"]
        C5["emission_time: float64"]
    end

    HotState -.->|События взаимодействий / вылета| Out["SimulationDataBuffer / DataManager"]
    ColdMetadata -.->|Запись в InitialStateBuffer по подписке ChunkType| Out
```

* **Горячее состояние (`ParticleState` и `NavigationState`):** упаковано в плоские массивы NumPy/Numba, непрерывно располагающиеся в кэш-памяти процессора (L1/L2). Все операции перемещения и рассеяния выполняются строго in-place без аллокаций памяти в цикле.
* **Независимость потоков ГСЧ:** `SimulationManager` порождает статистически независимые генераторы `np.random.Generator` через `base_rng.spawn()` отдельно для `ParticlePropagator` и каждого активного `Source`.

---

## 3. Компиляция сцены и навигация: `GeometryCompiler`

### 3.1 Плоский буфер геометрии (`geometry_buffer`)
Иерархический граф сцены (`SpatialNode`, `CompositeNode`, `GammaCameraNode`, `GantryNode`, `Volume`, `WoodcockVoxelVolume`, `ParametricParallelCollimator`, `DirectParallelCollimator`) компилируется классом `GeometryCompiler` в плоский типизированный массив структур Numba (`VolumeRecord`):
* Аффинные матрицы трансформации (`local_to_global`, `global_to_local`);
* AABB габариты объемов в локальных координатах (`bounds`);
* Целочисленные идентификаторы материалов и указатели на скомпилированные `cfunc`-функции для параметрических и воксельных объемов (`WoodcockParametricVolume`).

### 3.2 Трассировка Woodcock и аналитический RayCasting
Для гетерогенных объемов (`WoodcockVoxelVolume`, `ParametricParallelCollimator`) применяется метод дельта-трекинга Вудкока (Woodcock Tracking):
* Для каждого составного объема выбирается мажорантный материал `vol.majorant_material` (`majorant_material_map` в `PhysicsCompiler.compile_scene`) как материал с максимальным произведением $Z_{\text{eff}} \cdot \rho$ (`Material.__lt__` в `core/materials/materials.py`).
* Длина свободного пробега разыгрывается по линейному коэффициенту ослабления мажорантного материала $\mu_{\text{maj}}(E)$: $L = -\ln(\xi) / \mu_{\text{maj}}(E)$.
* В точке псевдовзаимодействия вычисляется отношение локального коэффициента ослабления $\mu_{\text{loc}}(E)$ к $\mu_{\text{maj}}(E)$.
* Для прямых периодических гексагональных коллиматоров (`PeriodicHexPrism` в `DirectParallelCollimator`) выполняется точный аналитический RayCasting по ячейкам решетки в `core/geometry/geometry_kernels.py`.

---

## 4. Конфигурация, `SourceMetadataConfig` и инвариант Roundtrip (`core/config/`)

Для соблюдения абсолютной физической чистоты `Source` (отсутствие строковых UI/паспортных атрибутов в физическом узле) и одновременной поддержки 100% симметрии сохранения/загрузки (`export(build(config)) == config`) в `core/config/` реализован внешний реестр метаданных источников:
* В `core/config/models.py` определена иммутабельная схема `SourceMetadataConfig(radionuclide, phantom_type)`.
* `SceneBuilder` при построении узла `Source` из `SourceConfig` сохраняет пару `(radionuclide, phantom_type)` в словаре `source_metadata_registry: Dict[str, SourceMetadataConfig]`.
* `SceneExporter` и `ConfigTemplateBinder` принимают `source_metadata_registry` и при экспорте восстанавливают паспортные поля `radionuclide` и `phantom_type` в `SourceConfig` без загрязнения класса `core.source.sources.Source`.

---

## 5. Система потоковой обработки данных (`DataManager`) и доменная структура `tests/`

```mermaid
sequenceDiagram
    participant Worker as SimulationManager (Numba Kernel)
    participant DB as SimulationDataBuffer
    participant DM as DataManager (Thread)
    participant H5 as HDF5 Storage

    Worker->>DB: Запись начальных состояний, взаимодействий и выбывших частиц
    DB->>DM: Передача чанков через Queue
    DM->>H5: Делегирование в BaseDataHandler (HistoryAssembler / SensitiveVolume / DoseMap)
```

* Обработчики `DirectStreamHandler`, `SensitiveVolumeHandler`, `HistoryAssemblerHandler` и `DoseMapHandler` поддерживают детерминированную финализацию (`finalize()`) и освобождение ресурсов разделяемой памяти (`close()`).
* Обработчики `DirectStreamHandler`, `SensitiveVolumeHandler`, `HistoryAssemblerHandler` пишут в HDF5 через [`DatasetWriteBuffer`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/dataset_write_buffer.py): дозаписи накапливаются по группам датасетов (`initial_states`, `interactions/<объём>`) и сбрасываются целыми чанками за один сеанс `_write_with_retry`; неполный хвост дописывается в `finalize()`. Объём буфера на группу задаёт поле `data_manager.buffer_size` (строка с единицей, по умолчанию `3 MiB`); он определяет размер чанка HDF5 (`chunk_rows = buffer_size // row_bytes`) и порог сброса.
* Ёмкость каналов телеметрии `SimulationDataBuffer` задаёт отдельное поле `simulation_manager.buffer_capacity` (`Optional[int]`, элементы; `None` → `particles_number`), не связанное с выходным буфером записи.
* Все модульные и интеграционные тесты расчетного ядра сгруппированы по доменным подпакетам внутри `tests/`: `tests/config/`, `tests/data/`, `tests/geometry/`, `tests/materials/`, `tests/other/`, `tests/particles/`, `tests/physics/`, `tests/source/` и `tests/transport/`.

