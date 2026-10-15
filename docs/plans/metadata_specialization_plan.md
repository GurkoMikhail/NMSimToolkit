# План 1: Рефакторинг и специализация метаданных HDF5 (Отказ от fallback-значений)

## 1. Цель и архитектурная концепция

Заменить универсальные дефолтные/fallback-значения (`0.0`, `"None"`) и дублирующиеся ключи в HDF5-метаданных на **строго специализированные схемы**, зависящие от фактического типа протокола исследования и состава сцены.

**Ключевой инвариант:**
В HDF5 записываются только первичные физические факты (Ground Truth), которые достоверно известны симулятору. Если параметр не применим к текущему протоколу или не задан в конфигурации — атрибут **не создается в HDF5 вообще**. Любые косвенные геометрические или кинематические аппроксимации (например, вычисление эквивалентного радиуса орбиты из матриц $4 \times 4$ для произвольного `CustomSweep`) делегируются внешним пост-обработчикам и конвертерам (DICOM NM / CASToR / STIR).

---

## 2. Декомпозиция изменений по компонентам

### Этап 1. Специализация провайдеров протокола и кинематики гантри (`core/data/metadata_collector.py`)

1. **`ProtocolMetadataProvider.collect()`:**
   * **Для `StepAndShootProtocolConfig`:**
     * Записывать: `modality = "StepAndShoot"`, `view_index`, `views_total = int(protocol.steps)`, `exposure_time = float(protocol.time_per_step) / float(units.s)` (привести к секундам согласованно с `units_map`).
     * Поле `orbit_radius` включать в словарь **только если `protocol.radius is not None`** (переводя из `units.mm`). Если `protocol.radius is None` — ключ `orbit_radius` не создается.
   * **Для `CustomSweepProtocolConfig`:**
     * Записывать только: `modality = "CustomSweep"`, `view_index`, `views_total`.
     * **Удалить** хардкод `"exposure_time": 0.0` и `"orbit_radius": 0.0`.
   * **Для отсутствующего протокола (`protocol is None`):**
     * Записывать только `modality = "Static"`, `view_index = 0`, `views_total = 1` без нулевых заглушек.

2. **`GantryMetadataProvider.collect()`:**
   * Добавить аргумент `sim_config: Optional[SimulationManagerConfig] = None` и `actual_stop_time: Optional[float] = None`, `actual_start_time: Optional[float] = None`.
   * **Угол гантри (`gantry_angle`):**
     * Записывать **только если** в сцене присутствует узел `GantryNode` либо угол явно задан в протоколе `StepAndShoot` / контексте. Если `GantryNode` отсутствует и угол не задан — не писать фиктивный `0.0`.
   * **Параметры круговой томографии (`orbit_radius`, `rotation_direction`, `angular_step`):**
     * Записывать **исключительно** при `isinstance(protocol, StepAndShootProtocolConfig)`.
     * Внутри `StepAndShoot`: `orbit_radius` записывать только при `protocol.radius is not None`.
     * Для `CustomSweep` и статических сцен эти ключи в `metadata/gantry` **не создаются**.
   * **Время экспозиции (`frame_duration`):**
     * Вычислять из фактического интервала времени задачи:
       $$\text{frame\_duration} = \frac{\text{actual\_stop\_time} - \text{actual\_start\_time}}{\text{units.s}}$$
       (где по умолчанию до завершения задачи используются `sim_config.stop_time` и `sim_config.start_time`, а при финализации — фактический `manager.global_timer`).
   * **Матрицы трансформации детекторов (`pose_matrix_detector_{i}`):**
     * Записывать строго под каноническими ключами `pose_matrix_detector_1`, `pose_matrix_detector_2`, ... (по индексу `GammaCameraNode`).
     * **Удалить** дублирующую запись `pose_matrix_{cam_name}` (`pose_matrix_GammaCamera_1` и т.д.).
     * При необходимости сохранения пользовательского имени узла записывать строковый атрибут `detector_{i}_name = camera_node.name`.

---

### Этап 2. Устранение дублирования в `GeometryMetadataProvider` (`core/data/metadata_collector.py`)

1. **Параметры кристаллов и коллиматоров:**
   * При отсутствии камер (`heads_count == 0`): записывать только `detector_heads_count = 0` (удалить запись нулевых массивов `crystal_dimensions = [0,0,0]` и строк `"None"`).
   * При наличии одной или нескольких **идентичных** камер (одинаковые размеры кристалла и параметры коллиматора):
     * Записывать единый канонический набор в `metadata/geometry`:
       `crystal_dimensions`, `collimator_type`, `collimator_hole_diameter`, `collimator_hole_length`, `collimator_septa_thickness`, `collimator_hole_shape`, `collimator_material`.
     * **Удалить** дублирующие ключи с суффиксами `_{cam_name}` (`crystal_dimensions_GammaCamera_1`, `collimator_hole_diameter_GammaCamera_2` и т.д.).
   * При наличии **различающихся** камер (разные кристаллы или коллиматоры на головках):
     * Записывать параметры каждой головки в структурированные подгруппы `metadata/geometry/detector_1/...`, `metadata/geometry/detector_2/...`.
2. **Изоляция физических объемов сцены:**
   * Перенести записи всех `Volume` сцены из плоского корня `metadata/geometry` во вложенную подгруппу `metadata/geometry/volumes/{volume_name}`, чтобы не смешивать паспорт детектора с узлами фантома и корпуса.

---

### Этап 3. Нормализация паспорта источника и очистка структуры задачи (`core/data/metadata_collector.py`, `core/data/data_manager.py`)

1. **`SourceMetadataProvider.collect()`:**
   * Округлять период полураспада при переводе в секунды для устранения артефактов IEEE-754 после конвертации `day -> ns -> s`:
     ```python
     half_life_val = round(float(primary_source.half_life) / float(units.s), 6)
     ```
     (для $^{161}\text{Tb}$ с $6.89\text{ сут}$ даст строго `595296.0` вместо `595295.9999999999`).
   * Аналогично округлять `simulated_activity` до разумной точности (`round(..., 6)`).
2. **Очистка структуры `/metadata/tasks/task_{task_id}`:**
   * Убрать из `ProcedureMetadataCollector.collect()` избыточный дублирующий блок `acquisition: {protocol, kinematics, detectors, geometry}`, который копировал все корневые метаданные внутрь каждой задачи по 2–3 раза.
   * В `/metadata/tasks/task_{task_id}` оставлять только специфичные для данного запуска параметры: `task_id`, `status`, `seed`, временные метки (`start_time`, `completion_time`, `elapsed_real_time`), физический интервал (`sim_start_time`, `sim_stop_time`, `frame_duration`, `simulated_histories`) и словарь переменных `context`.

---

## 3. Верификация и тесты (`tests/data/`)

1. Обновить модульные тесты в [`tests/data/test_metadata_collector.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_metadata_collector.py) и [`tests/data/test_hdf5_storage_and_metadata.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_hdf5_storage_and_metadata.py):
   * Проверить, что для `CustomSweep` в `metadata/gantry` и `metadata/protocol` отсутствуют ключи `orbit_radius`, `angular_step`, `rotation_direction`.
   * Проверить, что для `StepAndShoot` без заданного `radius` ключ `orbit_radius` не создается, а при заданном — равен точному значению в мм.
   * Проверить отсутствие дубликатов `pose_matrix_GammaCamera_N` и `crystal_dimensions_GammaCamera_N`.
   * Проверить точность `half_life = 595296.0` при входном значении `6.89 * units.day`.
