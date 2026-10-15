# Спецификация реализации: Унифицированные метаданные процедур и очистка DataHandler

## 1. Контекст и цели рефакторинга
* **Цель:** Полная изоляция низкоуровневых сборщиков физических взаимодействий (`core/data/data_handlers.py`) от процедур ядерной медицины (ОФЭКТ/ПЭТ/гентри), устранение эвристик и утиной типизации (`hasattr`), реализация декомпозированного сбора процедурных метаданных через специализированные провайдеры (`core/data/metadata_collector.py`).
* **Правила проекта:** Строгое следование принципам:
  - «Явное лучше неявного» (`02_code_contracts.md`);
  - Запрет суффиксов единиц измерения (`_mm`, `_deg`) и стандарт HepUnits (`03_physics_units.md`);
  - No Legacy Crutches и чистота ядра (`01_architecture_core.md`).

---

## 2. Пошаговый план изменений

### Шаг 1. Очистка `core/data/data_handlers.py`
1. Полностью удалить `extract_volume_metadata`.
2. Удалить искусственные роли (`role="detector"`, `role="scatter_history"`, `role="geometry"`), вычисления `orbit_radius_mm`, `detector_angle_deg`, `view_direction_angle_deg` и эвристики `"crystal" in node.name.lower()`.
3. Полностью исключить сбор метаданных из `DataHandler`:
   - Удалить `_build_volume_metadata`, `_apply_volume_attributes` и поле `self.volume_metadata` из `SensitiveVolumeHandler` и `HistoryAssemblerHandler`.
   - `DataHandler` отвечает строго за прием и сериализацию потока физических событий (`interactions`, `initial_states`), а не за инспекцию геометрии сцены.
   - Метаданные объемов не должны зависеть от стохастического попадания фотонов в объем.
4. Вся информация о детекторах, кристаллах, их матрицах трансформации (`global_matrix`) и тегах собирается централизованно в `core/data/metadata_collector.py` (`DetectorMetadataProvider`) и сохраняется в `/metadata` через `DataManager(metadata=...)`.

### Шаг 2. Создание модуля провайдеров метаданных `core/data/metadata_collector.py`
Создать модульную систему инспекции метаданных с классами:
1. `ProtocolMetadataProvider`:
   - Извлекает параметры модальности (`SPECT`, `StepAndShoot`, `CustomSweep`) без `hasattr` (через `isinstance`);
   - Записывает `modality`, `view_index`, `views_total`, `exposure_time` (в нс), `orbit_radius` (в мм).
2. `KinematicsMetadataProvider`:
   - Находит узлы `GantryNode` в сцене;
   - Фиксирует `gantry_name`, `gantry_angle` (в радианах).
3. `DetectorMetadataProvider`:
   - Находит узлы `GammaCameraNode` в сцене;
   - Через контракт слотов `camera.slots.get("crystal")` находит узел кристалла и сохраняет его `global_matrix` и имя.
4. `ProcedureMetadataCollector` (фасад):
   - Метод `collect(root_scene, protocol, context, task_id) -> Dict[str, Any]`.

### Шаг 3. Интеграция в `core/config/simulation_worker.py`
1. Использовать `ProcedureMetadataCollector` для формирования `acquisition_metadata`.
2. Передавать структуру в `task_metadata`:
   ```python
   task_metadata = {
       "task_id": task_id,
       "context": context_data,
       "acquisition": procedure_metadata,
   }
   ```
3. Инициализировать `DataManager(metadata=task_metadata)`.

### Шаг 4. Поддержка `tags` в конфигурации узлов (`core/config/models.py`, `core/config/builder.py`, `core/config/exporter.py`)
1. В `SpatialNodeConfig` добавить `tags: List[str] = Field(default_factory=list)`.
2. В `SceneBuilder._build_node`: проставлять `node.tags = list(config.tags)` при их наличии.
3. В `SceneExporter.export_node`: экспортировать `tags=list(node.tags) if node.tags else []`.

### Шаг 5. Верификация и тестирование
1. Запустить существующие тесты:
   - `tests/config/test_orchestrator.py`
   - `tests/test_full_benchmark.py`
   - `tests/test_stage1_core.py`
2. Добавить юнит-тесты на `metadata_collector.py` и очищенный `SensitiveVolumeHandler`.
