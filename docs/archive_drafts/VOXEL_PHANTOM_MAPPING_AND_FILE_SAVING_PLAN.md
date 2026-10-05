# Архитектурный план: Настройка Mapping материалов и сохранение путей воксельных распределений

## 1. Контекст и описание проблем

В ходе эксплуатации графического интерфейса пользователя (GUI) и подсистемы воксельных фантомов (`WoodcockVoxelVolume` / `VoxelVolumeViewModel` / `PropertyInspector`) были выявлены две критические проблемы:

1. **Отсутствие возможности настройки Mapping материалов в GUI:**
   - В расчетном ядре воксельный фантом (`WoodcockVoxelVolume`) хранит трехмерную матрицу дискретных идентификаторов тканей/органов как `MaterialArray`. Индексы вокселей (0, 1, 2, ...) сопоставляются с физическими материалами NIST (`core.materials.materials.Material`) через список `material_distribution.element_list`.
   - В панели `PropertyInspector` для `VoxelVolumeViewModel` отсутствуют элементы управления для инспекции и редактирования привязки целочисленных ID вокселей к материалам базы NIST. Пользователь не может посмотреть, какие ткани присутствуют в фантоме, и не может переназначить материал для конкретного ID (например, заменить мягкую ткань на кортикальную кость или титан).
   - Во `VoxelVolumeViewModel` отсутствуют метод `set_material_mapping` и сигнал `property_changed.emit('material_distribution', ...)`.

2. **Некорректное сохранение путей к файлам распределений и потеря mapping при экспорте в YAML:**
   - При выборе файла фантома или источника через кнопку «Обзор...» (`_on_browse_phantom_file`, `_on_browse_source_file`) путь обновляется во ViewModel (`self.file_path`), но не регистрируется в `SceneViewModel.distribution_registry`.
   - При экспорте сцены через `SceneExporter.export_to_yaml` данные берутся из `SceneViewModel.distribution_registry`. Если узел был создан или перезагружен в GUI, запись в реестре либо отсутствует (тогда генерируется заглушка `phantom.npy` с `mapping=None`), либо содержит устаревший путь и устаревший `mapping`.
   - В `SceneBuilder._build_woodcock_voxel_volume` для конфигурации `WoodcockVoxelVolumeConfig` наличие непустого словаря `mapping` является строгим предусловием (fail-fast: `ValueError("WoodcockVoxelVolumeConfig mapping requires an explicit mapping...")`). Конфигурация с дефолтной заглушкой без mapping становится невалидной и не может быть загружена повторно.
   - Экспортируемые пути сохраняются в исходном абсолютном виде, а не относительно директории сохраняемого YAML-файла (`base_dir`), что ломает переносимость проектов между машинами и директориями.
   - При загрузке YAML-файла через «Открыть конфигурацию...» пути из `distribution_registry` не проставлялись во `file_path` моделей представления, из-за чего в `PropertyInspector` поле пути оставалось пустым.

---

## 2. Архитектура решения и поток данных

```mermaid
flowchart TD
    subgraph UI["GUI: PropertyInspector & Viewport"]
        Inspector["PropertyInspector"]
        TableMapping["Таблица Mapping: ID | Цвет | QComboBox NIST"]
        BtnBrowse["Кнопка 'Обзор...'"]
        Viewport["SceneViewportController / VTK"]
    end

    subgraph VM["Слой ViewModel (MVVM)"]
        VoxelVM["VoxelVolumeViewModel\n- set_material_mapping(id, name)\n- reload_distribution(path)"]
        SourceVM["SourceViewModel\n- reload_distribution(path)"]
        SceneVM["SceneViewModel\n- update_distribution_path(node, path)\n- update_distribution_mapping(node, mapping)\n- distribution_registry: Dict[Node, Config]"]
    end

    subgraph Core["Расчетное ядро и сериализация"]
        CoreNode["WoodcockVoxelVolume\n- material_distribution.element_list\n- invalidate_geometry()"]
        Exporter["SceneExporter.export_to_yaml\n- Синхронизация mapping из element_list\n- Приведение к relative_path относительно YAML"]
        Builder["SceneBuilder\n- _resolve_dist_path(path)\n- build_scene()"]
    end

    BtnBrowse -->|Выбор файла| VoxelVM
    BtnBrowse -->|Выбор файла| SourceVM
    BtnBrowse -->|Обновление пути| SceneVM
    TableMapping -->|Выбор материала NIST| VoxelVM
    VoxelVM -->|Обновление element_list| CoreNode
    VoxelVM -->|property_changed('material_distribution')| Viewport
    VoxelVM -->|Обновление mapping| SceneVM
    SceneVM -->|distribution_registry| Exporter
    CoreNode -->|element_list| Exporter
    Exporter -.->|YAML файл| Builder
```

---

## 3. Спецификация изменений по компонентам

### 3.1. Расширение моделей представления узлов (`VoxelVolumeViewModel`, `SourceViewModel`)
- **Файл:** `gui/viewmodels/nodes/voxel_volume_vm.py`
  - Добавить публичный метод `set_material_mapping(material_id: int, material_name: str) -> None`:
    - Валидирует `material_id >= 0`.
    - Извлекает `Material` из `database_setting.material_database` или создает канонический `Material(name='Vacuum', ID=0)`.
    - Обновляет `self.core_node.material_distribution.element_list[material_id] = target_material` (при необходимости безопасно расширяя список).
    - Вызывает инвалидацию геометрии расчетного ядра: `self.core_node.invalidate_geometry()`.
    - Генерирует событие: `self.property_changed.emit('material_distribution', self.material_list)`.
  - В методе `reload_distribution`:
    - После обновления матрицы и геометрии эмитить `self.property_changed.emit('material_distribution', self.material_list)`.

- **Файл:** `gui/viewmodels/nodes/source_vm.py`
  - В методе `reload_distribution`:
    - Обеспечить гарантированную эмиссию `'distribution'` и `'file_path'`.

### 3.2. Управление реестром распределений в `SceneViewModel`
- **Файл:** `gui/viewmodels/scene_viewmodel.py`
  - Добавить методы управления реестром:
    ```python
    def update_distribution_path(
        self,
        core_node: SpatialNode,
        file_path: str,
        mapping: Optional[Dict[float, Union[float, str]]] = None,
    ) -> None:
        """
        Регистрирует или обновляет путь к файлу распределения в distribution_registry.
        Сохраняет существующий тип конфигурации (Numpy/Raw) или создает соответствующий.
        """
    ```
    ```python
    def update_distribution_mapping(
        self,
        core_node: SpatialNode,
        mapping: Dict[float, Union[float, str]],
    ) -> None:
        """
        Актуализирует словарь соответствия ID -> Material для узла распределения.
        """
    ```
  - В методе `load_scene(root_core_node, distribution_registry, ...)`:
    - Для всех загруженных узлов (`VoxelVolumeViewModel`, `SourceViewModel`) восстанавливать `node_vm.file_path` из `self.distribution_registry.get(node_vm.core_node)`.

### 3.3. Таблица Mapping и синхронизация в `PropertyInspector`
- **Файл:** `gui/views/property_inspector.py`
  - В секцию `voxel_group`:
    - Добавить таблицу `self.tbl_material_mapping = QTableWidget()`:
      - Колонки: `["ID", "Цвет", "Материал NIST"]`.
      - Колонка `ID`: целочисленный идентификатор, выравнивание по центру, `Qt.ItemIsEnabled`.
      - Колонка `Цвет`: визуальный индикатор цвета ткани (из `get_material_color(material.name)`).
      - Колонка `Материал NIST`: `QComboBox` с алфавитным списком всех 144 материалов NIST (`sorted(database_setting.material_database.keys())`) и `"Vacuum"`.
    - Подключить сигнал изменения комбобокса: при смене материала вызывать `current_vm.set_material_mapping(...)`, обновлять цвет индикатора и синхронизировать `distribution_registry` через `self.scene_vm.update_distribution_mapping(...)`.
  - В методах `_on_browse_phantom_file` и `_on_browse_source_file`:
    - После вызова `reload_distribution` вызывать `self.scene_vm.update_distribution_path(self.current_vm.core_node, path)`.
    - Для фантома перестраивать таблицу `self._update_material_mapping_table()`.
  - В методе `_on_property_changed_externally`:
    - Добавить обработку `prop_name == 'material_distribution'` для обновления таблицы маппинга при внешних изменениях.

### 3.4. Реактивная синхронизация во вьюпорте (`SceneViewportController`)
- **Файл:** `gui/controllers/viewport_controller.py`
  - В методе `on_node_property_changed`:
    - Добавить ветку `prop_name == 'material_distribution'`:
      вызывать `self.voxel_renderer.apply_material_transfer_functions(...)` для ступенчатого обновления цвета и непрозрачности вокселей без перезагрузки актора VTK и без сброса камеры.

### 3.5. Надежный экспорт в `SceneExporter`
- **Файл:** `core/config/exporter.py`
  - Для `WoodcockVoxelVolume`:
    - Гарантировать, что `dist_cfg.mapping` формируется напрямую из актуального `node.material_distribution.element_list`:
      `{float(idx): mat.name for idx, mat in enumerate(node.material_distribution.element_list)}`.
    - Это обеспечивает 100% валидность сохраненного YAML и исключает `ValueError` в `SceneBuilder`.
  - Для путей к файлам распределений (`path`):
    - В `export_to_yaml`: приводить пути к относительным относительно каталога сохраняемого `.yaml` файла (`base_dir`) через `os.path.relpath`.
    - Реализовать безопасный fallback на абсолютный путь при нахождении файлов на разных дисках в Windows (`ValueError: path is on mount 'C:', start on mount 'D:'`).

---

## 4. План реализации по шагам

1. **Шаг 1: ViewModel-слой**
   - Реализация `set_material_mapping` и эмиссии `material_distribution` в `gui/viewmodels/nodes/voxel_volume_vm.py`.
   - Добавление `update_distribution_path` и `update_distribution_mapping` в `gui/viewmodels/scene_viewmodel.py`.
   - Восстановление `file_path` во ViewModel при `SceneViewModel.load_scene`.

2. **Шаг 2: Расчетное ядро и экспорт**
   - Автоматическое формирование `mapping` из `material_distribution.element_list` в `core/config/exporter.py`.
   - Преобразование путей в относительные относительно сохраняемого YAML-файла в `SceneExporter.export_to_yaml`.

3. **Шаг 3: GUI и Viewport**
   - Создание и настройка таблицы `tbl_material_mapping` в `gui/views/property_inspector.py`.
   - Реактивная обработка `material_distribution` в `gui/controllers/viewport_controller.py`.
   - Синхронизация путей при выборе файлов в `_on_browse_phantom_file` и `_on_browse_source_file`.

4. **Шаг 4: Тестирование и верификация**
   - Модульные тесты `test_material_mapping_viewmodel` и `test_scene_vm_distribution_registry`.
   - Интеграционный тест полного цикла (round-trip): создание/загрузка фантома ➔ смена маппинга ➔ сохранение YAML ➔ загрузка через `SceneBuilder` ➔ проверка идентичности материалов и путей.
   - Запуск полного набора тестов (`pytest`) с достижением 100% прохождения.
