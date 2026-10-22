# Архитектурный бэклог и планы развития NMSimToolkit

В данном документе фиксируются отложенные архитектурные задачи, проектные решения и концепции для будущих итераций развития программного комплекса.

---

## 1. Кинематика и протоколы сканирования: Синхронизация протокол ↔ сцена и версионирование

### Проблема:
* На текущем этапе протоколы сканирования (ОФЭКТ, ПЭТ) могут приводить к неявной перезаписи геометрии сцены (дефолтные радиусы, углы головок) при открытии сцен без явного блока `protocol:`.
* Автоматическая обратная синхронизация «сцена → протокол» (инспекция координат камер для восстановления параметров орбиты) порождает сложности при миграции старых конфигураций на новые схемы протоколов.

### План реализации:
1. **Явное разделение режимов синхронизации:**
   * Протокол не должен мутировать граф сцены в фоновом режиме при загрузке файлов без протокола.
   * Рассмотреть явное действие в UI (кнопка / команда «Привязать протокол к сцене» / «Сформировать траекторию по геометрии») с предварительным предпросмотром изменений (diff).
2. **Система миграции схем конфигураций (Config Schema Migration):**
   * Введение версионирования структуры конфигурационных файлов (`schema_version: 1.x`).
   * Реализация изолированных пайплайнов миграции для автоматического обновления устаревших структур сцен и протоколов до актуального стандарта перед валидацией Pydantic.

---

## 2. Пространственная математика и библиотека 3D-трансформаций [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:** В [`core/other/transform.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/other/transform.py) реализован канонический класс [`Transform`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/other/transform.py) (TRS: `translation`, унитарный кватернион `rotation` на базе `scipy.spatial.transform.Rotation`, `scale`) со строгой валидацией ортогональности матриц поворота, поддержкой прямого и обратного преобразования аффинных матриц $4 \times 4$ и экспортом `MatrixTransformConfig` (`type: "matrix"`).

---

## 3. Модульная сборка детекторных головок (`GammaCamera` со слотами / sockets) [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:** Перевод `GammaCamera` на декларативную архитектуру со слотами (Slots / Sockets) полностью реализован и верифицирован на сквозной модели детекторной головки GE Discovery NM/CT 670 (`verification/collimators/exp06_nema_detector_head.py`).

### Реализованная архитектура:
1. **Узел расчетного ядра (`GammaCameraNode`):**
   * Класс [`GammaCameraNode(CompositeNode)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/scene/gamma_camera_node.py) является чистым составным узлом без собственного физического материала и геометрии.
   * Содержит открытый словарь слотов `slots: Dict[str, str]`, связывающий функциональные роли компонентов (`casing`, `detector_box`, `collimator`, `crystal`, `glass_backend`) с точными именами дочерних узлов `childs`.
   * Полностью удален процедурный генератор `core/geometry/gamma_cameras.py`.
2. **Конфигурационная схема и сериализация:**
   * В [`core/config/models.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/models.py) объявлены Pydantic-модели `GammaCameraSlotsConfig` и `GammaCameraConfig(CompositeNodeConfig)`.
   * Поле `children: List[AnyNodeConfig]` открыто для произвольных дочерних узлов.
   * `SceneBuilder` и `SceneExporter` оперируют строгими слотами без эвристик и угадывания имен по подстрокам.
3. **Инкапсуляция геометрии в ViewModel (`gui/viewmodels/nodes/gamma_camera_vm.py`):**
   * Вся логика параметризации (размер чувствительного поля `detector_size`, толщины слоев, зазор `gap`, защита `shielding_thickness`) и их пересчет инкапсулированы в `GammaCameraViewModel`.
   * Доступ к дочерним компонентам осуществляется через строго типизированные свойства слотов (`crystal_vm`, `collimator_vm`, `casing_vm`, `detector_box_vm`, `glass_backend_vm`).
   * Реализованы фабричные функции `create_default_gamma_camera(...)` (`gui/factories/gamma_camera_factory.py`) и `create_default_gamma_camera_vm(...)` (`gui/viewmodels/nodes/gamma_camera_vm.py`).

### Перспективы развития (Future Work):
* Поддержка пикселированных полупроводниковых детекторов (CZT) с сеточной компоновкой модулей в слоте `crystal`.
* Поддержка коллиматоров произвольной геометрии (веерные, конические, пинхол-коллиматоры) через расширение слотовой спецификации.

---

## 4. Детерминированный RayCasting для периодических коллиматоров [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:** В [`core/geometry/geometry_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/geometry_kernels.py) реализован и верифицирован аналитический детерминированный RayCasting в пространстве периодической гексагональной ячейки (`PeriodicHexPrism`, `_hex_prism_intersect`, `_trace_single_ray`) со 100% физической точностью (Zero-Compromise Physics).
* Выделены константы `GEOMETRY_EPSILON`, `SQRT_THREE`, `BOUNDARY_PUSH_EPSILON`, устранена слепая зона `[0, GEOMETRY_EPSILON]` и обрыв луча при наклонном входе через торец $Z$ в `_hex_prism_intersect`, исправлен клиппинг `closest_dist` родительским объемом в `_trace_single_ray`, а также выровнена фаза квадратных отверстий в `ParametricParallelCollimator`.
* Разработан фреймворк физической верификации коллиматоров (`verification/collimators/exp01..exp06`) со сверкой `DirectParallelCollimator` и `ParametricParallelCollimator` по теории Хола Ангера и паспортным эталонам GE NEMA NU-1 (`NemaPassportSpecification`).

---

## 5. Физическая визуализация воксельных фантомов на базе палитры материалов и глобального псевдорентгена [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:** В [`gui/viewport_3d/voxel_volume_renderer.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/voxel_volume_renderer.py) и [`gui/viewport_3d/material_palette.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/material_palette.py) реализованы ступенчатые передаточные функции VTK с прямым маппингом дискретных ID материалов на физические цвета палитры NIST и суммарный коэффициент линейного ослабления $\mu(E)$ по всем каналам сечений, точный фильтр газовых сред, синхронизация с глобальным режимом `pseudo_xray_mode` (`xray_energy`), ячеечная сетка `cell_data` (`dimensions = (Nx + 1, Ny + 1, Nz + 1)`) с интерполяцией Nearest Neighbor и янтарная рамка выделения фантома.

---

## 6. Строгая статическая типизация переменных биндинга (Property Binding Typing)

### Проблема:
* В первой итерации механизма привязки параметров (Property Binding / UE-Style Variables) переменные `grid_variables` и `zipped_variables` оперируют универсальными числами `float`.
* Отсутствует проверка соответствия физических размерностей на уровне интерфейса (например, попытка привязать переменную углов к координате смещения).

### Согласованное архитектурное решение:
* Введение метаданных типов для переменных протокола: `Length`, `Angle`, `Time`, `Energy`, `Unitless`.
* Фильтрация доступных для бинда переменных в выпадающем меню `[🔗 Bind]` по физической размерности целевого свойства.
* Интеграция валидации размерностей в `BindableDoubleField` при выборе переменной.

---

## 7. Глобальная унификация распределений и скалярных величин в расчетном ядре [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:** В [`core/source/sources.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py) (`Source.energy`) и [`core/config/models.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/models.py) (`SourceConfig`) энергетический спектр всегда хранится в унифицированном табличном представлении пар `[energy, probability]`, где моноэнергетический источник является строгим частным случаем таблицы из одной строки с вероятностью `1.0`. Также реализована нативная сериализация `PointSource` (`voxel_size = 0.0` без пространственного размытия) и параметров `RawDistributionConfig`.

---

## 8. Динамическое отображение значений привязанных переменных в BindableDoubleField при предпросмотре шагов симуляции (Simulation Step/Task Preview)

### Проблема:
* При связывании числового поля с переменной протокола (`Property Binding`) виджет `BindableDoubleField` переводится в режим `ReadOnly` и блокирует ручное редактирование, отображая имя привязанной переменной на кнопке связи (`🔗 ${var}`).
* При этом в самом поле `QDoubleSpinBox` сохраняется статическое базовое значение узла сцены. Поле не отражает фактическое вычисленное значение переменной для текущей выбранной задачи из пула свипа (`SweepTask`) или выбранного номера шага процедуры при предпросмотре.
* Механизм предпросмотра шага симуляции (`task/step preview`) в текущей реализации требует аудита и стабилизации консистенции состояния сцены и отображения значений.

### Согласованное архитектурное решение / План реализации:
1. **Аудит и стабилизация системы предпросмотра шагов (Step & Task Preview System):**
   - Провести ревизию механизма генерации пула задач симуляции (`SweepCompiler` / `SweepTask`) и предпросмотра шагов для всех типов процедур (`StepAndShootProcedureViewModel`, `SpectProcedureViewModel`, `CustomSweepProcedureViewModel`).
   - Обеспечить детерминированное извлечение словаря значений переменных `Dict[str, float]` для каждого конкретного шага симуляции.
2. **Реактивное обновление значений в `BindableDoubleField` при навигации по шагам:**
   - Реализовать во `ViewModel` / контроллере интерфейса передачу текущего контекста значений шага (`step_variable_values: Dict[str, float]`) в `PropertyInspector`.
   - При смене текущего шага предпросмотра, если поле привязано к переменной `${var}`, программно обновлять отображаемое в `QDoubleSpinBox` числовое значение (`setValue(val)`) с учетом единиц измерения, сохраняя блокировку редактирования (`ReadOnly`).
   - Реализовать информативный `tooltip` для заблокированного поля (например: *«Шаг 3: 45.00 deg (значение переменной: ${gantry_angle})»*), гарантируя наглядность и обратную связь для оператора.

---

## 9. Поддержка Property Binding для временных интервалов симуляции (`start_time` и `stop_time`) и динамического фреймирования [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:**
  1. **Временная шкала `StepAndShoot`, авто-подстановка макросов и запись в HDF5:**
     * В [`StepAndShootProtocolConfig`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/models.py), [`StepAndShootProcedureViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/procedure_viewmodel.py) и [`ProcedureInspectorSection`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/procedure_section.py) реализованы параметры и элементы управления `initial_time` и `sequential_time` для настройки начального смещения $t_0$ и последовательного хронологического отсчета времени между ракурсами.
     * В [`SweepCompiler.compile_protocol`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/sweep_compiler.py) реализована генерация переменных `start_time`, `stop_time` и `step_time` во внутренних единицах `hepunits`, а также автоматическая подстановка шаблонов `${start_time}` и `${stop_time}` в `simulation_manager` (`SweepCompiler.apply_protocol_defaults`, [`OrchestratorSession._get_start_time_cfg`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/orchestrator_session.py)).
     * В [`DataManager`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py) и [`ProcedureMetadataCollector`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/metadata_collector.py) реализована запись фактических границ интервала каждой задачи (`actual_start_time`, `actual_stop_time`, `frame_duration` в `/metadata/tasks/task_{id}`), кумулятивное обновление `/metadata/gantry/frame_duration` и бесшовный досчёт симуляции (`inspect_resume_state`) с сохранением `final_rng_state` и `particle_id_end`.
  2. **Интеграция `BindableDoubleField` для `start_time` и `stop_time` в GUI и полный Roundtrip:**
     * В [`BaseProcedureViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/procedure_viewmodel.py) и [`CustomSweepProcedureViewModel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/procedure_viewmodel.py) добавлены свойства `start_time` и `stop_time` с двусторонней синхронизацией `bindings["start_time"]`, `bindings["stop_time"]` и `bindings_registry["simulation_manager"]`.
     * В [`ProcedureInspectorSection`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/procedure_section.py) для `CustomSweep` добавлена секция временного интервала с полями [`BindableDoubleField`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/widgets/bindable_field.py) (`spin_start_time`, `spin_stop_time`, `unit="s"`), а в [`OrchestratorSession`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/orchestrator_session.py) и [`MainWindow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/main_window.py) обеспечен полный Roundtrip численных значений и строковых макросов `"${var} s"`.

---

## 10. Математический конвейер вычисляемых выражений над BindableVariable (Computed Variables / Math Operations Grid)

### Проблема:
* Физические параметры протокола часто задаются в канонической лабораторной системе отсчета (например, «чистый радиус орбиты» $R$ от изоцентра до коллиматора, или базовый угол поворота ротора $\theta$).
* При этом пространственное положение компонентов сцены требует учета их геометрических габаритов или фазовых сдвигов:
  - Координата $Y$ центра гамма-камеры равна $R + \text{half\_thickness}$;
  - Угол второй противоположной детекторной головки равен $\theta + 180^\circ$;
  - Смещение коллиматора по $Z$ требует масштабирования или интерполяции.
* В настоящее время для таких преобразований требуется либо зашивать формулы в специализированный код кинематики, либо вручную высчитывать массивы чисел в таблицах.

### Согласованное архитектурное решение / План реализации:
1. **Табличный конвейер математических операций (Math Operations Grid):**
   * Расширение функционала `CustomSweep` таблицей вычисляемых выражений из 4 столбцов:
     | Левый операнд (Left) | Операция (Op) | Правый операнд (Right) | Результирующий бинд (Result Target) |
     | :--- | :---: | :--- | :--- |
     | `radius` (🔗 Bind) | `+` | `half_thickness` (число / 🔗) | `camera_1_y` (🔗 Result Bind) |
     | `gantry_angle` (🔗 Bind) | `+` | `180.0` (число) | `camera_2_angle` (🔗 Result Bind) |
     | `bed_velocity` (🔗 Bind) | `*` | `step_time` (🔗 Bind) | `bed_delta_z` (🔗 Result Bind) |

2. **Правила операндов и поддерживаемые операции:**
   * Операнды могут быть:
     - Именем переменной протокола (`BindableVariable` из Grid или Zipped);
     - Числовой константой (`float`).
   * Поддерживаемые операции: `+`, `-`, `*`, `/`, `mod`, `deg2rad`, `rad2deg`.
   * Результирующее имя переменной (`Result Target`) автоматически регистрируется в реестре доступных переменных протокола и сразу появляется в выпадающем меню `[🔗 Bind]` для назначения на любые свойства узлов сцены.

3. **Интеграция с `SweepCompiler` и генератором задач:**
   * Перед генерацией задач компилятор траекторий вычисляет топологический порядок выражений (DAG).
   * На каждом шаге симуляции вычисляемые переменные автоматически рассчитываются и подставляются в контекст задачи `${camera_1_y}` для трансляции в геометрию.

---

## 11. Интеллектуальный парсинг и конвертация единиц измерения при вводе в текстовые и чипсовые поля (Natural Unit Input Parsing) [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:**
  1. **Единый парсер размерных величин и конвертация макросов (`core/config/units.py`, `core/config/sweep_compiler.py`):**
     * В [`core/config/units.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/units.py) сосредоточены реестр `ureg`, словарь русско-английских алиасов `UNIT_ALIASES`, функции [`parse_unit_value`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/units.py), [`parse_unit_tokens`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/units.py) (с проверкой физической совместимости категорий `length`, `energy`, `time`, `angle`, `activity`, поддержкой выражений с `pi`/`π` и конвертацией в `default_unit` или `hepunits`), `unit_validator_factory` и [`convert_from_hepunits`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/units.py).
     * В [`SweepCompiler.inject_variables`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/sweep_compiler.py) реализована поддержка синтаксиса `${var unit}` в одиночных полях и составных строковых шаблонах.
  2. **Виджеты с поддержкой размерного ввода (`UnitAwareDoubleSpinBox`, `BindableDoubleField`, `GridChip`, `GridVariableChipsRow`, `GridRangeDialog`):**
     * В [`gui/views/widgets/bindable_field.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/widgets/bindable_field.py) реализован подкласс [`UnitAwareDoubleSpinBox`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/widgets/bindable_field.py) и параметр `unit` в [`BindableDoubleField`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/widgets/bindable_field.py) (с автовыводом единицы из `suffix`), позволяющие вводить выражения вида `"25 cm"`, `"140 keV"`, `"pi/2 rad"`, `"2 min"` прямо в спинбокс с автоматическим пересчетом в целевую единицу поля.
     * В [`gui/views/widgets/grid_chips_widget.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/widgets/grid_chips_widget.py) реализован автоматический вывод базовых единиц стандартных переменных (`infer_variable_default_unit`) и парсинг размерных строк при инлайн-редактировании `GridChip`, пакетном добавлении значений в `GridVariableChipsRow` и генерации диапазонов в `GridRangeDialog`.

---

## 12. Расширение и многоуровневая архитектура телеметрии и логирования симуляций (Advanced Logging Architecture)

### Контекст и цели:
В ходе восстановления и модернизации файлового логирования через конфигурацию YAML (`LoggingConfig`) реализован базовый каркас: изоляция логов задач воркеров (`task_{task_id}.log`), логирование оркестратора (`orchestrator.log`) и строгое соблюдение Zero-Overhead на горячем пути переноса частиц (отсутствие микро-логирования отдельных частиц или проверок геометрии в циклах физики).
Для дальнейшего развития подсистемы логирования требуется детальный план расширения охвата событий расчетного ядра и интеграции с внешними каналами оповещения.

### Анализ необходимых уровней логирования и событий:

1. **Макро-события оркестрации и пакетирования (`core.config.orchestrator`):**
   * `INFO`: Старт пакета моделирования, суммарное число задач, размер пула (`pool_size`), сгенерированные сиды генератора случайных чисел `SeedSequence`.
   * `INFO`: Успешное завершение всего пакета задач, общее астрономическое время выполнения, средняя производительность (число задач в секунду).
   * `WARNING`: Обнаружение задержек ввода-вывода или повторных блокировок файлов.
   * `ERROR`: Фатальные исключения в оркестраторе или сбои рабочих процессов с фиксацией трассировки стека (`exc_info=True`).

2. **Жизненный цикл воркера симуляции (`core.config.simulation_worker`):**
   * `INFO`: Старт отдельной задачи с фиксацией `task_id`, `seed` и распакованного контекста варьируемых параметров (`_context`).
   * `DEBUG`: Детализированный отчет сборки графа сцены (`SceneBuilder`): количество загруженных узлов, воксельных матриц, материалов NIST, активных процессов переноса (`propagator.processes`).
   * `INFO`: Завершение симуляции: длительность расчета в секундах, итоговый путь к выходному HDF5-файлу.
   * `ERROR`: Исключения на этапе сборки сцены, инициализации метаданных или выполнения симуляции с немедленной записью в HDF5-метаданные (`status="failed"`) и файл лога задачи.

3. **Сбор данных, буферизация и I/O (`core.data.data_manager`):**
   * `DEBUG`: Инициализация схемы HDF5, регистрация обработчиков (`DirectStreamHandler`, `SensitiveVolumeHandler`, `HistoryAssemblerHandler`, `DoseMapHandler`).
   * `WARNING`: Срабатывание механизма повторных попыток записи `_write_with_retry` (`retry_index > 0`) при временной конкурентной блокировке файла HDF5 (`BlockingIOError` / `OSError`).
   * `INFO`: Завершение сессии записи HDF5: итоговое количество сохраненных чанков событий, размер HDF5-файла на диске.

4. **Физический менеджер моделирования (`core.transport.simulation_managers`):**
   * `INFO`: Старт и остановка физического таймера (`global_timer`), моделируемый интервал времени (`stop_time`).
   * `DEBUG`: Периодическая макро-статистика по завершении обработки крупных чанков буфера взаимодействия (например, каждые $10^5$ частиц): доля поглощенных частиц, частиц вышедших за границы мира (`escaped`), средняя глубина трекинга.
   * ⛔ **Строгий запрет (Zero-Overhead):** Никаких логов внутри горячих функций ядра: `_invalidate_by_volume`, `_step`, `woodcock_tracking_kernel`, циклов физических процессов.

5. **Каналы нотификации и алертинга (Monitoring & Notification):**
   * В [`infra/monitoring/telegram.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/infra/monitoring/telegram.py) реализован `TeleBotHandler` с чтением учетных данных из переменных окружения; для декларативной конфигурации требуется добавить в [`LoggingConfig`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/models.py#L315-L357) опциональную секцию `telegram: Optional[TelegramLoggingConfig] = None` (`token`, `chat_id`, `min_level="WARNING"`), позволяющую автоматически отправлять уведомления об ошибках и окончании длительных серий симуляций из YAML.

---

## 13. Архитектура параллельного ввода-вывода: Lock-Free шардирование и HDF5 Virtual Datasets (VDS)

### Проблема:
* В [`core/config/orchestrator.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/orchestrator.py) параллельный пул воркеров использует общий межпроцессный замок `multiprocessing_manager.Lock()` для каждого выходного файла HDF5.
* Внутри каждого воркера класс [`DataManager`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py) при каждом сбросе чанков данных вызывает `_write_with_retry` с циклом повторных попыток (до 100 раз с `time.sleep(0.1)`), открывая и закрывая `h5py.File(..., "a")`.
* По данным бенчмарка масштабируемости (`verification/profiling/orchestrator_scaling.py`, Раздел 6 `PROFILING_REPORT.md`) при `pool_size = 4` ускорение составляет `2.48x–2.49x` (эффективность `61%–62%`), а при масштабировании на десятки воркеров и частых сбросах детектора общая блокировка файла порождает конкуренцию за дисковый мьютекс (I/O lock contention) и риск `BlockingIOError`.

### Согласованное архитектурное решение / План реализации:
1. **Изолированная запись задач (Lock-Free Sharding):**
   * Каждый воркер пишет строго в собственный независимый файл-шард `output data/shards/{simulation_name}_task_{task_id}.h5` без межпроцессных блокировок и без `_write_with_retry` (с удержанием открытого дескриптора `h5py.File` на время работы задачи и поддержкой предварительной инспекции `inspect_resume_state`).
2. **Агрегация через HDF5 Virtual Datasets (VDS):**
   * По завершении всех задач оркестратор собирает единый мастер-файл `output data/{simulation_name}.h5` через стандартное API `h5py.VirtualLayout` / `VirtualSource` (HDF5 VDS).
   * Просмотрщик `ResultsViewer` и внешние алгоритмы реконструкции (CASToR, PyTomography) работают с единым непрерывным датасетом с нулевым оверхедом копирования.

---

## 14. $\mathcal{O}(1)$ поиск интервала энергетической сетки для интерполяции макроскопических сечений (двухуровневый лог-индекс)

### Проблема:
* В [`_get_macroscopic_cross_sections`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/physics_kernels.py#L14-L53) на каждом шаге трекинга выполняется бинарный поиск `np.searchsorted(mat_energy_grid, energy)` сложности $\mathcal{O}(\log K)$ по индивидуальной сетке материала ($K \approx 1110\text{–}1130$ узлов).
* По телеметрии переноса (`verification/profiling/transport_telemetry.py`) на одну историю приходится до `17.82` вызовов (`W3_Heart_Tc99m`), что порождает промахи предсказателя переходов. В [`make_transport_kernel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py#L202-L294) поиск выполняется дважды на шаг Вудкока (мажоранта и реальный материал вокселя) при неизменной энергии частицы.
* **Аудит данных `tables/NIST XCOM Elements MAC.h5`:** равномерный линейный шаг `1 кэВ` в диапазоне `1–1000 кэВ`, `22–51 кэВ` в диапазоне `1–10 МэВ`, верхняя граница `10 МэВ`. Края поглощения не представлены парами узлов (см. п. 26), поэтому точность на краях определяется данными, а не алгоритмом поиска интервала.

### Отклоненные варианты:
* **Пересэмплирование на равномерную лог-сетку $N = 1024 / 2048$:** шаг $\approx 1\%$ по $\ln E$ ($\sim 320$ эВ при 33 кэВ, $\sim 850$ эВ при 88 кэВ) размывает края поглощения; сверхплотная сетка ($N \geq 16384$) выводит таблицы за пределы L2/L3. Нарушает Zero-Compromise Physics.
* **Кусочно-равномерная сетка по краям подоболочек:** края различны у разных элементов, поиск сегмента возвращает ветвления; номер сегмента определяет лишь энергетически доступные оболочки, но не оболочку поглощения (см. п. 28).

### Согласованное архитектурное решение:
1. Точные узлы таблицы сохраняются без изменений (включая пары узлов краев поглощения из п. 26).
2. Вспомогательная общая для всех материалов равномерная сетка корзин по $\ln E$: `log_bin = int((log(E) - log_e_min) * inv_dlog_e)`. Число корзин выбирается при компиляции так, чтобы корзина покрывала не более 2–3 узлов точной сетки (проверяется предусловием).
3. Для каждого материала сцены таблица `bin_start: int32` (первая строка точной таблицы для корзины), адресуемая через новое поле `lookup_offset` в `MaterialPointerDType` (только материалы сцены, десятки КБ).
4. В ядре: корзина → `bin_start` → ограниченный досмотр вперед `while energy >= grid[idx + 1]` → прежняя линейная интерполяция. Результат побитово совпадает с бинарным поиском.
5. Индекс корзины не зависит от материала: вычисляется один раз на шаг частицы и переиспользуется для мажоранты и реального материала.

### План реализации:
* Фаза 3 плана [`docs/plans/cross_sections_and_source_sampling_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/cross_sections_and_source_sampling_plan.md).

---

## 15. Генерация первичных частиц: Vose's Alias, `@njit`-инжектор и пуассоновский поток распадов

### Проблема:
* **Хранение уже разреженное:** [`Source._generate_emission_table`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py#L186-L211) сохраняет координаты только ненулевых вокселей; полный `np.meshgrid` строится однократно при инициализации (`21.1 мс -> 4.7 мс` по `M2_EmissionTableInit`), что незначимо на фоне JIT-компиляции (п. 22) и не является целью оптимизации.
* **Горячая цена — [`generate_position`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py#L252-L260):** `self.rng.choice(voxel_origins, count, p=probability)` при каждом вызове заново строит CDF длины $K$ (до `~12 МБ`), проверяет `p` и выполняет бинарный поиск (`125.8 нс/частицу`, до `32.5%–34.8%` времени шага при малых батчах в `W2_NEMA_Tb161` / `W3_Heart_Tc99m`); [`inject`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py#L286-L321) дополнительно создает около десятка временных массивов `(N, 3)` / `(N, 4)` (см. п. 25.3).
* **Зависимость от разбиения на батчи:** блочная генерация (сначала $N$ энергий, затем $N$ направлений, позиций и времен) и розыгрыш времени эмиссии внутри окна батча `[time_start, time_end]` в [`generate_emission_time`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py#L262-L273) делают поток ГСЧ и времена эмиссии зависимыми от границ батчей, что блокирует точную обрезку по `stop_time` (п. 24).
* **Детерминированное число распадов:** число частиц за окно задается ожиданием (`get_expected_particles` + метод Ньютона в [`_calculate_time_step`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L274-L303) + [`_distribute_quotas`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L305-L325)), без пуассоновской статистики счета.

### Согласованное архитектурное решение:
* **Совместимость потока ГСЧ с предыдущими версиями не сохраняется:** эталоны пересчитываются, корректность выборки подтверждается статистическими критериями. Сохраняется инвариант **независимости от разбиения внутри версии**: `[0, T1] + [T1, T2] ≡ [0, T2]` побитово.
1. **Vose's Alias Method** для вокселей и линий спектра (`prob: float64`, `alias: int32`), построение за $\mathcal{O}(K)$ один раз в сеттерах `distribution` / `energy`. Таблица эмиссии хранит плоские индексы вокселей `int32` вместо координат `(K, 3)` в `float64`.
2. **`SourceCompiler` → `SourceBuffer`:** плоские массивы всех источников сцены (CSR alias-таблиц, матрицы `local -> global`, параметры распада) по аналогии с `PhysicsCompiler` / `PhysicsBuffer`.
3. **`@njit`-инжектор с по-частичной генерацией:** последовательность случайных чисел каждой частицы детерминирована и не зависит от границ батчей; запись напрямую в SoA [`ParticleBank`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/particles/particles.py) без промежуточных массивов (Zero-Allocation).
4. **Пуассоновский поток распадов (суперпозиция источников, прореживание Льюиса–Шедлера):** кандидаты с постоянной мажорантой $A_{\max} = \sum_s A_s(t_{\text{ref}})$ ($t_{\text{ref}}$ фиксируется в начале первого сегмента и сохраняется в состоянии досчета), $t \leftarrow t + \text{Exp}(1)/A_{\max}$; принятие кандидата и выбор источника — одним равномерным числом по долям $A_s(t)/A_{\max}$. Число частиц за интервал — пуассоновская случайная величина; обрезка по `stop_time` точная; при остановке сохраняется отложенное время кандидата `pending_arrival_time`.
5. **Удаление устаревшего кода (No Legacy Crutches):** `_calculate_time_step` (метод Ньютона), `_distribute_quotas`, `MIN_TIME_STEP`, блочные методы `generate_*` и поле `Source.rng`.

### План реализации:
* Фазы 4–5 плана [`docs/plans/cross_sections_and_source_sampling_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/cross_sections_and_source_sampling_plan.md).

---

## 16. Канонический кинематический тип `Transform` (TRS) и устранение Gimbal Lock [РЕАЛИЗОВАНО — см. Раздел 2]

### Статус:
* **Завершено:** Полностью консолидировано с Разделом 2 ([`core/other/transform.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/other/transform.py)).

---

## 17. Мульти-акторная визуализация нескольких сеток дозы в `DoseVolumeRenderer` [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:** В [`DoseVolumeRenderer`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/dose_volume_renderer.py) внедрены датакласс [`DoseActorEntry`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/dose_volume_renderer.py) и реестр `_entries: Dict[str, DoseActorEntry]`, хранящий независимые `pv.ImageData`, `volume_actor`, `volume_mapper`, `volume_property` и матрицы трансформации для каждой именованной сетки дозы (`grid_name`). В [`SceneViewportController`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/viewport_controller.py) (`on_dose_volume_received`, `on_node_transform_changed`, `on_node_property_changed`, `on_node_removed`) реализовано одновременное обновление всех активных сеток из словаря `Dict[str, np.ndarray]` с индивидуальными `voxel_size`, `origin`, `global_matrix`, синхронизацией видимости и удалением акторов при удалении `DoseGridViewModel`.

---

## 18. Декомпозиция процедурной геометрии манипулятора `TransformGizmo`

### Проблема:
* Модуль [`gui/viewport_3d/transform_gizmo.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/transform_gizmo.py) совмещает процедурную генерацию полигональных VTK-мешей стрелок, дуг, плоскостей и кубов масштабирования с математикой пересечения лучей мыши (Ray-Plane / Ray-Line Intersection, перевод вектора перемещения `filtered_delta` в базис родителя) и обработкой событий перетаскивания.

### Согласованное архитектурное решение / План реализации:
1. **Выделение построителя геометрии (`GizmoMeshBuilder`):**
   * Вынос процедурного создания `pv.PolyData` для всех осей и режимов (`TRANSLATE`, `ROTATE`, `SCALE`) в изолированный модуль `gui/viewport_3d/gizmo_mesh_builder.py`.
2. **Выделение математического солвера проекции луча (`GizmoRayProjector`):**
   * Изоляция чистых геометрических функций вычисления проекции курсора на ось/плоскость и дискретного снэппинга от состояния VTK-рендерера.

---

## 19. Предрасчет обратных матриц трансформации на шаге симуляции (`GeometryCompiler`)

### Проблема:
* При компиляции иерархии сцены или обновлении кинематического положения узлов на шаге томографического сканирования обратные матрицы перехода `global_to_local` в [`GeometryCompiler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/geometry_compiler.py) вычисляются через общий `np.linalg.inv(matrix)` для каждого объема.

### Согласованное архитектурное решение / План реализации:
1. **Аналитическое обращение ортонормированных аффинных матриц SE(3):**
   * Использование блочной структуры жесткого преобразования $\mathbf{M}^{-1} = \begin{bmatrix} \mathbf{R}^T & -\mathbf{R}^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$ вместо универсального LU-разложения `np.linalg.inv`, что ускоряет перекомпиляцию сцены между шагами свипа и исключает численную деградацию ортогональности.

---

## 20. Полная унификация имен свойств без суффиксов единиц в `gui/viewmodels/` и `gui/viewport_3d/` [РЕАЛИЗОВАНО]

### Статус:
* **Завершено:** Полностью устранены суффиксы единиц измерения (`_deg`, `_rad`) в [`core/geometry/spect_kinematics.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/spect_kinematics.py) (`compute_orbit_matrix`, `compute_spect_poses`), [`gui/viewmodels/nodes/gantry_vm.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewmodels/nodes/gantry_vm.py) (`gantry_angle`), [`gui/views/inspectors/gantry_section.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/views/inspectors/gantry_section.py), [`gui/viewport_3d/spect_manipulator.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/spect_manipulator.py) (`angle`, `delta_angle`), [`gui/viewport_3d/kinematic_constraints.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/kinematic_constraints.py) (`proposed_angle`, `angle`, `roll`, `gantry_angle`) и [`gui/controllers/viewport_controller.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/viewport_controller.py) (`_extract_camera_roll`, `angle`, `roll`).

---

## 21. Явное разделение режимов буферизации `DoseMapHandler` (In-Memory vs IPC `SharedMemory`)

### Проблема:
* В [`DoseMapHandler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/dose_map_handler.py) и [`DoseGridEntry`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/dose_map_handler.py) булевый флаг `create_shm: bool = True` совмещает две разные обязанности: выбор между выделением буфера в разделяемой памяти ОС (`multiprocessing.shared_memory.SharedMemory`) и ролью процесса (создатель против клиента).
* При `create_shm=True` (дефолт в headless/CLI-запуске [`simulation_worker_task`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/simulation_worker.py), когда `shm_name` не задан) воркер ненужно создает именованные сегменты `SharedMemory` в ядре ОС, хотя никакой процесс GUI их не читает.
* При `create_shm=False` `DoseGridEntry._init_shared_memory` безусловно пытается подключиться к существующему сегменту `SharedMemory(name=self.shm_name, create=False)`, перехватывает `FileNotFoundError`, спамит предупреждениями `WARNING: SharedMemory '...' для сетки '...' не найден.` и лишь затем откатывается на локальный массив `np.zeros(self.grid_shape, dtype=np.float64)`.

### Согласованное архитектурное решение / План реализации:
1. **Трехрежимная стратегия хранения буфера дозы (`shm_mode: Literal["none", "create", "attach"]` или `shm_name: Optional[str] = None`):**
   * Если `shm_name is None` (штатный CLI/headless-режим и автономные тесты без GUI-стриминга), `DoseGridEntry` сразу выделяет локальный массив `np.zeros(self.grid_shape, dtype=np.float64)` без системных вызовов `shared_memory.SharedMemory` и без ложных предупреждений в лог.
   * Если `shm_name` задан явно, режим `create_shm=True` создает сегмент в родительской сессии GUI (`OrchestratorSession`), а `create_shm=False` подключается к нему из воркера.

---

## 22. Устранение повторной компиляции фабрик замыканий Numba при инстанциации `ParticlePropagator` и `Process` (Приоритет P0)

### Проблема:
* По результатам комплексного профилирования (`verification/profiling/step_profiler.py`, Разделы 3.2 и 8.1 [`PROFILING_REPORT.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/reports/profiling/PROFILING_REPORT.md)) создание каждого нового экземпляра [`ParticlePropagator`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/propagator.py) приводит к принудительной перекомпиляции Numba/LLVM (**`~3.57 с` Cold Propagator Overhead** на инстанс, тогда как чистое время расчета `Warm Steady-State` для `150,000–300,000` историй с полной записью в HDF5 составляет всего **`0.32–0.80 с`**, т. е. перекомпиляция замедляет короткие запуски в **`5.4x – 12.3x` раз**):
  1. Конструкторы [`PhotoelectricEffect.__init__`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes.py), [`CoherentScattering.__init__`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes.py) и [`ComptonScattering.__init__`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes.py) при каждом создании объекта вызывают фабрики [`make_photoelectric_kernel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py), [`make_coherent_kernel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py) и [`make_compton_kernel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py) (причем `ComptonScattering` через `super().__init__` сначала вызывает `make_coherent_kernel` и тут же перезаписывает `self._kernel`).
  2. Конструктор [`ParticlePropagator.__init__`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/propagator.py) вызывает фабрику [`make_transport_kernel(self.process_ids)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py).
  3. Поскольку внутри этих фабрик функции `@njit(cache=True)` создаются заново как локальные замыкания (захватывающие свежие экземпляры `_device_func` и массив `process_ids`), Numba создает новые объекты `CPUDispatcher` и на первом шаге симуляции заново запускает полную компиляцию LLVM.

### Согласованное архитектурное решение / План реализации:
1. **Декомпозиция ядер процессов на уровень модуля и кэширование фабрик:**
   * Вынести `@njit(cache=True)`-ядра `_photoelectric_kernel`, `_compton_kernel`, `_coherent_kernel` и их `_device_func` на верхний уровень [`core/physics/processes_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py) (передавая `process_id` аргументом либо кэшируя фабрики через `@functools.lru_cache`).
   * Исключить лишний вызов `make_coherent_kernel` при инициализации `ComptonScattering`.
2. **Кэширование `make_transport_kernel` по неизменяемой сигнатуре процессов:**
   * Кэшировать скомпилированный диспетчер в [`core/transport/transport_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py) по ключу `tuple(int(pid) for pid in process_ids)` (либо передавать `process_ids` аргументом в top-level `@njit(cache=True)` ядро), сокращая инициализацию `ParticlePropagator` с `~3.57 с` до `< 1 мс`.

---

## 23. Алгоритмическая оптимизация и векторизация горячих узлов по результатам микробенчмарков `M1..M7`

### Проблема:
* Изолированные микробенчмарки ([`verification/profiling/micro_benchmarks.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/micro_benchmarks.py), Раздел 5 [`PROFILING_REPORT.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/reports/profiling/PROFILING_REPORT.md)) зафиксировали метрики 5 алгоритмических узких мест (`M1..M5`) и 2 эталонных соотношений ядер (`M6..M7`):
  1. **`M1_CrossSections` (Интерполяция сечений LAC в Вудкоке):** на каждую историю приходится до `17.82` вызовов бинарного поиска `np.searchsorted` (`39.0 нс/оп` против `7.2 нс/оп` на равномерной лог-сетке, ускорение **`5.44x`**).
  2. **`M2_EmissionTableInit` (Инициализация `EmissionTable` воксельного фантома):** построение полного 3D `np.meshgrid` + `np.stack` на сетках $128\times 128\times 92$ тратит `21.1 мс` против `4.7 мс` при разреженной инициализации (ускорение **`4.46x`**).
  3. **`M3_VoxelPositionSampling` (Генерация позиций в `Source`):** выбор активных вокселей через `rng.choice(origins, size=N, p=probabilities)` тратит `125.8 нс/частицу` против `19.5 нс/частицу` для `Vose's Alias + Direct SoA` (ускорение **`6.44x`**).
  4. **`M4_SensitiveVolumeMapping` ([`SensitiveVolumeHandler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_handlers.py)):** фильтрация и маппинг чанка через цикл Python с `dtype=object` тратит `87.2 нс/событие` против `2.5 нс/событие` для целочисленного LUT (ускорение **`34.63x`**).
  5. **`M5_HistoryAssemblerFiltering` ([`HistoryAssemblerHandler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_handlers.py)):** фильтрация взаимодействий через `np.intersect1d` и 4 повторных вызова `np.isin` тратит `67.5 нс/событие` против `2.3 нс/событие` при прямой булевой индексации (ускорение **`28.87x`**).
  6. **Справочные метрики ядер (`M6_GeometryPrimitives`, `M7_TransportVsProcessKernel`):** аналитическое пересечение `_hex_prism_intersect` (Voronoi 6-slab) занимает `22.0 нс/луч` против `3.0 нс/луч` для `_box_intersect` (`7.33x`); один шаг `cast_path + _transport_kernel` занимает `275.0 нс/частицу` против `335.4 нс/частицу` для `_compton_kernel + rotate` (`0.82x`).

### Согласованное архитектурное решение / План реализации:
1. **$\mathcal{O}(1)$ поиск интервала энергетической сетки (`M1`, см. Раздел 14):**
   * Двухуровневый лог-индекс (общая сетка корзин по $\ln E$ + таблица `bin_start` поверх точных узлов NIST) вместо бинарного поиска `np.searchsorted`; побитовое совпадение результата интерполяции. Прототип `M1` (`5.44x`) измерен для чисто равномерной таблицы — ускорение двухуровневой схемы подлежит повторному замеру.
2. **`Vose's Alias Method` и `@njit`-инжектор в `Source` (`M3` `6.44x`, см. Раздел 15):**
   * Разреженная инициализация (`M2`) незначима: хранение таблицы эмиссии уже разреженное, полный `np.meshgrid` строится однократно.
   * Предрасчет таблиц `prob` и `alias` (метод псевдонимов Уокера–Воза) для выборки индекса вокселя за $\mathcal{O}(1)$ и прямая генерация `x, y, z` в SoA-массивы `ParticleBank`.
3. **Векторизованные целочисленные LUT и булевы маски в [`core/data/data_handlers.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_handlers.py) (`M4` `34.63x`, `M5` `28.87x`):**
   * Замена Python-цикла со строками `dtype=object` в [`SensitiveVolumeHandler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_handlers.py) на предрассчитанную целочисленную таблицу отображения `volume_id`.
   * Замена `np.intersect1d` + `4x np.isin` в [`HistoryAssemblerHandler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_handlers.py) на прямую булеву индексацию по смещению `particle_ID - min_id`.

---

## 24. Клиппинг времени экспозиции в `SimulationManager._replenish_bank`, двухфазная динамика банка (`Full-Bank` vs `Tail Depletion`) и разделение `batch_capacity` и квоты историй

### Проблема:
* В [`SimulationManager.__init__`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L85-L150) аргумент `particles_number` задает емкость пула [`ParticleBank.allocate(self.particles_number)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/particles/particles.py) (размер батча одновременно летящих в памяти частиц), тогда как полное число моделируемых историй определяется интегралом активности источника за интервал `[start_time, stop_time]`.
* В методе [`SimulationManager._replenish_bank`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L327-L341) при вычислении `delta_time = self._calculate_time_step(num_to_inject)` **отсутствует ограничение целевого времени `target_time <= self.stop_time`**. В результате, если емкость банка `particles_number` превышает ожидаемое за `stop_time` число распадов ($N_{\text{expected}} = \int_{t_{\text{start}}}^{t_{\text{stop}}} A(t)\,dt$), на первом же шаге генерируется полный банк `particles_number` частиц, а модельное время `global_timer` перешагивает далеко за `stop_time`.
* **Связь с инвариантом побитового досчёта (`Stateful Resume`, [`docs/plans/simulation_resume_and_accumulation_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/simulation_resume_and_accumulation_plan.md)):** в коммите `48f146c` инжекция полными батчами `num_to_inject = bank.capacity - num_active` без обрезки `delta_time` была сохранена намеренно, поскольку текущая блочно-векторизованная реализация [`Source.inject`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py) генерирует сначала массив из $K$ позиций, затем $K$ направлений, затем $K$ энергий и времен. Из-за этого разбиение батча на $K_1 + K_2$ на границе $T_1$ меняет порядок выборки из `self.rng` и нарушает побитовую идентичность двухэтапного расчёта `[0, T_1] + [T_1, T_2]` относительно непрерывного `[0, T_2]`.
* **Двухфазная динамика банка на 6,000,000 историй ([`verification/profiling/batch_size_benchmark.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/batch_size_benchmark.py), Разделы 3.3, 7 и 8.2 [`PROFILING_REPORT.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/reports/profiling/PROFILING_REPORT.md)):**
  1. **Малые батчи (`2,500 – 5,000`):** требуют тысяч вызовов `Source.inject`; в воксельном фантоме `W2_NEMA_Tb161` генерация источника съедает **`32.5% – 34.8%`** времени стационарной фазы (`245–247 тыс. ист./с` против `413–437 тыс. ист./с` в оптимуме).
  2. **Оптимальный стационарный диапазон (`30,000 – 120,000`):** обеспечивает пик пропускной способности как в стационарной фазе `Full-Bank Sustained Phase`, так и по сквозному времени (**`2.13 – 2.16 млн ист./с`** в `W1_PointSource` при `30,000–60,000` и **`403.5 – 436.7 тыс. ист./с`** в `W2_NEMA_Tb161` при `60,000–120,000`), снижая долю `Source.inject` до `9.7%–10.8%` и удерживая долю хвостовой фазы `Tail Depletion Phase` в пределах `0.6%–1.1%`.
  3. **Сверхбольшие батчи (`250,000 – 500,000`) и замедление на хвосте (`Tail Depletion Phase`):** рабочий набор SoA-массивов (`20–40 МБ`) выходит за пределы L3-кэша CPU (снижая скорость `Full-Bank` в `W1` на `9%–14%`), а на хвостовой фазе долетывания последних частиц (когда среднее заполнение банка падает до `2.6%–30.1%`) удельная скорость чистого трекинга (`Треков/с`) падает в **`1.23x – 3.79x` раз** из-за разреженного массива `active_indices`.

### Согласованное архитектурное решение / План реализации:
1. **Точная обрезка по `stop_time` через пуассоновский поток распадов (реализуется в п. 15.4):**
   * `@njit`-инжектор генерирует кандидатов распада последовательно по времени и останавливается на первом кандидате с $t \geq$ `stop_time`, сохраняя его время как `pending_arrival_time` в состоянии досчета. Последовательность случайных чисел не зависит от границ батчей, поэтому досчет `Stateful Resume` побитово идентичен непрерывному расчету; метод Ньютона `_calculate_time_step` и `_distribute_quotas` удаляются.
2. **Явная параметризация `batch_capacity` и компактификация хвоста:**
   * Разделить емкость рабочего пула `ParticleBank` (оптимум `30,000 – 60,000` частиц) и квоту историй за время `stop_time`, а также рассмотреть периодическое уплотнение активных слотов на фазе `Tail Depletion Phase`.

---

## 25. Нагрузка на ядро ОС (Page Faults, переключения контекста) и NUMA-локальность при многопроцессном моделировании (`Orchestrator` / `SimulationManager` / `Source` / `WoodcockVoxelVolume`)

### Наблюдение:
* При боевом запуске 30 воркеров на 32-ядерном мульти-NUMA сервере (`AMD EPYC 7551P`, 4 кристалла / узла NUMA, 8 комплексов CCX по `8 МБ` L3-кэша) общесистемные счетчики Windows зафиксировали: `394,465 Page Faults/sec` (`~1.58 ГБ/с` страниц), `25%–48%` времени ядра (`Kernel Time`), `30,093 Context Switches/sec` (`~1,000/с` на процесс) и `> 4,400` потоков в системе.
* Атрибуция счетчиков к конкретным участкам кода выполнена по косвенным признакам (системные счетчики, а не стеки вызовов) и требует подтверждения измерением (см. п. 1 плана).

### Верифицированный по коду диагноз:
1. **Крупные временные массивы на шаге пополнения банка (подтверждено частично):**
   * В [`Source.generate_position`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py#L252-L260) вызов `self.rng.choice(voxel_origins, count, p=probability)` на каждом шаге строит массив `cdf` длины `K` (число активных вокселей; до `~12 МБ` при `K ≈ 1.5·10⁶`) и булеву маску проверки `p < 0` той же длины.
   * Это не единственный источник: на батч из `N` частиц `generate_position`, `generate_direction`, `convert_to_global_position` (`(N, 4)` + `matmul`) и `inject` (`astype`) создают около десятка временных массивов размером `N×3–4×8 байт`. Блоки свыше `~1 МБ` куча Windows (UCRT → `HeapAlloc`) выделяет через `VirtualAlloc` и возвращает через `VirtualFree`, поэтому каждое повторное касание порождает demand-zero page fault.
   * `generate_energy` к проблеме не относится: дискретный спектр содержит единицы линий, аллокации `< 1 КБ`.
2. **Безусловный поток `escaped_particles` (подтверждено и устранено в `460fbf4` / `9689ea6`):**
   * Ранее `SimulationManager._collect_escaped_particles` на каждом шаге `next_step` создавал 8 копий массивов (с двойным копированием: fancy indexing + `copy=True`) и выполнял `queue.put`, пробуждая поток `DataManager.run` и провоцируя передачу GIL, хотя ни один файловый обработчик ядра (`DirectStreamHandler`, `SensitiveVolumeHandler`, `HistoryAssemblerHandler`, `DoseMapHandler`) не потребляет `ESCAPED_PARTICLES` (единственный потребитель — [`GuiStreamDataHandler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/controllers/stream_handlers.py) при `show_escaped_tracks=True`).
3. **Карта материалов фантома в `int64` (арифметика подтверждено, механизм уточнен):**
   * В [`WoodcockVoxelVolume._compile_cfunc`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/geometry/voxel_volumes.py#L105-L134) массив `mat_dist_1d` хранится в `int64` (`12.06 МБ` для $128\times 128\times 92$) и захватывается замыканием `@cfunc`, т. е. вшивается константой в LLVM-модуль (раздувая также время и память компиляции).
   * Промах L3 на Zen 1 обслуживается контроллером памяти своего кристалла; межкристальный трафик по Infinity Fabric возникает только при обращении к удаленной NUMA-памяти (см. п. 4). Полное удержание фантома в L3 недостижимо при любом типе: на одном CCX работают 4+ воркера с собственными копиями фантома, таблиц сечений и SoA-банка частиц.
   * [`MaterialArray.ID`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/materials/materials.py#L111-L116) содержит **глобальный** идентификатор `material.ID` — порядковый номер `next(self.counter)` в `MaterialDataBase`, а не локальный индекс. Его значение может превышать `127`/`255` при любом числе материалов в фантоме, поэтому прямое приведение к `int8`/`uint8` недопустимо.
4. **Потоки BLAS и размещение процессов по NUMA (диагноз не подтвержден кодом):**
   * Лимиты `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `OMP_NUM_THREADS = 1` уже выставляются до импорта `numpy` в [`run_simulation.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/run_simulation.py#L1-L5) и [`gui/app.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/app.py#L13-L16); spawn-воркеры наследуют окружение, OpenBLAS при отсутствии `OPENBLAS_NUM_THREADS` использует `OMP_NUM_THREADS`. Наблюдение `> 4,400` потоков указывает на запуск через иную точку входа (verification-скрипты, `pytest`, Jupyter), в которой лимиты не заданы.
   * `NUMBA_NUM_THREADS` не влияет на работу: в `core/` отсутствуют `parallel=True` / `prange`.
   * Утверждение об «упаковке» процессов планировщиком Windows по 2 на физическое ядро не подтверждено измерением; планировщик SMT-осведомлен, а типичная причина подобной картины — парковка ядер (Core Parking) в плане электропитания `Balanced`.

### Согласованное архитектурное решение / План реализации:
1. **Измерение до оптимизации (обязательный первый этап):**
   * Расширить `step_profiler` пофазными дельтами `psutil.Process().num_page_faults()` и числом потоков процесса; для подтверждения стеков — трасса `WPR`/`WPA` с профилями `CPU` и `VirtualAllocation`.
   * Зафиксировать точку входа и окружение боевого запуска; проверить план электропитания (`High Performance`) и режим NUMA в BIOS сервера.
   * Все последующие пункты сопровождаются A/B-замером на том же стенде.
2. **[ВЫПОЛНЕНО] Декларативная подписка на `ChunkType(Enum)`, унификация `TelemetryBuffer` и `@njit`-буферизация с контролем переполнения (`460fbf4`, `9689ea6`, `f1da1f1`):**
   * Введены перечисление [`ChunkType`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/interaction_buffers.py#L15-L23) (`INITIAL_STATES`, `INTERACTIONS`, `DEAD_PARTICLES`, `ESCAPED_PARTICLES`), канонический причинно-следственный порядок каналов [`TELEMETRY_ORDER`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/interaction_buffers.py#L25-L30) (`INITIAL_STATES -> INTERACTIONS -> ESCAPED_PARTICLES -> DEAD_PARTICLES`) и протокол [`TelemetryBuffer`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/interaction_buffers.py#L36-L54) с единым методом `flush(clear=True)` и свойством `is_enabled`. В [`BaseDataHandler`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_handlers.py) введено свойство `consumed_chunk_types: FrozenSet[ChunkType]`, а [`DataManager`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py) предвычисляет таблицу маршрутизации `Dict[ChunkType, Tuple[BaseDataHandler, ...]]`.
   * Из [`ParticleBank`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/particles/particles.py) и [`InitialState`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/particles/initial_state.py) удален телеметрический признак `has_interacted`; [`EscapedParticlesBuffer`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/interaction_buffers.py#L340-L404) минимизирован до `(particle_ID, position)`. Факт отсутствия рассеяния определяется на лету через `@njit`-предикат [`_is_uncollided`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py) по неизменности энергии и направления относительно начального состояния.
   * Через фабрику [`SimulationDataBuffer.allocate_for(capacity, enabled)`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/interaction_buffers.py#L448-L459) неподписанным каналам выделяется `capacity == 0` без затрат памяти и записи в `@njit`-ядрах (с полным сохранением потока ГСЧ в `_sample_Z_scalar`). Во всех `@njit`-ядрах записи (`_write_single_initial_state`, `_push_to_dead_particles_kernel`, `_push_to_escaped_particles_kernel`, `_push_to_interaction_buffer`) кольцевой оператор `% capacity` заменен на линейный инкремент курсора с граничной отсечкой `idx >= capacity`.
   * В [`SimulationManager`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py) внедрены предусловие шага [`_ensure_step_capacity`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L236-L242) (каскадный сброс [`_flush_through`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L222-L227) в порядке `TELEMETRY_ORDER`), постусловие [`_check_no_overflow`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L244-L252) (строгий `RuntimeError`), кооперативная отправка в очередь [`_put_to_queue`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L195-L207) с таймаутом и проверкой `is_stopped`, а также устранены избыточные сканирования `bank.state.is_active` в `SimulationManager`, `ParticlePropagator.step` и `Source.inject`. По замерам `PROFILING_REPORT.md` доля буферного управления в `cProfile` снижена до **`1.3% – 2.4%`**, а доля фазы `P9_Invalidation_Escaped` — до **`2.0% – 2.8%`**.
3. **Устранение крупных временных аллокаций на шаге (реализация пп. 15 и 23.2):**
   * Выборка вокселя методом `Vose's Alias Method` по таблицам, построенным один раз при инициализации источника: `alias: int32`, `prob: float64` (точность `float32` недопустима по принципу Zero-Compromise Physics).
   * Генерация позиций, направлений и энергий в предвыделенные SoA-буферы (`out=`-аргументы или `@njit`-инжектор с явным `np.random.Generator`, пишущий напрямую в `ParticleBank`), без промежуточных `(N, 3)` / `(N, 4)` массивов.
   * Смена потока случайных чисел требует замены побитовых эталонных тестов статистическими (критерий $\chi^2$ частот выборки вокселей).
4. **Компактная карта материалов в `WoodcockVoxelVolume`:**
   * Хранить в `mat_dist_1d` локальный индекс материала в `material_list` типа `uint8` и передавать в `@cfunc` малую таблицу отображения `local_to_global: int64[≤256]` (размещается в L1).
   * При компиляции проверять предусловие `len(material_list) <= 256` с немедленным отказом (Fail-Fast) при нарушении.
   * Ожидаемый эффект умеренный (снижение промахов кэша и объема вшитой константы `12.06 МБ → 1.50 МБ`); подтверждается A/B-замером.
5. **Лимит потоков BLAS и опциональная привязка воркеров к ядрам:**
   * Дополнительно к установке в точках входа (необходимой для собственного процесса и режима `pool_size == 1`, т. к. к моменту вызова оркестратора `numpy` уже импортирован) выставлять `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` в [`Orchestrator.run`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/orchestrator.py#L72-L143) до создания `Pool` (наследуется spawn-воркерами независимо от точки входа) либо применять `threadpoolctl.threadpool_limits(1)` в воркере. Установка переменных окружения в инициализаторе пула неэффективна: на Windows модули с `numpy` импортируются при распаковке задачи до вызова инициализатора.
   * Привязку воркеров к физическим ядрам равномерно по узлам NUMA реализовать как опцию конфигурации (`cpu_binding: Literal["none", "physical_cores"] = "none"`), выполняемую в начале воркера до построения сцены (NUMA-локальность по first-touch). Соседние SMT-потоки определять по топологии ОС (`GetLogicalProcessorInformationEx` / `os.sched_getaffinity`), а не по формуле `[2*k, 2*k+1]`. По умолчанию выключено, чтобы не лишать GUI процессорного времени на рабочих станциях.


---

## 26. Перегенерация таблиц NIST XCOM с краями поглощения и согласование энергетического диапазона

### Проблема:
* **Края поглощения размыты в самих данных** `tables/NIST XCOM Elements MAC.h5`: повторяющихся узлов на энергиях краев нет. Пример Pb: K-край `88.005 кэВ` лежит между узлами `88.0 кэВ` ($\mu_{pe} = 2.48 \cdot 10^{-20}$) и `89.0 кэВ` ($1.14 \cdot 10^{-19}$) — скачок в `4.6` раза линейно размазан на `1 кэВ`; для L3-края (`13.035 кэВ`) присутствует только узел `13.04 кэВ` со значением **ниже** края, значение над краем отсутствует.
* Ветка сдвига дублирующихся узлов на `-1 эВ` в [`AttenuationDataBase._add_material`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/materials/attenuation_database.py#L82-L84) на текущих данных не срабатывает (мертвый код), а `np.unique` при смешивании сеток элементов схлопнул бы пары узлов краев.
* **Рассогласование диапазона:** таблица ограничена `10 МэВ`, тогда как [`Process._energy_range`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes.py#L48) объявляет `20 МэВ`; выше `10 МэВ` [`_get_macroscopic_cross_sections`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/physics_kernels.py#L40-L42) молча держит последнее значение.

### Согласованное архитектурное решение:
1. Воспроизводимый офлайн-скрипт генерации HDF5-таблицы: края поглощения кодируются парой узлов `(E_edge, μ_ниже)`, `(E_edge, μ_выше)`; дополнительно для каждого элемента сохраняется датасет энергий и меток краев (`K`, `L1`, `L2`, `L3`, `M1…`) для п. 28.
2. Смешивание материалов с сохранением пар узлов: объединение сеток элементов без схлопывания пар краев, лево-/право-непрерывная интерполяция каждого элемента на краях; удаление сдвига `-1 эВ`.
3. Согласование диапазона: либо данные до `20 МэВ`, либо `energy_range = [1 кэВ, E_max таблицы]`; предусловие (Fail-Fast) в `PhysicsCompiler`: все линии спектров источников лежат внутри диапазона таблицы.

### План реализации:
* Фаза 1 плана [`docs/plans/cross_sections_and_source_sampling_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/cross_sections_and_source_sampling_plan.md).

---

## 27. Выбор элемента-мишени взаимодействия по парциальным сечениям процесса

### Проблема:
* [`_sample_Z_scalar`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py#L82-L104) выбирает атомный номер по **массовой доле** элемента (`ElementCSR.element_fraction`), тогда как физически вероятность выбора элемента $i$ для процесса $p$ равна $w_i (\mu/\rho)_{p,i}(E) / \sum_j w_j (\mu/\rho)_{p,j}(E)$.
* Пример NaI при 140 кэВ: массовая доля Na `≈ 15.3%`, а его вклад в фотоэффект `≈ 0.1%` — около 15% событий фотоэффекта получают неверный $Z$.
* Сейчас $Z$ используется как метка в `InteractionBuffer.Z`; при появлении атомной релаксации (п. 28) и функций $S(x, Z)$ / $F(x, Z)$ для рассеяния ошибка становится физической.

### Согласованное архитектурное решение:
1. Замена `ElementCSR.element_fraction` таблицей парциальных макроскопических сечений элементов `element_lac_table` на строках точной сетки материала (`[строка, слот элемента, процесс]`, дополнение нулями до максимального числа элементов в материале).
2. В ядре процесса: та же строка и доля интерполяции, что и в поиске интервала (п. 14); кумулятивный проход по элементам с порогом $u \cdot \mu_p(E)$. Линейность интерполяции гарантирует согласованность суммы парциальных сечений с полным $\mu_p(E)$.

### План реализации:
* Фаза 2 плана [`docs/plans/cross_sections_and_source_sampling_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/cross_sections_and_source_sampling_plan.md).

---

## 28. Атомная релаксация после фотоэффекта: выбор оболочки и характеристическая флуоресценция

### Проблема:
* Фотоэффект ([`make_photoelectric_kernel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py#L107-L170)) депонирует всю энергию фотона локально; характеристическое излучение не моделируется.
* Пропущенные эффекты, значимые для ОФЭКТ: K-излучение свинца коллиматора и защиты (`72–87 кэВ`, попадает в окна $^{201}\text{Tl}$ и низкоэнергетических линий $^{161}\text{Tb}$), пики вылета йода в NaI (фотопик минус `≈ 28.5 / 32.3 кэВ`, линии Kα / Kβ), искажение спектра и окон рассеяния.
* Номер сегмента энергетической сетки определяет лишь энергетически доступные оболочки ($E > E_K$), но не оболочку поглощения — для выбора требуются вероятности по каналам.

### Согласованное архитектурное направление:
1. **Выбор оболочки** (после выбора элемента по п. 27): либо метод скачков $P_K = 1 - 1/J_K$ с рекурсивным делением остатка по L1/L2/L3 (скачки $J$ извлекаются из пар узлов краев п. 26), либо точные сечения подоболочек (EPDL97 / Scofield) как дополнительные столбцы на той же сетке — выбор на этапе проектирования.
2. **Релаксация:** выход флуоресценции $\omega_s$, энергии линий и радиационные вероятности (Kα1, Kα2, Kβ1, Kβ2, опционально L-линии); Оже- и Костер–Крониг-переходы депонируются локально; каскад вакансий K → L — опционально.
3. **Вторичный фотон:** изотропная эмиссия из точки взаимодействия, что требует нового механизма инжекции вторичных частиц в `ParticleBank` во время переноса с идентификатором родителя (влияет на `InitialState`, `HistoryAssemblerHandler` и схему HDF5).
4. **Данные атомной релаксации** запекаются офлайн в HDF5 (источник — EADL / `xraylib` только в скрипте генерации, без зависимости во время выполнения).

### План реализации:
* Фаза 6 плана [`docs/plans/cross_sections_and_source_sampling_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/cross_sections_and_source_sampling_plan.md) (требует отдельной проектной спецификации перед реализацией).

---

## 29. Атомарность сброса группы в `DatasetWriteBuffer` при частичной записи

> **Статус: отложено.**

### Проблема:
* [`DatasetWriteBuffer`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/dataset_write_buffer.py) сбрасывает группу датасетов (`initial_states`, `interactions/<объём>`) поле за полем внутри одного вызова `write_func`, который выполняется через `DataManager._write_with_retry`.
* Если `write_func` падает на середине группы (например, `OSError` на записи второго поля), ретрай `_write_with_retry` повторно вызывает запись, а в файле часть полей группы уже стала длиннее (после `resize` и частичной записи).
* Результат: проверка согласованности длин полей группы бросает `RuntimeError` «длины различаются» (Fail-Fast). Тихой порчи данных нет, но прогон завершается ошибкой вместо успешного ретрая.

### Согласованное архитектурное решение:
1. Фиксировать длину $L$ датасетов группы до начала сброса; при исключении откатывать `resize` всех уже изменённых полей до $L$ (`try/finally` без широкого `except`), чтобы повторный вызов `write_func` стартовал из согласованного состояния.
2. Покрыть откат тестом: инъекция `OSError` на записи второго поля группы, затем успешный ретрай и проверка равенства длин всех полей.

### План реализации:
* Доработка `DatasetWriteBuffer.write_full_chunks` / `write_all` (откат `resize` по `try/finally`) и тест в `tests/data/test_dataset_write_buffer.py`; выполняется отдельной задачей после этапов 1–4 плана [`docs/plans/data_manager_write_buffer_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/data_manager_write_buffer_plan.md).
