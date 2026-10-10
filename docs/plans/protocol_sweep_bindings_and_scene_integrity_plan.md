# План архитектурных доработок: Протоколы сканирования, переменные биндингов (UE-Style) и целостность сцены

## 1. Контекст и цели

Настоящий документ регламентирует комплексные архитектурные и инженерные доработки подсистем протоколов сканирования, сериализации конфигураций, привязки параметров (Property Binding) и моделирования спектров источников излучения в программном комплексе **NMSimToolkit**.

### Ключевые выявленные дефекты:
1. **Потеря пула задач (`pool_size`)**:
   - `yaml_dumper.py` отбрасывает ключ `pool_size` при формировании словаря с якорными материалами `Materials`.
   - Рассинхронизация между `sim_settings.pool_size` и `orchestrator_session.pool_size` в GUI.
2. **Деградация сцены при цикле сохранения/загрузки**:
   - В `WoodcockVoxelVolume` экспортер заменяет оригинальную карту плотностей `mapping` на фиктивные индексы `0.0: Vacuum, 1.0: Air`, из-за чего при повторной загрузке воксели не сопоставляются материалам и становятся вакуумом (`opacity = 0`).
   - В `Source` активность экспортируется как голое число без единиц измерения, что приводит к занижению на 6 порядков при парсинге через Pint.
   - Инициализация `SourceViewModel._sync_from_core()` через setter `gui_field` затирает сложный спектр ядра одноэлементным массивом (Feedback Mutation).
3. **Отсутствие поддержки сложных спектров в GUI**:
   - В ядре и GUI сохраняются дублирующие ветвления «скаляр vs список» и флаги вида `is_discrete_spectrum`.
4. **Дублирование интерфейса протоколов ОФЭКТ/ПЭТ**:
   - Необходимость полного упразднения протоколов `SPECT` и `PET` (без алиасов обратной совместимости, строго по правилу *No Legacy Crutches*).
   - Переход на обобщенный `StepAndShoot` (шаги гантри `steps` вместо `views`, массив радиусов `radius`) и параметрический `CustomSweep`.
5. **Отсутствие связи параметров сцены с переменными сканирования (`${variable}`)**:
   - Необходимость введения системы биндингов (Property Binding) по аналогии с Variables в Blueprints / UMG Unreal Engine с поддержкой двустороннего импорта/экспорта.

---

## 2. Архитектура ядра и протоколов (Core & Sweep)

### 2.1. Контракт `StepAndShootProtocolConfig` ([core/config/models.py](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/models.py))
* Полное исключение геометрических параметров камер (`gamma_cameras`, `head_angles`). Геометрия детекторов определяется исключительно структурой графа сцены (`GantryNode -> GammaCameraNode`).
* Параметр количества шагов: строго **`steps: int = Field(ge=1)`** (число ракурсов не зависит от детекторов).
* Переменный радиус орбиты: **`radius: Optional[Union[LengthConfig, List[LengthConfig]]] = None`** (поддержка контурных орбит Body Contour).
* Полное удаление классов `SpectProtocolConfig` и `PetProtocolConfig` без создания промежуточных алиасов:

```python
class StepAndShootProtocolConfig(BaseProtocolConfig):
    type: Literal['StepAndShoot'] = 'StepAndShoot'
    steps: int = Field(ge=1, description="Количество физических угловых положений (шагов) гантри")
    start_angle: AngleConfig = 0.0 * units.deg
    end_angle: AngleConfig = 360.0 * units.deg
    endpoint: bool = False
    time_per_step: TimeConfig = 1.0 * units.s
    radius: Optional[Union[LengthConfig, List[LengthConfig]]] = None

    @model_validator(mode='after')
    def validate_radius_length(self) -> 'StepAndShootProtocolConfig':
        if isinstance(self.radius, list) and len(self.radius) != self.steps:
            raise ValueError(
                f"Длина массива радиусов ({len(self.radius)}) обязана строго совпадать "
                f"с числом шагов steps ({self.steps})"
            )
        return self

AnyProtocolConfig = Annotated[
    Union[StepAndShootProtocolConfig, CustomSweepProtocolConfig],
    Field(discriminator='type')
]
```

### 2.2. Генератор матрицы задач ([core/config/sweep_compiler.py](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/sweep_compiler.py))
* `SweepCompiler.compile_protocol(protocol)`:
  - Формирует `zipped_variables`:
    - `gantry_angle`: список углов `np.linspace(start, end, steps, endpoint=endpoint)`.
    - `step_time`: список времен `[time_per_step] * steps`.
    - `radius` (если задан): список радиусов на каждый шаг.
  - Поворот камер происходит автоматически за счет матричной кинематики родительского узла `GantryNode`.

---

## 3. Система биндингов параметров в GUI (UE-Style Property Binding)

### 3.1. Базовый виджет числовых свойств с опцией бинда (`BindableDoubleField`)
Создается универсальный компонент пользовательского интерфейса, объединяющий:
1. Поле ввода (`QDoubleSpinBox`) с поддержкой суффиксов единиц измерения (`мм`, `°`, `с`, `кэВ`).
2. Компактную кнопку-индикатор бинда **`[🔗 Bind]`**:
   - При клике открывается меню `QMenu`:
     - `[None] (Статическое значение)` — отвязать свойство;
     - Разделитель;
     - Список доступных переменных из активного протокола (например: `gantry_angle`, `radius`, `step_time`);
     - Разделитель;
     - `[➕ Создать переменную протокола...]` — вызывает инлайн-диалог создания переменной в `CustomSweep` (имя, распределение: Grid/Zipped, начальный диапазон).
3. При активной привязке:
   - Спинбокс блокируется для ручного ввода (`read-only`) и подсвечивается акцентным цветом (синяя рамка / бейдж `🔗 gantry_angle`).
   - Значение в поле динамически отображает текущую величину переменной для выбранного в `JobsWidget` ракурса (Preview View).

### 3.2. Архитектура связывания и двусторонний импорт/экспорт
* **Без промежуточных словарей**: у каждого поля ввода `BindableDoubleField` хранится простая строка:
  ```python
  bound_variable: Optional[str] = None  # например, "gantry_angle" или None
  ```
* **Автоматические Zero-Config бинды для типовых узлов**:
  - Узел `Gantry`: поле поворота станины `gantry_angle` связывается с переменной `gantry_angle` автоматически по умолчанию.
  - Каретки детекторов: при наличии переменной `radius` в протоколе — радиальное смещение автоматически связывается с `radius`.
  - Ручная настройка через `[🔗 Bind]` требуется только для пользовательских нестандартных сценариев (`CustomSweep`).
* **Двусторонний импорт**:
  - При парсинге YAML в `SceneBuilder`: если значение свойства представляет собой шаблон интерполяции (например, `alpha: "${gantry_angle} deg"`), у соответствующего поля проставляется `bound_variable = 'gantry_angle'`.
* **Экспорт**:
  - В `SceneExporter`: если `bound_variable is not None`, экспортер генерирует строковый шаблон вида `f"${bound_variable} deg"`, иначе пишет статическое число.


---

## 4. Унификация спектров источника и устранение деградации данных

### 4.1. Единый контракт спектра (список пар `[энергия, вес]`)
* Отказ от флагов `is_discrete_spectrum` и скалярных развилок.
* Моноэнергетический источник представляется единым списком из одной пары: `[[140.5, 1.0]]`.
* Сложный полиэнергетический спектр: список пар `[[74.57, 10.3], [48.9, 17.0], ...]`.
* Нормализация суммы весов ($\sum w_i = 1.0$) производится автоматически «под капотом» при передаче в `core_node.energy`.
* **Устранение Feedback-мутации**:
  - В `SourceViewModel._sync_from_core()` мутация ядра блокируется контекстным менеджером `with self._block_mutation():`, исключая уничтожение многолинейного спектра при инициализации интерфейса.

### 4.2. Корректная сериализация активности и карты материалов
* В `SceneExporter.export_node`:
  - `activity` источника сохраняется с явной строковой единицей измерения (`f"{act_mbq:.4f} MBq"`).
  - Исходный `mapping` в `WoodcockVoxelVolume` сохраняется из `distribution_registry` без перезаписи индексами `0.0, 1.0`.

---

## 5. Пошаговый план реализации

| Этап | Задачи | Затронутые модули |
| :--- | :--- | :--- |
| **Этап 1** | Доработка `yaml_dumper.py` (сохранение `pool_size`, `protocol`), синхронизация в `main_window.py`. Устранение перезаписи `mapping` и искажения активности. | `core/config/yaml_dumper.py`, `core/config/exporter.py`, `gui/views/main_window.py` |
| **Этап 2** | Рефакторинг `StepAndShootProtocolConfig` (`steps`, переменный `radius`), удаление `SpectProtocolConfig`. Доработка `SweepCompiler`. | `core/config/models.py`, `core/config/sweep_compiler.py` |
| **Этап 3** | Унификация спектра в `Source` и `SourceViewModel` как списка пар `[E, w]` без флагов. Блокировка feedback-мутации. | `core/source/sources.py`, `gui/viewmodels/nodes/source_vm.py` |
| **Этап 4** | Создание виджета `BindableDoubleField`, интеграция меню биндов в `PropertyInspector`, поддержка двустороннего импорта/экспорта `${var}`. | `gui/views/widgets/bindable_field.py`, `gui/views/property_inspector.py`, `core/config/builder.py`, `core/config/exporter.py` |
| **Этап 5** | Обновление `ProcedureSelectorWidget` (`StepAndShoot`, `CustomSweep`) и таблицы спектра в `PropertyInspector`. | `gui/views/procedure_selector_widget.py`, `gui/viewmodels/procedure_viewmodel.py` |
| **Этап 6** | Сквозные тесты: полный цикл round-trip для `tb161_nema_protocol direct.yaml` с проверкой сохранения всех 10 линий спектра, фантома и пула задач. | `tests/config/`, `tests/test_gui_integration_and_fixes.py` |
