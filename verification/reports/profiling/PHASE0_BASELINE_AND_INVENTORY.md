# Фаза 0: Базовые замеры и архитектурная инвентаризация (`cross_sections_and_source_sampling_plan.md`)

Данный документ фиксирует результаты выполнения **Фазы 0 (`Базовые замеры и инвентаризация`)** плана оптимизации интерполяции макроскопических сечений (`MaterialBank` / `AttenuationDataBase`) и генерации первичных частиц (`Source` / `ParticleBank`).

---

## 1. Базовые замеры производительности и Page Faults (`W1–W3`, `M1`, `M3`)

### 1.1. Микробенчмарки `M1` (`M1a_CompactLinear`, `M1b_CompactLogLog`) и `M3` (`micro_benchmarks.py`)

В `verification/profiling/micro_benchmarks.py` микробенчмарк `M1` дополнен прямым сопоставлением двух вариантов интерполяции на компактной сетке NIST (`~70` узлов на материал с парами точек $(E_{\text{edge}}, E_{\text{edge}})$ на всех краях фотоэлектрического поглощения и двухуровневым логарифмическим индексом `bin_start[mat_id, b]` на `256` корзин) против текущего бинарного поиска `np.searchsorted` на сетке `MaterialBank` (`~1117` узлов в среднем по материалам сцены `W2_NEMA_Tb161`). Кроме того, устранено искажение старого `M1`, в котором `test_mat_ids` выбирались по всем `126` слотам `mat_pointers` (где `120` слотов имели `length == 0` и вызывали ранний `return`).

| ID микробенчмарка | Описание узла | База (Текущая реализация) | Альтернатива (Целевая схема) | База (нс/оп) | Альтернатива (нс/оп) | Ускорение (`Speedup`) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `M1a_CompactLinear` | Интерполяция 3 компонент $(\sigma_{\text{pe}}, \sigma_{\text{comp}}, \sigma_{\text{coh}})$ (5 млн запросов) | `np.searchsorted` $O(\log K)$ (`~1117` узлов) | 2-уровневый лог-индекс (`256` корзин) + линейная интерполяция (`~70` узлов) | **57.6** | **11.5** | **5.01x** |
| `M1b_CompactLogLog` | Интерполяция 3 компонент $(\sigma_{\text{pe}}, \sigma_{\text{comp}}, \sigma_{\text{coh}})$ (5 млн запросов) | `np.searchsorted` $O(\log K)$ (`~1117` узлов) | 2-уровневый лог-индекс (`256` корзин) + кусочная log-log (`math.exp`, `~70` узлов) | **57.6** | **28.5** | **2.02x** |
| `M3_VoxelPositionSampling` | Генерация 500 тыс. позиций частиц в 3D воксельном фантоме `W2` | `rng.choice(origins, p=prob)` $O(N \log M)$ + временные массивы | Таблица алиасов Воза (Vose's Alias $O(1)$) + прямая запись в SoA | **124.0** | **19.8** | **6.28x** |

* **Соотношение `M1a` vs `M1b`:** Кусочно-линейная интерполяция с предвычисленными наклонами (`11.5 нс/оп`) работает в **2.48x раза быстрее**, чем кусочная log-log интерполяция с вызовом `math.exp` (`28.5 нс/оп`, где `~17.0 нс` тратится на трансцендентные функции `math.log` и `3x math.exp`).

---

### 1.2. Пошаговый профиль времени и дельты Page Faults на стационарной фазе (`step_profiler.py`)

В `verification/profiling/step_profiler.py` внедрен пофазный сбор дельт страничных прерываний ОС (`PageFaultCount` через `PROCESS_MEMORY_COUNTERS`) для всех 10 фаз шага и подфаз `Source.inject`. Замеры разделены на:
* **`Full-Bank Phase`** — все шаги с полным банком частиц (`25,000` частиц), включая шаг №1;
* **`Steady Full-Bank`** (в скобках) — шаги полного банка со **2-го шага и далее**, когда предвыделенные рабочие буферы `ParticleBank`, `TransportBuffer`, `InteractionBuffer` и `SimulationDataBuffer` уже полностью отображены на физические страницы ОЗУ.

| Сцена и коллиматор | Историй (`Шагов Full/Depl`) | Скорость Warm (ист./с) | `P2_Source` доля (%) / время (мс) | `P5_Woodcock` доля (%) / время (мс) | `Full` P2 Source PF (`Steady PF`) | `Full` P4 Geom PF (`Steady PF`) | `Full` P5 Woodcock PF (`Steady PF`) | `src.position` PF | `src.direction` PF | `src.bank_inject` PF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | 300,000 (`24 / 5`) | **933,546** | 8.4% (`26.98`) | 16.7% (`53.61`) | **3,086** (`2,058`) | 0 (`0`) | 92 (`0`) | 1,743 | 726 | 456 |
| `W1_PointSource_Direct` | 300,000 (`29 / 29`) | **872,758** | 7.9% (`27.18`) | 10.6% (`36.46`) | **1,812** (`762`) | 0 (`0`) | 96 (`0`) | 1,088 | 541 | 42 |
| `W2_NEMA_Tb161_Parametric` | 150,000 (`18 / 18`) | **264,843** | 8.6% (`48.68`) | 27.0% (`152.82`) | **7,821** (`5,713`) | 0 (`0`) | 96 (`0`) | **6,654** | 569 | 764 |
| `W2_NEMA_Tb161_Direct` | 150,000 (`19 / 69`) | **276,218** | 9.7% (`52.67`) | 22.2% (`120.55`) | **7,722** (`6,141`) | 0 (`0`) | 95 (`0`) | **7,102** | 628 | 258 |
| `W3_Heart_Tc99m_Parametric` | 150,000 (`27 / 21`) | **177,744** | 10.6% (`89.46`) | 34.3% (`289.49`) | **21,810** (`19,986`) | 0 (`0`) | 48 (`0`) | **21,929** | 519 | 58 |
| `W3_Heart_Tc99m_Direct` | 150,000 (`29 / 34`) | **179,667** | 11.3% (`94.36`) | 31.9% (`266.37`) | **23,490** (`21,839`) | 0 (`0`) | 0 (`0`) | **23,573** | 407 | 54 |

* **Ключевой вывод для критерия приемки Фазы 4 (Zero-Allocation инжектора):**
  На установившихся шагах полного банка (`Steady Full-Bank`) транспортные JIT-ядра `P4_Geometry_Raycast` и `P5_Woodcock_Transport` дают **строго `0` page faults**, тогда как `P2_Source_Inject` порождает **`5,713..6,141` page faults** на `W2_NEMA_Tb161` и **`19,986..21,839` page faults** на `W3_Heart_Tc99m` из-за аллокаций временных массивов в `Source.generate_position` (`rng.choice` + `rng.uniform`), `Source.generate_direction`, упаковки `Vector3DSoA` и `ParticleBank.inject_particles`.

---

### 1.3. Алгоритмическая телеметрия переноса (`transport_telemetry.py`, `12,000` историй)

| Сцена и коллиматор | Лучей / историю | Примитивов / луч | Шагов Вудкока / историю | Доля холостых шагов Вудкока (%) | Вызовов `np.searchsorted` / историю | Вызовов `@cfunc` / историю |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | 2.03 | 2.09 | 1.53 | **83.1%** | **2.77** | 1.51 |
| `W1_PointSource_Direct` | 2.63 | 3.30 | 0.26 | **0.0%** | **1.86** | 0.00 |
| `W2_NEMA_Tb161_Parametric` | 3.40 | 3.89 | 7.99 | **72.4%** | **11.02** | 7.96 |
| `W2_NEMA_Tb161_Direct` | 3.58 | 4.19 | 4.64 | **51.5%** | **7.72** | 4.46 |
| `W3_Heart_Tc99m_Parametric` | 5.00 | 3.81 | 13.46 | **78.5%** | **17.82** | 13.44 |
| `W3_Heart_Tc99m_Direct` | 5.41 | 4.27 | 11.74 | **75.3%** | **16.28** | 11.49 |

---

## 2. Критическое открытие по структуре `tables/NIST XCOM Elements MAC.h5` и `AttenuationDataBase`

В ходе выполнения Фазы 0 проведена прямая инспекция HDF5-базы `tables/NIST XCOM Elements MAC.h5` и класса `AttenuationDataBase` (`core/materials/attenuation_database.py`), выявившая фундаментальную причину раздувания таблиц `MaterialBank` до `~1117` узлов и скрытый дефект обработки краев поглощения:

1. **Пред-интерполированная 1-кэВ сетка вместо компактной сетки NIST XCOM:**
   В файле `tables/NIST XCOM Elements MAC.h5` для всех 100 элементов ($Z = 1..100$) записана **не** оригинальная логарифмическая сетка NIST XCOM (`~50–85` узлов), а плотная сетка из **`1113..1131` узлов** на диапазоне `[0.001, 10.0]` МэВ:
   * `998` узлов с равномерным шагом **`1 keV`** на интервале `[0.001, 1.0]` МэВ (именно этот участок доминирует в ядерной медицине);
   * `113` узлов на интервале `(1.0, 10.0]` МэВ;
   * от `0` до `18` дополнительных точек вблизи оболочечных краев поглощения (`H`, `O`: `1113` точек; `Na`: `1114`; `I`: `1120`; `Pb`: `1125`).
2. **Полное отсутствие дублирующихся узлов `E[k] == E[k+1]` в HDF5-файле (`counts > 1` всегда пусто):**
   Проверка всех 100 элементов в `tables/NIST XCOM Elements MAC.h5` показала, что **`np.unique(energy, return_counts=True)` не находит ни одного дубликата ни для одного элемента** (`elements_with_duplicates == []`).
   Вследствие этого блок кода в `core/materials/attenuation_database.py:82-84`:
   ```python
   _, indices, counts = np.unique(energy, return_index=True, return_counts=True)
   indices_of_shells = indices[counts > 1]
   energy[indices_of_shells] -= 1.0 * units.eV
   ```
   является **на 100% мертвым кодом** (`indices_of_shells` всегда пустой массив).
3. **Размытие скачков краев поглощения на шаг сетки (~1 кэВ) в `NIST XCOM Elements MAC.h5`:**
   При прежней генерации `NIST XCOM Elements MAC.h5` пары точек $(E_{\text{edge}}, \mu^-)$ и $(E_{\text{edge}}, \mu^+)$ были схлопнуты: левый предел сечения $\mu^-$ остался в точке $E_{\text{edge}}$, а правый предел $\mu^+$ оказался на следующем узле регулярной 1-кэВ сетки. Например, для свинца (`Pb`, $Z=82$):
   * $K$-край (`88.005 кэВ`): узел `100` имеет $E = 0.088\text{ МэВ}, \sigma_{\text{pe}} = 2.478 \times 10^{-20}\text{ см}^2$, а правый предел скачка находится в узле `101` при $E = 0.089\text{ МэВ}, \sigma_{\text{pe}} = 1.139 \times 10^{-19}\text{ см}^2$ (ширина перехода `1 кэВ`);
   * $L_3$-край (`13.035 кэВ`): узел `18` имеет $E = 0.01304\text{ МэВ}$, а правый предел — в узле `19` при $E = 0.014\text{ МэВ}$ (ширина перехода `0.96 кэВ`).
4. **Архитектурное следствие для Фазы 1:**
   Простая замена `np.interp` на объединении сырых массивов `energy` из `NIST XCOM Elements MAC.h5` **не уменьшит** число узлов с `~1117` до `~70`, так как сами элементные таблицы в `.h5` содержат по `1113+` точек. Чтобы получить в `MaterialBank` компактную сетку NIST (`~70` узлов) или сохранить точность краев поглощения, в Фазе 1 необходимо либо перегенерировать/очистить элементные сетки до опорных логарифмических узлов NIST + истинных пар краев поглощения (детектируемых по условию `pe[i+1] > pe[i]`), либо явно учесть 1-кэВ природу текущего `.h5` файла.

---

## 3. Полная архитектурная инвентаризация кодовой базы (5 пунктов Фазы 0)

### 3.1. Пункт 1: Все потребители `AttenuationDataBase`, `AttenuationFunction` и `np.interp`

1. **`AttenuationDataBase`** ([`core/materials/attenuation_database.py:24-103`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/materials/attenuation_database.py#L24-L103)):
   * Наследует `Dict[Material, NDArray[Any]]` и хранит поле `_elements_MAC: Dict[str, NDArray[Any]]` (`строка 28`).
   * Использует модульный словарь `processes_names` (`строки 15-19`: `'Photoelectric absorption' -> 'PhotoelectricEffect'`, `'Incoherent scattering' -> 'ComptonScattering'`, `'Coherent scattering' -> 'CoherentScattering'`), `processes_dtype` (`строка 20`) и структурированный тип `MAC_dtype = np.dtype([('Energy', Float), ('Coefficient', processes_dtype)])` (`строка 21`).
   * При инициализации (`__init__`, `строки 30-35`) вызывает `self._load_elements_MAC()` (`строки 46-58`), который читает `tables/{self._base_name}.h5` и заполняет `self._elements_MAC[str(element)]` структурированными массивами `MAC_dtype` с полями `'Energy'` и `'Coefficient'` (`'PhotoelectricEffect'`, `'ComptonScattering'`, `'CoherentScattering'`).
   * Методы `add_material(material)` (`строки 60-62`), `add_materials(materials)` (`строки 64-67`) и `_add_material(material)` (`строки 69-102`) формируют объединенную сетку энергий материала `array_of_energy = np.unique(np.concatenate(list_of_energy))` (`строка 90`), создают массив `array_of_mac` типа `MAC_dtype` (`строки 92-93`), суммируют взвешенные элементные коэффициенты через `np.interp(array_of_energy, energy, mac_arr[process_name])` (`строки 95-100`) и сохраняют таблицу в `self[material]` через `self.update({material: array_of_mac})` (`строка 102`).
   * [`settings/database_setting.py:5-7`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/settings/database_setting.py#L5-L7) — создает синглтоны `material_database = MaterialDataBase()`, `attenuation_database = AttenuationDataBase()` и заполняет базу вызовом `attenuation_database.add_materials(material_database.values())` (`строка 7`).
   * [`core/materials/__init__.py:8-9,17-18`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/materials/__init__.py#L8-L9) — реэкспорт `AttenuationDataBase` и `AttenuationFunction`.
   * [`core/physics/processes.py:27-155`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes.py#L27-L155) — базовый класс `Process`:
     * В `__init__` (`строки 38-49`) сохраняет `self.attenuation_database` (по умолчанию `settings.attenuation_database`), задает `self._energy_range = np.array([1.0 * units.keV, 20.0 * units.MeV], dtype=Float)` (`строка 48`) и вызывает `self._construct_attenuation_function()` (`строки 51-52`, также вызывается в `energy_range.setter` на `строке 67`), создавая `self.attenuation_function = AttenuationFunction(self, self.attenuation_database)`.
     * Свойство `name` (`строки 54-57`) возвращает `self.__class__.__name__`.
     * Подклассы в [`core/physics/processes.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes.py):
       * `PhotoelectricEffect(Process)` (`строки 93-105`, `name == 'PhotoelectricEffect'`, `process_id = ProcessID(0)`);
       * `CoherentScattering(Process)` (`строки 107-125`, `name == 'CoherentScattering'`, `process_id = ProcessID(2)`);
       * `ComptonScattering(CoherentScattering)` (`строки 127-149`, `name == 'ComptonScattering'`, `process_id = ProcessID(1)`);
       * `PairProduction(Process)` (`строки 151-155`, `name == 'PairProduction'`, `process_id = ProcessID(3)`).
   * [`core/transport/propagator.py:35-56`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/propagator.py#L35-L56) — `ParticlePropagator.__init__` разрешает `resolved_database` (по умолчанию `database_setting.attenuation_database`, `строки 44-46`) и инстанциирует процессы `self.processes = [process_cls(resolved_database, self.rng) for process_cls in resolved_processes]` (`строка 49`).
   * [`core/physics/physics_compiler.py:28-85`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/physics_compiler.py#L28-L85) — `PhysicsCompiler._build_material_bank(materials_list, processes_list)` извлекает сетку энергий из первого процесса `energy_grid, _ = first_proc.attenuation_function[material]` (`строки 52-54`) и заполняет столбцы матрицы ЛКО `lac_matrix[:, p_idx] = process.attenuation_function(material, energy_grid)` (`строки 66-69`).
   * [`gui/viewport_3d/material_palette.py:241-285`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/material_palette.py#L241-L285) — функция `compute_material_linear_attenuation(material_name, energy=140.0 * units.keV)`: обращается к `mac_table = attenuations_db[target_material]` (`строка 267`), берет `energy_array = mac_table['Energy']` (`строка 268`) и `coefficients_table = mac_table['Coefficient']` (`строка 269`), суммирует все поля `coefficients_table.dtype.names` (`строки 271-280`), интерполирует массовый коэффициент через `np.interp(energy_table_scale, energy_array, total_mass_attenuation_array)` (`строка 282`) и умножает на `target_material.density` (`строка 283`).
   * [`verification/common/presets.py:287-315`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/common/presets.py#L287-L315) — функция `compute_total_linear_attenuation(material, energy, processes=None)`: при `processes is None` создает `[PhotoelectricEffect(database_setting.attenuation_database, rng), ComptonScattering(database_setting.attenuation_database, rng), CoherentScattering(database_setting.attenuation_database, rng)]` (`строки 299-305`) и суммирует `proc.attenuation_function(material, query_energy)` (`строки 311-314`).
   * Профилировочные модули и примеры: [`verification/profiling/micro_benchmarks.py:226,267-306`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/micro_benchmarks.py#L226-L306) (обращение к `attenuation_database._elements_MAC` в `_build_compact_nist_bank`), передача `attenuation_database` в `ParticlePropagator` в [`verification/profiling/workloads.py:48,607`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/workloads.py#L607), [`verification/profiling/cprofile_harness.py:28,453,595,703`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/cprofile_harness.py#L453), [`verification/profiling/batch_size_benchmark.py:30,198,486`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/batch_size_benchmark.py#L198) и скриптах `examples/` (`tb161_nema_simulation.py:289`, `lung_cancer_main.py:205`, `heart_main.py:204`, `heart_32_main.py:205`, `brain_main.py:195`).

2. **`AttenuationFunction` и все вызовы `np.interp`**:
   * [`core/materials/attenuation_functions.py:15-76`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/materials/attenuation_functions.py#L15-L76):
     * `@njit(cache=True) def fast_interp(query_points, grid_x, grid_y)` (`строки 15-18`) — вызывает `np.interp(query_points, grid_x, grid_y)`.
     * Класс `AttenuationFunction(Dict[Material, Tuple[NDArray[Float], NDArray[Float]]])` (`строки 21-76`):
       * В `__init__(self, process, attenuation_database, kind='linear')` (`строки 28-45`) для каждого материала берет `energy = np.copy(attenuation_data['Energy'])` (`строка 37`), вычисляет ЛКО `attenuation_coefficient = attenuation_data['Coefficient'][process.name] * material.density` (`строка 38`), обрезает оба массива по `process.energy_range` через `np.searchsorted` (`строки 39-42`) и сохраняет `(np.ascontiguousarray(energy), np.ascontiguousarray(attenuation_coefficient))` в `self[material]` (`строка 45`).
       * В `__call__(self, material, energy)` (`строки 53-75`) для `Material` и массива `energy` вызывает `fast_interp(np.asarray(energy, dtype=Float), grid_x, grid_y)` (`строка 62`), для скалярного `energy` — `Float(np.interp(float(energy), grid_x, grid_y))` (`строка 63`), а для `MaterialArray` итерируется по `material.inverse_indices.items()` с вызовом `fast_interp` (`строки 65-73`).
   * [`core/materials/attenuation_database.py:98`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/materials/attenuation_database.py#L98) — в `AttenuationDataBase._add_material` вызывает `np.interp(array_of_energy, energy, mac_arr[process_name])`.
   * [`gui/viewport_3d/material_palette.py:282`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/material_palette.py#L282) — `np.interp` в `compute_material_linear_attenuation` (указано выше).
   * [`verification/profiling/micro_benchmarks.py:306`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/micro_benchmarks.py#L306) — `np.interp(e_query, el_e, el_mac)` при построении эталонной компактной сетки микробенчмарка `M1`.
   * [`gui/viewport_3d/dicom_colormaps.py:69-71`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/gui/viewport_3d/dicom_colormaps.py#L69-L71) — интерполяция RGB-каналов цветовой палитры DICOM в `get_colormap_lut` (не связана с физическими сечениями).

---

### 3.2. Пункт 2: Потребители `ElementCSR.element_fraction`, `InteractionBuffer.Z` и маппинг `Process.process_id` vs индекс столбца в `physics_lac_table`

1. **Все потребители `ElementCSR.element_fraction`**:
   * [`core/physics/physics_buffer.py:13-20`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/physics_buffer.py#L13-L20) — `NamedTuple` `ElementCSR(element_offsets: NDArray[Index], element_Z: NDArray[Charge], element_fraction: NDArray[Float])`.
   * [`core/physics/physics_compiler.py:87-116`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/physics_compiler.py#L87-L116) — метод `PhysicsCompiler._build_element_csr(materials_list, capacity)` сейчас заполняет `element_fraction` **нормированными массовыми долями** элементов `float(weight) / total_weight` (`строка 109`), не зависящими ни от энергии, ни от типа процесса.
   * [`core/physics/processes_kernels.py:82-104`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py#L82-L104) — функция `_sample_Z_scalar(mat_id: Index, physics_buffer: PhysicsBuffer, rng_ctx: RNGContext) -> Charge` разыгрывает атомный номер $Z$ линейным обходом `physics_buffer.element_csr.element_fraction[start:end]` (выборка по массовым долям вместо физических парциальных сечений $w_k \cdot (\mu/\rho)_{k,p}(E)$).
   * [`core/physics/processes_kernels.py:124-128,193,266`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py#L124-L128) — вызовы `atomic_z = _sample_Z_scalar(mat_id, physics_buffer, rng_ctx)` внутри `_photoelectric_device_func` (`строка 128`, при этом на `строке 125` энергия частицы уже обнулена `state.energy[p_idx] = 0.0`, поэтому в Фазе 2 при энергетически-зависимом отборе $Z$ необходимо передавать сохраненную на `строке 124` величину `energy_deposit`), `_compton_device_func` (`строка 193`) и `_coherent_device_func` (`строка 266`).
   * Тесты и фикстуры, обращающиеся к `ElementCSR`:
     * [`tests/physics/test_physics_compiler.py:33-61`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/physics/test_physics_compiler.py#L33-L61) — `test_build_element_csr_basic` проверяет `csr.element_offsets`, `csr.element_Z` и `csr.element_fraction`;
     * [`tests/physics/test_processes.py:176-189`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/physics/test_processes.py#L176-L189) — `_create_mock_physics_buffer` конструирует `MockCSR(element_offsets, element_Z, element_fraction)`;
     * [`tests/transport/conftest.py:72-76`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/transport/conftest.py#L72-L76) — конструирует `ElementCSR` в фикстуре `mock_physics_buffer`;
     * [`tests/transport/test_step5_transport.py:58-62`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/transport/test_step5_transport.py#L58-L62) — конструирует `ElementCSR` в `test_transport_kernel_execution`.

2. **Все потребители `atomic_z` и `InteractionBuffer.Z`**:
   * **Влияние `atomic_z` на физику рассеяния (КРИТИЧЕСКИЙ ФАКТ):** В отличие от упрощенного Клейна–Нишины, в `NMSimToolkit` разыгранный атомный номер `atomic_z` **напрямую влияет на физическую кинематику рассеяния** в каждом акте взаимодействия:
     * В [`core/physics/processes_kernels.py:195`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py#L195) (`_compton_device_func`) `atomic_z` передается в `_generate_compton_theta_scalar(energy, atomic_z, rng_ctx)` ([`core/physics/g4compton.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/g4compton.py)), где определяет функцию некогерентного рассеяния $S(x, Z)$ для отклонения углового распределения от формулы Клейна–Нишины!
     * В [`core/physics/processes_kernels.py:268`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py#L268) (`_coherent_device_func`) `atomic_z` передается в `_generate_coherent_theta_scalar(energy, atomic_z, rng_ctx)` ([`core/physics/g4coherent.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/g4coherent.py)), где определяет атомный форм-фактор $F(x, Z)$ рэлеевского рассеяния!
     * Следовательно, переход в Фазе 2 от выборки $Z$ по массовым долям к выборке по парциальным сечениям $w_k \cdot (\mu/\rho)_{k,p}(E)$ физически уточнит угловые распределения комптоновского и рэлеевского рассеяния в композитных средах (`NaI`, вода, кость, мягкие ткани, боросиликатное стекло).
   * Запись в буфер взаимодействий: [`core/physics/processes_kernels.py:70,141,215,286`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py#L70) (`_push_to_interaction_buffer` записывает `inter_buffer.Z[idx] = Z` в поле `Z: NDArray[Charge]` структуры `InteractionBuffer` в [`core/physics/interaction_buffers.py:89,167,188`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/interaction_buffers.py#L89)).
   * Сохранение в HDF5: [`core/data/data_handlers.py:194-195,210,480-484,504`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_handlers.py#L194) (`DirectStreamHandler._format_raw_interactions`, `SensitiveVolumeHandler._format_and_write_interactions` и наследующий его `HistoryAssemblerHandler`).
   * Проверки в тестах: [`tests/physics/test_processes.py:80`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/physics/test_processes.py#L80), [`tests/transport/test_step5_transport.py:133`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/transport/test_step5_transport.py#L133), [`tests/data/test_data_handlers.py:109,189,315`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_data_handlers.py#L109), [`tests/data/test_hdf5_storage_and_metadata.py:144,234`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_hdf5_storage_and_metadata.py#L144), [`tests/data/test_spect_simulation_hdf5.py:187`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_spect_simulation_hdf5.py#L187).

3. **Различие `Process.process_id` и порядкового индекса столбца `p_idx` в `physics_lac_table`**:
   * В [`core/physics/processes.py:96,110,130,154`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes.py#L96) каждому классу процесса жестко назначен `process_id`:
     * `PhotoelectricEffect.process_id = ProcessID(0)` (`строка 96`);
     * `ComptonScattering.process_id = ProcessID(1)` (`строка 130`);
     * `CoherentScattering.process_id = ProcessID(2)` (`строка 110`);
     * `PairProduction.process_id = ProcessID(3)` (`строка 154`).
     Именно этот `process_id` запекается в замыкание `make_*_kernel(self.process_id)` и записывается в `InteractionBuffer.process_id`.
   * В [`core/physics/physics_compiler.py:65-70`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/physics_compiler.py#L65-L70) столбцы матрицы `MaterialBank.physics_lac_table` индексируются **порядковым номером процесса `p_idx` (`0..len(processes_list)-1`) в переданном списке `processes_list`**: `for p_idx, process in enumerate(processes_list): lac_matrix[:, p_idx] = lacs`.
   * В [`core/transport/propagator.py:50-55`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/propagator.py#L50-L55) `ParticlePropagator.__init__` собирает массив `self.process_ids = np.array([proc.process_id if proc.process_id is not None else -1 for proc in self.processes], dtype=Index)` и передает его как `mapped_process_ids` в `make_transport_kernel(self.process_ids)`.
   * В [`core/transport/transport_kernels.py:207-215,276-280`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py#L207-L215) функция `_sample_process_id(majorant_lac, process_lacs, rng_ctx)` обходит `for i in range(num_processes):` по столбцам `process_lacs[i]` и возвращает `Index(mapped_process_ids[i])` (то есть `process.process_id`), который записывается в `transport_buffer.process_ids[p_idx]`.
   * В [`core/transport/propagator.py:108-120`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/propagator.py#L108-L120) метод `ParticlePropagator.step` итерируется `for process in self.processes:`, фильтрует частицы по `mask = self.transport_buffer.process_ids[active_indices] == process.process_id` (`строка 109`) и вызывает `process.apply(...)`.
   * **Следствие для Фазы 2:** В `transport_buffer.process_ids` и `InteractionBuffer.process_id` хранится `process.process_id` (`mapped_process_ids[i]`), тогда как столбцы в `MaterialBank.physics_lac_table` и целевой таблице парциальных/кумулятивных долей элементов в `ElementCSR` индексируются порядковым номером `p_idx` (`0..len(processes_list)-1`). Если в `ParticlePropagator` передан неполный список процессов или список в произвольном порядке (например, `[ComptonScattering]` или `[PhotoelectricEffect, CoherentScattering]`), то `process.process_id` не совпадает с `p_idx`. Поэтому в Фазе 2 в `Process.apply(...)` и в `@njit`-ядра процессов необходимо передавать `process_col_idx: int` (порядковый индекс `p_idx` из `for p_idx, process in enumerate(self.processes):` в `ParticlePropagator.step`), сохраняя запись `self.process_id` в `InteractionBuffer.process_id`.

---

### 3.3. Пункт 3: Все потребители `Source.generate_*`, `Source.inject`, `Source.emission_table`, `Source.rng` и `ParticleBank.inject_particles`

1. **Иерархия источников в `core/source/`**:
   * [`core/source/sources.py:26-322`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py#L26-L322) — базовый воксельный класс **`Source(CompositeNode)`**:
     * Хранит атрибуты (`строки 37-45`): `_distribution: NDArray[Float]`, `initial_activity: Float`, `_voxel_size: Length`, `size: Vector3D`, `radiation_type: str`, `_energy: NDArray[Any]` (структурированный массив с полями `'energy'` и `'probability'`), `half_life: Time`, `rng: np.random.Generator`, `emission_table: List[NDArray[Float]]` (заполняется в `_generate_emission_table`, `строки 186-211`, при установке `distribution` или `voxel_size`).
     * Методы расчета кинетики распада (`строки 213-237`): свойство `decay_constant`, `_get_effective_dt(delta_time)`, `get_activity(current_time)`, `get_expected_particles(time_start, time_end)`, а также `set_state(rng_state)` (`строки 239-245`).
     * Методы генерации и инжекции частиц:
       * `generate_energy(count)` (`строки 247-250`) — вызывает `self.rng.choice(self.energy["energy"], count, p=self.energy["probability"])`;
       * `generate_position(count)` (`строки 252-260`) — вызывает `self.rng.choice(voxel_origins, count, p=probability)` по `self.emission_table[0], self.emission_table[1]`, прибавляет `self.rng.uniform(0.0, self._voxel_size, sampled_positions.shape)` при `self._voxel_size > 0.0` и переводит координаты в глобальную СК через `self.convert_to_global_position(sampled_positions)`;
       * `generate_emission_time(count, time_start, time_end)` (`строки 262-273`);
       * `generate_direction(count)` (`строки 275-284`);
       * `inject(bank, batch_size, time_start, time_end)` (`строки 286-321`) — вызывает четыре метода `generate_*`, упаковывает координаты и направления в `Vector3DSoA` и вызывает `bank.inject_particles(...)` (`строка 313`).
   * [`core/source/sources.py:324-348`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py#L324-L348) — класс **`PointSource(Source)`**:
     * Не переопределяет `generate_position`, а в `__init__` (`строки 331-347`) задает `distribution = np.ones((1, 1, 1), dtype=Float)` и передает `voxel_size=size` (по умолчанию `Float(0.0)`) в `super().__init__(...)`.
   * [`core/source/source_compiler.py:11-48`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/source_compiler.py#L11-L48) — класс **`SourceCompiler`**:
     * Сейчас обходит дерево сцены в `compile_scene(root_node: SpatialNode) -> List[Source]` (`строки 21-28`) через `_extract_sources` (`строки 30-47`) и возвращает плоский список листовых узлов `Source`. В **Фазе 4** (п. 4.1 плана) этот модуль будет расширен компиляцией `List[Source]` в `SourceBuffer` с таблицами алиасов Воза (`Vose's Alias Method`).

2. **Подклассы и переопределения вне `core/` (КРИТИЧЕСКАЯ НАХОДКА)**:
   * [`verification/common/sources.py:17-119`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/common/sources.py#L17-L119) — класс **`ConePointSource(PointSource)`**, используемый в верификационных скриптах `verification/collimators/exp01..exp06`.
   * `ConePointSource` переопределяет **`generate_position(count)`** (`строки 84-99`) и **`generate_direction(count)`** (`строки 101-118`, генерация в телесном угле конуса к апертуре коллиматора).
   * При реализации Фазы 4 (`SourceBuffer` и `@njit`-инжектор в `core/source/source_kernels.py`) необходимо обеспечить поддержку `ConePointSource` (либо через поля конуса/джиттера в `SourceBuffer`, либо через сохранение контракта для специализированных верификационных подклассов).

3. **Все потребители `Source.inject`, `generate_*`, `emission_table`, `Source.rng` и `ParticleBank.inject_particles`**:
   * [`core/transport/simulation_managers.py:105,140-141,274-341`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L274-L341) — в `SimulationManager.__init__` извлекает `self.active_sources = SourceCompiler().compile_scene(scene)` (`строка 105`) и синхронизирует `source_node.rng = self.rng` (`строки 140-141`). Расчет шага времени и пополнение банка выполняются внутри `SimulationManager`:
     * `_calculate_time_step(num_to_inject)` (`строки 274-303`) — решает уравнение числа распадов методом Ньютона–Рафсона через `source_node.get_activity(...)` и `source_node.get_expected_particles(...)`;
     * `_distribute_quotas(target_time, num_to_inject)` (`строки 305-325`) — распределяет целочисленные квоты между источниками и вызывает `source_obj.inject(self.bank, int(quota_count), self.global_timer, target_time)` (`строка 325`);
     * `_replenish_bank()` (`строки 327-341`) — вычисляет `num_to_inject`, вызывает `_calculate_time_step` и `_distribute_quotas`, после чего сдвигает `self.global_timer = target_time`.
   * Профилировочный стенд в `verification/profiling/`:
     * [`verification/profiling/step_profiler.py:472,507-552`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/step_profiler.py#L472-L552) — для пофазного замера `P2_Source_Inject` напрямую вызывает `manager._calculate_time_step(num_to_inject)` (`строка 472`), `source_obj.generate_energy(num_p)` (`строка 507`), `source_obj.generate_direction(num_p)` (`строка 513`), `source_obj.generate_position(num_p)` (`строка 519`), `source_obj.generate_emission_time(num_p, manager.global_timer, target_time)` (`строки 525-527`) и `manager.bank.inject_particles(...)` (`строка 545`);
     * [`verification/profiling/micro_benchmarks.py:683-684,700,854-857`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/micro_benchmarks.py#L683-L700) — обращается к `workload.source_node.emission_table[0..1]` (`строки 683-684`), `workload.source_node.generate_position(...)` (`строки 700, 854`), `generate_direction(...)` (`строка 855`), `generate_energy(...)` (`строка 856`) и `m7_bank.inject_particles(...)` (`строка 857`).
   * [`core/particles/particles.py:53-110`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/particles/particles.py#L53-L110) — метод **`ParticleBank.inject_particles(...)`**:
     * Вызывается в `Source.inject` ([`core/source/sources.py:313`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py#L313)), в профилировщике ([`verification/profiling/workloads.py:512`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/workloads.py#L512), [`verification/profiling/step_profiler.py:545`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/step_profiler.py#L545), [`verification/profiling/micro_benchmarks.py:857`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/verification/profiling/micro_benchmarks.py#L857)), а также напрямую в unit-тестах:
       * [`tests/particles/test_particles.py:55`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/particles/test_particles.py#L55);
       * [`tests/geometry/test_geometry_compiler.py:93,196`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/geometry/test_geometry_compiler.py#L93).
     * Поэтому метод `ParticleBank.inject_particles` должен быть сохранен для прямой инжекции детерминированных пучков в тестах (п. 4.3 плана).
   * Прямые обращения к `emission_table`, `generate_position`, `source.inject` и `source.rng` в тестах:
     * [`tests/source/test_sources.py:24,31,80,85,93,97,109-113,131-148`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/source/test_sources.py#L24) — проверяет `PointSource(..., rng=...)`, `source.inject(...)`, `src.generate_position(...)`, `src.emission_table[1]`;
     * [`tests/geometry/test_geometry_stability.py:51-70`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/geometry/test_geometry_stability.py#L51-L70) — `test_source_zero_activity_and_normalization` проверяет `src2.emission_table[1]`;
     * [`tests/data/test_hdf5_storage_and_metadata.py:762-800`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_hdf5_storage_and_metadata.py#L762-L800) — `test_simulation_manager_unified_rng` проверяет `self.assertIs(active_src.rng, mgr.rng)`;
     * [`tests/transport/test_simulation_benchmark.py:87-91`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/transport/test_simulation_benchmark.py#L87-L91) — передает `rng=np.random.default_rng(42)` в `PointSource(...)`.

---

### 3.4. Пункт 4: Владельцы ГСЧ и состав состояния досчета (`Stateful Resume`)

1. **Единая архитектура ГСЧ (`np.random.Generator` + `RNGContext` поверх CFFI-указателя `bit_generator`)**:
   * [`core/config/orchestrator.py:112-137`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/orchestrator.py#L112-L137) — `Orchestrator.run()` создает `seed_sequence = SeedSequence(self.parsed_config.simulation_manager.seed)` (`строка 112`), вызывает `seeds = seed_sequence.spawn(len(task_contexts))` (`строка 113`) и генерирует `seed_value = seed_item.generate_state(1)[0]` (`строка 127`) для каждой задачи.
   * [`core/config/simulation_worker.py:149-181,268-282`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/simulation_worker.py#L149-L181) — `simulation_worker_task` вызывает `DataManager.inspect_resume_state(final_config.data_manager.filename, lock=file_lock, context=context_data)` (`строки 149-153`), извлекает `initial_particle_id`, `effective_task_id`, `effective_start_time`, `effective_seed` и `rng_state` (`строки 155-176`), после чего создает `ParticlePropagator(rng=np.random.default_rng(effective_seed))` (`строка 268`) и `SimulationManager(..., stop_time=effective_stop_time, start_time=effective_start_time, seed=effective_seed, initial_particle_id=initial_particle_id, rng_state=rng_state, ...)` (`строки 269-282`).
   * [`core/transport/simulation_managers.py:134-143`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L134-L143) — `SimulationManager.__init__`:
     * Создает единый генератор `self.rng = np.random.default_rng(seed) if seed is not None else self.propagator.rng` (`строка 134`);
     * Если передан `rng_state is not None`, восстанавливает состояние битового генератора: `self.rng.bit_generator.state = rng_state` (`строки 135-136`);
     * Назначает `self.propagator.rng = self.rng`, `process.rng = self.rng` и `source_node.rng = self.rng` (`строки 137-141`);
     * Оборачивает **тот же самый экземпляр `self.rng`** в `self.rng_ctx = RNGContext.from_numpy_rng(self.rng)` (`строка 143`; см. [`core/physics/interaction_buffers.py:56-71`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/interaction_buffers.py#L56-L71), где извлекаются CFFI-поля `rng.bit_generator.cffi.next_double` и `rng.bit_generator.cffi.state_address` для прямых вызовов из `@njit`-ядер Numba).
   * **Ключевое архитектурное свойство для Фаз 4 и 5:** Сейчас и Python-методы `Source`, и все скомпилированные `@njit`-ядра транспорта (`transport_kernel`) и физических процессов (`_photoelectric_kernel`, `_compton_kernel`, `_coherent_kernel`) через `self.rng_ctx` читают и продвигают **один и тот же поток `PCG64` (`self.rng.bit_generator`)**. В **Фазе 4** перенос генерации частиц из Python (`self.rng`) в `@njit`-инжектор через `RNGContext` сохраняет прямую работу с `PCG64`, а в **Фазе 5** (разделы 5.2–5.3 плана) поток ГСЧ разделяется на два независимых генератора от `SeedSequence` задачи — `emission_rng` (эмиссия первичных частиц) и `transport_rng` (перенос и взаимодействия), чтобы поток `InitialState` не зависел от `batch_capacity` и шагов транспорта.

2. **Состав метаданных состояния досчета (`Stateful Resume`) в HDF5**:
   * Сбор в [`core/data/metadata_collector.py:513-567`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/metadata_collector.py#L513-L567) (`ProcedureMetadataCollector.collect`), ранняя запись в [`core/data/data_manager.py:354-410`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py#L354-L410) (`DataManager.initialize_metadata`), финализация в [`core/data/data_manager.py:411-500`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py#L411-L500) (`DataManager.finalize_metadata`) и чтение чекпоинта в [`core/data/data_manager.py:128-255`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_manager.py#L128-L255) (`DataManager.inspect_resume_state`):
     * В группе `/metadata/tasks/task_<N>` сохраняются атрибуты (`simulation_worker.py:353-363`):
       * `status` (`"in_progress"` -> `"completed"` / `"failed"`);
       * `seed` (`int`);
       * `actual_start_time`, `actual_stop_time`, `frame_duration` (`float`);
       * `particle_id_start`, `particle_id_end`, `simulated_histories` (`int`, берутся из `manager.bank.count`);
       * `final_rng_state` (`str` — JSON-дамп словаря `manager.rng.bit_generator.state`, формируемый в [`core/config/simulation_worker.py:361`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/config/simulation_worker.py#L361)).
     * На уровне `/metadata/simulation` и `/metadata/gantry` в `DataManager.finalize_metadata` (`строки 457-496`) кумулятивно агрегируются `simulated_histories` и `frame_duration`.
     * **Следствие для Фазы 5 (п. 5.3 плана):** В `runtime_summary` (`simulation_worker.py:353-363`) и `DataManager.inspect_resume_state` (`data_manager.py:128-255`) потребуется добавить сохранение и восстановление состояния `emission_rng` (отдельно от `transport_rng`) и параметров пуассоновского потока прореживания Льюиса–Шедлера (`pending_arrival_time`, `t_ref`, `t_last`).

---

### 3.5. Пункт 5: Полный список тестов с жесткими и статистическими эталонами, зависящими от сечений и потока ГСЧ источника

1. **Тесты с прямыми проверками структуры `MaterialBank` и `ElementCSR` (`tests/physics/`, `tests/transport/`)**:
   * [`tests/physics/test_physics_compiler.py:9-62`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/physics/test_physics_compiler.py#L9-L62) — `test_build_element_csr_basic` проверяет поля `ElementCSR` (`element_offsets`, `element_Z`, `element_fraction`). **Подлежит обновлению в Фазе 2** при замене `element_fraction` на таблицу парциальных/кумулятивных долей элементов и расширении тестов `MaterialBank` в **Фазах 1 и 3**.
   * [`tests/physics/test_processes.py:56-81,176-298`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/physics/test_processes.py#L176-L189) — `_create_mock_physics_buffer` (`строки 176-189`) создает синтетический `MockCSR(element_offsets, element_Z, element_fraction)`, используемый в `test_photoelectric_kernel`, `test_compton_kernel`, `test_coherent_kernel`. **Подлежит обновлению в Фазе 2** (передача парциальных долей элементов и `process_col_idx`).
   * [`tests/transport/conftest.py:41-83`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/transport/conftest.py#L41-L83) и [`tests/transport/test_step5_transport.py:28-88`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/transport/test_step5_transport.py#L28-L88) — вручную конструируют синтетические `MaterialBank` и `ElementCSR` для тестирования `transport_kernel`. **Подлежат обновлению в Фазе 2** (структура `ElementCSR`) **и в Фазе 3** (поля двухуровневого лог-индекса в `MaterialBank` и `MaterialPointerDType`).

2. **Тесты с проверками `Source`, `emission_table`, `rng` и инжекции (`tests/source/`, `tests/geometry/`, `tests/data/`)**:
   * [`tests/source/test_sources.py:17-150`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/source/test_sources.py#L17-L150) — `TestSourcesSoA` проверяет `PointSource`, `Source`, `source.inject(...)`, `src.generate_position(...)` и `src.emission_table[1]`. **Подлежит обновлению в Фазах 4 и 5** (проверка `SourceCompiler` / `SourceBuffer`, таблиц Воза и `@njit`-инжектора в `core/source/source_kernels.py`).
   * [`tests/geometry/test_geometry_stability.py:51-70`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/geometry/test_geometry_stability.py#L51-L70) — `test_source_zero_activity_and_normalization` обращается к `src2.emission_table[1]`. **Подлежит обновлению в Фазе 4** при замене `self.emission_table` на плоские индексы активных вокселей и таблицы алиасов Воза в `Source`.
   * [`tests/data/test_hdf5_storage_and_metadata.py:762-800`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_hdf5_storage_and_metadata.py#L762-L800) — `test_simulation_manager_unified_rng` проверяет `for active_src in mgr.active_sources: self.assertIs(active_src.rng, mgr.rng)`. **Подлежит обновлению в Фазе 4** (удаление `self.rng` из `Source`) **и в Фазе 5** (разделение на `transport_rng` и `emission_rng`).

3. **Интеграционные и статистические тесты переноса и досчета (`tests/transport/`, `tests/data/`, `tests/config/`)**:
   * [`tests/transport/test_simulation_benchmark.py:40-226`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/transport/test_simulation_benchmark.py#L40-L226) — `TestFullBenchmark` (`test_run_benchmark` на `строках 43-122` и `test_profiling_harness_components` на `строках 124-225`) проверяет полный цикл симуляции ОФЭКТ и инварианты профилировочного стенда (**подлежит обновлению в Фазе 4** при удалении параметра `rng` у `PointSource`).
   * [`tests/data/test_hdf5_storage_and_metadata.py:847-1017`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_hdf5_storage_and_metadata.py#L847-L1017) — `test_stateful_resume_bitwise_identity`, `test_skip_already_completed_time_range`, `test_multi_step_protocol_shared_hdf5_resume_and_no_collision` (проверяют побитовую воспроизводимость и непрерывность `Stateful Resume`; **подлежат расширению в Фазе 5** под раздельные `transport_rng`/`emission_rng` и независимость `InitialState` от `batch_capacity`).
   * [`tests/config/test_orchestrator.py:83-208`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/config/test_orchestrator.py#L83-L208) и [`tests/data/test_spect_simulation_hdf5.py:116-237`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/tests/data/test_spect_simulation_hdf5.py#L116-L237) — сквозные тесты `Orchestrator` и сохранения результатов SPECT-симуляции в HDF5.



