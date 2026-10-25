# План 2: Механизм предварительного сканирования HDF5 и бесшовного досчёта симуляции (Stateful Resume)

## 1. Цель и физико-математический инвариант

Обеспечить возможность **побитово идентичного продолжения (досчёта) симуляции** при увеличении конечного времени `stop_time` (например, сначала посчитали интервал `[0, 10] с`, затем указали в конфигурации `[0, 20] с` — симулятор автоматически сканирует целевой HDF5-файл, обнаруживает завершенный расчёт до `~10 с`, восстанавливает состояние ГСЧ, счётчик `particle_ID` и досчитывает только недостающий интервал `[10, 20] с`).

**Главный инвариант:**
Результат двухэтапного расчёта `[0, T_1]` + досчёт до `T_2` должен быть **побитово (на 100.0%) идентичен** непрерывному одноэтапному расчёту `[0, T_2]` при том же начальном `seed`.

---

## 2. Декомпозиция изменений по компонентам

### Этап 1. Единый ГСЧ симуляции и точное усечение по `stop_time` (`core/transport/simulation_managers.py`, `core/source/source_kernels.py`)

1. **Единый детерминированный генератор случайных чисел (`self.rng` / `self.rng_ctx`):**
   * В `SimulationManager` используется единый генератор `self.rng` и его контекст `self.rng_ctx = RNGContext.from_numpy_rng(self.rng)` на весь поток симуляции:
     * `self.rng_ctx` передается и в `@njit`-ядро пуассоновской инжекции первичных частиц [`inject_poisson_sources_kernel`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/source_kernels.py), и в `ParticlePropagator.step` / физические процессы (`PhotoelectricEffect`, `ComptonScattering`, `CoherentScattering`).
   * Благодаря проверке наличия свободного слота в `ParticleBank` **до** генерации кандидата распада последовательность рожденных частиц (`InitialState`) на прямом цикле инжекции побитово не зависит от размера батча (`batch_capacity`).

2. **Пуассоновский поток распадов (прореживание Льюиса–Шедлера) и точная обрезка по `stop_time`:**
   * Методы `_calculate_time_step` (решатель Ньютона) и `_distribute_quotas` полностью удалены.
   * На каждом вызове `_replenish_bank()` ядро `inject_poisson_sources_kernel` генерирует времена кандидатов $t \leftarrow t_{\text{last}} - \ln(1 - u_1) / A_{\max}$ с постоянной мажорантой $A_{\max} = \sum_s A_{0,s} e^{-\lambda_s t_{\text{ref}}}$ ($t_{\text{ref}}$ фиксируется в начале первого сегмента).
   * Как только очередной кандидат удовлетворяет $t \ge \text{stop\_time}$, он **не** разыгрывает прореживание и параметры частицы, а сохраняется в `pending_arrival_time = t`; таймер `global_timer` устанавливается строго в `stop_time`, и генерация новых частиц на текущем этапе прекращается (все активные частицы в банке долетают до конца).

3. **Поддержка старта `SimulationManager` с сохраненного состояния (Checkpoint State):**
   * В `SimulationManager.__init__(..., resume_state: Optional[Mapping[str, Any]] = None)` и методе `restore_checkpoint(resume_state)` поддерживается компактное восстановление состояния из словаря чекпоинта:
     * `initial_particle_id: int` — записывается в `self.bank.count_array[0] = initial_particle_id` для сквозной нумерации `particle_ID` без коллизий;
     * `rng_state: Dict[str, Any]` — восстанавливает состояние единого `self.rng`;
     * `t_ref: Float | float` — опорное время мажоранты $A_{\max}$ первого сегмента;
     * `t_last: Float | float` — время последнего кандидата $< \text{stop\_time}$;
     * `pending_arrival_time: Float | float` — первый отложенный кандидат $t \ge \text{stop\_time}$ предыдущего сегмента.

---

### Этап 2. Сохранение финального слепка состояния в HDF5 (`core/data/data_manager.py`, `core/data/metadata_collector.py`, `core/config/simulation_worker.py`)

1. **Фиксация фактических итогов работы `SimulationManager` при финализации:**
   По завершении `manager.join()` в `simulation_worker_task` извлекаются:
   * `actual_start_time = float(effective_start_time)`;
   * `actual_stop_time = float(manager.global_timer)` (в режиме `stop_time` в точности равен `stop_time`);
   * `actual_frame_duration = (actual_stop_time - actual_start_time) / float(units.s)`;
   * `emitted_particles_in_task = int(manager.bank.count) - int(initial_particle_id)`;
   * `next_particle_id = int(manager.bank.count)`;
   * `rng_state = json.dumps(manager.rng.bit_generator.state)`;
   * `t_ref = float(manager.t_ref)`, `t_last = float(manager.t_last)`, `pending_arrival_time = float(manager.pending_arrival_time)`;
   * `batch_capacity = int(manager.batch_capacity)`.

2. **Запись в `/metadata/tasks/task_{task_id}` при `finalize_metadata()`:**
   * Сохраняются атрибуты завершенной задачи `task_{task_id}`:
     * `status = "completed"`
     * `seed` (исходный `uint32/uint64` seed первого запуска)
     * `actual_start_time` (в нс) и `actual_stop_time` (в нс)
     * `frame_duration` (в с)
     * `particle_id_start = initial_particle_id`, `particle_id_end = next_particle_id`, `simulated_histories = emitted_particles_in_task`
     * `rng_state`
     * `t_ref`, `t_last`, `pending_arrival_time`
     * `batch_capacity`

3. **Кумулятивное обновление корневых метаданных файла (`/metadata/gantry`, `/metadata/simulation`):**
   * При финализации задачи обновляются корневые показатели всего HDF5-файла по совокупности всех завершенных задач:
     * `/metadata/gantry/frame_duration` = суммарное время экспозиции по завершенным задачам файла;
     * `/metadata/simulation/simulated_histories` = `next_particle_id` (общее число рожденных за все этапы частиц).

---

### Этап 3. Предварительное сканирование HDF5 перед запуском задачи (`core/data/data_manager.py` / `core/config/simulation_worker.py`)

1. **Инспекция существующего файла перед стартом воркера (`DataManager.inspect_resume_state`):**
   * Проверяется наличие всех обязательных атрибутов досчёта в завершенных задачах `task_*` (`seed`, `actual_stop_time`, `particle_id_end`, `rng_state`, `t_ref`, `t_last`, `pending_arrival_time`, `batch_capacity`); при отсутствии любого из них немедленно возбуждается `ValueError` (Fail-Fast).
   * Для совпадающего `context` извлекаются сохраненные `seed`, `actual_stop_time`, `max_particle_id_end`, `rng_state`, `t_ref`, `t_last`, `pending_arrival_time`, `batch_capacity`.

2. **Логика принятия решения (Skip vs Resume):**
   * **Случай 1 (`saved_stop_time >= req_stop_time`):** задача пропускается без запуска `SimulationManager`.
   * **Случай 2 (`saved_stop_time < req_stop_time`):** проверяется совпадение `batch_capacity` с сохраненным, и `SimulationManager` запускается с `start_time = saved_stop_time`, `stop_time = req_stop_time`, восстановленными `rng_state`, `t_ref`, `t_last`, `pending_arrival_time` и продолжает пуассоновский поток ровно с отложенного кандидата `pending_arrival_time`.

---

## 3. Верификация и тесты (`tests/data/`, `tests/transport/`)

1. **Тест побитовой идентичности досчёта (`test_stateful_resume_bitwise_identity`):**
   * Прогон 1 (эталон): запустить симуляцию с `seed = 42` на интервале `[0.0, 0.02] с` в файл `ref.h5`.
   * Прогон 2 (двухэтапный):
     * запустить симуляцию с `seed = 42` на интервале `[0.0, 0.01] с` в файл `resumed.h5`;
     * изменить `stop_time = 0.02 с` и повторно запустить расчёт в тот же `resumed.h5`.
   * **Проверки:**
     * Массивы `initial_states/emission_position`, `emission_time`, `emission_energy`, `particle_ID` и все массивы `interactions/*` в `ref.h5` и `resumed.h5` **совпадают побитово (`np.array_equal`)**!
     * В `resumed.h5` присутствуют `metadata/tasks/task_0` и `metadata/tasks/task_1`.
     * Итоговый `metadata/gantry/frame_duration` и `metadata/simulation/simulated_histories` в `resumed.h5` в точности равны значениям в `ref.h5`.
2. **Тест пропуска уже рассчитанной задачи (`test_skip_already_completed_time_range`):**
   * Повторный запуск с тем же `stop_time = 0.02 с` не добавляет новых частиц и мгновенно завершается.
