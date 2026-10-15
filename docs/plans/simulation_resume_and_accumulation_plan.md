# План 2: Механизм предварительного сканирования HDF5 и бесшовного досчёта симуляции (Stateful Resume)

## 1. Цель и физико-математический инвариант

Обеспечить возможность **побитово идентичного продолжения (досчёта) симуляции** при увеличении конечного времени `stop_time` (например, сначала посчитали интервал `[0, 10] с`, затем указали в конфигурации `[0, 20] с` — симулятор автоматически сканирует целевой HDF5-файл, обнаруживает завершенный расчёт до `~10 с`, восстанавливает состояние ГСЧ, счётчик `particle_ID` и досчитывает только недостающий интервал `[10, 20] с`).

**Главный инвариант:**
Результат двухэтапного расчёта `[0, T_1]` + досчёт до `T_2` должен быть **побитово (на 100.0%) идентичен** непрерывному одноэтапному расчёту `[0, T_2]` при том же начальном `seed`.

---

## 2. Декомпозиция изменений по компонентам

### Этап 1. Унификация ГСЧ и возврат оригинальной логики полного батча (`core/transport/simulation_managers.py`, `core/config/simulation_worker.py`)

1. **Единый генератор случайных чисел (`rng`) для транспорта и всех источников:**
   * В [`SimulationManager.__init__()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L104-L112) удалить вызов `base_rng.spawn(1 + len(self.active_sources))`.
   * Использовать единый экземпляр `self.rng` (созданный из `seed` или переданный через `propagator.rng`):
     * `self.propagator.rng = self.rng`
     * для каждого физического процесса в `self.propagator.processes`: `process.rng = self.rng`
     * для каждого источника в `self.active_sources`: `source_node.rng = self.rng`
     * `self.rng_ctx = RNGContext.from_numpy_rng(self.rng)`
   * Так как `_replenish_bank()` и `propagator.step()` выполняются строго последовательно в одном потоке, единый `self.rng` полностью детерминирован и описывается **одним-единственным состоянием `self.rng.bit_generator.state`**.

2. **Возврат оригинальной логики полного батча (Вариант А — без искусственной обрезки `delta_time` по `stop_time`):**
   * В [`SimulationManager._calculate_time_step()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L298-L300) убрать искусственный клиппинг `min(delta_time, Float(self.stop_time - self.global_timer))`.
   * Шаг времени $\Delta t$ всегда вычисляется методом Ньютона-Рафсона ровно под заполнение всех свободных слотов банка `num_to_inject = self.bank.capacity - num_active`.
   * В [`SimulationManager._distribute_quotas()`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py#L301-L320) распределять квоты строго на `num_to_inject` частиц.
   * На последнем шаге перед завершением `self.global_timer` естественным образом слегка перескакивает за `self.stop_time` (например, останавливается на `10.00034 с` при `stop_time = 10.0 с`), после чего условие `self.global_timer < self.stop_time` прекращает генерацию новых частиц, а активные частицы в банке долетают до конца.
   * *(Примечание для синтетических тестов с микро-активностью вроде `test_simulation_manager_termination_no_hang`: скорректировать параметры теста так, чтобы `particles_number` соответствовал активности и времени).*

3. **Поддержка старта `SimulationManager` с сохраненного состояния (Checkpoint State):**
   * Добавить в `SimulationManager.__init__()` опциональные параметры восстановления:
     * `initial_particle_id: int = 0` — записывается в `self.bank.count_array[0] = initial_particle_id`, чтобы сквозная нумерация `particle_ID` продолжалась без коллизий (`N, N+1, ...`);
     * `rng_state: Optional[Dict[str, Any]] = None` — если передан, восстанавливает `self.rng.bit_generator.state = rng_state` до создания `self.rng_ctx`.

---

### Этап 2. Сохранение финального слепка состояния в HDF5 (`core/data/data_manager.py`, `core/config/simulation_worker.py`)

1. **Фиксация фактических итогов работы `SimulationManager` при финализации:**
   По завершении `manager.join()` в `simulation_worker_task` извлекать из `manager`:
   * `actual_start_time = float(effective_start_time)`;
   * `actual_stop_time = float(manager.global_timer)`;
   * `actual_frame_duration = (actual_stop_time - actual_start_time) / float(units.s)`;
   * `emitted_particles_in_task = int(manager.bank.count) - int(initial_particle_id)`;
   * `next_particle_id = int(manager.bank.count)`;
   * `final_rng_state_json = json.dumps(manager.rng.bit_generator.state)`.

2. **Запись в `/metadata/tasks/task_{task_id}` при `finalize_metadata()`:**
   * Сохранять в атрибутах завершенной задачи `task_{task_id}`:
     * `status = "completed"`
     * `seed` (исходный `uint32/uint64` seed первого запуска)
     * `actual_start_time` (в нс)
     * `actual_stop_time` (в нс)
     * `frame_duration` (в с)
     * `particle_id_start = initial_particle_id`
     * `particle_id_end = next_particle_id`
     * `simulated_histories = emitted_particles_in_task`
     * `final_rng_state = final_rng_state_json`

3. **Кумулятивное обновление корневых метаданных файла (`/metadata/gantry`, `/metadata/simulation`):**
   * При финализации задачи обновлять корневые показатели всего HDF5-файла по совокупности всех завершенных задач:
     * `/metadata/gantry/frame_duration` = суммарное время экспозиции по завершенным задачам файла;
     * `/metadata/simulation/simulated_histories` = `next_particle_id` (общее число рожденных за все этапы частиц).

---

### Этап 3. Предварительное сканирование HDF5 перед запуском задачи (`core/data/data_manager.py` / `core/config/simulation_worker.py`)

1. **Инспекция существующего файла перед стартом воркера:**
   * Перед запуском расчёта проверять, существует ли целевой файл `final_config.data_manager.filename`.
   * Если файл существует и содержит группу `/metadata/tasks`:
     * Найти все подгруппы `task_*` со `status == "completed"`, вычислив сквозной `max_particle_id_end` и `max_task_index` по всему файлу, а кандидата на досчёт искать среди задач с совпадающим `context`.
     * Если завершенных задач нет — начинать расчёт с нуля.
     * Если есть последняя завершенная задача `last_task` для данного `context`:
       * Считать:
         * `saved_seed = int(last_task.attrs["seed"])`
         * `saved_stop_time = float(last_task.attrs["actual_stop_time"])`
         * `saved_next_particle_id = max_particle_id_end`
         * `saved_rng_state = json.loads(last_task.attrs["final_rng_state"])`
       * Вычислить `next_task_index` (например, если в файле есть `task_0`, следующая задача для этого файла получает имя `task_1`).

2. **Логика принятия решения (Skip vs Resume):**
   * Пусть в конфигурации запрошен интервал `[req_start_time, req_stop_time]`:
   * **Случай 1 (`saved_stop_time >= req_stop_time`):**
     * Вся запрошенная длительность (например, `10.0 с`) уже рассчитана в файле (`10.00034 с >= 10.0 с`).
     * Воркер логирует: `"Задача уже полностью рассчитана (actual_stop_time >= stop_time), пропуск."` и мгновенно завершается без запуска `SimulationManager`.
   * **Случай 2 (`saved_stop_time < req_stop_time`):**
     * Режим досчёта (Resume):
       * `effective_start_time = saved_stop_time` (стартуем ровно с `10.00034 с`, где остановился предыдущий батч!);
       * `effective_stop_time = req_stop_time` (например, `20.0 с`);
       * `effective_seed = saved_seed`;
       * `initial_particle_id = saved_next_particle_id`;
       * `rng_state = saved_rng_state`.
     * `SimulationManager` запускается с этими параметрами, дописывает новые события в `interactions` и `initial_states`, а `DataManager` сохраняет паспорт нового этапа в `metadata/tasks/task_{next_task_index}` и обновляет кумулятивные `frame_duration` и `simulated_histories` в корне `/metadata`.

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
