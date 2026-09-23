# План реализации Runtime-накопления дозы в воксельной структуре

## 1. Контекст и архитектурное обоснование

### 1.1. Текущее состояние Runtime-трекинга
В проекте реализован и функционирует конвейер передачи и отображения телеметрии треков частиц:
1. **SimulationManager** аккумулирует взаимодействия в SoA-буфер `InteractionBuffer`.
2. При заполнении буфера или завершении шага вызывается `flush_interactions()`, который отправляет сырые словари в `multiprocessing.Queue`.
3. Поток **DataManager** читает очередь и распределяет чанки по цепочке `BaseDataHandler`.
4. **GuiStreamDataHandler** прореживает треки (`stride`) и складывает их в `track_queue`, а также накапливает 2D-проекцию детектора в `SharedMemory`.
5. **IPCReceiver** (QThread в процессе GUI) вычитывает `track_queue` и опрашивает SharedMemory с частотой кадров (FPS=30).
6. **TrackRenderer** визуализирует траектории в PyVista/VTK вьюпорте (облако точек / полилинии с цветовой кодировкой по `process_id`).

### 1.2. Архитектурная позиция аккумулятора дозы
- **Почему НЕ внутри ядра транспорта (Core Transport):** Доза является вторичной интегральной метрикой взаимодействия излучения с веществом. Встраивание 3D-матрицы в вычислительные циклы `SimulationManager` / `propagator` нарушит Data-Oriented Design ядра, увеличит нагрузку на кэш процессора и свяжет физический движок с сеточной дискретизацией визуализации.
- **Почему НЕ на стороне GUI:** Для сохранения производительности визуализации треков `GuiStreamDataHandler` выполняет субдискретизацию (`sub_indices = slice(0, n_points, stride)`). Накопление дозы на прореженных данных приводит к значительным артефактам и недооценке энергии.
- **Целевое решение:** Создание специализированного обработчика **`DoseMapHandler(BaseDataHandler)`**. Он получает **полный (неразреженный)** поток взаимодействий `InteractionBuffer` в потоке `DataManager`, векторизованно разносит энергию по вокселям в отдельный блок `multiprocessing.shared_memory.SharedMemory` и предоставляет GUI доступ к актуальному 3D-снимку через IPC.

---

## 2. Схема потоков данных (Data Flow)

```
[SimulationManager] (Ядро моделирования)
       │ (multiprocessing.Queue: interactions chunk)
       ▼
  [DataManager] (Поток диспетчеризации данных)
   ├── [HistoryAssemblerHandler] ──> Запись полного HDF5
   ├── [GuiStreamDataHandler]    ──> Прореженные 3D-треки (Queue) + 2D-проекция (SHM 1)
   └── [DoseMapHandler] (NEW)    ──> Полное 3D-накопление энергии (SHM 2: float64)
                                              │
       ┌──────────────────────────────────────┘ (Zero-Copy SharedMemory)
       ▼
 [IPCReceiver] (QThread GUI, независимый тайминг опроса ~2 FPS)
       │ dose_volume_received.emit(snapshot)
       ▼
[DoseVolumeRenderer] (NEW, PyVista / VTK ImageData)
       │ Обновление vtkSmartVolumeMapper / PiecewiseFunction
       ▼
  [VTK Viewport] (3D-сцена: совмещение геометрии, треков и полупрозрачной дозы)
```

---

## 3. Детальный план реализации по компонентам

### Шаг 1: `DoseMapHandler` (`core/data/dose_map_handler.py`)
Наследник `BaseDataHandler`, отвечающий за сеточное накопление энерговыделения.

* **Параметры инициализации:**
  * `grid_shape: Tuple[int, int, int]` — размерность сетки вокселей $(N_x, N_y, N_z)$, например `(64, 64, 64)`.
  * `voxel_size: Union[float, Tuple[float, float, float]]` — шаг сетки в мм.
  * `origin: Tuple[float, float, float]` — минимальный угол ограничивающего параллелепипеда (min bounds) в мм.
  * `shm_name: str` — уникальное имя сегмента POSIX/Windows SharedMemory.
  * `create_shm: bool = True` — флаг создания/подключения блока памяти.
* **Структура памяти:**
  * Буфер `SharedMemory` размером $N_x \times N_y \times N_z \times 8$ байт (`np.float64` во избежание потери точности при длительном суммировании малых депонирований).
  * Массив `self._dose_grid = np.ndarray(grid_shape, dtype=np.float64, buffer=self._shm.buf)`.
* **Логика метода `process_chunk(chunk: Dict[str, Any])`:**
  1. Фильтрация по `chunk.get('type') == 'interactions'`.
  2. Извлечение `pos_x`, `pos_y`, `pos_z`, `energy_deposit`.
  3. Векторизованный пересчет координат в индексы сетки:
     $$i_x = \left\lfloor \frac{pos_x - origin_x}{spacing_x} \right\rfloor, \quad i_y = \left\lfloor \frac{pos_y - origin_y}{spacing_y} \right\rfloor, \quad i_z = \left\lfloor \frac{pos_z - origin_z}{spacing_z} \right\rfloor$$
  4. Формирование булевой маски нахождения точек внутри воксельного пространства:
     $0 \le i_x < N_x \land 0 \le i_y < N_y \land 0 \le i_z < N_z$.
  5. Векторизованное накопление с использованием `np.add.at`:
     ```python
     np.add.at(self._dose_grid, (ix[mask], iy[mask], iz[mask]), energy_deposit[mask])
     ```
* **Метод `close()`:** Корректное закрытие дескриптора и вызов `unlink()`, если хэндлер владеет памятью.

---

### Шаг 2: Расширение `IPCReceiver` (`gui/controllers/ipc_receiver.py`)
* **Новые сигналы:**
  * `dose_volume_received = Signal(object)` — передача `np.ndarray` среза 3D-сетки.
* **Новые параметры конструктора:**
  * `dose_shm_name: Optional[str] = None`
  * `dose_grid_shape: Tuple[int, int, int] = (64, 64, 64)`
  * `dose_fps: float = 2.0` (частота съема дозы должна быть значительно ниже треков для экономии ресурсов GPU/VTK).
* **Логика опроса:**
  * В методе `run()` добавляется интервальный таймер для дозы (`dose_poll_interval = 1.0 / dose_fps`).
  * По таймеру выполняется безопасный snapshot:
    ```python
    if self._dose_buf is not None and (now - self._last_dose_time) >= self.dose_poll_interval:
        snapshot = self._dose_buf.copy()
        self.dose_volume_received.emit(snapshot)
        self._last_dose_time = now
    ```
* **Очистка:** Отключение от блока SharedMemory при завершении потока.

---

### Шаг 3: `DoseVolumeRenderer` (`gui/viewport_3d/dose_volume_renderer.py`)
Специализированный 3D-рендерер воксельной дозы на базе PyVista / VTK Volume Rendering.

* **Отличия от `VoxelVolumeRenderer`:**
  * Фантом статический, а дозовая матрица динамически обновляется на лету без пересоздания структуры сетки.
  * Требуется специализированная передаточная функция непрозрачности (Opacity Transfer Function): нулевые и околонулевые значения должны быть абсолютно прозрачными ($\alpha = 0$), чтобы не скрывать фантом и треки.
  * Поддержка цветовых шкал: *Dose Wash*, *Jet*, *Hot Iron*, *Rainbow*.
* **Ключевые методы:**
  * `setup_grid(grid_shape, voxel_size, origin)`: создание `pv.ImageData`, привязка скалярного массива нулей, инициализация `vtkSmartVolumeMapper`.
  * `update_dose_data(dose_3d: np.ndarray)`:
    * Прямая запись данных: `self.grid.point_data['dose'] = dose_3d.flatten(order='F').astype(np.float32)`.
    * Сигнализирование VTK об обновлении: `self.grid.GetPointData().GetScalars().Modified()` / `self.grid.Modified()`.
    * Динамическая подстройка `scalar_range = (0.0, max(1e-6, np.max(dose_3d)))`.
    * Обновление `vtkColorTransferFunction` и `vtkPiecewiseFunction` под новый максимум.
    * Запрос перерисовки сцены: `self.viewport.render()`.
  * `set_visible(visible: bool)`: включение/выключение отображения актора дозы.
  * `set_threshold(min_dose_ratio: float)`: отсечение фоновых вокселей ниже определенного процента от пиковой дозы (например, $< 5\%$).

---

### Шаг 4: Расширение `SimulationSession` (`gui/controllers/simulation_session.py`)
Фасад сессии координирует создание пайплайна и управление жизненным циклом.

* **Параметры:**
  * `dose_accumulation_enabled: bool = True`
  * `dose_grid_shape: Tuple[int, int, int] = (64, 64, 64)`
  * `dose_voxel_size: float = 5.0`
  * `dose_origin: Optional[Tuple[float, float, float]] = None`
  * `dose_shm_name: str = "nmsim_dose_shm"`
* **Конфигурирование конвейера в `_setup_pipeline()`:**
  * Если `dose_accumulation_enabled` включен:
    1. Автоматический расчет `origin` и `grid_shape` по габаритам сцены (Bounding Box корневого объема), если они не заданы явно.
    2. Создание `DoseMapHandler` с `create_shm=True`.
    3. Добавление `DoseMapHandler` в список `handlers` диспетчера `DataManager`.
    4. Передача параметров дозовой памяти в `IPCReceiver`.
    5. Подключение сигнала: `self.ipc_receiver.dose_volume_received.connect(self.dose_volume_received)`.
* **Очистка в `close()`:**
  * Вызов `self.dose_handler.close()` с удалением блока SharedMemory из ОС.

---

### Шаг 5: Интеграция в `MainWindow` (`gui/views/main_window.py`)
* **Элементы управления в UI:**
  * Добавление чекбокса / Action в панель инструментов: *"Накопление дозы"* (Toggle Dose Map).
  * Выпадающий список выбора палитры дозы (*Jet*, *Hot*, *CoolWarm*).
  * Ползунок порога отсечения дозы (Threshold).
* **Связка с вьюпортом:**
  * Создание экземпляра `DoseVolumeRenderer(self.viewport)`.
  * При старте симуляции: подписка `self.session.dose_volume_received.connect(self.dose_renderer.update_dose_data)`.
  * При сбросе сцены: очистка сетки дозы `dose_renderer.clear()`.

---

## 4. План тестирования и верификации

### 4.1. Модульное тестирование (`tests/data/test_dose_map_handler.py`)
1. **Тест точности биннинга:**
   * Создание сетки $4 \times 4 \times 4$ с `voxel_size=10.0`, `origin=(0, 0, 0)`.
   * Подача синтетического чанка с 3 точками: внутри вокселя (0,0,0), внутри вокселя (3,3,3) и точки за пределами сетки.
   * Проверка корректного накопления энергии в ячейках (0,0,0) и (3,3,3) и игнорирования внешней точки.
2. **Тест работы SharedMemory:**
   * Проверка совместного доступа к буферу из двух независимых структур `np.ndarray`.
   * Проверка корректности `close()` и `unlink()`.

### 4.2. Тестирование контроллера и визуализатора (`tests/viewport_3d/test_dose_volume_renderer.py`)
1. **Тест инкрементального обновления:**
   * Проверка, что вызов `update_dose_data` не пересоздает актор VTK, а мутирует существующие скаляры `pv.ImageData`.
   * Проверка работы без падений при передаче массива из одних нулей.

### 4.3. Интеграционный тест сессии (`tests/controllers/test_simulation_session_dose.py`)
1. Запуск тестовой симуляции с `dose_accumulation_enabled=True`.
2. Проверка эмита сигнала `dose_volume_received` с ненулевым массивом после шага физики.
3. Проверка детерминированной остановки и освобождения SharedMemory без утечек дескрипторов ОС.

---

## 5. Очередность выполнения задач

| № | Этап | Затрагиваемые файлы |
|---|---|---|
| **1** | Создание `DoseMapHandler` | `core/data/dose_map_handler.py`, `core/data/__init__.py` |
| **2** | Модульные тесты хэндлера | `tests/data/test_dose_map_handler.py` |
| **3** | Расширение `IPCReceiver` | `gui/controllers/ipc_receiver.py` |
| **4** | Создание `DoseVolumeRenderer` | `gui/viewport_3d/dose_volume_renderer.py`, `gui/viewport_3d/__init__.py` |
| **5** | Интеграция в `SimulationSession` | `gui/controllers/simulation_session.py` |
| **6** | Интеграция в интерфейс `MainWindow` | `gui/views/main_window.py` |
| **7** | Комплексные интеграционные тесты | `tests/test_gui_integration_and_fixes.py` |
