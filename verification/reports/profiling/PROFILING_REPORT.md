# Комплексный отчет по профилированию и анализу узких мест NMSimToolkit

**Общее время работы профилировочного стенда:** `478.80 с` | **Время холодного прогрева JIT/cfunc:** `3.97 с` | **Режим:** 100% Zero-Overhead для боевого ядра `core/` (Warm Steady-State + диагностика Cold Compile).

## 0. Контроль теплового режима и стабильности частоты CPU (Thermal Guard)

| Контрольная точка этапа | Время CPU-маяка (мс) | Относительная частота CPU (%) | Пауза охлаждения (с) |
| :--- | :---: | :---: | :---: |
| `01_Baseline_After_JIT_Warmup` | 35.90 | **100.0%** | 0.0 |
| `02_Before_cProfile_Suite` | 34.44 | **104.2%** | 1.5 |
| `03_Before_DataHandlers_Suite` | 35.33 | **101.6%** | 1.5 |
| `04_Before_StepProfiler_Suite` | 35.68 | **100.6%** | 1.5 |
| `05_Before_Telemetry_Suite` | 36.08 | **99.5%** | 1.5 |
| `06_Before_MicroBenchmarks` | 36.10 | **99.4%** | 1.5 |
| `07_Before_Orchestrator_Scaling` | 35.88 | **100.1%** | 1.5 |
| `08_Before_BatchSize_Sweep` | 35.54 | **101.0%** | 1.5 |
| `09_After_Full_Pipeline` | 36.96 | **97.1%** | 0.5 |

## 1. Профилирование конвейера CPython (`cProfile` / `pstats`, Warm Steady-State)

![Распределение времени cProfile по доменам](01_cprofile_domain_breakdown.png)

### 1.1. Сводная декомпозиция времени по доменам (сквозной однопоточный режим)

| Сцена | Коллиматор | Шагов | Историй | Steady-State (с) | Историй/с | JIT Транспорт (%) | JIT Процессы (%) | Источник (%) | Буферы+Escaped (%) | DataHandler CPU (%) | HDF5 I/O (%) | Дамп `.prof` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `W1_PointSource_Parametric` | `parametric` | 6 | 40,000 | 0.089 | **447,105** | 13.6% | 2.4% | 16.2% | 15.6% | 6.2% | 16.3% | `W1_PointSource_Parametric_end_to_end.prof` |
| `W1_PointSource_Direct` | `direct` | 32 | 40,000 | 0.100 | **401,950** | 13.2% | 3.0% | 14.0% | 15.8% | 5.6% | 16.7% | `W1_PointSource_Direct_end_to_end.prof` |
| `W2_NEMA_Tb161_Parametric` | `parametric` | 18 | 25,000 | 0.124 | **202,275** | 31.7% | 10.6% | 7.8% | 7.9% | 3.8% | 13.5% | `W2_NEMA_Tb161_Parametric_end_to_end.prof` |
| `W2_NEMA_Tb161_Direct` | `direct` | 92 | 25,000 | 0.169 | **148,351** | 23.3% | 7.9% | 5.8% | 6.0% | 3.2% | 20.7% | `W2_NEMA_Tb161_Direct_end_to_end.prof` |
| `W3_Heart_Tc99m_Parametric` | `parametric` | 21 | 25,000 | 0.187 | **133,487** | 33.5% | 9.2% | 5.6% | 6.1% | 3.4% | 15.0% | `W3_Heart_Tc99m_Parametric_end_to_end.prof` |
| `W3_Heart_Tc99m_Direct` | `direct` | 32 | 25,000 | 0.179 | **139,369** | 36.3% | 10.0% | 6.0% | 6.2% | 3.8% | 14.3% | `W3_Heart_Tc99m_Direct_end_to_end.prof` |

### 1.2. Штатный двухпоточный режим (`SimulationManager` + `DataManager` в разных потоках)

| Сцена | Коллиматор | Wall-Time (с) | Поток Sim (с) | Поток Data (с) | Ожидание `queue.put` (с) | Ожидание `queue.get` (с) | Активная работа Data (с) | Tail Drain Lag (мс) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | 0.093 | 0.047 | 0.086 | 0.0001 | 0.0459 | 0.040 | 45.93 |
| `W1_PointSource_Direct` | `direct` | 0.105 | 0.055 | 0.118 | 0.0000 | 0.0540 | 0.064 | 49.07 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | 0.133 | 0.081 | 0.126 | 0.0001 | 0.0790 | 0.047 | 51.83 |
| `W2_NEMA_Tb161_Direct` | `direct` | 0.158 | 0.089 | 0.160 | 0.0001 | 0.0863 | 0.074 | 68.07 |
| `W3_Heart_Tc99m_Parametric` | `parametric` | 0.192 | 0.114 | 0.189 | 0.0000 | 0.1123 | 0.077 | 77.80 |
| `W3_Heart_Tc99m_Direct` | `direct` | 0.180 | 0.122 | 0.193 | 0.0001 | 0.1187 | 0.074 | 57.29 |

### 1.3. Топ-12 функций по собственному времени (`tottime`, без JIT-компиляции) — `W2_NEMA_Tb161_Parametric`

| № | Функция | Файл и строка | Вызовы (`ncalls`) | `tottime` (с) | `cumtime` (с) | Доля (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `step` | `core/transport/propagator.py:61` | 18 | 0.0391 | 0.0576 | **31.7%** |
| 2 | `_flush_group` | `core/data/dataset_write_buffer.py:313` | 2 | 0.0229 | 0.0274 | **18.5%** |
| 3 | `_process_kernel` | `core/physics/processes_kernels.py:200` | 46 | 0.0128 | 0.0168 | **10.4%** |
| 4 | `make_fid` | `h5py/_hl/files.py:212` | 2 | 0.0112 | 0.0112 | **9.1%** |
| 5 | `_replenish_bank` | `core/transport/simulation_managers.py:331` | 18 | 0.0096 | 0.0107 | **7.7%** |
| 6 | `_record_finished_histories` | `core/transport/simulation_managers.py:399` | 18 | 0.0046 | 0.0046 | **3.7%** |
| 7 | `flush` | `h5py/_hl/files.py:625` | 2 | 0.0025 | 0.0025 | **2.0%** |
| 8 | `_flush_dead_particles` | `core/data/data_handlers.py:506` | 1 | 0.0015 | 0.0047 | **1.2%** |
| 9 | `<built-in method numpy._core._multiarray_umath._unique_hash>` | `~:0` | 2 | 0.0013 | 0.0013 | **1.1%** |
| 10 | `make_new_dset` | `h5py/_hl/dataset.py:39` | 16 | 0.0013 | 0.0018 | **1.0%** |
| 11 | `<method 'copy' of 'numpy.ndarray' objects>` | `~:0` | 26 | 0.0013 | 0.0013 | **1.0%** |
| 12 | `__contains__` | `h5py/_hl/group.py:605` | 52 | 0.0010 | 0.0013 | **0.8%** |

### 1.3. Топ-12 функций по собственному времени (`tottime`, без JIT-компиляции) — `W2_NEMA_Tb161_Direct`

| № | Функция | Файл и строка | Вызовы (`ncalls`) | `tottime` (с) | `cumtime` (с) | Доля (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `step` | `core/transport/propagator.py:61` | 92 | 0.0393 | 0.0654 | **23.3%** |
| 2 | `_flush_group` | `core/data/dataset_write_buffer.py:313` | 4 | 0.0389 | 0.0494 | **23.1%** |
| 3 | `make_fid` | `h5py/_hl/files.py:212` | 2 | 0.0240 | 0.0241 | **14.3%** |
| 4 | `_process_kernel` | `core/physics/processes_kernels.py:200` | 62 | 0.0130 | 0.0183 | **7.7%** |
| 5 | `_replenish_bank` | `core/transport/simulation_managers.py:331` | 92 | 0.0096 | 0.0108 | **5.7%** |
| 6 | `_record_finished_histories` | `core/transport/simulation_managers.py:399` | 23 | 0.0048 | 0.0048 | **2.8%** |
| 7 | `make_new_dset` | `h5py/_hl/dataset.py:39` | 38 | 0.0031 | 0.0044 | **1.9%** |
| 8 | `__contains__` | `h5py/_hl/group.py:605` | 118 | 0.0021 | 0.0029 | **1.2%** |
| 9 | `_flush_dead_particles` | `core/data/data_handlers.py:506` | 1 | 0.0016 | 0.0056 | **0.9%** |
| 10 | `<built-in method numpy._core._multiarray_umath._unique_hash>` | `~:0` | 2 | 0.0014 | 0.0014 | **0.9%** |
| 11 | `<method 'copy' of 'numpy.ndarray' objects>` | `~:0` | 26 | 0.0014 | 0.0014 | **0.9%** |
| 12 | `<built-in method __new__ of type object at 0x00007FF8451B4D90>` | `~:0` | 804 | 0.0014 | 0.0014 | **0.8%** |

## 2. Сравнительный анализ обработчиков данных (`DataHandlers`) и HDF5 I/O (Warm Steady-State)

![Сравнение DataHandlers](02_data_handlers_comparison.png)

| Сцена | Коллиматор | Код обработчика | Транспорт (с) | CPU хэндлера (с) | HDF5 I/O (с) | Доля Data (%) | Оверхед к Baseline (%) | Wall-Time 2 потока (с) | Tail Lag (мс) | Размер HDF5 (МБ) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | `H0_NullDrain_NoEscaped` | 0.027 | 0.000 | 0.000 | **0.1%** | +0.0% | 0.041 | 11.61 | 0.00 |
| `W1_PointSource_Parametric` | `parametric` | `H0_NullDrain_WithEscaped` | 0.027 | 0.000 | 0.000 | **0.1%** | +2.2% | 0.032 | 1.94 | 0.00 |
| `W1_PointSource_Parametric` | `parametric` | `H1_SensitiveVolume` | 0.020 | 0.000 | 0.031 | **62.0%** | +89.5% | 0.050 | 31.21 | 0.03 |
| `W1_PointSource_Parametric` | `parametric` | `H2_HistoryAssembler_NoInit` | 0.031 | 0.002 | 0.044 | **60.1%** | +189.4% | 0.050 | 30.50 | 0.03 |
| `W1_PointSource_Parametric` | `parametric` | `H3_HistoryAssembler_Full_Gzip` | 0.029 | 0.002 | 0.040 | **59.6%** | +166.0% | 0.072 | 40.25 | 0.03 |
| `W1_PointSource_Parametric` | `parametric` | `H4_HistoryAssembler_Full_NoComp` | 0.029 | 0.002 | 0.023 | **46.7%** | +103.6% | 0.057 | 27.62 | 6.03 |
| `W1_PointSource_Parametric` | `parametric` | `H5_DirectStream` | 0.031 | 0.000 | 0.065 | **68.2%** | +258.6% | 0.087 | 57.92 | 0.68 |
| `W1_PointSource_Parametric` | `parametric` | `H6_DoseMap` | 0.020 | 0.000 | 0.030 | **61.1%** | +84.9% | 0.035 | 16.00 | 0.21 |
| `W1_PointSource_Parametric` | `parametric` | `H7_Combined_History_And_Dose` | 0.029 | 0.002 | 0.055 | **66.4%** | +221.7% | 0.080 | 51.60 | 0.23 |
| `W1_PointSource_Direct` | `direct` | `H0_NullDrain_NoEscaped` | 0.033 | 0.000 | 0.000 | **0.1%** | +0.0% | 0.038 | 1.82 | 0.00 |
| `W1_PointSource_Direct` | `direct` | `H0_NullDrain_WithEscaped` | 0.032 | 0.000 | 0.000 | **0.1%** | -3.2% | 0.037 | 1.62 | 0.00 |
| `W1_PointSource_Direct` | `direct` | `H1_SensitiveVolume` | 0.027 | 0.000 | 0.024 | **48.3%** | +47.9% | 0.050 | 25.99 | 0.03 |
| `W1_PointSource_Direct` | `direct` | `H2_HistoryAssembler_NoInit` | 0.026 | 0.002 | 0.028 | **53.4%** | +70.2% | 0.061 | 34.59 | 0.03 |
| `W1_PointSource_Direct` | `direct` | `H3_HistoryAssembler_Full_Gzip` | 0.033 | 0.003 | 0.050 | **61.0%** | +157.3% | 0.082 | 46.77 | 0.03 |
| `W1_PointSource_Direct` | `direct` | `H4_HistoryAssembler_Full_NoComp` | 0.034 | 0.002 | 0.025 | **44.0%** | +85.1% | 0.059 | 26.78 | 6.03 |
| `W1_PointSource_Direct` | `direct` | `H5_DirectStream` | 0.032 | 0.000 | 0.068 | **68.4%** | +199.4% | 0.088 | 53.74 | 0.69 |
| `W1_PointSource_Direct` | `direct` | `H6_DoseMap` | 0.025 | 0.000 | 0.033 | **58.2%** | +70.5% | 0.040 | 15.27 | 0.21 |
| `W1_PointSource_Direct` | `direct` | `H7_Combined_History_And_Dose` | 0.034 | 0.003 | 0.065 | **66.4%** | +206.7% | 0.084 | 51.15 | 0.23 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H0_NullDrain_NoEscaped` | 0.073 | 0.000 | 0.000 | **0.0%** | +0.0% | 0.096 | 20.75 | 0.00 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H0_NullDrain_WithEscaped` | 0.074 | 0.000 | 0.000 | **0.0%** | +1.1% | 0.081 | 3.46 | 0.00 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H1_SensitiveVolume` | 0.058 | 0.000 | 0.028 | **33.1%** | +16.8% | 0.089 | 30.46 | 0.03 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H2_HistoryAssembler_NoInit` | 0.071 | 0.003 | 0.043 | **39.4%** | +60.5% | 0.089 | 32.09 | 0.03 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H3_HistoryAssembler_Full_Gzip` | 0.071 | 0.004 | 0.047 | **41.7%** | +68.3% | 0.113 | 41.84 | 0.03 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H4_HistoryAssembler_Full_NoComp` | 0.070 | 0.003 | 0.035 | **35.5%** | +50.0% | 0.092 | 19.39 | 6.03 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H5_DirectStream` | 0.075 | 0.000 | 0.149 | **66.6%** | +208.7% | 0.209 | 134.24 | 2.67 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H6_DoseMap` | 0.059 | 0.000 | 0.032 | **35.0%** | +23.9% | 0.072 | 15.19 | 0.73 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `H7_Combined_History_And_Dose` | 0.078 | 0.006 | 0.067 | **48.3%** | +107.7% | 0.129 | 55.07 | 0.75 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H0_NullDrain_NoEscaped` | 0.078 | 0.000 | 0.000 | **0.0%** | +0.0% | 0.082 | 1.76 | 0.00 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H0_NullDrain_WithEscaped` | 0.075 | 0.000 | 0.000 | **0.0%** | -4.7% | 0.077 | 1.73 | 0.00 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H1_SensitiveVolume` | 0.060 | 0.000 | 0.058 | **50.4%** | +46.2% | 0.115 | 56.43 | 0.04 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H2_HistoryAssembler_NoInit` | 0.057 | 0.003 | 0.053 | **49.7%** | +45.7% | 0.111 | 51.09 | 0.05 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H3_HistoryAssembler_Full_Gzip` | 0.081 | 0.004 | 0.082 | **51.7%** | +113.1% | 0.146 | 67.36 | 0.06 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H4_HistoryAssembler_Full_NoComp` | 0.078 | 0.003 | 0.031 | **30.8%** | +43.4% | 0.109 | 32.27 | 12.04 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H5_DirectStream` | 0.076 | 0.001 | 0.161 | **68.0%** | +203.5% | 0.217 | 139.40 | 2.65 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H6_DoseMap` | 0.062 | 0.000 | 0.019 | **23.9%** | +3.3% | 0.086 | 25.16 | 0.73 |
| `W2_NEMA_Tb161_Direct` | `direct` | `H7_Combined_History_And_Dose` | 0.077 | 0.005 | 0.070 | **49.4%** | +94.8% | 0.170 | 96.89 | 0.78 |

### 2.1. Зависимость времени обработки от емкости буфера (`buffer_capacity`)

| Сцена | Коллиматор | `buffer_capacity` | Чанков | Транзакций HDF5 | Транспорт (с) | CPU фильтрация (с) | HDF5 I/O (с) | Суммарно (с) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `5,000` | 7 | 3 | 0.076 | 0.004 | 0.035 | 0.114 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `15,000` | 7 | 3 | 0.078 | 0.003 | 0.046 | 0.126 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `50,000` | 5 | 3 | 0.079 | 0.005 | 0.053 | 0.137 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `200,000` | 3 | 3 | 0.076 | 0.004 | 0.049 | 0.129 |
| `W2_NEMA_Tb161_Direct` | `direct` | `5,000` | 7 | 3 | 0.076 | 0.002 | 0.064 | 0.142 |
| `W2_NEMA_Tb161_Direct` | `direct` | `15,000` | 7 | 3 | 0.079 | 0.003 | 0.084 | 0.166 |
| `W2_NEMA_Tb161_Direct` | `direct` | `50,000` | 5 | 3 | 0.080 | 0.003 | 0.073 | 0.156 |
| `W2_NEMA_Tb161_Direct` | `direct` | `200,000` | 3 | 3 | 0.079 | 0.004 | 0.068 | 0.151 |

## 3. Детальная 10-фазная декомпозиция пошагового цикла симуляции (Warm Steady-State)

![10-фазная декомпозиция шага](03_step_phase_breakdown.png)

| Сцена | Коллиматор | Шагов | Историй | Steady-State (с) | Историй/с | P2: Source (%) | P4: Geometry (%) | P5: Woodcock (%) | P8: Processes (%) | P9: Invalid+Esc (%) | P10: Data+HDF5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | 29 | 300,000 | 0.401 | **749,017** | 27.9% | 4.6% | 12.5% | 4.4% | 20.0% | 24.2% |
| `W1_PointSource_Direct` | `direct` | 55 | 300,000 | 0.432 | **694,346** | 26.5% | 8.7% | 7.9% | 4.6% | 19.3% | 26.5% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | 35 | 150,000 | 0.534 | **281,153** | 11.9% | 4.6% | 26.3% | 14.2% | 7.8% | 21.8% |
| `W2_NEMA_Tb161_Direct` | `direct` | 137 | 150,000 | 0.545 | **275,045** | 12.2% | 5.4% | 22.2% | 14.9% | 8.5% | 22.5% |
| `W3_Heart_Tc99m_Parametric` | `parametric` | 47 | 150,000 | 0.762 | **196,954** | 9.1% | 5.0% | 34.9% | 14.3% | 6.0% | 18.9% |
| `W3_Heart_Tc99m_Direct` | `direct` | 83 | 150,000 | 0.763 | **196,541** | 9.5% | 6.5% | 33.9% | 14.4% | 6.1% | 17.6% |

### 3.1. Внутренняя декомпозиция инжекции источника и физических процессов (Warm Steady-State)

| Сцена | Коллиматор | `inject_poisson_sources_kernel` (мс) | `_record_finished` (мс) | Фотоэффект (мс) | Комптон (мс) | Рэлей (мс) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | 111.42 | 67.97 | 11.37 | 3.54 | 2.63 |
| `W1_PointSource_Direct` | `direct` | 113.72 | 69.45 | 12.34 | 4.24 | 3.33 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | 63.25 | 25.93 | 16.92 | 48.26 | 10.44 |
| `W2_NEMA_Tb161_Direct` | `direct` | 66.04 | 26.98 | 17.71 | 52.72 | 10.49 |
| `W3_Heart_Tc99m_Parametric` | `parametric` | 68.87 | 28.52 | 13.22 | 86.63 | 8.48 |
| `W3_Heart_Tc99m_Direct` | `direct` | 71.80 | 27.18 | 13.00 | 88.42 | 8.00 |

### 3.2. Сравнение `Cold Propagator Overhead` vs `Warm Steady-State`

| Сцена | Коллиматор | Историй | Cold JIT пропагатора (с) | Полное время Cold (с) | Чистое время Warm (с) | Speedup (Warm/Cold) | Скорость Cold (ист./с) | Скорость Warm (ист./с) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | 300,000 | 1.607 | 2.008 | **0.401** | **5.0x** | 149,416 | **749,017** |
| `W1_PointSource_Direct` | `direct` | 300,000 | 1.607 | 2.039 | **0.432** | **4.7x** | 147,106 | **694,346** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | 150,000 | 1.607 | 2.141 | **0.534** | **4.0x** | 70,067 | **281,153** |
| `W2_NEMA_Tb161_Direct` | `direct` | 150,000 | 1.607 | 2.153 | **0.545** | **3.9x** | 69,681 | **275,045** |
| `W3_Heart_Tc99m_Parametric` | `parametric` | 150,000 | 1.607 | 2.369 | **0.762** | **3.1x** | 63,321 | **196,954** |
| `W3_Heart_Tc99m_Direct` | `direct` | 150,000 | 1.607 | 2.370 | **0.763** | **3.1x** | 63,278 | **196,541** |

### 3.3. Разделение стационарной фазы полного банка (`Full-Bank Phase`) и фазы опустошения (`Tail Depletion Phase`)

| Сцена | Коллиматор | Емкость банка | Историй | Шагов `Full-Bank` | Шагов `Depletion` | Транспорт `Full-Bank` (мс) | Транспорт `Depletion` (мс) | Треков/с `Full-Bank` | Треков/с `Depletion` | Замедление на хвосте (`x`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | `25,000` | 300,000 | 24 | 5 | 296.45 | 6.08 | **3,234,698** | 1,727,025 | **1.87x** |
| `W1_PointSource_Direct` | `direct` | `25,000` | 300,000 | 29 | 26 | 287.45 | 26.19 | **4,109,106** | 2,113,573 | **1.94x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `25,000` | 150,000 | 18 | 17 | 368.69 | 45.80 | **1,465,269** | 1,352,784 | **1.08x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `25,000` | 150,000 | 19 | 118 | 366.15 | 51.56 | **1,575,038** | 1,164,630 | **1.35x** |
| `W3_Heart_Tc99m_Parametric` | `parametric` | `25,000` | 150,000 | 27 | 20 | 552.11 | 63.79 | **1,394,532** | 1,195,572 | **1.17x** |
| `W3_Heart_Tc99m_Direct` | `direct` | `25,000` | 150,000 | 29 | 54 | 562.34 | 64.96 | **1,474,528** | 1,360,208 | **1.08x** |

### 3.4. Пофазные дельты Page Faults (`psutil.Process().memory_info()`) на стационарной фазе (`Full-Bank Phase`) и в `inject_poisson_sources_kernel`

| Сцена | Коллиматор | `Full` P2 Source PF (`Steady`) | `Full` P4 Geom PF (`Steady`) | `Full` P5 Woodcock PF (`Steady`) | `Full` P8 Phys PF (`Steady`) | `Full` P9 Comp PF (`Steady`) | `Full` P10 Data PF | `src.kernel` PF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | **790** (`0`) | 0 (`0`) | 94 (`2`) | 744 (`738`) | 9,114 (`9,009`) | 11,565 | 790 |
| `W1_PointSource_Direct` | `direct` | **804** (`0`) | 0 (`0`) | 94 (`2`) | 492 (`492`) | 5,876 (`5,752`) | 7,606 | 804 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | **565** (`0`) | 0 (`0`) | 94 (`0`) | 2,313 (`1,698`) | 9,202 (`8,636`) | 9,484 | 565 |
| `W2_NEMA_Tb161_Direct` | `direct` | **568** (`0`) | 0 (`0`) | 96 (`0`) | 1,666 (`1,218`) | 10,071 (`9,503`) | 9,701 | 568 |
| `W3_Heart_Tc99m_Parametric` | `parametric` | **344** (`0`) | 0 (`0`) | 96 (`0`) | 1,831 (`1,380`) | 8,612 (`8,047`) | 12,029 | 344 |
| `W3_Heart_Tc99m_Direct` | `direct` | **576** (`0`) | 0 (`0`) | 96 (`0`) | 1,937 (`1,451`) | 10,929 (`10,387`) | 11,961 | 576 |

## 4. Алгоритмическая телеметрия ядер переноса (Geometry & Woodcock)

![Телеметрия счетчиков переноса](04_transport_telemetry_comparison.png)

| Сцена | Коллиматор | Историй | Лучей/ист. | Примитивов/луч | `_box_intersect` | `_hex_prism_intersect` | Шагов Вудкока/ист. | Холостых Вудкока (%) | `searchsorted`/ист. | `@cfunc`/ист. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | 12,000 | 2.03 | 2.08 | 50,788 | 0 | 1.54 | **83.4%** | 2.77 | 1.51 |
| `W1_PointSource_Direct` | `direct` | 12,000 | 2.59 | 3.24 | 91,345 | 9,531 | 0.26 | **0.0%** | 1.82 | 0.00 |
| `W2_NEMA_Tb161_Parametric` | `parametric` | 12,000 | 3.38 | 3.91 | 158,668 | 0 | 7.87 | **72.0%** | 10.90 | 7.85 |
| `W2_NEMA_Tb161_Direct` | `direct` | 12,000 | 3.55 | 4.21 | 175,143 | 4,043 | 4.59 | **51.7%** | 7.64 | 4.41 |
| `W3_Heart_Tc99m_Parametric` | `parametric` | 12,000 | 4.98 | 3.81 | 227,512 | 0 | 13.28 | **78.3%** | 17.60 | 13.25 |
| `W3_Heart_Tc99m_Direct` | `direct` | 12,000 | 5.42 | 4.25 | 268,532 | 7,716 | 11.67 | **75.3%** | 16.21 | 11.42 |

## 5. Изолированные микробенчмарки горячих узлов (`M1..M7`)

![Микробенчмарки горячих узлов](05_micro_benchmarks_speedup.png)

| ID | Узел конвейера | Текущая реализация / База | Альтернатива / Сравнение | Текущее (нс/оп) | Альтернатива (нс/оп) | Отношение (`Speedup / Ratio`) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `M1a_LogIndexSingle` | Интерполяция LAC в MaterialBank (~54 узлов, одиночный запрос, 5 млн зап.) | `np.searchsorted O(log K) (~54 узлов)` | `2-ур. лог-индекс O(1) (512 корзин)` | 71.3 | 35.2 | **2.03x** |
| `M1b_LogIndexWoodcockPair` | Пара запросов Вудкока (мажоранта + воксель, общий log(E), ~54 узлов, 5 млн шагов) | `2x np.searchsorted + 2x log(E)` | `1x log(E) + 2x _locate_row (MaterialBank)` | 176.2 | 60.3 | **2.92x** |
| `M2_EmissionTableInit` | Инициализация таблицы эмиссии 3D-фантома 128x128x92 | `3D np.meshgrid + stack` | `Разреженный flatnonzero + unravel` | 21.8 | 4.9 | **4.44x** |
| `M3_VoxelPositionSampling` | Генерация 500 тыс. позиций частиц в воксельном фантоме | `rng.choice(origins, p=prob)` | `Vose's Alias O(1) + Direct SoA` | 142.5 | 19.1 | **7.48x** |
| `M4_SensitiveVolumeMapping` | Маппинг идентификаторов объемов в SensitiveVolumeHandler (200 тыс. событий) | `Python for-loop + dtype=object` | `Векторизованный целочисленный LUT` | 85.7 | 2.2 | **38.51x** |
| `M5_HistoryAssemblerFiltering` | Фильтрация историй по particle_ID в HistoryAssemblerHandler (450 тыс. взаимод.) | `intersect1d + 4x np.isin` | `Прямая булева индексация O(1)` | 61.8 | 2.2 | **27.89x** |
| `M6_GeometryPrimitives` | Аналитическое пересечение примитивов (5 млн лучей) | `_hex_prism_intersect (Voronoi 6-slab)` | `_box_intersect (AABB 3-slab)` | 23.5 | 3.1 | **7.48x** |
| `M7_TransportVsProcessKernel` | 1 шаг Raycast+Woodcock против 1 шага ComptonScattering (100 тыс. частиц) | `cast_path + _transport_kernel` | `_compton_kernel + rotate` | 279.4 | 236.4 | **1.18x** |

## 6. Масштабируемость параллельного `Orchestrator` и конкуренция за HDF5

![Масштабируемость Orchestrator](06_orchestrator_scaling.png)

| Коллиматор | `pool_size` | Задач | Режим файлов HDF5 | Время `Wall-Time` (с) | Задач/с | Ускорение (`Speedup`) | Параллельная эффективность (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `parametric` | 1 | 4 | `per_task_file` | 10.429 | 0.38 | **1.00x** | 100.0% |
| `parametric` | 2 | 4 | `per_task_file` | 7.010 | 0.57 | **1.49x** | 74.4% |
| `parametric` | 4 | 4 | `per_task_file` | 4.731 | 0.85 | **2.20x** | 55.1% |
| `parametric` | 4 | 4 | `shared_single_file` | 4.893 | 0.82 | **2.13x** | 53.3% |
| `direct` | 1 | 4 | `per_task_file` | 9.730 | 0.41 | **1.00x** | 100.0% |
| `direct` | 2 | 4 | `per_task_file` | 6.580 | 0.61 | **1.48x** | 73.9% |
| `direct` | 4 | 4 | `per_task_file` | 4.473 | 0.89 | **2.18x** | 54.4% |
| `direct` | 4 | 4 | `shared_single_file` | 4.618 | 0.87 | **2.11x** | 52.7% |

## 7. Исследование размера батча частиц (`ParticleBank` capacity / Batch Size Sweep, 6,000,000 историй)

![Масштабируемость от размера батча частиц](07_batch_size_scaling.png)

### 7.1. Сравнение стационарной фазы полного банка (`Full-Bank Sustained Phase`) и фазы опустошения (`Tail Depletion Phase`)

| Сцена | Коллиматор | `batch_size` | Историй | Шагов (`Full / Depl`) | Время `Full-Bank` (с) | Время `Depletion` (с) | Доля хвоста (%) | Скорость `Full-Bank` (ист./с) | Скорость сквозная (ист./с) | Треков/с (`Full / Depl`) | Замедление на хвосте (`x`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | `2,500` | 6,000,000 | 4882 / 5 | 7.523 | 0.002 | 0.0% | **797,278** | 795,705 | 2,398,451 / 1,251,239 | **1.92x** |
| `W1_PointSource_Parametric` | `parametric` | `5,000` | 6,000,000 | 2440 / 5 | 6.769 | 0.005 | 0.1% | **885,635** | 884,663 | 2,747,050 / 1,913,714 | **1.44x** |
| `W1_PointSource_Parametric` | `parametric` | `15,000` | 6,000,000 | 813 / 5 | 6.436 | 0.008 | 0.1% | **930,742** | 930,618 | 3,003,972 / 1,882,454 | **1.60x** |
| `W1_PointSource_Parametric` | `parametric` | `30,000` | 6,000,000 | 406 / 6 | 6.202 | 0.015 | 0.2% | **963,989** | 964,660 | 3,120,645 / 2,138,432 | **1.46x** |
| `W1_PointSource_Parametric` | `parametric` | `60,000` | 6,000,000 | 202 / 14 | 6.101 | 0.045 | 0.7% | **973,870** | 975,935 | 3,180,316 / 2,573,876 | **1.24x** |
| `W1_PointSource_Parametric` | `parametric` | `120,000` | 6,000,000 | 101 / 12 | 5.909 | 0.045 | 0.8% | **1,002,765** | 1,007,454 | 3,271,402 / 2,134,287 | **1.53x** |
| `W1_PointSource_Parametric` | `parametric` | `250,000` | 6,000,000 | 48 / 7 | 5.894 | 0.105 | 1.7% | **992,788** | 1,000,151 | 3,255,778 / 2,443,883 | **1.33x** |
| `W1_PointSource_Parametric` | `parametric` | `500,000` | 6,000,000 | 24 / 16 | 5.891 | 0.111 | 1.9% | **990,392** | 999,520 | 3,292,724 / 2,125,130 | **1.55x** |
| `W1_PointSource_Direct` | `direct` | `2,500` | 6,000,000 | 6252 / 32 | 7.894 | 0.006 | 0.1% | **759,847** | 757,739 | 2,904,078 / 703,216 | **4.13x** |
| `W1_PointSource_Direct` | `direct` | `5,000` | 6,000,000 | 3126 / 22 | 6.966 | 0.007 | 0.1% | **860,859** | 859,217 | 3,376,014 / 1,237,811 | **2.73x** |
| `W1_PointSource_Direct` | `direct` | `15,000` | 6,000,000 | 1040 / 31 | 6.441 | 0.016 | 0.3% | **929,404** | 928,394 | 3,807,661 / 2,091,083 | **1.82x** |
| `W1_PointSource_Direct` | `direct` | `30,000` | 6,000,000 | 519 / 29 | 6.208 | 0.028 | 0.4% | **962,466** | 961,674 | 3,977,347 / 2,325,173 | **1.71x** |
| `W1_PointSource_Direct` | `direct` | `60,000` | 6,000,000 | 259 / 32 | 6.134 | 0.040 | 0.6% | **971,157** | 971,455 | 4,032,367 / 2,620,753 | **1.54x** |
| `W1_PointSource_Direct` | `direct` | `120,000` | 6,000,000 | 128 / 33 | 5.923 | 0.109 | 1.8% | **993,999** | 994,439 | 4,114,397 / 2,826,013 | **1.46x** |
| `W1_PointSource_Direct` | `direct` | `250,000` | 6,000,000 | 61 / 37 | 5.970 | 0.151 | 2.5% | **976,842** | 980,111 | 4,096,464 / 2,639,890 | **1.55x** |
| `W1_PointSource_Direct` | `direct` | `500,000` | 6,000,000 | 29 / 42 | 5.678 | 0.447 | 7.3% | **972,357** | 979,376 | 4,146,419 / 2,935,747 | **1.41x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `2,500` | 6,000,000 | 8175 / 14 | 19.610 | 0.007 | 0.0% | **305,857** | 305,394 | 1,232,945 / 816,286 | **1.51x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `5,000` | 6,000,000 | 4087 / 15 | 18.962 | 0.012 | 0.1% | **316,198** | 315,870 | 1,274,668 / 1,016,881 | **1.25x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `15,000` | 6,000,000 | 1361 / 16 | 17.381 | 0.023 | 0.1% | **344,594** | 344,560 | 1,390,961 / 1,221,423 | **1.14x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `30,000` | 6,000,000 | 679 / 17 | 16.943 | 0.060 | 0.4% | **352,408** | 352,741 | 1,420,996 / 1,459,188 | **0.97x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `60,000` | 6,000,000 | 339 / 19 | 16.364 | 0.083 | 0.5% | **363,917** | 364,714 | 1,464,859 / 1,466,274 | **1.00x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `120,000` | 6,000,000 | 168 / 19 | 16.025 | 0.200 | 1.2% | **367,983** | 369,724 | 1,487,217 / 1,482,432 | **1.00x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `250,000` | 6,000,000 | 79 / 21 | 15.748 | 0.487 | 3.0% | **365,485** | 369,542 | 1,484,126 / 1,511,253 | **0.98x** |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `500,000` | 6,000,000 | 39 / 21 | 15.459 | 0.630 | 3.9% | **364,716** | 372,887 | 1,495,123 / 1,529,252 | **0.98x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `2,500` | 6,000,000 | 8594 / 56 | 18.991 | 0.011 | 0.1% | **315,823** | 315,256 | 1,356,384 / 648,851 | **2.09x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `5,000` | 6,000,000 | 4297 / 59 | 17.506 | 0.013 | 0.1% | **342,504** | 342,121 | 1,458,978 / 891,144 | **1.64x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `15,000` | 6,000,000 | 1430 / 104 | 16.342 | 0.036 | 0.2% | **366,296** | 366,113 | 1,569,530 / 1,210,317 | **1.30x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `30,000` | 6,000,000 | 714 / 80 | 15.880 | 0.046 | 0.3% | **376,365** | 376,597 | 1,611,743 / 1,466,403 | **1.10x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `60,000` | 6,000,000 | 356 / 164 | 15.337 | 0.151 | 1.0% | **388,063** | 387,259 | 1,661,201 / 929,060 | **1.79x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `120,000` | 6,000,000 | 177 / 109 | 14.885 | 0.166 | 1.1% | **396,860** | 398,558 | 1,701,292 / 1,611,617 | **1.06x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `250,000` | 6,000,000 | 83 / 122 | 14.820 | 0.468 | 3.1% | **388,078** | 392,419 | 1,681,020 / 1,695,251 | **0.99x** |
| `W2_NEMA_Tb161_Direct` | `direct` | `500,000` | 6,000,000 | 40 / 175 | 14.256 | 0.920 | 6.1% | **385,976** | 395,315 | 1,686,123 / 1,732,264 | **0.97x** |

### 7.2. Пофазная структура времени в стационарном режиме полного банка (`Full-Bank Sustained Phase`)

| Сцена | Коллиматор | `batch_size` | Скорость `Full-Bank` (ист./с) | `Full` Source (%) | `Full` Geometry (%) | `Full` Woodcock (%) | `Full` Processes (%) | `Full` Compaction+Esc (%) | Заполнение банка общ. / в `Depletion` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PointSource_Parametric` | `parametric` | `2,500` | **797,278** | 32.4% | 6.0% | 16.3% | 14.0% | 31.3% | 99.9% / 19.2% |
| `W1_PointSource_Parametric` | `parametric` | `5,000` | **885,635** | 34.4% | 6.5% | 16.5% | 9.9% | 32.8% | 99.9% / 28.6% |
| `W1_PointSource_Parametric` | `parametric` | `15,000` | **930,742** | 36.9% | 6.9% | 16.0% | 6.8% | 33.5% | 99.5% / 16.7% |
| `W1_PointSource_Parametric` | `parametric` | `30,000` | **963,989** | 37.1% | 6.9% | 16.0% | 5.8% | 34.2% | 98.8% / 15.1% |
| `W1_PointSource_Parametric` | `parametric` | `60,000` | **973,870** | 37.5% | 6.9% | 16.1% | 5.2% | 34.2% | 94.2% / 10.5% |
| `W1_PointSource_Parametric` | `parametric` | `120,000` | **1,002,765** | 37.3% | 6.9% | 16.1% | 5.1% | 34.6% | 90.0% / 6.1% |
| `W1_PointSource_Parametric` | `parametric` | `250,000` | **992,788** | 37.5% | 6.6% | 16.0% | 4.9% | 35.0% | 88.8% / 11.8% |
| `W1_PointSource_Parametric` | `parametric` | `500,000` | **990,392** | 38.1% | 6.4% | 16.0% | 4.7% | 34.7% | 61.0% / 2.6% |
| `W1_PointSource_Direct` | `direct` | `2,500` | **759,847** | 31.8% | 9.9% | 12.4% | 15.0% | 30.9% | 99.5% / 5.2% |
| `W1_PointSource_Direct` | `direct` | `5,000` | **860,859** | 33.5% | 11.0% | 11.1% | 11.3% | 33.0% | 99.4% / 7.8% |
| `W1_PointSource_Direct` | `direct` | `15,000` | **929,404** | 36.4% | 11.7% | 10.3% | 7.0% | 34.5% | 97.3% / 6.7% |
| `W1_PointSource_Direct` | `direct` | `30,000` | **962,466** | 36.9% | 12.2% | 10.1% | 5.9% | 34.9% | 95.1% / 6.8% |
| `W1_PointSource_Direct` | `direct` | `60,000` | **971,157** | 37.2% | 12.4% | 10.1% | 5.3% | 35.0% | 89.6% / 5.1% |
| `W1_PointSource_Direct` | `direct` | `120,000` | **993,999** | 37.0% | 12.5% | 10.0% | 5.1% | 35.5% | 80.9% / 6.8% |
| `W1_PointSource_Direct` | `direct` | `250,000` | **976,842** | 37.6% | 12.1% | 9.8% | 4.8% | 35.7% | 63.8% / 4.1% |
| `W1_PointSource_Direct` | `direct` | `500,000` | **972,357** | 38.4% | 11.9% | 9.8% | 4.5% | 35.4% | 44.0% / 5.4% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `2,500` | **305,857** | 15.5% | 5.1% | 32.4% | 22.4% | 24.6% | 99.9% / 16.5% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `5,000` | **316,198** | 15.5% | 5.3% | 32.9% | 20.7% | 25.7% | 99.7% / 15.8% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `15,000` | **344,594** | 15.6% | 5.6% | 33.1% | 19.5% | 26.2% | 99.0% / 11.6% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `30,000` | **352,408** | 15.4% | 5.8% | 33.5% | 19.0% | 26.3% | 98.0% / 16.3% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `60,000` | **363,917** | 15.1% | 5.9% | 32.7% | 19.1% | 27.1% | 95.3% / 10.5% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `120,000` | **367,983** | 15.4% | 6.0% | 32.3% | 19.0% | 27.4% | 91.1% / 12.5% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `250,000` | **365,485** | 15.5% | 5.9% | 31.5% | 19.6% | 27.5% | 81.8% / 13.2% |
| `W2_NEMA_Tb161_Parametric` | `parametric` | `500,000` | **364,716** | 15.6% | 6.0% | 31.5% | 19.4% | 27.5% | 68.2% / 9.1% |
| `W2_NEMA_Tb161_Direct` | `direct` | `2,500` | **315,823** | 16.6% | 6.2% | 27.3% | 24.3% | 25.6% | 99.4% / 5.0% |
| `W2_NEMA_Tb161_Direct` | `direct` | `5,000` | **342,504** | 15.9% | 6.5% | 27.9% | 22.4% | 27.2% | 98.7% / 4.0% |
| `W2_NEMA_Tb161_Direct` | `direct` | `15,000` | **366,296** | 16.4% | 7.0% | 28.1% | 20.9% | 27.7% | 93.4% / 2.7% |
| `W2_NEMA_Tb161_Direct` | `direct` | `30,000` | **376,365** | 16.3% | 7.2% | 28.4% | 20.0% | 28.1% | 90.2% / 2.8% |
| `W2_NEMA_Tb161_Direct` | `direct` | `60,000` | **388,063** | 16.2% | 7.4% | 27.5% | 20.1% | 29.0% | 68.9% / 1.4% |
| `W2_NEMA_Tb161_Direct` | `direct` | `120,000` | **396,860** | 16.1% | 7.6% | 26.7% | 19.9% | 29.6% | 62.7% / 2.0% |
| `W2_NEMA_Tb161_Direct` | `direct` | `250,000` | **388,078** | 16.7% | 7.4% | 26.5% | 19.9% | 29.6% | 41.9% / 2.4% |
| `W2_NEMA_Tb161_Direct` | `direct` | `500,000` | **385,976** | 16.8% | 7.4% | 26.2% | 20.3% | 29.3% | 20.0% / 1.7% |

---

## 8. Главные аналитические выводы и приоритизированная матрица оптимизаций

### 8.1. Методологическое очищение профиля и эффект внедренных оптимизаций (`Warm Steady-State`)

* **Холодный старт (`Cold Compile`) vs Установившийся режим (`Warm Steady-State`):** консолидация `@njit`-ядер в `processes_kernels.py` сократила время холодного прогрева `ParticlePropagator` более чем вдвое (**`3.55 с -> 1.61 с`**), а время выполнения пакета из 4 задач в `Orchestrator` (Раздел 6) уменьшилось с **`18.21–18.71 с` до `9.73–10.43 с`** при `pool_size = 1` и с **`7.30–7.55 с` до `4.47–4.89 с`** при `pool_size = 4` (**ускорение в `1.6x–1.8x` раза**).
* **Компактная сетка NIST XCOM с краями поглощения и $\mathcal{O}(1)$ лог-индекс (`M1a`, `M1b`):** переход от плотной 1-кэВ сетки (`~1117` узлов) к родной сетке NIST XCOM (`54–95` узлов с дуплетами K/L/M/N-краев поглощения) и двухуровневому логарифмическому индексу `energy_bin_map` ($512$ корзин) ускорил одиночный запрос сечения в **`2.03x` раза** (`35.2 нс`), а спаренный запрос Вудкока (`1x log(E) + 2x _locate_row` с кусочно-степенной `log-log` интерполяцией и выборкой $Z$ по парциальным сечениям) — в **`2.92x` раза** (`60.3 нс` против `176.2 нс`).
* **Zero-Allocation инжекция первичных частиц (`inject_poisson_sources_kernel`, `M2`, `M3`):** переход на таблицы алиасов Воза (`Vose's Alias` $\mathcal{O}(1)$, **`19.1 нс/оп`, ускорение `7.48x`**) и прямую запись в `ParticleBank` полностью обнулил страничные прерывания ОС в установившемся режиме (**`Steady Page Faults = 0`** во всех 6 сценах `W1–W3` против прежних `762–21,839`).
* **Ускорение объемных клинических и фантомных сцен (`W2_NEMA_Tb161`, `W3_Heart_Tc99m`):** в 10-фазном профилировщике (Раздел 3) сквозная скорость на 5-тканевом фантоме миокарда `W3_Heart_Tc99m` выросла до **`196.5–197.0 тыс. ист./с` (`+9.4%..+10.8%`)**, а чистая скорость трекинга полного банка — до **`1.39–1.47 млн треков/с` (`+7.6%..+12.8%`)** в `W3` и **`1.47–1.58 млн треков/с` (`+7.9%..+12.3%`)** в `W2_NEMA_Tb161`.

### 8.2. Двухфазная динамика банка частиц (`Full-Bank Sustained Phase` vs `Tail Depletion Phase`) на выборке 6,000,000 историй

* **Инвариантность генерации источника к размеру батча в воксельных фантомах (Раздел 7):** благодаря $\mathcal{O}(1)$ выборке по таблицам Воза и методу прореживания Льюиса–Шедлера в `inject_poisson_sources_kernel`, на малых батчах (`batch_size = 2,500`, `8,175–8,594` шагов пополнения) доля времени источника в `W2_NEMA_Tb161` сократилась с прежних **`32.5%–34.8%` до `15.5%–16.6%`**, а скорость на `batch_size = 2,500` выросла с `244.6–246.9 тыс. ист./с` до **`305.9–315.8 тыс. ист./с` (`+25.0%..+27.9%`)**. Во всем диапазоне `batch_size` от `2,500` до `500,000` доля `Full Source (%)` в `W2_NEMA_Tb161` остается постоянной (**`15.1%–16.8%`**).
* **Оптимальный диапазон `batch_size` (`60,000 – 120,000`):** максимизирует сквозную производительность как для точечного источника `W1_PointSource` (**до `1.00–1.01 млн историй/с` и `3.27–4.11 млн треков/с`** при `batch_size = 120,000`), так и для воксельного фантома `W2_NEMA_Tb161` (**`369.7 тыс. ист./с` в Parametric и `398.6 тыс. ист./с` в Direct**, до **`1.70 млн треков/с`** при `batch_size = 120,000`), удерживая долю хвостовой фазы опустошения `Tail Depletion Phase` в пределах `0.8%–1.2%`.

### 8.3. Приоритизированная матрица дальнейших оптимизаций (Roadmap)

| Статус / Приоритет | Узел / Модуль | Описание | Реализация / Следующий шаг | Эффект (`Speedup`) |
| :---: | :--- | :--- | :--- | :---: |
| **Выполнено** | `core/materials/`, `core/physics/` (`M1a`, `M1b`) | Компактная сетка NIST XCOM с дуплетами краев поглощения, 2-уровневый лог-индекс `energy_bin_map` и парциальные доли `element_partial_fractions` | Внедрено в `AttenuationDatabase`, `MaterialBank`, `PhysicsCompiler`, `transport_kernels.py` | **`2.03x–2.92x`** на запросах сечений, размер `.h5` **`3.52 -> 0.74 МБ`** |
| **Выполнено** | `core/source/` (`M2`, `M3`) | Таблицы алиасов Воза (`Vose's Alias`), `SourceBuffer` и `@njit`-инжектор `inject_poisson_sources_kernel` | Внедрено в `alias_tables.py`, `source_compiler.py`, `source_kernels.py` | **`7.48x`** выборка вокселя, **`0` Steady Page Faults**, **`+25%..+28%`** на малых батчах `W2` |
| **Выполнено** | `core/data/data_handlers.py` (`M4`, `M5`) | Векторизованный LUT `volume_id` и прямая булева фильтрация `HistoryAssemblerHandler` | Внедрено в `SensitiveVolumeHandler` и `HistoryAssemblerHandler` | **`38.51x`** и **`27.89x`** на фильтрации чанков |
| **P1 (Next)** | [`core/source/source_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/source_kernels.py), [`docs/plans/composable_sources_and_secondary_radiation_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/composable_sources_and_secondary_radiation_plan.md) | В `W1_PointSource` пошаговое прореживание Льюиса–Шедлера при $\lambda = 0$ и матричное умножение нулевых локальных координат тратят `~350 нс/частицу` | В рамках композиционной модели `SpatialDistribution` (`PointDistribution`) + быстрый путь для стационарного одиночного источника ($\lambda = 0$) исключить холостой розыгрыш `u2` и матричное умножение `(0,0,0)` | Сокращение времени `P2_Source` для `PointSource` в **`2.5x–3.0x` раза** |
| **P2 (Next)** | [`core/physics/processes_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py), [`core/transport/transport_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py) | Компиляция замыканий `ParticlePropagator` при холодном старте процесса (`1.61 с`) | Кэширование фабрик `@njit`-замыканий по `process_id` и `tuple(process_ids)` | **`1.61 с -> < 5 мс`** на холодный старт воркера |
