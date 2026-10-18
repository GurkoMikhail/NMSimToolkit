# Примеры использования (Examples)

В этом каталоге собраны клинические сценарии и примеры моделирования для ядерной медицины (ОФЭКТ), использующие актуальный SoA API расчетного ядра (`Source`, `GammaCameraNode`, `WoodcockVoxelVolume`, `ParticlePropagator`, `SimulationManager`, `DataManager`):

- **`brain_main.py`** — Моделирование ОФЭКТ головного мозга (Brain SPECT) с фантомом Хоффмана и четырехголовочной системой гамма-камер.
- **`heart_main.py`** — Моделирование перфузионной ОФЭКТ миокарда (60 проекций) с радиофармпрепаратом 99mTc-MIBI (`Source`, 140.511 кэВ, $T_{1/2} = 6.0067$ ч) и воксельным фантомом грудной клетки.
- **`heart_32_main.py`** — Сценарий томографического сканирования миокарда на 32 угловых проекциях с двухголовочной гамма-камерой.
- **`lung_cancer_main.py`** — Сценарий моделирования накопления радиофармпрепарата при новообразованиях легких.
- **`tb161_nema_simulation.py`** — Двухкамерный протокол ОФЭКТ/КТ $^{161}\text{Tb}$ (GE Discovery LEHR, 60 ракурсов по 30 с) с воксельным фантомом NEMA IEC и 21-линейным спектром излучения.
- **`tb161_nema_protocol.yaml`** — Декларативная YAML-конфигурация двухкамерного протокола `StepAndShoot` для $^{161}\text{Tb}$ и фантома NEMA IEC.
- **`nema_1_cam.yaml`** — Декларативная YAML-конфигурация однокамерного протокола `CustomSweep` с фантомом NEMA IEC.

Все пути к фантомам, логам и выходным файлам HDF5 вычисляются относительно корня репозитория (`PROJECT_ROOT`).

### Запуск примеров

Из корневой директории репозитория:
```bash
python examples/brain_main.py
python examples/heart_main.py
python examples/heart_32_main.py
python examples/lung_cancer_main.py
python examples/tb161_nema_simulation.py
python run_simulation.py examples/tb161_nema_protocol.yaml
python run_simulation.py examples/nema_1_cam.yaml
```


