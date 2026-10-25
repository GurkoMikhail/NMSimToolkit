# NMSimToolkit (Nuclear Medicine Simulation Toolkit)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

**NMSimToolkit** — современный программный комплекс и интерактивное 3D-приложение для Монте-Карло моделирования систем ядерной медицины (однофотонной эмиссионной компьютерной томографии **ОФЭКТ/SPECT** и позитронно-эмиссионной томографии **ПЭТ/PET**).

---

## Ключевые возможности

* **Высокопроизводительный транспорт фотонов (Data-Oriented Design / SoA):** Моделирование методом Монте-Карло в плоских буферах Numba JIT с алгоритмом отслеживания Вудкока (Woodcock tracking) и аналитическим RayCasting-ом периодических коллиматоров.
* **Физические взаимодействия (Zero-Compromise Physics):** Фотоэлектрическое поглощение, комптоновское рассеяние и когерентное (рэлеевское) рассеяние с CSR-таблицей элементного состава материалов (`ElementCSR`, выбор атомного номера $Z$ по предрассчитанным нормированным парциальным сечениям элементов `element_partial_fractions` для каждого процесса) и сечениями взаимодействия из базы данных NIST XCOM.
* **Интерактивный 3D-вьюпорт (PyVista / VTK):**
  * Отображение геометрии коллиматоров, кристаллов детектора и воксельных фантомов (с физической палитрой материалов и режимом псевдорентгена DRR).
  * Runtime-трекинг траекторий частиц с цветовой дифференциацией физических процессов в реальном времени.
  * Манипуляторы томографических орбит ОФЭКТ (`GantryNode`, `GammaCameraNode`) и колец детекторов ПЭТ, а также UE-Style 3D Gizmo (`TransformGizmo`).
* **3D-визуализация накопления дозы (Volume Rendering):**
  * In-place отображение воксельного энерговыделения (`DoseGridNode`) без задержек и пересоздания шейдеров GPU.
  * Медицинские палитры псевдоокрашивания (*Hot Iron*, *Rainbow*, *GE Color* и др.) с динамическим отсечением фона.
* **Анализатор результатов и спектрометрия:**
  * Накопление 2D-проекций гамма-камеры в разделяемой памяти (`SharedMemory` IPC).
  * Энергетический спектр с идентификацией фотопиков радиоизотопов (99mTc, 123I, 161Tb и др.).
  * Построение горизонтальных и вертикальных профилей счета в реальном времени.
* **Декларативная архитектура и строгий Roundtrip:** Конфигурирование сцен и протоколов сканирования (`StepAndShoot`, `CustomSweep`) через YAML-файлы с поддержкой привязки переменных (`Property Binding`), реестром метаданных источников (`SourceMetadataConfig` / `source_metadata_registry`) и инспектором свойств узлов (паттерн MVVM).

---

## Установка (требуется Python 3.12+)

### Вариант 1: Через `uv` (Рекомендуется)
[uv](https://github.com/astral-sh/uv) обеспечивает быструю установку и гарантирует воспроизводимость зависимостей:
```bash
git clone https://github.com/GurkoMikhail/NMSimToolkit.git
cd NMSimToolkit
uv sync
```

### Вариант 2: Стандартный `pip` и виртуальное окружение `.venv`
```bash
python -m venv .venv
# Активация окружения (Windows PowerShell):
.venv\Scripts\Activate.ps1
# Активация окружения (Linux / macOS):
# source .venv/bin/activate

pip install -e .
```
Либо установка зависимостей по файлу:
```bash
pip install -r requirements.txt
```

---

## Запуск

### 1. Графический интерфейс (GUI)
После установки пакета доступна CLI-команда:
```bash
nmsim-gui
```
Или прямой вызов модуля:
```bash
python -m gui.app
```

### 2. Пакетный расчет через CLI
Для запуска моделирования без графического интерфейса по YAML-конфигурации:
```bash
nmsim-cli examples/nema_1_cam.yaml
# или
python run_simulation.py examples/tb161_nema_protocol.yaml
```

### 3. Клинические сценарии и примеры
В каталоге [`examples/`](examples/README.md) доступны готовые скрипты и YAML-конфигурации экспериментов:
```bash
python examples/brain_main.py             # Томография головного мозга (фантом Хоффмана)
python examples/heart_main.py             # Сцинтиграфия миокарда (99mTc-MIBI, 60 ракурсов)
python examples/heart_32_main.py          # 32-проекционная ОФЭКТ миокарда
python examples/lung_cancer_main.py       # Моделирование патологий легких
python examples/tb161_nema_simulation.py  # Двухкамерный протокол ОФЭКТ 161Tb с фантомом NEMA IEC
```

---

## Структура репозитория

```
NMSimToolkit/
├── core/                # Вычислительное ядро Монте-Карло (DoD/SoA), физика, геометрия, конфигурация, HDF5
├── gui/                 # MVVM-архитектура графического интерфейса (PySide6 / VTK)
│   ├── controllers/     # Медиаторы сессий (OrchestratorSession, SceneViewportController, IPCReceiver)
│   ├── factories/       # Фабрики декларативной сборки составных узлов (GammaCameraNode)
│   ├── models/          # Модели настроек GUI и импорта распределений
│   ├── viewmodels/      # Обобщенные ViewModels графа сцены (NodeViewModel[T]), процедур и обработчиков
│   ├── views/           # Главное окно, секции инспектора свойств, виджеты биндинга и просмотрщик результатов
│   └── viewport_3d/     # 3D-вьюпорт VTK, рендереры треков/дозы/фантомов и кинематические манипуляторы
├── infra/               # Инфраструктурные сервисы и мониторинг (Telegram-уведомления)
├── settings/            # Конфигурация подключения к базам данных материалов и процессов
├── docs/                # Архитектурная документация, описание кинематики, бэклог и планы ревью
├── examples/            # Скрипты и YAML-протоколы клинических сценариев моделирования ОФЭКТ
├── phantoms/            # Цифровые анатомические и тестовые фантомы (NEMA IEC, Hoffman)
├── presets/             # Базовые пресеты параметров и геометрии
├── tests/               # Модульные и интеграционные тесты, сгруппированные по доменным пакетам:
│   ├── config/          # Тесты Pydantic-схем, Roundtrip, SweepCompiler и TemplateBinder
│   ├── controllers/     # Тесты сессий оркестратора, воркеров и IPC
│   ├── data/            # Тесты DataManager, обработчиков HDF5 и DistributionLoader
│   ├── geometry/        # Тесты геометрии, компилятора сцены, коллиматоров и RayCasting
│   ├── materials/       # Тесты базы материалов и функций ослабления
│   ├── other/           # Тесты пространственных трансформаций и векторов SoA
│   ├── particles/       # Тесты банков частиц SoA
│   ├── physics/         # Тесты кернелов Комптона, Рэлея, фотоэффекта и PhysicsCompiler
│   ├── source/          # Тесты воксельных и точечных источников излучения
│   ├── transport/       # Тесты пропагатора, Woodcock-трекинга и SimulationManager
│   ├── viewmodels/      # Тесты NodeViewModel, процедур и кинематических ограничений
│   ├── viewport_3d/     # Тесты 3D-вьюпорта, гизмо и рендереров
│   └── views/           # Тесты главного окна, секций инспектора и виджетов
├── pyproject.toml       # Метаданные пакета и настройки инструментов (Python 3.12+)
└── run_simulation.py    # Точка входа для пакетного запуска симуляций из CLI
```

---

## Тестирование и проверка типов

Запуск полного тестового сьюта и статического анализатора типов `mypy`:
```bash
pytest
mypy --python-version 3.12 --ignore-missing-imports core gui infra settings run_simulation.py examples
```

---

## Документация

Подробные архитектурные спецификации и инженерная документация находятся в каталоге [`docs/`](docs/README.md):
* [Оглавление технической документации](docs/README.md)
* [Архитектура расчетного ядра (`core_architecture.md`)](docs/core_architecture.md)
* [Архитектура графического интерфейса (`gui_architecture.md`)](docs/gui_architecture.md)
* [Кинематическая система и детекторы (`kinematic_system.md`)](docs/kinematic_system.md)
* [Архитектурный бэклог проекта (`architecture_backlog.md`)](docs/architecture_backlog.md)
* [Сводный отчет комплексного ревью (`docs/plans/project_review_report.md`)](docs/plans/project_review_report.md)

---

## Лицензия

Проект распространяется под открытой лицензией [MIT License](LICENSE).

