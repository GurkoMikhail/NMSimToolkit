# Фреймворк физической верификации и бенчмаркинга NMSimToolkit (`verification/`)

Автономный пакет `verification/` предназначен для воспроизводимой физико-математической валидации, перекрестной верификации моделей и профилирования производительности вычислительного ядра `NMSimToolkit`.

В отличие от модульных тестов `tests/` (выполняемых за миллисекунды при каждом коммите), эксперименты в `verification/` запускают полномасштабные Монте-Карло симуляции ($10^5\text{–}10^7$ историй), сверяют распределения с аналитическими законами медицинской физики и формируют самодостаточные отчеты (`.md`, `.json`, `.png`).

---

## 1. Структура пакета и масштабирование на все подсистемы

```text
verification/
├── __init__.py
├── README.md                          # Архитектура и регламент верификации
├── run_verification.py                # Единая CLI-точка входа (--domain, --experiments)
├── common/                            # Переиспользуемое ядро для всех доменов
│   ├── __init__.py
│   ├── presets.py                     # Пресеты GE Healthcare (LEHR, LEUHR, MEGP, HEGP) и нуклидов
│   ├── sources.py                     # Специализированные источники (ConePointSource)
│   ├── history_classifier.py          # Потоковый классификатор фотонов (Geometric / Septal / Scatter)
│   ├── scene_builders.py              # Сборщики верификационных сцен без искажений детектора
│   ├── metrics.py                     # FWHM/FWTM, чувствительность, KS-тест, хи-квадрат, Z-score
│   └── plotting.py                    # Отрисовка 2D/1D PSF, кривых Ангера и бенчмарков (Agg)
├── collimators/                       # Домен 1: Параллельные коллиматоры (Parametric vs Direct)
│   ├── __init__.py
│   ├── exp01_ray_optical_depth.py     # Детерминированный аудит геометрии и оптического пути в свинце
│   ├── exp02_psf_ge_series.py         # Покомпонентная 2D/1D PSF на серии GE (LEHR, LEUHR, MEGP, HEGP)
│   ├── exp03_anger_distance_curve.py  # Кривые разрешения FWHM(Z) и инвариантности S(Z) по Ангеру
│   ├── exp04_septal_star_stress.py    # Стресс-тест септального пробоя I-131 («эффект 6-лучевой звезды»)
│   └── exp05_speed_benchmark.py       # Бенчмарк скорости счета Parametric vs Direct
└── reports/                           # Автогенерируемые отчеты, JSON-метрики и PNG-графики
    └── collimators/
```

### Планируемые домены расширения
- `verification/physics/` — сверка сечений XCOM, угловых распределений Клейна-Нишины и спектров Комптона/Рэлея;
- `verification/sources/` — проверка изотропии, воксельных фантомов и цепочек радиоактивного распада;
- `verification/detectors/` — энергетическое разрешение, тормозной путь и параллакс глубины взаимодействия в кристаллах $\text{NaI(Tl)}$ / $\text{CZT}$;
- `verification/spect/` — сквозные фантомные тесты NEMA NU 1.

---

## 2. Запуск экспериментов

Полный прогон всех 5 экспериментов верификации коллиматоров:
```powershell
.venv\Scripts\python.exe -m verification.run_verification --domain collimators --experiments all
```

Быстрый экспресс-прогон (`--quick`):
```powershell
.venv\Scripts\python.exe -m verification.run_verification --domain collimators --quick
```

Запуск отдельных экспериментов:
```powershell
.venv\Scripts\python.exe -m verification.run_verification --experiments exp01 exp05
```
