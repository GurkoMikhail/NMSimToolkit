---
description: Незыблемые архитектурные, физические и инженерные контракты проекта NMSimToolkit
trigger: always_on
---

# Архитектурные и инженерные правила NMSimToolkit

Данный документ является главным индексом архитектурных, физических и стилистических контрактов проекта `NMSimToolkit`. Все агенты обязаны неукоснительно соблюдать приведенные правила.

---

## ⚡ Экспресс-памятка инвариантов (Cheat Sheet)

* **Физика и единицы (`hepunits`):** Базовая длина — **мм** (`1.0`), энергия — **МэВ** (`1.0`), время — **нс** (`1.0`), угол — **рад** (`1.0`). Категорический запрет суффиксов единиц измерения в именах идентификаторов (`energy`, но не `energy_kev`). Zero-Compromise Physics: никаких упрощений физики транспорта ради скорости.
* **Изоляция Core, JIT и ГСЧ:** В `core/` категорически запрещены импорты Qt/VTK (`PySide6`, `vtk`, `pyvista`) и атрибуты отображения (`color`, `is_sensitive_detector`). В `@njit`-ядрах — Zero-Allocation и Zero-Logging. Полный запрет глобального `np.random.*` (только явный генератор с `SeedSequence`).
* **Контракты (DbC / LBYL) и Roundtrip:** Абсолютный запрет утиной типизации (`hasattr`, `getattr`, `setattr`). Проверка предусловий до выполнения, Fail-Fast без broad `try...except`. Сквозной инвариант сериализации: `export(build(config)) == config`.
* **Архитектурная целостность (SRP) и импорты:** Запрет искусственного распила классов на сателлиты `*_helper.py` с передачей `section: Any` или `parent: Any`. Только Top-Level импорты (блок `if TYPE_CHECKING:` в начале файла разрешен).
* **Окружение, гигиена тестов и граф:** Запуск строго через `.venv\Scripts\python.exe` и `.venv\Scripts\pytest.exe`. Запрет создания файлов-однодневок `test_*_fixes.py` в корне `tests/`. Обязательная переиндексация `codebase-memory` (`index_repository`) после структурных изменений.
* **Git и язык:** Все docstrings, комментарии и коммиты — строго на русском языке. Коммиты Result-Oriented (только итоговый результат).

---

## 📑 Тематические модули правил

1. **[01. Архитектура ядра и физическая модель (`01_architecture_and_physics.md`)](01_architecture_and_physics.md)**
   * Чистое разделение `core/` и `gui/`;
   * Zero-Compromise Physics и стандарт размерностей `hepunits`;
   * Иерархия сцены (`CompositeNode`, `SpatialNode`, `Volume`), Data Purity и Roundtrip-инвариант;
   * Модульность подпакетов, фасадные `__init__.py` и изоляция Numba-кернелов/Pydantic-схем;
   * Полное отсутствие костылей обратной совместимости (No Legacy Crutches);
   * Контракты Numba JIT (Zero-Allocation) и воспроизводимость ГСЧ (Seed Reproducibility).

2. **[02. Контрактное программирование, типизация и стиль (`02_code_contracts_and_style.md`)](02_code_contracts_and_style.md)**
   * Абсолютный запрет утиной типизации (`hasattr`/`getattr`) и моков в ядре;
   * C-подобная надежность LBYL, явные инварианты, Headless-тестируемость GUI/VTK;
   * Рациональная типизация: отсутствие проверок `isinstance` при однозначном type hint;
   * Принцип SRP и категорический запрет псевдо-хелперов `*_helper.py`;
   * Top-Level импорты (с поддержкой `if TYPE_CHECKING:`) и запрет однобуквенных наименований.

3. **[03. Процессы разработки, окружение и документация (`03_dev_workflow.md`)](03_dev_workflow.md)**
   * Окружение строго в `.venv` и доменная гигиена тестов в `tests/`;
   * Ведение архитектурного бэклога в [`docs/architecture_backlog.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/architecture_backlog.md);
   * Консолидированная структура каталога `docs/` и изоляция планов в `docs/plans/`;
   * Стандарты Result-Oriented Git-коммитов на русском языке;
   * Ограничения моделей Claude, актуализация `codebase-memory` и инварианты Handoff Spec.
