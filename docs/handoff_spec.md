# Спецификация передачи задачи (Handoff Specification)
## Реализация универсального диалога импорта воксельных данных и нецелочисленного маппинга

---

### 1. Область изменений и целевые файлы

| Файл | Назначение | Характер изменений |
|---|---|---|
| `gui/models/distribution_import_params.py` | Модель данных параметров импорта | **Новый файл:** `ImportTargetKind`, `DistributionImportParameters` |
| `core/data/distribution_loader.py` | Сервис загрузки распределений | Добавление статического метода `inspect_metadata(file_path)` |
| `gui/views/distribution_import_dialog.py` | Универсальный мастер импорта | **Новый файл:** `DistributionImportDialog` с 3 блоками |
| `gui/viewmodels/nodes/voxel_volume_vm.py` | ViewModel воксельного фантома | Поддержка вещественного маппинга `dict[float, str]` без `astype(int)` |
| `gui/viewmodels/nodes/source_vm.py` | ViewModel источника излучения | Поддержка расширенных параметров формы, порядка и шага вокселей |
| `gui/views/property_inspector.py` | Инспектор свойств | Интеграция вызова диалога для фантома и источника |
| `tests/test_distribution_import_dialog.py` | Набор автоматических тестов | **Новый файл:** модульные и интеграционные тесты |

---

### 2. Сигнатуры интерфейсов и типы данных

#### 2.1. `gui/models/distribution_import_params.py`
```python
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

class ImportTargetKind(Enum):
    PHANTOM = "phantom"
    SOURCE = "source"

@dataclass
class DistributionImportParameters:
    target_kind: ImportTargetKind
    file_path: Path
    shape: Tuple[int, int, int]
    order: str  # 'F' или 'C'
    dtype: np.dtype
    encoding: str  # 'binary' или 'text'
    voxel_size: Union[float, Tuple[float, float, float]]
    mapping: Optional[Dict[float, str]] = None
    fill_value: str = "Vacuum"
    total_activity: Optional[float] = None
    noise_threshold: Optional[float] = None
    is_npy: bool = False
```

#### 2.2. `core/data/distribution_loader.py`
```python
@classmethod
def inspect_metadata(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Легковесная инспекция метаданных файла распределения без полного чтения массива в память.
    Возвращает:
      - 'suffix': расширение файла
      - 'file_size': размер файла в байтах
      - 'is_npy': bool
      - 'shape': Optional[Tuple[int, ...]] (если .npy)
      - 'order': Optional[str] ('C'/'F', если .npy)
      - 'dtype': Optional[np.dtype] (если .npy)
    """
```

#### 2.3. `gui/views/distribution_import_dialog.py`
```python
class DistributionImportDialog(QDialog):
    def __init__(
        self,
        file_path: Union[str, Path],
        target_kind: ImportTargetKind,
        current_voxel_size: Union[float, Tuple[float, float, float]],
        current_shape: Optional[Tuple[int, ...]] = None,
        existing_mapping: Optional[Dict[float, str]] = None,
        scene_phantom_dims: Optional[Tuple[int, int, int]] = None,
        scene_phantom_voxel_size: Optional[Tuple[float, float, float]] = None,
        parent: Optional[QWidget] = None,
    ) -> None: ...

    def get_import_parameters(self) -> DistributionImportParameters: ...
```

---

### 3. Алгоритм нецелочисленного сопоставления вокселей (Float-Safe Mapping)

В `VoxelVolumeViewModel.reload_distribution`:
1. Загрузка массива:
   ```python
   raw_data = DistributionLoader.load(
       target_path,
       target_shape=target_shape,
       order=order,
       dtype=dtype,
       encoding=encoding
   )
   ```
2. Формирование списка уникальных материалов `element_list`:
   - Начинается с `Material(name=fill_value, ID=0)` (по умолчанию `Vacuum`).
   - Для каждого уникального имени материала из `mapping` регистрируется объект `Material` из базы данных NIST с последовательным `ID`.
3. Создание `MaterialArray(raw_data.shape)`.
4. Заполнение массива индексов через буфер `view(np.ndarray)`:
   - Все ячейки инициализируются индексом `0` (`fill_value`).
   - Для каждого значения `float_value, material_name` из `mapping`:
     ```python
     material_idx = material_to_idx[material_name]
     mask = np.isclose(raw_data, float(float_value), atol=1e-5)
     underlying_buffer[mask] = material_idx
     ```
   Это полностью исключает округление `int()` и сохраняет любые дробные калибровочные метки.

---

### 4. Ограничения и инварианты (DbC)

1. **Разделение подсистем:**
   - Никаких графических библиотек (`PySide6`, `pyqtgraph`, `vtk`) в каталоге `core/`.
   - `DistributionImportParameters` размещается строго в `gui/models/`.
2. **Запрет утиной типизации:**
   - Запрещены `hasattr()`, `getattr()`, динамический `setattr()`.
   - Взаимодействие только через типизированные методы базовых классов.
3. **Запрет однобуквенных переменных:**
   - Имена только предметные: `dimension_x`, `voxel_step_x`, `material_name`, `buffer_byte_size`.
4. **Конвенция единиц:**
   - Все длины — в миллиметрах (`mm`), активности — в мегабеккерелях (`MBq`).
