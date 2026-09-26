"""
Подпакет вспомогательных структур данных, SoA-векторов и определений типов ядра.
"""

from core.other.vectors import Vector3DSoA
from core.other.nonunique_array import NonuniqueArray
from core.other.typing_definitions import (
    Float,
    Length,
    Energy,
    Time,
    Vector3D,
    Index,
    ID,
    ProcessID,
    Species,
    Charge,
)
from core.other.transform import Matrix3x3DType, Vector3DDType, TransformDType

__all__ = [
    'Vector3DSoA',
    'NonuniqueArray',
    'Float',
    'Length',
    'Energy',
    'Time',
    'Vector3D',
    'Index',
    'ID',
    'ProcessID',
    'Species',
    'Charge',
    'Matrix3x3DType',
    'Vector3DDType',
    'TransformDType',
]
