"""
Определения базовых физических и индексных псевдонимов типов (TypeAlias) расчетного ядра.
"""
import ctypes
from enum import IntEnum
from typing import TypeAlias
import numpy as np
from numba import from_dtype
from numpy.typing import NDArray

Float: TypeAlias = np.float64
Length: TypeAlias = Float
Energy: TypeAlias = Float
Time: TypeAlias = Float
Activity: TypeAlias = Float
Angle: TypeAlias = Float
Density: TypeAlias = Float

Vector3D: TypeAlias = NDArray[Float]
ID: TypeAlias = np.int64
Species: TypeAlias = np.uint8
Index: TypeAlias = np.int64


class ParticleSpecies(IntEnum):
    """Канонические коды типов частиц в расчётном ядре."""

    PHOTON = 0
    ELECTRON = 1
    POSITRON = 2
    ALPHA = 3


class ProcessType(IntEnum):
    """Канонические коды физических процессов взаимодействия в расчётном ядре."""

    PHOTOELECTRIC = 0
    COMPTON = 1
    COHERENT = 2
    PAIR_PRODUCTION = 3


ShapeID: TypeAlias = np.int32

# Сигнатура CFUNCTYPE для функций пространственного поиска материала Вудкока
CMaterialFunc = ctypes.CFUNCTYPE(
    np.ctypeslib.as_ctypes_type(Index),
    np.ctypeslib.as_ctypes_type(Float),
    np.ctypeslib.as_ctypes_type(Float),
    np.ctypeslib.as_ctypes_type(Float),
)

CFuncAddress: TypeAlias = np.uint64

NumbaFloat = from_dtype(np.dtype(Float))
NumbaIndex = from_dtype(np.dtype(Index))

ProcessID: TypeAlias = np.uint8
Charge: TypeAlias = np.int8

__all__ = [
    "Float",
    "Length",
    "Energy",
    "Time",
    "Activity",
    "Angle",
    "Density",
    "Vector3D",
    "ID",
    "Species",
    "ParticleSpecies",
    "ProcessType",
    "Index",
    "ShapeID",
    "CMaterialFunc",
    "CFuncAddress",
    "NumbaFloat",
    "NumbaIndex",
    "ProcessID",
    "Charge",
]
