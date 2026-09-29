"""
Модуль перечислений типов, режимов и систем координат 3D-манипулятора трансформаций (TransformGizmo).
Вынесен в автономный модуль для развязки циклических зависимостей между вьюпортом и кинематическими ограничениями.
"""

from enum import Enum


class GizmoMode(Enum):
    """Режим трансформации 3D-манипулятора."""
    TRANSLATE = "translate"
    ROTATE = "rotate"
    SCALE = "scale"


class GizmoSpace(Enum):
    """Система координат трансформации."""
    WORLD = "world"
    LOCAL = "local"


class GizmoAxis(Enum):
    """Активная ось или плоскость взаимодействия манипулятора."""
    NONE = "none"
    X = "x"
    Y = "y"
    Z = "z"
    XY = "xy"
    XZ = "xz"
    YZ = "yz"
    XYZ = "xyz"


__all__ = [
    "GizmoMode",
    "GizmoSpace",
    "GizmoAxis",
]
