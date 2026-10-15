"""
Модуль составного узла детекторной головки гамма-камеры в графе сцены расчетного ядра.
"""

from typing import Dict, Optional
from core.scene.nodes import CompositeNode


class GammaCameraNode(CompositeNode):
    """
    Семантический составной узел детекторной головки гамма-камеры.
    Является чистым составным узлом (CompositeNode), содержащим дочерние физические
    объемы (корпус, коллиматор, сцинтилляционный кристалл и т.д.) согласно открытой
    декларативной схеме сборки со слотами.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        slots: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(name=name or "GammaCamera")
        self.slots: Dict[str, str] = dict(slots) if slots is not None else {}


__all__ = ["GammaCameraNode"]
