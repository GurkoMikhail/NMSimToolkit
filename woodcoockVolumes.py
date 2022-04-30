import numpy as np
from volumes import TransformableVolume
from geometries import Box


class WoodcockVolume(TransformableVolume):
    """
    Базовый класс Woodcock объёма
    """


class WoodcockParameticVolume(WoodcockVolume):
    """
    Класс параметричекого Woodcock объёма
    """

    def _parametricFunction(self, position):
        return [], None

    def getMaterialByPosition(self, position, local=False, asParent=True):
        if not local:
            position = self.convertToLocalPosition(position, asParent)
        materials = super().getMaterialByPosition(position, True, asParent)
        inside = np.array([i for i, material in enumerate(materials) if material is not None], dtype=int)
        indices, newMaterial = self._parametricFunction(position[inside])
        materials[inside[indices]] = newMaterial
        return materials

