from core.geometry.volumes import TransformableVolume


class WoodcockVolume(TransformableVolume):
    """
    Базовый класс Woodcock объёма
    """


class WoodcockParameticVolume(WoodcockVolume):
    """
    Класс параметричекого Woodcock объёма
    """

    def _parametric_function(self, position):
        return [], None

    def get_material_by_position(self, position, local=False, as_parent=True):
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        material = super().get_material_by_position(position, True, as_parent)
        inside = (material != 0).nonzero()[0]
        indices, new_material = self._parametric_function(position[inside])
        material[inside[indices]] = new_material
        return material

