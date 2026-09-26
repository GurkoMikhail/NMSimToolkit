import hepunits as units
from core.other.typing_definitions import Float, Length
from core.scene.nodes import CompositeNode


class PetScanner(CompositeNode):
    """
    Базовый класс геометрического узла ПЭТ-сканера в графе сцены расчетного ядра.
    Представляет цилиндрическое кольцо детекторов позитронно-эмиссионного томографа.
    """

    def __init__(
        self,
        name: str = "PetScanner",
        diameter: Length = Float(600.0 * units.mm),
        axial_length: Length = Float(200.0 * units.mm),
        num_sectors: int = 32,
    ) -> None:
        if diameter <= 0:
            raise ValueError(f"Диаметр ПЭТ-сканера должен быть положительным: {diameter}")
        if axial_length <= 0:
            raise ValueError(f"Аксиальная длина ПЭТ-сканера должна быть положительной: {axial_length}")
        if num_sectors <= 0:
            raise ValueError(f"Число секторов ПЭТ-сканера должно быть положительным: {num_sectors}")

        super().__init__(name=name)
        self.diameter = Float(diameter)
        self.axial_length = Float(axial_length)
        self.num_sectors = int(num_sectors)

