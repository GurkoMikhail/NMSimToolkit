"""
Реестр сопоставления моделей представления и секций инспектора свойств.
"""
from typing import Any, List

from gui.views.inspectors.base_section import BaseInspectorSection


class InspectorRegistry:
    """
    Реестр сопоставления ViewModel -> Секции инспектора.
    Управляет жизненным циклом и динамическим выбором активных секций для отображения.
    """

    def __init__(self) -> None:
        """Инициализирует пустой реестр секций инспектора."""
        self._sections: List[BaseInspectorSection] = []

    def register_section(self, section: BaseInspectorSection) -> None:
        """Регистрирует экземпляр секции в общем реестре инспектора."""
        if section not in self._sections:
            self._sections.append(section)

    def register_sections(self, sections: List[BaseInspectorSection]) -> None:
        """Регистрирует список секций."""
        for section in sections:
            self.register_section(section)

    def get_all_sections(self) -> List[BaseInspectorSection]:
        """Возвращает все зарегистрированные секции."""
        return list(self._sections)

    def get_active_sections(self, view_model: Any) -> List[BaseInspectorSection]:
        """
        Возвращает список секций, применимых для указанной модели представления.
        Секции опрашиваются через can_handle(view_model).
        """
        if view_model is None:
            return []
        return [section for section in self._sections if section.can_handle(view_model)]


__all__ = ["InspectorRegistry"]
