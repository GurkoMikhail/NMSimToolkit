from typing import Any, Annotated
import re
import pint
from pydantic import BeforeValidator
import hepunits as units

ureg = pint.UnitRegistry()


def unit_validator_factory(target_unit: str, to_hepunits_factor: float = 1.0):
    """
    Фабрика Pydantic-валидатора для преобразования физических величин 
    в когерентную внутреннюю систему единиц HepUnits.

    Инвариант валидации единиц измерения:
    - Числовые значения (int, float) обязаны передаваться уже во внутренних единицах HepUnits (например, 4 * units.mm).
    - Строковые значения парсятся через Pint к целевой единице target_unit и домножаются на to_hepunits_factor.
    - Шаблоны интерполяции вида ${var} пропускаются без изменений для отложенной подстановки оркестратором.
    """
    def validator(input_value: Any) -> Any:
        if isinstance(input_value, (int, float)):
            # Числовые значения обязаны передаваться уже во внутренних единицах HepUnits
            return float(input_value)
        if isinstance(input_value, str):
            # Пропуск шаблонов интерполяции вида ${var} для отложенной подстановки в оркестраторе
            if re.search(r'\$\{[^}]+\}', input_value):
                return input_value
            if input_value.strip() == '':
                raise ValueError("Пустая строка не является допустимой физической величиной")
            try:
                quantity = ureg(input_value)
                magnitude_val = float(quantity.to(target_unit).magnitude)
                return float(magnitude_val * to_hepunits_factor)
            except (pint.DimensionalityError, pint.UndefinedUnitError) as unit_err:
                raise ValueError(f"Невозможно преобразовать '{input_value}' к единицам {target_unit}: {unit_err}")
            except Exception as parse_err:
                raise ValueError(f"Ошибка парсинга физической величины '{input_value}': {parse_err}")
        raise ValueError(f"Ожидалось число или строка с единицами, получено {type(input_value)}")
    return validator


LengthConfig = Annotated[Any, BeforeValidator(unit_validator_factory('mm', float(units.mm)))]
EnergyConfig = Annotated[Any, BeforeValidator(unit_validator_factory('MeV', float(units.MeV)))]
TimeConfig = Annotated[Any, BeforeValidator(unit_validator_factory('ns', float(units.ns)))]
ActivityConfig = Annotated[Any, BeforeValidator(unit_validator_factory('Bq', float(units.Bq)))]
AngleConfig = Annotated[Any, BeforeValidator(unit_validator_factory('rad', float(units.radian)))]

