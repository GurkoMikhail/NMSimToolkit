from typing import Any, Annotated
import pint
from pydantic import BeforeValidator

ureg = pint.UnitRegistry()

def unit_validator_factory(target_unit: str):
    def validator(v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                quantity = ureg(v)
                return quantity.to(target_unit).magnitude
            except (pint.DimensionalityError, pint.UndefinedUnitError) as e:
                raise ValueError(f"Cannot convert '{v}' to {target_unit}: {e}")
            except Exception as e:
                raise ValueError(f"Failed to parse '{v}' as physical quantity: {e}")
        raise ValueError(f"Expected number or string, got {type(v)}")
    return validator

LengthConfig = Annotated[float, BeforeValidator(unit_validator_factory('mm'))]
EnergyConfig = Annotated[float, BeforeValidator(unit_validator_factory('MeV'))]
TimeConfig = Annotated[float, BeforeValidator(unit_validator_factory('ns'))]
ActivityConfig = Annotated[float, BeforeValidator(unit_validator_factory('Bq'))]
