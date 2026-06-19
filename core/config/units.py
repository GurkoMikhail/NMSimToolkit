from typing import Any, Annotated
import pint
from pydantic import BeforeValidator

ureg = pint.UnitRegistry()

import re

def unit_validator_factory(target_unit: str):
    def validator(v: Any) -> Any:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # Check for interpolation pattern (e.g. ${angle})
            if re.search(r'\$\{[^}]+\}', v):
                return v # Return as raw string for orchestrator injection later
            if v.strip() == '':
                raise ValueError("Empty string is not a valid unit")
            try:
                quantity = ureg(v)
                return quantity.to(target_unit).magnitude
            except (pint.DimensionalityError, pint.UndefinedUnitError) as e:
                raise ValueError(f"Cannot convert '{v}' to {target_unit}: {e}")
            except Exception as e:
                raise ValueError(f"Failed to parse '{v}' as physical quantity: {e}")
        raise ValueError(f"Expected number or string, got {type(v)}")
    return validator

LengthConfig = Annotated[Any, BeforeValidator(unit_validator_factory('mm'))]
EnergyConfig = Annotated[Any, BeforeValidator(unit_validator_factory('MeV'))]
TimeConfig = Annotated[Any, BeforeValidator(unit_validator_factory('ns'))]
ActivityConfig = Annotated[Any, BeforeValidator(unit_validator_factory('Bq'))]
AngleConfig = Annotated[Any, BeforeValidator(unit_validator_factory('rad'))]
