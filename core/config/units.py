from typing import Any, Annotated, Dict, List, Optional, Tuple, Union
import math
import re
import pint
from pydantic import BeforeValidator
import hepunits as units

ureg = pint.UnitRegistry()

# Словарь канонической нормализации русских, символьных и регистронезависимых единиц измерения
UNIT_ALIASES: Dict[str, str] = {
    # Длина
    "мм": "mm",
    "см": "cm",
    "дм": "dm",
    "м": "m",
    "км": "km",
    "мкм": "um",
    "микрометр": "um",
    "нм": "nm",
    "Å": "angstrom",
    "ангстрем": "angstrom",
    "μm": "um",
    # Угол
    "°": "deg",
    "град": "deg",
    "градус": "deg",
    "градуса": "deg",
    "градусов": "deg",
    "рад": "rad",
    "радиан": "rad",
    "мрад": "mrad",
    "мкрад": "urad",
    "μrad": "urad",
    # Время
    "с": "s",
    "сек": "s",
    "секунда": "s",
    "секунды": "s",
    "секунд": "s",
    "мс": "ms",
    "мсек": "ms",
    "мкс": "us",
    "мксек": "us",
    "μs": "us",
    "нс": "ns",
    "нсек": "ns",
    "пс": "ps",
    "псек": "ps",
    "мин": "min",
    "минута": "min",
    "минуты": "min",
    "минут": "min",
    "ч": "hour",
    "час": "hour",
    "часа": "hour",
    "часов": "hour",
    "сут": "day",
    "дн": "day",
    # Энергия
    "эв": "eV",
    "эВ": "eV",
    "ev": "eV",
    "eV": "eV",
    "кэв": "keV",
    "кэВ": "keV",
    "КэВ": "keV",
    "kev": "keV",
    "keV": "keV",
    "мэв": "MeV",
    "МэВ": "MeV",
    "Мэв": "MeV",
    "mev": "MeV",
    "MeV": "MeV",
    "гэв": "GeV",
    "ГэВ": "GeV",
    "Гэв": "GeV",
    "gev": "GeV",
    "GeV": "GeV",
    "тэв": "TeV",
    "ТэВ": "TeV",
    "tev": "TeV",
    "TeV": "TeV",
    "дж": "J",
    "Дж": "J",
    # Активность
    "бк": "Bq",
    "Бк": "Bq",
    "bq": "Bq",
    "Bq": "Bq",
    "кбк": "kBq",
    "кБк": "kBq",
    "КБк": "kBq",
    "kbq": "kBq",
    "kBq": "kBq",
    "мбк": "MBq",
    "МБк": "MBq",
    "Мбк": "MBq",
    "mbq": "MBq",
    "MBq": "MBq",
    "гбк": "GBq",
    "ГБк": "GBq",
    "Гбк": "GBq",
    "gbq": "GBq",
    "GBq": "GBq",
    "тбк": "TBq",
    "ТБк": "TBq",
    "tbq": "TBq",
    "TBq": "TBq",
    "ки": "Ci",
    "Ки": "Ci",
    "ci": "Ci",
    "Ci": "Ci",
    "мки": "mCi",
    "мКи": "mCi",
    "mci": "mCi",
    "mCi": "mCi",
    "мкки": "uCi",
    "мкКи": "uCi",
    "uci": "uCi",
    "uCi": "uCi",
    "μci": "uCi",
    "μCi": "uCi",
}

_LOWERCASE_UNIT_ALIASES: Dict[str, str] = {
    alias_key.lower(): canonical_val for alias_key, canonical_val in UNIT_ALIASES.items()
}

# Базовые единицы измерения по физическим категориям (для пользовательского ввода без явного default_unit)
DEFAULT_CATEGORY_UNITS: Dict[str, str] = {
    "length": "mm",
    "energy": "MeV",
    "time": "s",
    "angle": "deg",
    "activity": "MBq",
}

_ANGLE_CANONICAL_UNITS: frozenset[str] = frozenset({
    "rad",
    "radian",
    "radians",
    "mrad",
    "milliradian",
    "milliradians",
    "urad",
    "microradian",
    "microradians",
    "deg",
    "degree",
    "degrees",
    "arcmin",
    "arcminute",
    "arcminutes",
    "arcsec",
    "arcsecond",
    "arcseconds",
    "turn",
    "turns",
    "grad",
})

_NUM_ATOM_REGEX = r"(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*pi(?![a-zA-Zа-яА-ЯёЁ_]))?|pi(?![a-zA-Zа-яА-ЯёЁ_]))"
_NUMERIC_PREFIX_PATTERN = re.compile(
    rf"^\s*([+-]?\s*{_NUM_ATOM_REGEX}(?:\s*[*\/]\s*[+-]?\s*{_NUM_ATOM_REGEX})*)\s*(.*)$",
    re.IGNORECASE,
)


def normalize_unit_name(unit_str: str) -> str:
    """
    Нормализует строковое обозначение единицы измерения (включая русские суффиксы и спецсимволы)
    в каноническое имя единицы, поддерживаемое реестром Pint и HepUnits.
    """
    cleaned = unit_str.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()
    if len(cleaned) > 1 and cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return ""
    if cleaned in UNIT_ALIASES:
        return UNIT_ALIASES[cleaned]
    lower_cleaned = cleaned.lower()
    if lower_cleaned in _LOWERCASE_UNIT_ALIASES:
        return _LOWERCASE_UNIT_ALIASES[lower_cleaned]
    return cleaned


def get_unit_category(unit_str: str) -> str:
    """
    Определяет физическую категорию единицы измерения ('length', 'time', 'energy', 'activity', 'angle').
    Выбрасывает ValueError, если единица неизвестна или имеет неподдерживаемую размерность.
    """
    canonical_unit = normalize_unit_name(unit_str)
    if not canonical_unit:
        raise ValueError("Единица измерения не может быть пустой")

    if canonical_unit in _ANGLE_CANONICAL_UNITS or canonical_unit.lower() in _ANGLE_CANONICAL_UNITS:
        return "angle"

    try:
        quantity = ureg.Quantity(1.0, canonical_unit)
    except (pint.UndefinedUnitError, pint.PintError, ValueError, TypeError) as err:
        raise ValueError(f"Неизвестная единица измерения '{unit_str}': {err}") from err

    dim = quantity.dimensionality
    if dim == ureg.mm.dimensionality:
        return "length"
    if dim == ureg.s.dimensionality:
        return "time"
    if dim == ureg.MeV.dimensionality:
        return "energy"
    if dim == ureg.Bq.dimensionality:
        return "activity"

    raise ValueError(f"Неподдерживаемая размерность единицы измерения '{unit_str}'")


def get_hepunits_factor(unit_str: str) -> float:
    """
    Возвращает коэффициент перевода из указанной единицы измерения
    во внутреннюю когерентную систему единиц HepUnits (мм = 1.0, МэВ = 1.0, нс = 1.0, рад = 1.0, Бк = 1e-9).
    """
    canonical_unit = normalize_unit_name(unit_str)
    if not canonical_unit:
        raise ValueError("Целевая единица измерения не может быть пустой")

    if (
        canonical_unit in units.__dict__
        and isinstance(units.__dict__[canonical_unit], (int, float))
        and not isinstance(units.__dict__[canonical_unit], bool)
    ):
        return float(units.__dict__[canonical_unit])

    category = get_unit_category(canonical_unit)
    quantity = ureg.Quantity(1.0, canonical_unit)
    if category == "length":
        return float(quantity.to("mm").magnitude) * float(units.mm)
    if category == "time":
        return float(quantity.to("ns").magnitude) * float(units.ns)
    if category == "energy":
        return float(quantity.to("MeV").magnitude) * float(units.MeV)
    if category == "activity":
        return float(quantity.to("Bq").magnitude) * float(units.Bq)
    if category == "angle":
        return float(quantity.to("rad").magnitude) * float(units.radian)

    raise ValueError(f"Неподдерживаемая размерность единицы '{unit_str}'")


def _parse_single_factor(factor_token: str) -> float:
    """Вычисляет одиночный числовой множитель или константу pi со знаком."""
    trimmed = factor_token.strip()
    if not trimmed:
        raise ValueError("Пустой множитель в числовом выражении")
    sign = 1.0
    if trimmed[0] == "+":
        trimmed = trimmed[1:].strip()
    elif trimmed[0] == "-":
        sign = -1.0
        trimmed = trimmed[1:].strip()

    if not trimmed:
        raise ValueError("Отсутствует число после знака")

    if trimmed.lower() == "pi":
        return sign * math.pi

    parsed_float = sign * float(trimmed)
    if not math.isfinite(parsed_float):
        raise ValueError(f"Числовое значение должно быть конечным, получено '{factor_token}'")
    return parsed_float


def _evaluate_numeric_expression(expr_text: str) -> float:
    """
    Безопасно вычисляет числовое выражение, поддерживающее научную нотацию,
    константу pi и базовые операции умножения/деления (например, '2*pi', 'pi/2', '1e-3').
    """
    normal_expr = _IMPLICIT_PI_MUL_PATTERN.sub(r"\1*\2", expr_text.strip())
    parts = _MUL_DIV_SPLIT_PATTERN.split(normal_expr)
    if not parts:
        raise ValueError(f"Некорректное числовое выражение: '{expr_text}'")

    result = _parse_single_factor(parts[0])
    idx = 1
    while idx < len(parts):
        operator_symbol = parts[idx]
        if idx + 1 >= len(parts):
            raise ValueError(f"Незавершенная операция '{operator_symbol}' в выражении '{expr_text}'")
        operand_val = _parse_single_factor(parts[idx + 1])
        if operator_symbol == "*":
            result *= operand_val
        elif operator_symbol == "/":
            if abs(operand_val) == 0.0:
                raise ValueError("Деление на ноль в числовом выражении")
            result /= operand_val
        idx += 2

    if not math.isfinite(result):
        raise ValueError(f"Результат выражения '{expr_text}' не является конечным числом")
    return float(result)


def _normalize_numeric_punctuation(raw_text: str) -> str:
    """Нормализует символ числа пи и заменяет десятичную запятую на точку."""
    prepared = raw_text.strip().replace("π", "pi").replace("Π", "pi")
    prepared = _DECIMAL_COMMA_MID_PATTERN.sub(r"\1.\2", prepared)
    return _DECIMAL_COMMA_LEAD_PATTERN.sub(r"\1.\2", prepared)


def _split_numeric_and_unit(raw_text: str) -> Tuple[float, str]:
    """
    Разделяет входную строку на числовое значение (включая выражения с pi)
    и строковый суффикс единицы измерения.
    """
    prepared = _normalize_numeric_punctuation(raw_text)
    if not prepared:
        raise ValueError("Пустая строка не является допустимой физической величиной")

    match = _NUMERIC_PREFIX_PATTERN.match(prepared)
    if match is None:
        raise ValueError(f"Ошибка парсинга физической величины '{raw_text}'")

    numeric_part = match.group(1).strip()
    unit_part = match.group(2).strip()
    numeric_value = _evaluate_numeric_expression(numeric_part)
    return numeric_value, unit_part


def _clean_floating_noise(val: float) -> float:
    """Устраняет микропогрешности IEEE 754 при конвертации единиц (например, 0.14000000000000004 -> 0.14)."""
    if val == 0.0:
        return 0.0
    return float(f"{val:.12g}")


def parse_unit_value(
    value: Union[int, float, str],
    default_unit: Optional[str] = None,
    to_hepunits: bool = False,
) -> float:
    """
    Универсальный парсер и конвертер числовых и размерных пользовательских значений.

    Правила работы:
    - Если передано число или строка с чистым числом (включая научную нотацию, десятичную запятую
      или выражение с pi без единицы) — величина трактуется в единицах default_unit
      (или переводится в HepUnits при to_hepunits=True).
    - Если передана строка с единицей измерения (например, '25 cm', '25 см', '0.5 m', '140 keV',
      '180 deg', 'pi rad', '2 min', '500 ms', '10 MBq') — проверяется физическая совместимость
      размерности с default_unit (при несовместимости выбрасывается ValueError) и выполняется
      конвертация в default_unit (или в HepUnits при to_hepunits=True).
    - Если default_unit не задан (None или ''), размерные строки при to_hepunits=False
      конвертируются в базовую единицу соответствующей категории (mm, MeV, s, deg, MBq).
    """
    if isinstance(value, bool):
        raise ValueError("Булево значение не является допустимой физической величиной")

    if isinstance(value, (int, float)):
        numeric_val = float(value)
        if not math.isfinite(numeric_val):
            raise ValueError(f"Числовое значение должно быть конечным, получено {value}")
        if not to_hepunits or not default_unit:
            return numeric_val
        cleaned_default_unit = normalize_unit_name(default_unit)
        if cleaned_default_unit:
            return _clean_floating_noise(numeric_val * get_hepunits_factor(cleaned_default_unit))
        return numeric_val

    cleaned_default_unit = normalize_unit_name(default_unit) if default_unit else ""

    if isinstance(value, str):
        numeric_val, raw_unit_part = _split_numeric_and_unit(value)
        if not raw_unit_part:
            if to_hepunits and cleaned_default_unit:
                return _clean_floating_noise(numeric_val * get_hepunits_factor(cleaned_default_unit))
            return numeric_val

        canonical_input_unit = normalize_unit_name(raw_unit_part)
        input_category = get_unit_category(canonical_input_unit)

        if cleaned_default_unit:
            target_category = get_unit_category(cleaned_default_unit)
            if input_category != target_category:
                raise ValueError(
                    f"Несовместимая физическая размерность: невозможно преобразовать '{value}' "
                    f"({canonical_input_unit}, {input_category}) к единицам '{default_unit}' "
                    f"({cleaned_default_unit}, {target_category})"
                )
            target_unit = cleaned_default_unit
        else:
            target_unit = DEFAULT_CATEGORY_UNITS[input_category]

        if to_hepunits:
            return _clean_floating_noise(numeric_val * get_hepunits_factor(canonical_input_unit))

        try:
            quantity = ureg.Quantity(numeric_val, canonical_input_unit)
            converted_magnitude = float(quantity.to(target_unit).magnitude)
        except (pint.DimensionalityError, pint.UndefinedUnitError, pint.PintError) as err:
            raise ValueError(f"Невозможно преобразовать '{value}' к единицам '{target_unit}': {err}") from err

        return _clean_floating_noise(converted_magnitude)

    raise ValueError(f"Ожидалось число или строка с единицами, получено {type(value)}")


_BATCH_TOKEN_REGEX = re.compile(
    rf"([+-]?\s*{_NUM_ATOM_REGEX}(?:\s*[*\/]\s*[+-]?\s*{_NUM_ATOM_REGEX})*(?:\s*(?:°|[a-zA-Zа-яА-ЯёЁμÅ]+))?)",
    re.IGNORECASE,
)
_IMPLICIT_PI_MUL_PATTERN = re.compile(r"(\d)\s*(pi)\b", re.IGNORECASE)
_MUL_DIV_SPLIT_PATTERN = re.compile(r"([*\/])")
_DECIMAL_COMMA_MID_PATTERN = re.compile(r"(\d),(\d)")
_DECIMAL_COMMA_LEAD_PATTERN = re.compile(r"^([+-]?\s*),(\d)")
_BATCH_DELIMITER_SEARCH_PATTERN = re.compile(r",\s+|[a-zA-Zа-яА-ЯёЁ°]\s*,")
_BATCH_DELIMITER_SPLIT_PATTERN = re.compile(r"[;]|,\s+|(?<=[a-zA-Zа-яА-ЯёЁ°])\s*,")
_TEMPLATE_MACRO_SEARCH_PATTERN = re.compile(r"\$\{[^}]+\}")
_STRICT_MACRO_FULLMATCH_PATTERN = re.compile(r"\$\{[a-zA-Z_][a-zA-Z0-9_]*\}")
_FLOAT_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_unit_tokens(
    raw_text: str,
    default_unit: Optional[str] = None,
    to_hepunits: bool = False,
) -> List[float]:
    """
    Разбирает строку пакетного ввода (через запятую, точку с запятой или пробелы)
    с поддержкой единиц измерения у каждого элемента (например, '10 cm, 25 cm, 0.5 m' или '40, 50, 60').
    Выбрасывает ValueError, если строка пуста или хотя бы один токен невалиден / несовместим по размерности.
    """
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("Пустая строка пакетного ввода")

    # Разделение по ';', по запятой с пробелом, по запятой после буквенной единицы измерения,
    # либо по запятым, если их в строке больше одной и нет буквенных единиц с десятичной запятой
    if ";" in stripped or _BATCH_DELIMITER_SEARCH_PATTERN.search(stripped):
        segments = [seg.strip() for seg in _BATCH_DELIMITER_SPLIT_PATTERN.split(stripped) if seg.strip()]
    elif stripped.count(",") >= 2:
        segments = [seg.strip() for seg in stripped.split(",") if seg.strip()]
    else:
        segments = [stripped]

    parsed_results: List[float] = []
    for segment in segments:
        normalized_seg = _normalize_numeric_punctuation(segment)

        # Извлекаем все токены вида <числовое_выражение> [единица_измерения] внутри сегмента
        pos = 0
        seg_len = len(normalized_seg)
        tokens_in_segment: List[str] = []
        while pos < seg_len:
            while pos < seg_len and normalized_seg[pos].isspace():
                pos += 1
            if pos >= seg_len:
                break
            token_match = _BATCH_TOKEN_REGEX.match(normalized_seg, pos)
            if token_match is None:
                raise ValueError(f"Нераспознанный токен в строке '{raw_text}' начиная с '{normalized_seg[pos:]}'")
            tokens_in_segment.append(token_match.group(1).strip())
            pos = token_match.end()

        if not tokens_in_segment:
            raise ValueError(f"Не удалось извлечь значения из сегмента '{segment}'")

        for token_str in tokens_in_segment:
            parsed_results.append(
                parse_unit_value(token_str, default_unit=default_unit, to_hepunits=to_hepunits)
            )

    if not parsed_results:
        raise ValueError(f"Не найдено числовых значений в строке '{raw_text}'")

    return parsed_results


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
        if isinstance(input_value, (int, float)) and not isinstance(input_value, bool):
            # Числовые значения обязаны передаваться уже во внутренних единицах HepUnits
            return float(input_value)
        if isinstance(input_value, str):
            # Пропуск шаблонов интерполяции вида ${var} для отложенной подстановки в оркестраторе
            if _TEMPLATE_MACRO_SEARCH_PATTERN.search(input_value):
                return input_value
            if input_value.strip() == '':
                raise ValueError("Пустая строка не является допустимой физической величиной")
            try:
                quantity = ureg(input_value)
                magnitude_val = float(quantity.to(target_unit).magnitude)
                return float(magnitude_val * to_hepunits_factor)
            except (pint.DimensionalityError, pint.UndefinedUnitError) as unit_err:
                raise ValueError(f"Невозможно преобразовать '{input_value}' к единицам {target_unit}: {unit_err}")
            except (ValueError, TypeError, ArithmeticError, pint.PintError) as parse_err:
                raise ValueError(f"Ошибка парсинга физической величины '{input_value}': {parse_err}")
        raise ValueError(f"Ожидалось число или строка с единицами, получено {type(input_value)}")
    return validator


LengthConfig = Annotated[Any, BeforeValidator(unit_validator_factory('mm', float(units.mm)))]
EnergyConfig = Annotated[Any, BeforeValidator(unit_validator_factory('MeV', float(units.MeV)))]
TimeConfig = Annotated[Any, BeforeValidator(unit_validator_factory('ns', float(units.ns)))]
ActivityConfig = Annotated[Any, BeforeValidator(unit_validator_factory('Bq', float(units.Bq)))]
AngleConfig = Annotated[Any, BeforeValidator(unit_validator_factory('rad', float(units.radian)))]


def integer_count_validator(input_value: Any) -> Any:
    """
    Pydantic-валидатор для целочисленных счетчиков и квантованных величин
    с поддержкой строгих шаблонных макросов вида ${variable_name}.
    """
    if isinstance(input_value, int) and not isinstance(input_value, bool):
        if input_value <= 0:
            raise ValueError(f"Целочисленный счетчик должен быть положительным (> 0), получено {input_value}")
        return int(input_value)
    if isinstance(input_value, float) and input_value.is_integer():
        int_val = int(input_value)
        if int_val <= 0:
            raise ValueError(f"Целочисленный счетчик должен быть положительным (> 0), получено {int_val}")
        return int_val
    if isinstance(input_value, str):
        trimmed = input_value.strip()
        # Пропуск строгих шаблонов интерполяции вида ${var} для отложенной подстановки в оркестраторе
        if _STRICT_MACRO_FULLMATCH_PATTERN.fullmatch(trimmed):
            return trimmed
        if _FLOAT_NUMBER_PATTERN.fullmatch(trimmed):
            num = float(trimmed)
            if math.isfinite(num) and num.is_integer():
                val = int(num)
                if val <= 0:
                    raise ValueError(f"Целочисленный счетчик должен быть положительным (> 0), получено {val}")
                return val
        raise ValueError(f"Ожидалось положительное целое число или макрос ${{var}}, получено: '{input_value}'")
    raise ValueError(f"Ожидалось целое число или строковый макрос ${{var}}, получено {type(input_value)}")


CountConfig = Annotated[Any, BeforeValidator(integer_count_validator)]


def convert_from_hepunits(value: float, target_unit: str) -> float:
    """
    Преобразует числовое значение из внутренней когерентной системы единиц HepUnits
    (мм = 1.0, МэВ = 1.0, нс = 1.0, рад = 1.0, Бк = 1e-9) в указанную целевую единицу измерения.
    Используется при подстановке шаблонных макросов с явным указанием единицы вида ${var unit}.
    """
    cleaned_unit = target_unit.strip()
    if not cleaned_unit:
        raise ValueError("Целевая единица измерения в макросе не может быть пустой")

    unit_factor = get_hepunits_factor(cleaned_unit)
    converted_value = round(float(value) / unit_factor, 10)
    if converted_value == 0.0:
        converted_value = 0.0
    return converted_value


__all__ = [
    "ureg",
    "UNIT_ALIASES",
    "DEFAULT_CATEGORY_UNITS",
    "normalize_unit_name",
    "get_unit_category",
    "get_hepunits_factor",
    "parse_unit_value",
    "parse_unit_tokens",
    "unit_validator_factory",
    "integer_count_validator",
    "convert_from_hepunits",
    "LengthConfig",
    "EnergyConfig",
    "TimeConfig",
    "ActivityConfig",
    "AngleConfig",
    "CountConfig",
]



