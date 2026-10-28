"""Предобработка данных для ML-модели.

Модуль — единственный источник правды для признаков: и обучение (scripts/train_model.py,
src/training/train.py), и инференс (PredictionService) строят их через extract_features /
build_feature_vector. Раньше формулы дублировались и расходились (floor_ratio), из-за чего
модель получала на входе не то, на чём училась.
"""

import math
import numbers
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.core.config import Settings

# Признаки, которые приходят от пользователя / из объявления
BASE_FEATURES = [
    "rooms",
    "area",
    "floor",
    "total_floors",
    "year_built",
    "distance_to_metro_min",
]
# Производные признаки (считаются из базовых)
DERIVED_FEATURES = ["area_per_room", "floor_ratio", "building_age"]
# Порядок колонок, на котором обучается модель
FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES

_UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)(млн|тыс)")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+")
_AREA_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:м[²2]|кв\.?\s*м)", re.IGNORECASE)


def _get_current_year(settings: "Settings | None" = None) -> int:
    """Возвращает текущий год из настроек или по умолчанию."""
    if settings is not None:
        return settings.current_year
    from datetime import UTC, datetime

    return datetime.now(UTC).year


def _as_float(value: Any) -> float | None:
    """Число -> float (NaN -> None). Для не-чисел возвращает None."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return None if math.isnan(value) else float(value)
    return None


def parse_price(text: Any) -> float | None:
    """Парсит цену ('10 млн' -> 10000000, '5,5 млн' -> 5500000, '5 500 000' -> 5500000)."""
    if text is None:
        return None
    if isinstance(text, numbers.Real) and not isinstance(text, bool):
        return _as_float(text)

    # \s+ убирает и обычные пробелы, и неразрывные (\xa0, \u202f), которыми сайты разделяют разряды
    normalized = re.sub(r"\s+", "", str(text).lower())

    match = _UNIT_RE.search(normalized)
    if match:
        # Запятая в русских объявлениях — десятичный разделитель: '5,5 млн' = 5.5 млн
        value = float(match.group(1).replace(",", "."))
        return value * (1_000_000 if match.group(2) == "млн" else 1_000)

    match = _NUMBER_RE.search(normalized)
    if not match:
        return None
    token = match.group(0)
    if _THOUSANDS_RE.fullmatch(token):
        token = re.sub(r"[.,]", "", token)  # '5.500.000' / '5,500,000' — разделители разрядов
    else:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def clean_price(df: pd.DataFrame, column: str = "price") -> pd.DataFrame:
    """Очищает колонку с ценой."""
    df = df.copy()
    df[column] = df[column].apply(parse_price)
    df = df.dropna(subset=[column])
    df[column] = df[column].astype(float)
    return df


def parse_area(text: Any) -> float | None:
    """Парсит площадь ('50 м²' -> 50.0). Числа проходят как есть."""
    if text is None:
        return None
    if isinstance(text, numbers.Real) and not isinstance(text, bool):
        return _as_float(text)
    match = _AREA_RE.search(str(text))
    return float(match.group(1).replace(",", ".")) if match else None


def clean_area(df: pd.DataFrame, column: str = "area") -> pd.DataFrame:
    """Очищает колонку с площадью (уже числовые значения не теряются)."""
    df = df.copy()
    df[column] = df[column].apply(parse_area)
    df = df.dropna(subset=[column])
    df[column] = df[column].astype(float)
    return df


def extract_features(df: pd.DataFrame, reference_year: int | None = None) -> pd.DataFrame:
    """Извлекает фичи для модели.

    Создает колонки: rooms, area, floor, total_floors, year_built,
    distance_to_metro_min, area_per_room, floor_ratio, building_age.

    Args:
        df: Исходные данные.
        reference_year: Год, от которого считается возраст здания (по умолчанию — текущий).
            Нужен, чтобы результат (и тесты) не зависели от даты запуска.
    """
    features = pd.DataFrame(index=df.index)

    # Основные фичи
    for col in BASE_FEATURES:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    # Дополнительные фичи
    if "area" in features.columns and "rooms" in features.columns:
        features["area_per_room"] = features["area"] / features["rooms"].clip(lower=1)

    if "floor" in features.columns and "total_floors" in features.columns:
        features["floor_ratio"] = (features["floor"] - 1) / features["total_floors"].clip(lower=1)

    if "year_built" in features.columns:
        year = reference_year if reference_year is not None else _get_current_year()
        features["building_age"] = year - features["year_built"]

    return features


def build_feature_vector(
    features: Mapping[str, Any], reference_year: int | None = None
) -> list[float]:
    """Собирает вектор признаков для ОДНОГО объекта в порядке FEATURE_COLUMNS.

    Использует ту же extract_features, что и обучение, поэтому train/serve skew невозможен.

    Raises:
        ValueError: если не хватает базовых признаков или они не числовые.
    """
    missing = [col for col in BASE_FEATURES if col not in features]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")

    frame = extract_features(
        pd.DataFrame([{col: features[col] for col in BASE_FEATURES}]),
        reference_year=reference_year,
    )
    vector = [float(v) for v in frame.iloc[0][FEATURE_COLUMNS].tolist()]
    if any(math.isnan(v) for v in vector):
        raise ValueError("All features must be numeric")
    return vector
