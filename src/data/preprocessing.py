"""Предобработка данных для ML-модели."""

import re

import pandas as pd


def parse_price(text: str) -> float | None:
    """Парсит цену из строки ('10 млн' -> 10000000, '5 500 000' -> 5500000)."""
    text = str(text).strip().lower().replace(" ", "")

    match = re.search(r"(\d+(?:\.\d+)?)\s*млн", text)
    if match:
        return float(match.group(1)) * 1_000_000

    match = re.search(r"(\d+(?:\.\d+)?)\s*тыс", text)
    if match:
        return float(match.group(1)) * 1_000

    match = re.search(r"([\d]+)", text)
    if match:
        return float(match.group(1))

    return None


def clean_price(df: pd.DataFrame, column: str = "price") -> pd.DataFrame:
    """Очищает колонку с ценой."""
    df = df.copy()
    df[column] = df[column].apply(parse_price)
    df = df.dropna(subset=[column])
    df[column] = df[column].astype(float)
    return df


def parse_area(text: str) -> float | None:
    """Парсит площадь ('50 м²' -> 50.0)."""
    match = re.search(r"([\d]+(?:\.\d+)?)\s*м[²2]", str(text))
    return float(match.group(1)) if match else None


def clean_area(df: pd.DataFrame, column: str = "area") -> pd.DataFrame:
    """Очищает колонку с площадью."""
    df = df.copy()
    df[column] = df[column].apply(parse_area)
    df = df.dropna(subset=[column])
    df[column] = df[column].astype(float)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает фичи для модели.

    Создает колонки: rooms, area, floor, total_floors, year_built, distance_to_metro_min.
    """
    features = pd.DataFrame()

    # Основные фичи
    for col in ["rooms", "area", "floor", "total_floors", "year_built", "distance_to_metro_min"]:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    # Дополнительные фичи
    if "area" in features.columns and "rooms" in features.columns:
        features["area_per_room"] = features["area"] / features["rooms"].clip(lower=1)

    if "floor" in features.columns and "total_floors" in features.columns:
        features["floor_ratio"] = (features["floor"] - 1) / features["total_floors"].clip(lower=1)

    if "year_built" in features.columns:
        features["building_age"] = 2025 - features["year_built"]

    return features
