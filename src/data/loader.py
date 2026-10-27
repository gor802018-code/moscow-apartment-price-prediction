"""Загрузчик данных."""

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Загружает CSV-файл с данными о квартирах."""
    df = pd.read_csv(path)
    return df


def load_from_dict(data: list[dict]) -> pd.DataFrame:
    """Создает DataFrame из списка словарей (например, после парсинга)."""
    return pd.DataFrame(data)
