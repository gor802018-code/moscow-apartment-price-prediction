"""Тесты для data preprocessing."""

import pandas as pd

from src.data.preprocessing import (
    clean_area,
    clean_price,
    extract_features,
    parse_area,
    parse_price,
)


class TestParsePrice:
    """Тесты парсинга цены."""

    def test_million(self):
        assert parse_price("10 млн") == 10_000_000

    def test_million_float(self):
        assert parse_price("12.5 млн") == 12_500_000

    def test_full_number(self):
        assert parse_price("5500000") == 5_500_000

    def test_thousand(self):
        assert parse_price("500 тыс") == 500_000

    def test_invalid(self):
        assert parse_price("Договорная") is None


class TestCleanPrice:
    """Тесты очистки цены."""

    def test_clean_price(self):
        df = pd.DataFrame({"price": ["10 млн", "5 500 000", "invalid"]})
        result = clean_price(df)
        assert len(result) == 2
        assert result.iloc[0]["price"] == 10_000_000
        assert result.iloc[1]["price"] == 5_500_000


class TestParseArea:
    """Тесты парсинга площади."""

    def test_area(self):
        assert parse_area("50 м²") == 50.0

    def test_area_float(self):
        assert parse_area("45.5 м2") == 45.5

    def test_no_area(self):
        assert parse_area("Нет данных") is None


class TestCleanArea:
    """Тесты очистки площади."""

    def test_clean_area(self):
        df = pd.DataFrame({"area": ["50 м²", "30 м2", "N/A"]})
        result = clean_area(df)
        assert len(result) == 2
        assert result.iloc[0]["area"] == 50.0


class TestExtractFeatures:
    """Тесты извлечения фич."""

    def test_basic_features(self):
        df = pd.DataFrame({
            "rooms": [2, 3],
            "area": [50.0, 75.0],
            "floor": [5, 10],
            "total_floors": [10, 16],
            "year_built": [2005, 2015],
            "distance_to_metro_min": [10, 5],
        })
        result = extract_features(df)

        assert "rooms" in result.columns
        assert "area" in result.columns
        assert "area_per_room" in result.columns
        assert "floor_ratio" in result.columns
        assert "building_age" in result.columns

        assert result.iloc[0]["area_per_room"] == 25.0
        assert result.iloc[0]["building_age"] == 20
        assert abs(result.iloc[0]["floor_ratio"] - 0.4) < 0.01
