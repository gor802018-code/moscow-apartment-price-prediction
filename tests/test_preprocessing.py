"""Тесты для data preprocessing."""

import pandas as pd

from src.data.preprocessing import (
    FEATURE_COLUMNS,
    build_feature_vector,
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

    def test_comma_is_decimal_separator(self):
        """Регрессия: '5,5 млн' раньше превращалось в 55 млн."""
        assert parse_price("5,5 млн") == 5_500_000

    def test_non_breaking_spaces(self):
        """Сайты разделяют разряды неразрывными пробелами — раньше цена обрезалась до '5'."""
        assert parse_price("5\u202f500\u202f000 ₽") == 5_500_000
        assert parse_price("10\u00a0млн") == 10_000_000

    def test_numeric_and_missing_values(self):
        assert parse_price(7_000_000.0) == 7_000_000.0
        assert parse_price(float("nan")) is None
        assert parse_price(None) is None


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

    def test_numeric_area_passes_through(self):
        assert parse_area(62.5) == 62.5
        assert parse_area(float("nan")) is None


class TestCleanArea:
    """Тесты очистки площади."""

    def test_clean_area(self):
        df = pd.DataFrame({"area": ["50 м²", "30 м2", "N/A"]})
        result = clean_area(df)
        assert len(result) == 2
        assert result.iloc[0]["area"] == 50.0

    def test_clean_area_keeps_numeric_column(self):
        """Регрессия: числовая колонка раньше полностью удалялась (parse_area не понимал числа)."""
        df = pd.DataFrame({"area": [50.0, 30.0]})
        assert len(clean_area(df)) == 2


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
        result = extract_features(df, reference_year=2025)

        assert "rooms" in result.columns
        assert "area" in result.columns
        assert "area_per_room" in result.columns
        assert "floor_ratio" in result.columns
        assert "building_age" in result.columns

        assert result.iloc[0]["area_per_room"] == 25.0
        assert result.iloc[0]["building_age"] == 20  # 2025 - 2005
        assert abs(result.iloc[0]["floor_ratio"] - 0.4) < 0.01

    def test_keeps_index_and_column_order(self):
        df = pd.DataFrame(
            {
                "rooms": [2], "area": [50.0], "floor": [5], "total_floors": [10],
                "year_built": [2005], "distance_to_metro_min": [10], "price": [1e7],
            },
            index=[42],
        )
        result = extract_features(df, reference_year=2025)
        assert list(result.columns) == FEATURE_COLUMNS
        assert list(result.index) == [42]  # индекс сохраняется — к признакам можно вернуть price


class TestBuildFeatureVector:
    """Вектор для инференса должен совпадать с признаками при обучении (нет train/serve skew)."""

    FEATURES = {
        "rooms": 2, "area": 50.0, "floor": 5, "total_floors": 10,
        "year_built": 2005, "distance_to_metro_min": 10,
    }

    def test_matches_extract_features(self):
        vector = build_feature_vector(self.FEATURES, reference_year=2025)
        expected = extract_features(pd.DataFrame([self.FEATURES]), reference_year=2025)
        assert vector == [float(v) for v in expected.iloc[0][FEATURE_COLUMNS]]
        # floor_ratio = (5 - 1) / 10 — та же формула, что и при обучении
        assert abs(vector[FEATURE_COLUMNS.index("floor_ratio")] - 0.4) < 1e-9

    def test_missing_features_raise(self):
        import pytest

        with pytest.raises(ValueError, match="Missing features"):
            build_feature_vector({"rooms": 2})

    def test_non_numeric_raises(self):
        import pytest

        with pytest.raises(ValueError, match="numeric"):
            build_feature_vector({**self.FEATURES, "area": "много"})
