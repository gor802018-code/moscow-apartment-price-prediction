"""Тесты для парсера (scraper)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.scraper import (
    _parse_area,
    _parse_price,
    _parse_rooms,
    parse_listing,
    scrape_listings,
)


class TestParsePrice:
    """Тесты парсинга цены."""

    def test_million(self):
        assert _parse_price("10 млн") == 10_000_000

    def test_million_float(self):
        assert _parse_price("12.5 млн") == 12_500_000

    def test_full_number(self):
        assert _parse_price("5 500 000") == 5_500_000

    def test_thousand(self):
        assert _parse_price("500 тыс") == 500_000

    def test_invalid(self):
        assert _parse_price("Договорная") is None


class TestParseArea:
    """Тесты парсинга площади."""

    def test_area_m2(self):
        assert _parse_area("50 м²") == 50.0

    def test_area_m2_float(self):
        assert _parse_area("45.5 м2") == 45.5

    def test_no_area(self):
        assert _parse_area("Без площади") is None


class TestParseRooms:
    """Тесты парсинга комнат."""

    def test_rooms(self):
        assert _parse_rooms("2 комнаты") == 2

    def test_rooms_singular(self):
        assert _parse_rooms("1 комната") == 1

    def test_no_rooms(self):
        assert _parse_rooms("Студия") is None


class TestParseListing:
    """Тесты парсинга карточки."""

    def test_full_listing(self):
        html = """
        <div class="listing-card">
            <span class="price">10 млн</span>
            <span class="area">50 м²</span>
            <span class="rooms">2 комнаты</span>
            <span class="district">р-н Арбат</span>
        </div>
        """
        result = parse_listing(html)
        assert result["price"] == 10_000_000
        assert result["area"] == 50.0
        assert result["rooms"] == 2
        assert result["district"] == "р-н Арбат"

    def test_price_not_confused_with_rooms_or_area(self):
        """Регрессия: цена бралась из первого попавшегося числа (например, '2 комнаты')."""
        html = """
        <div class="card">
            <span>2 комнаты</span><span>50 м²</span>
            <span>5,5 млн ₽</span><span>тел. 8 495 123-45-67</span>
        </div>
        """
        result = parse_listing(html)
        assert result["price"] == 5_500_000
        assert result["area"] == 50.0
        assert result["rooms"] == 2

    def test_price_with_non_breaking_spaces(self):
        html = '<div class="card"><span>12\u202f500\u202f000 ₽</span><span>40 м²</span></div>'
        assert parse_listing(html)["price"] == 12_500_000

    def test_empty_listing(self):
        html = "<div class='listing-card'><p>No data</p></div>"
        result = parse_listing(html)
        assert result["price"] is None
        assert result["area"] is None


class TestScrapeListings:
    """Тесты парсинга списка."""

    @pytest.mark.asyncio
    async def test_scrape_listings(self):
        """Парсинг списка квартир."""
        html = """
        <div class="item-card" data-id="1">
            <span class="price">10 млн</span>
            <span class="area">50 м²</span>
            <span class="rooms">2 комнаты</span>
        </div>
        <div class="item-card" data-id="2">
            <span class="price">15 млн</span>
            <span class="area">75 м²</span>
            <span class="rooms">3 комнаты</span>
        </div>
        """

        with patch("src.ingestion.scraper.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = html
            results = await scrape_listings("http://example.com")
            assert len(results) == 2
            assert results[0]["price"] == 10_000_000
            assert results[1]["price"] == 15_000_000

    @pytest.mark.asyncio
    async def test_scrape_empty(self):
        """Парсинг пустого списка."""
        html = "<div class='listings'></div>"

        with patch("src.ingestion.scraper.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = html
            results = await scrape_listings("http://example.com")
            assert results == []
