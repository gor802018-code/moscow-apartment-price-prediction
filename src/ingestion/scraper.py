"""Парсер объявлений о квартирах (Playwright + BeautifulSoup)."""

import re
from typing import Any

from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.logging import get_logger

# Единая логика разбора цены/площади (раньше была скопирована сюда и расходилась с preprocessing).
# Алиасы с подчёркиванием оставлены для обратной совместимости импортов.
from src.data.preprocessing import parse_area as _parse_area
from src.data.preprocessing import parse_price as _parse_price

log = get_logger(__name__)

# Цена — это число С денежным маркером/единицей. Раньше искалось любое число в карточке,
# и вместо цены могла быть взята площадь, этаж или номер телефона.
_PRICE_RE = re.compile(r"\d[\d\s.,]*\s*(?:млн|тыс|₽|руб|р\.)", re.IGNORECASE)


class ScraperError(Exception):
    """Базовое исключение для ошибок парсинга."""


class ScraperParseError(ScraperError):
    """Ошибка разбора данных."""


class ScraperEmptyResultError(ScraperError):
    """Парсер не нашёл никаких объявлений."""


def _parse_rooms(text: str) -> int | None:
    """Парсит количество комнат."""
    match = re.search(r"(\d+)\s*комн", text)
    if match:
        return int(match.group(1))
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=(
        retry_if_exception_type(PlaywrightTimeoutError)
        | retry_if_exception_type(ConnectionError)
        | retry_if_exception_type(TimeoutError)
    ),
    reraise=True,
)
async def fetch_page(url: str) -> str:
    """Загружает HTML страницы через Playwright с ретраями.

    Ретраит только при таймаутах и сетевых ошибках.
    """
    log.info("Fetching page", url=url)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(2000)  # ждем подгрузки динамического контента
                html = await page.content()
                log.info("Page fetched successfully", url=url, length=len(html))
                if not html or len(html) < 100:
                    raise ScraperParseError(f"Empty or too short page content: {len(html)} bytes")
                return html
            finally:
                await browser.close()
    except PlaywrightTimeoutError:
        log.warning("Page timeout", url=url)
        raise
    except ScraperParseError:
        raise
    except Exception as e:
        log.error("Page fetch error", url=url, error=str(e))
        raise ScraperError(f"Failed to fetch page: {e}") from e


def parse_listing(html: str, url: str = "") -> dict[str, Any]:
    """Парсит одну карточку квартиры из HTML.

    Args:
        html: HTML-разметка карточки объявления.
        url: URL страницы-источника (сохраняется в поле source_url).

    Returns:
        Словарь с полями: price, area, rooms, district, url.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Парсим цену
    price_el = soup.find(string=_PRICE_RE)
    price = _parse_price(str(price_el)) if price_el else None

    # Парсим площадь
    area_el = soup.find(string=re.compile(r"\d+\s*м[²2]"))
    area = _parse_area(str(area_el)) if area_el else None

    # Парсим комнаты
    rooms_el = soup.find(string=re.compile(r"\d+\s*комн"))
    rooms = _parse_rooms(str(rooms_el)) if rooms_el else None

    # Парсим район
    district_el = soup.find(string=re.compile(r"р-н|район|м\."))
    district = district_el.string if district_el else None

    return {
        "price": price,
        "area": area,
        "rooms": rooms,
        "district": district,
        "source_url": url,
    }


async def scrape_listings(url: str) -> list[dict[str, Any]]:
    """Основная функция парсинга списка квартир.

    Args:
        url: URL страницы с объявлениями.

    Returns:
        Список словарей с данными квартир (без дубликатов).
    """
    html = await fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")

    # Ищем все карточки (адаптировать под реальный сайт)
    seen_keys: set[tuple] = set()
    listings = []
    for item in soup.find_all("div", class_=re.compile(r"item|card|listing")):
        listing_html = str(item)
        data = parse_listing(listing_html, url=url)
        if not data.get("price") or not data.get("area"):
            continue

        # Дедупликация по ключу (rooms, area, district, price)
        dedup_key = (
            data["rooms"],
            round(data["area"], 1),
            data["district"],
            round(data["price"], -3),
        )
        if dedup_key in seen_keys:
            log.debug("Skipping duplicate listing", key=dedup_key)
            continue

        seen_keys.add(dedup_key)
        listings.append(data)

    log.info("Scraped listings", count=len(listings), source=url, unique=len(seen_keys))
    return listings
