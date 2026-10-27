"""Парсер объявлений о квартирах (Playwright + BeautifulSoup)."""

import re
from typing import Any

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)


def _parse_price(text: str) -> float | None:
    """Парсит цену из строки ('10 млн' -> 10000000, '5 500 000' -> 5500000)."""
    text = text.strip().lower().replace(" ", "")

    # "X млн"
    match = re.search(r"(\d+(?:\.\d+)?)\s*млн", text)
    if match:
        return float(match.group(1)) * 1_000_000

    # "X тыс." (тысячи)
    match = re.search(r"(\d+(?:\.\d+)?)\s*тыс", text)
    if match:
        return float(match.group(1)) * 1_000

    # Просто число
    match = re.search(r"([\d\s]+)", text)
    if match:
        cleaned = match.group(1).replace(" ", "")
        return float(cleaned)

    return None


def _parse_area(text: str) -> float | None:
    """Парсит площадь ('50 м²' -> 50.0)."""
    match = re.search(r"([\d]+(?:\.\d+)?)\s*м[²2]", text)
    if match:
        return float(match.group(1))
    return None


def _parse_rooms(text: str) -> int | None:
    """Парсит количество комнат."""
    match = re.search(r"(\d+)\s*комн", text)
    if match:
        return int(match.group(1))
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def fetch_page(url: str) -> str:
    """Загружает HTML страницы через Playwright с ретраями."""
    log.info("Fetching page", url=url)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(2000)  # ждем подгрузки динамического контента
            html = await page.content()
            log.info("Page fetched successfully", url=url, length=len(html))
            return html
        finally:
            await browser.close()


def parse_listing(html: str) -> dict[str, Any]:
    """Парсит одну карточку квартиры из HTML.

    Args:
        html: HTML-разметка карточки объявления.

    Returns:
        Словарь с полями: price, area, rooms, district, url.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Парсим цену
    price_el = soup.find(string=re.compile(r"млн|тыс|\d[\d\s]*"))
    price = _parse_price(price_el.string if price_el else "")

    # Парсим площадь
    area_el = soup.find(string=re.compile(r"\d+\s*м[²2]"))
    area = _parse_area(area_el.string if area_el else "")

    # Парсим комнаты
    rooms_el = soup.find(string=re.compile(r"\d+\s*комн"))
    rooms = _parse_rooms(rooms_el.string if rooms_el else "")

    # Парсим район
    district_el = soup.find(string=re.compile(r"р-н|район|м\."))
    district = district_el.string if district_el else None

    return {
        "price": price,
        "area": area,
        "rooms": rooms,
        "district": district,
    }


async def scrape_listings(url: str) -> list[dict[str, Any]]:
    """Основная функция парсинга списка квартир.

    Args:
        url: URL страницы с объявлениями.

    Returns:
        Список словарей с данными квартир.
    """
    html = await fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")

    # Ищем все карточки (адаптировать под реальный сайт)
    listings = []
    for item in soup.find_all("div", class_=re.compile(r"item|card|listing")):
        listing_html = str(item)
        data = parse_listing(listing_html)
        if data.get("price") and data.get("area"):
            listings.append(data)

    log.info("Scraped listings", count=len(listings), source=url)
    return listings
