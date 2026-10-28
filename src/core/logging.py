"""Конфигурация логирования через structlog."""

import logging
import uuid

import structlog
from structlog.contextvars import clear_contextvars


def setup_logging(
    level: int = logging.INFO,
    structured: bool = False,
) -> None:
    """Настройка structlog для продакшена и разработки.

    Args:
        level: Уровень логирования.
        structured: True для JSON-лога (продакшен), False для консоли.
    """
    # merge_contextvars подмешивает всё, что добавлено через bind_contextvars (в т.ч. request_id)
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if structured:
        processors = [
            *shared,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            *shared,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        # False, чтобы повторный вызов setup_logging() (например, в lifespan) реально применялся
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None):
    """Получить структурированный логгер."""
    return structlog.get_logger(name)


def generate_request_id() -> str:
    """Генерирует уникальный ID для запроса."""
    return f"req-{uuid.uuid4().hex[:12]}"


def clear_request_context() -> None:
    """Очищает контекст запроса (для middleware cleanup)."""
    clear_contextvars()
