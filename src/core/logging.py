"""Конфигурация логирования через structlog."""

import logging
import uuid

import structlog
from structlog.contextvars import clear_contextvars


def _request_id_processor(logger, method_name, event_dict):
    """Добавляет request_id в каждый лог."""
    from structlog.contextvars import get_contextvars
    ctx = get_contextvars()
    request_id = ctx.get("request_id")
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def setup_logging(
    level: int = logging.INFO,
    structured: bool = False,
) -> None:
    """Настройка structlog для продакшена и разработки.

    Args:
        level: Уровень логирования.
        structured: True для JSON-лога (продакшен), False для консоли.
    """
    base_processors = [
        structlog.contextvars.merge_contextvars,
        _request_id_processor,
        structlog.processors.add_log_level,
    ]

    if structured:
        base_processors.append(structlog.processors.JSONRenderer())
    else:
        base_processors.extend([
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ])

    structlog.configure(
        processors=base_processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
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
