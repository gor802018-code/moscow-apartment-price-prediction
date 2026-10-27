"""Конфигурация логирования через structlog."""

import logging

import structlog


def setup_logging(
    level: int = logging.INFO,
    structured: bool = True,
) -> None:
    """Настройка structlog для продакшена и разработки."""

    if structured:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.JSONLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str | None = None):
    """Получить структурированный логгер."""
    return structlog.get_logger(name)


# Инициализация по умолчанию
setup_logging()
