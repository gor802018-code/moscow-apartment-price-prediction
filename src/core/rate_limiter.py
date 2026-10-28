"""Rate limiting configuration for the API."""

import sys
import warnings

from slowapi import Limiter
from slowapi.util import get_remote_address

# Disable rate limiting in tests
if "pytest" in sys.modules:
    # Create a no-op limiter for tests
    class _NoOpLimiter:
        def limit(self, limit_str):
            def decorator(fn):
                return fn
            return decorator

    limiter = _NoOpLimiter()  # type: ignore
else:
    # slowapi при создании читает .env через starlette Config системной кодировкой (cp1251 на
    # Windows) и падает на кириллических комментариях. Файл конфигурации отключаем: пустая строка
    # вместо None — единственный способ не открывать .env, настройки приложения всё равно читает
    # pydantic-settings с явным utf-8. Заодно гасим UserWarning «Config file not found».
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Config file .* not found",
            category=UserWarning,
        )
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=["100/hour"],
            config_filename="",
        )
