"""Rate limiting configuration for the API."""

import sys

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
    limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])
