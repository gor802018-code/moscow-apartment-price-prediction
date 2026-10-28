"""Роут health check."""

import asyncio

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.core.logging import get_logger
from src.repositories.model_repository import ModelRepository

log = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """Проверка работоспособности API, БД и ML-модели.

    Возвращает 200, если всё в порядке, и 503, если что-то недоступно
    (раньше код считал нужный статус, но всегда отдавал 200).
    """
    checks: dict = {}

    # Проверка БД
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        # Детали ошибки (хост, имя БД) наружу не отдаём — только в лог
        log.exception("Health check: database failed")
        checks["database"] = {"status": "error"}

    # Проверка ML-модели
    try:
        # Чтение файла модели синхронное — выносим из event loop
        await asyncio.to_thread(ModelRepository().load)
        checks["model"] = {"status": "ok"}
    except FileNotFoundError:
        checks["model"] = {"status": "missing"}
    except Exception:
        log.exception("Health check: model failed")
        checks["model"] = {"status": "error"}

    all_ok = all(
        v == "ok" or (isinstance(v, dict) and v.get("status") == "ok") for v in checks.values()
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "healthy" if all_ok else "degraded", "checks": checks},
    )
