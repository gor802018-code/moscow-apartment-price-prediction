"""Роут health check."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.core.config import get_settings
from src.core.logging import get_logger
from src.repositories.model_repository import ModelRepository

log = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """Проверка работоспособности API, БД и ML-модели."""
    checks = {}

    # Проверка БД
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = {"status": "error", "detail": str(e)}

    # Проверка ML-модели
    try:
        model_repo = ModelRepository()
        model_repo.load()
        checks["model"] = {"status": "ok", "path": str(settings.model_path)}
    except FileNotFoundError:
        checks["model"] = {"status": "missing", "path": str(settings.model_path)}
    except Exception as e:
        checks["model"] = {"status": "error", "detail": str(e)}

    # Общий статус
    all_ok = all(
        v == "ok" or (isinstance(v, dict) and v.get("status") == "ok")
        for v in checks.values()
    )

    status_code = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
    }
