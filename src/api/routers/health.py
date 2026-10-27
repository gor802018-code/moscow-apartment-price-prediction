"""Роут health check."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Проверка работоспособности API."""
    return {"status": "ok"}
