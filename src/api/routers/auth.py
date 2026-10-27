"""Роутер аутентификации: /register, /login."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.schemas import LoginRequest, RegisterRequest, TokenResponse
from src.core.logging import get_logger
from src.core.rate_limiter import limiter
from src.services.auth_service import AuthService

log = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/hour")
async def register(
    request: Request,
    body: RegisterRequest,
    session: AsyncSession = Depends(get_db),
):
    """Регистрация пользователя. Лимит: 5 запросов в час."""
    service = AuthService(session)
    try:
        return await service.register(body.email, body.password)
    except ValueError as e:
        detail = str(e)
        if "already exists" in detail.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=detail,
            ) from None
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(
    request: Request,
    body: LoginRequest,
    session: AsyncSession = Depends(get_db),
):
    """Вход пользователя. Лимит: 10 запросов в минуту."""
    service = AuthService(session)
    try:
        return await service.login(body.email, body.password)
    except ValueError:
        # Одинаковая ошибка для защиты от user enumeration
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        ) from None
