"""Роутер аутентификации: /register, /login."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_db, get_current_user
from src.api.schemas import RegisterRequest, LoginRequest, TokenResponse
from src.core.logging import get_logger
from src.services.auth_service import AuthService
from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    session: AsyncSession = Depends(get_db),
):
    """Регистрация пользователя."""
    try:
        service = AuthService(session)
        return await service.register(body.email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    session: AsyncSession = Depends(get_db),
):
    """Вход пользователя."""
    try:
        service = AuthService(session)
        return await service.login(body.email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
