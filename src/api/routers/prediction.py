"""Роутер предсказаний: /predict, /history."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_current_user, get_db
from src.api.schemas import HistoryResponse, PredictionRequest, PredictionResponse
from src.core.logging import get_logger
from src.core.rate_limiter import limiter
from src.services.prediction_service import PredictionService
from src.repositories.model_repository import ModelRepository

log = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    body: PredictionRequest,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Предсказание цены квартиры. Лимит: 10 запросов в минуту."""
    try:
        service = PredictionService(session)
        result = await service.predict(
            user_id=current_user["user_id"],
            input_features=body.model_dump(),
        )
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None


@router.get("/history", response_model=HistoryResponse)
@limiter.limit("30/minute")
async def history(
    request: Request,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """История предсказаний пользователя. Лимит: 30 запросов в минуту."""
    try:
        service = PredictionService(session)
        predictions = await service.get_history(user_id=current_user["user_id"])
        return HistoryResponse(
            predictions=[PredictionResponse(**p) for p in predictions]
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None
