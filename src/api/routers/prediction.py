"""Роутер предсказаний: /predict, /history."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_db, get_current_user
from src.api.schemas import PredictionRequest, PredictionResponse, HistoryResponse
from src.core.logging import get_logger
from src.services.prediction_service import PredictionService
from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def predict(
    body: PredictionRequest,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Предсказание цены квартиры."""
    try:
        service = PredictionService(session)
        result = await service.predict(
            user_id=int(current_user.get("user_id", 0)),
            input_features=body.model_dump(),
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/history", response_model=HistoryResponse)
async def history(
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """История предсказаний пользователя."""
    try:
        service = PredictionService(session)
        predictions = await service.get_history(user_id=int(current_user.get("user_id", 0)))
        return HistoryResponse(predictions=[PredictionResponse(**p) for p in predictions])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
