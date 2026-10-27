"""Pydantic-схемы для API."""

from pydantic import BaseModel, Field

# --- Auth ---


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int


# --- Prediction ---


class PredictionRequest(BaseModel):
    rooms: int = Field(ge=1, le=10)
    area: float = Field(gt=0, le=1000)
    floor: int = Field(ge=1)
    total_floors: int = Field(ge=1)
    year_built: int = Field(ge=1800, le=2030)
    distance_to_metro_min: float = Field(ge=0, le=120)


class PredictionResponse(BaseModel):
    id: int
    input_features: dict
    predicted_price: float
    created_at: str


class HistoryResponse(BaseModel):
    predictions: list[PredictionResponse]
