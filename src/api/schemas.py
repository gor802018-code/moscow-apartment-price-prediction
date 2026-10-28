"""Pydantic-схемы для API."""

from pydantic import BaseModel, EmailStr, Field, model_validator

# --- Auth ---


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
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

    @model_validator(mode="after")
    def _floor_not_above_total(self) -> "PredictionRequest":
        if self.floor > self.total_floors:
            raise ValueError("floor must not exceed total_floors")
        return self


class PredictionResponse(BaseModel):
    id: int
    input_features: dict
    predicted_price: float
    created_at: str


class HistoryResponse(BaseModel):
    predictions: list[PredictionResponse]
