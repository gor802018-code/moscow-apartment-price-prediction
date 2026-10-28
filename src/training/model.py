"""Обёртка над обученной моделью: предсказывает цену в рублях.

Модель обучается на log1p(price) (цена сильно скошена), поэтому «сырой» predict() CatBoost
возвращает логарифм. Обёртка применяет обратное преобразование, чтобы API отдавал рубли.
Класс лежит в отдельном модуле, чтобы joblib мог распаковать модель в API.
"""

from typing import Any

import numpy as np


class LogTargetRegressor:
    """predict() возвращает expm1 от предсказания внутренней модели."""

    def __init__(self, model: Any, feature_cols: list[str]):
        self.model = model
        self.feature_cols = feature_cols

    def predict(self, X) -> np.ndarray:
        return np.expm1(np.asarray(self.model.predict(X), dtype=float))
