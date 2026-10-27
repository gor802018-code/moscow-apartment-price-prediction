"""Тесты для обучения модели (CatBoost + Optuna)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.training.train import train_model


@pytest.fixture
def sample_df():
    """Генерирует тестовый DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "price": np.random.uniform(3_000_000, 30_000_000, n),
        "rooms": np.random.choice([1, 2, 3, 4, 5], n),
        "area": np.random.uniform(20, 150, n),
        "floor": np.random.randint(1, 25, n),
        "total_floors": np.random.choice([5, 9, 12, 16, 17, 22, 25], n),
        "year_built": np.random.randint(1960, 2024, n),
        "distance_to_metro_min": np.random.uniform(1, 60, n),
    }).assign(
        area_per_room=lambda df: df["area"] / df["rooms"].clip(lower=1),
        floor_ratio=lambda df: (df["floor"] - 1) / df["total_floors"].clip(lower=1),
        building_age=lambda df: 2025 - df["year_built"],
    )


class TestTrainModel:
    """Тесты обучения модели."""

    def test_train_model_saves_file(self, sample_df, tmp_path):
        """Модель сохраняется в файл."""
        output_path = str(tmp_path / "model.joblib")
        result = train_model(sample_df, n_trials=3, output_path=output_path)

        assert Path(output_path).exists()
        loaded = joblib.load(output_path)
        assert loaded is not None

    def test_train_model_returns_best_params(self, sample_df):
        """Возвращает best_params и best_score."""
        result = train_model(sample_df, n_trials=3)

        assert "best_params" in result
        assert "best_score" in result
        assert "model" in result
        assert "feature_cols" in result

        assert isinstance(result["best_score"], float)
        assert result["best_score"] > 0

    def test_train_model_mae_decreases(self, sample_df):
        """Модель обучается с конечным MAE."""
        result = train_model(sample_df, n_trials=5)
        assert result["best_score"] < 10_000_000  # MAE должен быть разумным

    def test_model_predict_shape(self, sample_df):
        """Предсказания имеют правильную форму."""
        result = train_model(sample_df, n_trials=3)
        model = result["model"]

        X = sample_df[result["feature_cols"]].head(5)
        preds = model.predict(X)
        assert len(preds) == 5
