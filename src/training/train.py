"""Обучение модели CatBoost с оптимизацией гиперпараметров через Optuna."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.core.logging import get_logger

log = get_logger(__name__)


def objective(trial, X, y) -> float:
    """Optuna objective function for CatBoost hyperparameter tuning."""
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "random_seed": 42,
        "verbose": 0,
    }

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=Pool(X_val, y_val),
        early_stopping_rounds=20,
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    return mae


def train_model(
    df: pd.DataFrame,
    target_col: str = "price",
    n_trials: int = 50,
    test_size: float = 0.2,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Обучает модель CatBoost с Optuna.

    Args:
        df: DataFrame с данными.
        target_col: Название целевой колонки.
        n_trials: Количество триалов Optuna.
        test_size: Доля тестовой выборки.
        output_path: Путь для сохранения модели (.joblib).

    Returns:
        Словарь с результатами: best_params, best_score, model.
    """
    feature_cols = [
        "rooms", "area", "floor", "total_floors",
        "year_built", "distance_to_metro_min",
        "area_per_room", "floor_ratio", "building_age",
    ]

    # Удаляем строки с NaN
    df = df.dropna(subset=feature_cols + [target_col])

    # Логарифмируем цену (скошенная распределение)
    df = df.copy()
    df["price_log"] = np.log1p(df[target_col])

    X = df[feature_cols]
    y = df["price_log"]

    log.info("Training model", samples=len(X), trials=n_trials)

    # Optuna на обучающей + валидационной выборке
    study = optuna.create_study(direction="minimize", study_name="price_prediction")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    best_score = study.best_value
    best_params = study.best_params

    log.info("Best score", mae=best_score, params=best_params)

    # Разделяем на train/val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Финальная модель на train+val
    final_model = CatBoostRegressor(
        **best_params,
        random_seed=42,
        verbose=0,
    )
    final_model.fit(X_trainval, y_trainval)

    # Оценка на hold-out тесте
    test_preds = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mae_actual = np.expm1(test_preds).mean()  # approximate

    log.info("Test MAE", mae=test_mae)

    # Сохраняем модель
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, output_path)
        log.info("Model saved", path=output_path)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "test_mae": test_mae,
        "model": final_model,
        "feature_cols": feature_cols,
    }
