"""Обучение модели CatBoost с оптимизацией гиперпараметров через Optuna."""

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
from src.data.preprocessing import FEATURE_COLUMNS, extract_features
from src.training.model import LogTargetRegressor

log = get_logger(__name__)

_SEED = 42


def objective(trial, X, y) -> float:
    """Optuna objective: MAE (в лог-пространстве) на валидационной части переданных данных."""
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "random_seed": _SEED,
        "verbose": 0,
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=_SEED)

    model = CatBoostRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=Pool(X_val, y_val),
        early_stopping_rounds=20,
    )
    # Запоминаем число деревьев, на котором сработал early stopping: финальная модель
    # должна обучаться ровно столько же, а не с дефолтными 1000 итераций.
    trial.set_user_attr("best_iteration", int(model.get_best_iteration()))

    return float(mean_absolute_error(y_val, model.predict(X_val)))


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Достраивает недостающие производные признаки (если переданы только базовые)."""
    df = df.copy()
    if all(col in df.columns for col in FEATURE_COLUMNS):
        return df
    derived = extract_features(df)
    for col in FEATURE_COLUMNS:
        if col not in df.columns and col in derived.columns:
            df[col] = derived[col]
    return df


def train_model(
    df: pd.DataFrame,
    target_col: str = "price",
    n_trials: int = 50,
    test_size: float = 0.2,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Обучает модель CatBoost с Optuna.

    Схема без утечки данных:
      1. hold-out тест отделяется ПЕРВЫМ и не участвует ни в подборе гиперпараметров,
         ни в early stopping;
      2. Optuna работает только на train+val (внутри objective — свой train/val сплит);
      3. финальная модель обучается на всём train+val с числом итераций лучшего триала;
      4. качество считается на hold-out в рублях.

    Args:
        df: DataFrame с данными (признаки + целевая колонка с ценой в рублях).
        target_col: Название целевой колонки.
        n_trials: Количество триалов Optuna.
        test_size: Доля hold-out тестовой выборки.
        output_path: Путь для сохранения модели (.joblib).

    Returns:
        Словарь: best_params, best_score (MAE в лог-пространстве ≈ относительная ошибка),
        best_iteration, test_mae (в рублях), test_mape, model (предсказывает рубли),
        feature_cols.
    """
    data = _ensure_features(df).dropna(subset=FEATURE_COLUMNS + [target_col])
    data = data[data[target_col] > 0]  # log1p определён только для положительной цены

    X = data[FEATURE_COLUMNS]
    # Цена сильно скошена — обучаемся на логарифме, обратное преобразование делает обёртка
    y = np.log1p(data[target_col])

    log.info("Training model", samples=len(X), trials=n_trials)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=_SEED
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name="price_prediction",
        sampler=optuna.samplers.TPESampler(seed=_SEED),
    )
    study.optimize(lambda trial: objective(trial, X_trainval, y_trainval), n_trials=n_trials)

    best_score = float(study.best_value)
    best_params = study.best_params
    best_iteration = int(study.best_trial.user_attrs.get("best_iteration", 500))
    log.info("Best trial", log_mae=best_score, params=best_params, iterations=best_iteration)

    final_model = CatBoostRegressor(
        **best_params,
        iterations=best_iteration + 1,
        random_seed=_SEED,
        verbose=0,
    )
    final_model.fit(X_trainval, y_trainval)
    model = LogTargetRegressor(final_model, list(FEATURE_COLUMNS))

    # Оценка на hold-out в рублях
    y_true = np.expm1(y_test.to_numpy())
    y_pred = model.predict(X_test)
    test_mae = float(mean_absolute_error(y_true, y_pred))
    test_mape = float(np.mean(np.abs(y_true - y_pred) / y_true))
    log.info("Hold-out metrics", mae_rub=round(test_mae), mape=round(test_mape, 4))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        log.info("Model saved", path=output_path)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "test_mae": test_mae,
        "test_mape": test_mape,
        "model": model,
        "feature_cols": list(FEATURE_COLUMNS),
    }
