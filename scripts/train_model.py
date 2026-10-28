"""Скрипт для обучения и сохранения ML-модели.

Запуск из корня проекта:
    python -m scripts.train_model                       # синтетические данные
    python -m scripts.train_model --data data/train.csv # реальные данные
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.data.preprocessing import clean_area, clean_price, extract_features
from src.training.train import train_model

log = get_logger(__name__)


def generate_sample_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Генерирует синтетические данные для обучения (только для демо!)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "rooms": rng.choice([1, 2, 3, 4, 5], n),
            "area": rng.uniform(20, 150, n),
            "floor": rng.integers(1, 25, n),
            "total_floors": rng.choice([5, 9, 12, 16, 17, 22, 25], n),
            "year_built": rng.integers(1960, 2024, n),
            "distance_to_metro_min": rng.uniform(1, 60, n),
        }
    )
    # Этаж не может быть выше этажности дома
    df["floor"] = np.minimum(df["floor"], df["total_floors"])

    base_price = 3_000_000
    price = (
        base_price
        + df["rooms"] * 500_000
        + df["area"] * 80_000
        - df["distance_to_metro_min"] * 50_000
        + rng.normal(0, 500_000, n)
    )
    df["price"] = np.maximum(price, 1_000_000).astype(float)
    return df


def load_dataset(data_path: Path, n_samples: int) -> pd.DataFrame:
    """Загружает CSV (или генерирует данные) и возвращает признаки + колонку price."""
    if data_path.exists():
        log.info("Loading training data", path=str(data_path))
        raw = pd.read_csv(data_path)
        raw = clean_price(raw, "price")
        raw = clean_area(raw, "area")
    else:
        log.warning("No training data found, generating SYNTHETIC data", path=str(data_path))
        raw = generate_sample_data(n_samples)

    # extract_features возвращает только признаки, поэтому цену добавляем обратно
    # (раньше она терялась, и train_model падал с KeyError: 'price')
    dataset = extract_features(raw)
    dataset["price"] = pd.to_numeric(raw["price"], errors="coerce")
    return dataset.dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить и сохранить ML-модель")
    parser.add_argument("--data", default="data/train.csv", help="CSV с обучающими данными")
    parser.add_argument(
        "--output", default=str(get_settings().model_path), help="Куда сохранить модель"
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Количество триалов Optuna")
    parser.add_argument("--n-samples", type=int, default=1000, help="Размер синтетической выборки")
    args = parser.parse_args()

    setup_logging()

    dataset = load_dataset(Path(args.data), args.n_samples)
    log.info("Training data", samples=len(dataset), columns=list(dataset.columns))

    result = train_model(
        dataset,
        target_col="price",
        n_trials=args.n_trials,
        output_path=args.output,
    )

    log.info("Model trained and saved", path=args.output)
    log.info("Best params", params=json.dumps(result["best_params"]))
    log.info(
        "Hold-out quality",
        mae_rub=round(result["test_mae"]),
        mape=round(result["test_mape"], 4),
    )


if __name__ == "__main__":
    main()
