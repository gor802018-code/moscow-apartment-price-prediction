# Kwork ML — Предсказание цен на квартиры в Москве

Production-ready SaaS для портфолио: ML-модель предсказания цен + автосбор данных + API + Frontend.

## Стек

| Категория | Технологии |
|-----------|------------|
| **Язык** | Python 3.11+, Poetry |
| **API** | FastAPI, Pydantic v2, JWT (python-jose), passlib (PBKDF2-SHA256), slowapi (rate limiting) |
| **БД** | PostgreSQL 16, SQLAlchemy 2.0 (async), Alembic, asyncpg |
| **ML** | CatBoost, Optuna (hyperparameter tuning), scikit-learn, pandas |
| **Парсинг** | Playwright, BeautifulSoup4, tenacity (retries) |
| **Frontend** | Streamlit |
| **Тесты** | pytest, pytest-asyncio, httpx |
| **Инфраструктура** | Docker, docker-compose, GitHub Actions CI/CD |
| **Логирование** | structlog |

## Архитектура

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI (v1)   │────▶│ PostgreSQL  │
│  (Frontend) │     │   :8000          │     │   :5432     │
└─────────────┘     └──────────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Ingestion   │
                    │ Playwright   │
                    └──────────────┘
```

**Layered Architecture:**
```
┌─────────────────────────────────────────────┐
│              API Layer (FastAPI)             │
│  routers/auth.py | prediction.py | health.py │
├─────────────────────────────────────────────┤
│           Service Layer (Business Logic)     │
│  AuthService | PredictionService             │
├─────────────────────────────────────────────┤
│          Repository Layer (Data Access)      │
│  UserRepository | PredictionRepository       │
│  ModelRepository                           │
├─────────────────────────────────────────────┤
│              Model Layer (SQLAlchemy)        │
│  User | Prediction                          │
├─────────────────────────────────────────────┤
│               Core Layer                    │
│  config.py | security.py | logging.py       │
├─────────────────────────────────────────────┤
│            Ingestion Layer                  │
│  scraper.py (Playwright + BS4)              │
└─────────────────────────────────────────────┘
```

## Быстрый старт

### Локально

```bash
# Установка зависимостей
poetry install

# Настройки (при необходимости поправьте значения)
cp .env.example .env

# Применить миграции БД (PostgreSQL должен быть запущен)
poetry run alembic upgrade head

# Обучить модель -> artifacts/price_model.joblib
# Без --data берутся СИНТЕТИЧЕСКИЕ данные (подходит только для демо)
poetry run python -m scripts.train_model
poetry run python -m scripts.train_model --data data/train.csv --n-trials 50

# Запуск API (dev mode)
poetry run uvicorn src.api.main:app --reload --port 8000

# Запуск Streamlit
poetry run streamlit run src/frontend/app.py --server.port 8501
```

### Docker Compose

```bash
# 1. Обучить модель на хосте (папка artifacts/ монтируется в контейнер API)
poetry run python -m scripts.train_model

# 2. Запуск всех сервисов (API + DB + Frontend). Миграции применяются при старте API.
docker compose up --build

# Сервисы доступны по адресам:
# API:      http://localhost:8000
# Frontend: http://localhost:8501
# DB:       localhost:5432
```

Без файла `artifacts/price_model.joblib` API стартует, но `/health` вернёт 503, а `/predict` — 503
(«Model is not available»). В продакшене задайте `ENVIRONMENT=production` и свой `SECRET_KEY`
(с дефолтным ключом приложение не запустится).

Если порт `5432` на хосте занят (например, локальным PostgreSQL), создайте локальный
`docker-compose.override.yml` — он в git не коммитится:

```yaml
services:
  db:
    ports: !override
      - "5433:5432"
```

API внутри сети compose по-прежнему подключается к `db:5432`, снаружи БД будет доступна на
`localhost:5433`.

## Данные и качество модели

Модель обучается на CSV с колонками `rooms, area, floor, total_floors, year_built,
distance_to_metro_min, price` (цена в рублях). Файл кладётся в `data/train.csv`; и данные, и
обученная модель в git не коммитятся.

**Источник данных текущей модели** — открытый датасет
[Moscow-apartment-price-prediction](https://github.com/ded0inside/Moscow-apartment-price-prediction)
(MIT): 6983 объявления о продаже квартир в Москве с Cian, ноябрь 2023. Поля источника
переименовываются в признаки модели: `number_of_rooms → rooms`, `total_area → area`,
`floor → floor`, `number_of_floors → total_floors`, `construction_year → year_built`,
`min_to_metro → distance_to_metro_min`, `price → price`.

Подготовка `data/train.csv` (6983 → 5625 строк: отброшено 1307 строк с пропусками — в основном
1222 без года постройки, и 51 выброс по диапазонам цены, площади, этажа, года и удалённости
от метро):

```python
import pandas as pd

raw = pd.read_csv("moscow_flats_dataset_eng.csv")
df = pd.DataFrame(
    {
        "rooms": raw["number_of_rooms"],
        "area": raw["total_area"],
        "floor": raw["floor"],
        "total_floors": raw["number_of_floors"],
        "year_built": raw["construction_year"],
        "distance_to_metro_min": raw["min_to_metro"],
        "price": raw["price"],
    }
).dropna()
df = df[
    df["price"].between(1_000_000, 500_000_000)
    & df["area"].between(9, 300)
    & df["rooms"].between(1, 6)
    & (df["floor"] >= 1)
    & (df["floor"] <= df["total_floors"])
    & df["total_floors"].between(1, 100)
    & df["year_built"].between(1800, 2026)
    & df["distance_to_metro_min"].between(0, 120)
]
df.to_csv("data/train.csv", index=False)
```

Обучение: `poetry run python -m scripts.train_model --data data/train.csv --n-trials 50`
(CatBoost + подбор гиперпараметров Optuna, таргет `log1p(price)`).

| Выборка | MAE | MAPE |
|---|---|---|
| hold-out 20 % реальных данных (1125 строк) | 12.7 млн ₽ | 24.2 % |
| синтетика (`--data` не передан, 1000 строк) | 0.4 млн ₽ | 4.8 % |

Синтетический режим — только для демо запуска: данные генерируются формулой, поэтому его
метрики ничего не говорят о реальном качестве. Ошибка около 24 % на реальных объявлениях —
следствие узкого набора признаков (нет района, класса жилья, ремонта и состояния дома).

Лицензия датасета — MIT, но исходные объявления принадлежат Cian, поэтому данные в этом
репозитории не публикуются: их скачивают из источника, соблюдая условия сайта.

## API Документация

Базовый URL: `http://localhost:8000/api/v1`

### Health Check

| Method | Path | Auth | Описание |
|--------|------|------|----------|
| GET | `/health` | ✗ | Проверка API, БД и ML-модели |

**Response (200, всё доступно):**
```json
{"status": "healthy", "checks": {"database": "ok", "model": {"status": "ok"}}}
```

Если БД или модель недоступны, статус ответа **503**, а `status` = `"degraded"`.

### Аутентификация

| Method | Path | Auth | Описание |
|--------|------|------|----------|
| POST | `/auth/register` | ✗ | Регистрация нового пользователя |
| POST | `/auth/login` | ✗ | Вход, возвращает JWT токен |

**Request (register/login):**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response (201 / 200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "user_id": 1
}
```

### Предсказания

| Method | Path | Auth | Описание |
|--------|------|------|----------|
| POST | `/predict` | ✓ | Предсказание цены квартиры |
| GET | `/predict/history` | ✓ | История предсказаний пользователя |

**Request (POST /predict):**
```json
{
  "rooms": 2,
  "area": 50.0,
  "floor": 5,
  "total_floors": 10,
  "year_built": 2005,
  "distance_to_metro_min": 10
}
```

**Response (201):**
```json
{
  "id": 1,
  "input_features": {"rooms": 2, "area": 50.0, ...},
  "predicted_price": 8500000.0,
  "created_at": "2026-10-27T14:30:00+00:00"
}
```

**Коды ошибок `/predict`:** `401` — нет/просрочен токен, `422` — невалидные данные
(в т.ч. `floor` > `total_floors`), `429` — превышен лимит, `503` — модель недоступна.

**Response (GET /history):**
```json
{
  "predictions": [
    {
      "id": 1,
      "input_features": {...},
      "predicted_price": 8500000.0,
      "created_at": "2026-10-27T14:30:00+00:00"
    }
  ]
}
```

### Авторизация

Все защищённые эндпоинты требуют заголовок:
```
Authorization: Bearer <jwt_token>
```

Subject токена — `user_id`. Email регистронезависим (`A@x.com` и `a@x.com` — один пользователь).

### Rate limiting

| Эндпоинт | Лимит |
|----------|-------|
| `POST /auth/register` | 5 в час |
| `POST /auth/login` | 10 в минуту |
| `POST /predict` | 10 в минуту |
| `GET /predict/history` | 30 в минуту |

Лимиты считаются по IP и хранятся в памяти процесса (при нескольких воркерах/репликах
нужно общее хранилище, например Redis). В тестах лимитер отключён.

## Схема БД

### Таблица `users`

| Колонка | Тип | Описание |
|---------|-----|----------|
| `id` | INTEGER (PK) | Автоинкремент |
| `email` | VARCHAR(255) | Уникальный, индекс |
| `hashed_password` | VARCHAR(255) | PBKDF2 хэш |
| `created_at` | TIMESTAMP | server_default=now() |

### Таблица `predictions`

| Колонка | Тип | Описание |
|---------|-----|----------|
| `id` | INTEGER (PK) | Автоинкремент |
| `user_id` | INTEGER (FK → users) | CASCADE DELETE, индекс |
| `input_features` | JSONB (PostgreSQL) / JSON (SQLite в тестах) | Входные фичи |
| `predicted_price` | FLOAT | Предсказанная цена |
| `created_at` | TIMESTAMP | server_default=now() |

## Структура проекта

```
src/
├── api/
│   ├── routers/          # FastAPI роутеры
│   │   ├── auth.py       # /register, /login
│   │   ├── prediction.py # /predict, /history
│   │   └── health.py     # /health
│   ├── schemas.py        # Pydantic модели
│   ├── dependencies.py   # DI: get_db, get_token
│   └── main.py           # FastAPI app
├── services/             # Бизнес-логика
│   ├── auth_service.py
│   └── prediction_service.py
├── repositories/         # Доступ к БД
│   ├── user_repository.py
│   ├── prediction_repository.py
│   └── model_repository.py
├── models/               # SQLAlchemy модели
│   ├── base.py
│   ├── user.py
│   └── prediction.py
├── core/                 # Ядро
│   ├── config.py         # pydantic-settings
│   ├── security.py       # JWT, хэширование
│   ├── rate_limiter.py   # slowapi
│   └── logging.py        # structlog
├── ingestion/            # Парсинг
│   └── scraper.py
├── data/                 # Preprocessing
│   ├── loader.py
│   └── preprocessing.py
├── training/             # Обучение ML
│   ├── train.py          # CatBoost + Optuna (hold-out без утечки)
│   └── model.py          # обёртка: predict() возвращает рубли (модель учится на log1p)
└── frontend/             # Streamlit
    └── app.py
scripts/
└── train_model.py        # python -m scripts.train_model
alembic/                  # миграции БД
artifacts/                # обученная модель (не в git)
tests/
├── test_core.py          # Security + Config
├── test_models.py        # SQLAlchemy models
├── test_repositories.py  # CRUD operations
├── test_scraper.py       # Парсинг
├── test_preprocessing.py # Data pipeline
├── test_services.py      # Business logic
├── test_training.py      # ML training
├── test_api.py           # API endpoints
└── test_imports.py       # Import validation
```

## Тесты

```bash
# Все тесты
poetry run pytest tests/ -v

# С покрытием
poetry run pytest tests/ --cov=src --cov-report=html

# Только API
poetry run pytest tests/test_api.py -v
```

## CI/CD

GitHub Actions запускает:
- **lint** — `ruff check` (пока не блокирует сборку)
- **test** — pytest (БД в тестах — SQLite in-memory)

Конфиг: `.github/workflows/ci.yml`

## Лицензия

MIT
