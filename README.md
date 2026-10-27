# Kwork ML — Предсказание цен на квартиры в Москве

Production-ready SaaS для портфолио: ML-модель предсказания цен + автосбор данных + API + Frontend.

## Стек

| Категория | Технологии |
|-----------|------------|
| **Язык** | Python 3.11+, Poetry |
| **API** | FastAPI, Pydantic v2, JWT (python-jose), passlib/bcrypt |
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

# Запуск API (dev mode)
poetry run uvicorn src.api.main:app --reload --port 8000

# Запуск Streamlit
poetry run streamlit run src/frontend/app.py --server.port 8501
```

### Docker Compose

```bash
# Запуск всех сервисов (API + DB + Frontend)
docker compose up --build

# Сервисы доступны по адресам:
# API:      http://localhost:8000
# Frontend: http://localhost:8501
# DB:       localhost:5432
```

## API Документация

Базовый URL: `http://localhost:8000/api/v1`

### Health Check

| Method | Path | Auth | Описание |
|--------|------|------|----------|
| GET | `/health` | ✗ | Проверка работоспособности |

**Response:**
```json
{"status": "ok"}
```

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
  "created_at": "2026-10-27 14:30:00"
}
```

**Response (GET /history):**
```json
{
  "predictions": [
    {
      "id": 1,
      "input_features": {...},
      "predicted_price": 8500000.0,
      "created_at": "2026-10-27 14:30:00"
    }
  ]
}
```

### Авторизация

Все защищённые эндпоинты требуют заголовок:
```
Authorization: Bearer <jwt_token>
```

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
| `user_id` | INTEGER (FK → users) | CASCADE DELETE |
| `input_features` | JSONB | Входные фичи |
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
│   └── logging.py        # structlog
├── ingestion/            # Парсинг
│   └── scraper.py
├── data/                 # Preprocessing
│   ├── loader.py
│   └── preprocessing.py
├── training/             # Обучение ML
│   └── train.py
└── frontend/             # Streamlit
    └── app.py
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
- **lint** — ruff check + format
- **test** — pytest с PostgreSQL сервисом
- **docker-build** — сборка образа API

Конфиг: `.github/workflows/ci.yml`

## Лицензия

MIT
