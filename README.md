# Kwork — ML Prediction SaaS

ML-модель предсказания цен на квартиры в Москве с автоматическим сбором данных и API.

## Стек

- **Python 3.11+**, Poetry
- **API**: FastAPI, Pydantic, JWT
- **БД**: PostgreSQL, SQLAlchemy (async), Alembic
- **ML**: CatBoost, Optuna, scikit-learn
- **Парсинг**: Playwright, BeautifulSoup4
- **Frontend**: Streamlit
- **Инфраструктура**: Docker, docker-compose

## Архитектура

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│  PostgreSQL │
│  (Frontend) │     │   (API)     │     │   (DB)      │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Ingestion  │
                   │  (Scraper)  │
                   └─────────────┘
```

**Layered Architecture:**
- **API Layer** — роутеры, схемы, зависимости
- **Service Layer** — бизнес-логика (auth, prediction)
- **Repository Layer** — доступ к БД
- **Model Layer** — SQLAlchemy модели
- **Core** — конфиг, логирование, безопасность
- **Ingestion** — парсинг данных

## Запуск

### Локально

```bash
poetry install
poetry run uvicorn src.api.main:app --reload
```

### Docker

```bash
docker compose up --build
```

## API Endpoints

| Method | Path           | Description          | Auth |
|--------|----------------|----------------------|------|
| POST   | /api/auth/register | Регистрация      | ✗    |
| POST   | /api/auth/login    | Вход (JWT)         | ✗    |
| POST   | /api/predict       | Предсказание цены  | ✓    |
| GET    | /api/predict/history | История предсказаний | ✓ |
| GET    | /api/health        | Health check       | ✗    |

## Структура проекта

```
src/
├── api/          — FastAPI роутеры, схемы, зависимости
├── services/     — бизнес-логика
├── repositories/ — доступ к БД
├── models/       — SQLAlchemy модели
├── core/         — конфиг, логирование, JWT
└── ingestion/    — парсеры
```
