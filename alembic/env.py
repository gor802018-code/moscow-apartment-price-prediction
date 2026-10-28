"""Конфигурация Alembic для async SQLAlchemy."""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from src.core.config import get_settings
from src.models.base import Base
from src.models.prediction import Prediction  # noqa: F401

# Импортируем все модели, чтобы Alembic их увидел
from src.models.user import User  # noqa: F401

settings = get_settings()
config = context.config

# Настройка URL БД
# configparser трактует % как интерполяцию, поэтому экранируем (пароли бывают с %)
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("%", "%%"))

# Конфигурация логирования
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Запуск миграций в офлайн-режиме."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Внутренняя функция миграций."""
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def async_run_migrations() -> None:
    """Запуск миграций в онлайн-режиме (async)."""
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await engine.dispose()


def run_migrations_online() -> None:
    """Запуск миграций в онлайн-режиме (async)."""
    asyncio.run(async_run_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
