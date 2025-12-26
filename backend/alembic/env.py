"""Alembic environment configuration for async SQLAlchemy."""
import sys
import os
from pathlib import Path

# #region agent log - DEBUG: Hypothesis A - Check calculated path
_env_file = Path(__file__).resolve()
_calculated_path = str(_env_file.parent.parent)
print(f"[DEBUG-A] __file__={_env_file}", flush=True)
print(f"[DEBUG-A] calculated_path={_calculated_path}", flush=True)
# #endregion

# Add the backend directory to Python path for Railway deployment
sys.path.insert(0, _calculated_path)

# #region agent log - DEBUG: Hypothesis D/E - Check sys.path and PYTHONPATH
print(f"[DEBUG-D] sys.path after insert: {sys.path[:5]}", flush=True)
print(f"[DEBUG-E] PYTHONPATH env: {os.environ.get('PYTHONPATH', 'NOT SET')}", flush=True)
# #endregion

# #region agent log - DEBUG: Hypothesis B/C - Check directory and file existence
_app_dir = Path(_calculated_path) / "app"
_models_dir = _app_dir / "models"
_app_init = _app_dir / "__init__.py"
_models_init = _models_dir / "__init__.py"
print(f"[DEBUG-B] app_dir exists: {_app_dir.exists()} -> {_app_dir}", flush=True)
print(f"[DEBUG-B] models_dir exists: {_models_dir.exists()} -> {_models_dir}", flush=True)
print(f"[DEBUG-C] app/__init__.py exists: {_app_init.exists()}", flush=True)
print(f"[DEBUG-C] app/models/__init__.py exists: {_models_init.exists()}", flush=True)
# List app directory contents if it exists
if _app_dir.exists():
    print(f"[DEBUG-B] app_dir contents: {list(_app_dir.iterdir())[:10]}", flush=True)
else:
    # Try listing parent to see what's there
    print(f"[DEBUG-B] calculated_path contents: {list(Path(_calculated_path).iterdir())[:15]}", flush=True)
# #endregion

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Import all models to ensure they're registered with Base
from app.database import Base
from app.models import KYCSession, KYCResult, KYCReason, KYCFrameMetric, KYCJob
from app.config import settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Override sqlalchemy.url from settings (sync URL for offline migrations)
config.set_main_option("sqlalchemy.url", settings.database_url.replace("+asyncpg", ""))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    # Use asyncpg URL directly for async migrations
    connectable = create_async_engine(
        settings.async_database_url,
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
