"""Pytest fixtures for testing."""

from collections.abc import AsyncGenerator
from typing import Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.main import app
from app.database import Base, get_db
from app.api.security import require_auth


TEST_TENANT_ID = "org_testtenant"


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type, _compiler, **_kwargs):
    """Allow Postgres JSONB columns to work in SQLite test database."""
    return "JSON"


def pytest_addoption(parser):
    """Add --runslow option for running slow ML tests."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Register the 'slow' marker."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow is given."""
    if config.getoption("--runslow"):
        # --runslow given: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# Test database URL (use SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
)

# Create test session factory
TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with overridden database dependency."""
    
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    async def override_require_auth() -> None:
        return None

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[require_auth] = override_require_auth
    
    async with AsyncClient(
        app=app,
        base_url="http://test",
        headers={"X-Authenticated-Org-Id": TEST_TENANT_ID},
    ) as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def no_tenant_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Client fixture without tenant header for negative tests."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    async def override_require_auth() -> None:
        return None

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[require_auth] = override_require_auth

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sync_client() -> Generator[TestClient, None, None]:
    """Synchronous test client for simple tests."""
    with TestClient(app) as c:
        yield c
