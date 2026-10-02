import os

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
# Rate-limit counters in memory during tests (set before the app is imported): no Redis
# state leaks between runs and nothing is left blocked in the dev Redis.
os.environ.setdefault("RATE_LIMIT_STORAGE_URI", "memory://")

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.core.dependencies import get_db
from app.main import app

# Overridable because the host port (5433) is not reachable from inside the compose network,
# where the URL is postgresql+asyncpg://postgres:postgres@db:5432/lastseen_test
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5433/lastseen_test",
)

# NullPool: each test runs in its own event loop, so connections cannot be reused across tests
engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
TestSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def setup_db():
    """Creates and tears down all tables. Only used by integration tests.

    Skips (instead of failing) when the test database is not reachable.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except (OSError, SQLAlchemyError) as exc:
        pytest.skip(f"test database not reachable: {type(exc).__name__}")
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session(setup_db):
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
