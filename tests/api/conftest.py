import pytest
import pytest_asyncio

from app.api.v1.routes.auth import _create_token
from app.models.analysis import Analysis, AnalysisStatus
from app.models.user import User

# Every test here needs the real database; `setup_db` skips them when it is unreachable.
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Counters live in memory for the whole process; start every test from zero."""
    from app.core.ratelimit import limiter

    limiter.reset()
    yield
    limiter.reset()


@pytest_asyncio.fixture
async def make_user(db_session):
    async def _make(email: str = "a@example.com", credits: int = 0, premium: bool = False) -> User:
        user = User(email=email, hashed_password="x", credits=credits, is_premium=premium)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        return user

    return _make


@pytest_asyncio.fixture
async def make_analysis(db_session):
    async def _make(
        user: User,
        unlocked: bool = False,
        credits_spent: int = 0,
        status: AnalysisStatus = AnalysisStatus.completed,
        result: dict | None = None,
    ) -> Analysis:
        analysis = Analysis(
            user_id=user.id,
            platform="whatsapp",
            original_filename="chat.txt",
            status=status,
            result=result,
            unlocked=unlocked,
            credits_spent=credits_spent,
        )
        db_session.add(analysis)
        await db_session.commit()
        await db_session.refresh(analysis)
        return analysis

    return _make


def auth(user: User) -> dict:
    return {"Authorization": f"Bearer {_create_token(user.id)}"}


FULL_RESULT = {
    "temporal": {"overview": {"total_messages": 10}, "initiative_balance": {"Alice": 1}},
    "sentiment": {"per_person": {}},
    "conflict": {"episodes": [{"date": "2026-01-01"}]},
    "narrative": {"resumen": "summary", "dinamica": "secret", "punto_de_quiebre": None},
}
