import pytest
from sqlalchemy import select

from app.api.v1.routes import upload as upload_routes
from app.models.analysis import Analysis, AnalysisStatus
from app.models.user import User
from app.services.credits import refund_analysis_credit_async
from tests.api.conftest import FULL_RESULT, auth

pytestmark = pytest.mark.asyncio

CHAT = "[1/9/26, 10:00:00] Ana: hola\n[1/9/26, 10:01:00] Beto: hey"


class _Queue:
    """Stands in for the Celery task: records calls instead of queueing."""

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def apply_async(self, **kwargs):
        if self.fail:
            raise RuntimeError("broker down")
        self.calls.append(kwargs)

        class _Result:
            id = "task-1"

        return _Result()


@pytest.fixture
def queue(monkeypatch):
    q = _Queue()
    monkeypatch.setattr(upload_routes, "process_chat_upload", q)
    return q


async def _upload(client, user=None):
    headers = auth(user) if user else {}
    return await client.post(
        "/api/v1/upload/",
        files={"file": ("chat.txt", CHAT.encode(), "text/plain")},
        headers=headers,
    )


async def _row(db_session, analysis_id: int) -> Analysis:
    await db_session.rollback()
    return (await db_session.execute(select(Analysis).where(Analysis.id == analysis_id))).scalar_one()


async def _credits(db_session, user_id: int) -> int:
    await db_session.rollback()
    return (await db_session.execute(select(User.credits).where(User.id == user_id))).scalar_one()


# ── Upload tiers ──────────────────────────────────────────────────────────────

async def test_guest_gets_preview_tier(client, queue):
    resp = await _upload(client)
    assert resp.status_code == 202
    assert resp.json()["tier"] == "preview"
    assert queue.calls[0]["kwargs"]["tier"] == "preview"


async def test_enqueue_hides_chat_content_from_logs(client, queue):
    await _upload(client)
    call = queue.calls[0]
    assert "hola" not in call["kwargsrepr"]
    assert "hola" not in call["argsrepr"]


async def test_user_without_credits_gets_locked_preview(client, make_user, queue, db_session):
    user = await make_user(credits=0)
    user_id = user.id
    resp = await _upload(client, user)
    body = resp.json()
    assert body["tier"] == "preview"
    analysis = await _row(db_session, body["analysis_id"])
    assert (analysis.unlocked, analysis.credits_spent) == (False, 0)
    assert await _credits(db_session, user_id) == 0
    assert queue.calls[0]["kwargs"]["tier"] == "preview"


async def test_user_with_credit_pays_and_unlocks(client, make_user, queue, db_session):
    user = await make_user(credits=1)
    user_id = user.id
    body = (await _upload(client, user)).json()
    assert body["tier"] == "full"
    analysis = await _row(db_session, body["analysis_id"])
    assert (analysis.unlocked, analysis.credits_spent) == (True, 1)
    assert await _credits(db_session, user_id) == 0
    assert queue.calls[0]["kwargs"]["tier"] == "full"


async def test_second_upload_with_one_credit_is_preview(client, make_user, queue, db_session):
    user = await make_user(credits=1)
    first = (await _upload(client, user)).json()
    second = (await _upload(client, user)).json()
    assert (first["tier"], second["tier"]) == ("full", "preview")
    assert (await _row(db_session, second["analysis_id"])).unlocked is False


async def test_premium_is_full_without_charge(client, make_user, queue, db_session):
    user = await make_user(credits=0, premium=True)
    user_id = user.id
    body = (await _upload(client, user)).json()
    assert body["tier"] == "full"
    analysis = await _row(db_session, body["analysis_id"])
    assert (analysis.unlocked, analysis.credits_spent) == (True, 0)
    assert await _credits(db_session, user_id) == 0


async def test_enqueue_failure_refunds_credit(client, make_user, monkeypatch, db_session):
    monkeypatch.setattr(upload_routes, "process_chat_upload", _Queue(fail=True))
    user = await make_user(credits=1)
    user_id = user.id
    resp = await _upload(client, user)
    assert resp.status_code == 503
    assert await _credits(db_session, user_id) == 1
    analysis = (await db_session.execute(select(Analysis))).scalar_one()
    assert (analysis.status, analysis.credits_spent) == (AnalysisStatus.failed, 0)


# ── Refund ────────────────────────────────────────────────────────────────────

async def test_refund_is_idempotent(make_user, make_analysis, db_session):
    user = await make_user(credits=0)
    user_id = user.id
    analysis = await make_analysis(user, unlocked=True, credits_spent=1)
    analysis_id = analysis.id
    assert await refund_analysis_credit_async(db_session, analysis_id) == 1
    await db_session.commit()
    assert await refund_analysis_credit_async(db_session, analysis_id) == 0
    await db_session.commit()
    assert await _credits(db_session, user_id) == 1
    assert (await _row(db_session, analysis_id)).credits_spent == 0


# ── Read ──────────────────────────────────────────────────────────────────────

async def test_get_locked_returns_preview(client, make_user, make_analysis):
    user = await make_user()
    analysis = await make_analysis(user, result=FULL_RESULT)
    body = (await client.get(f"/api/v1/analysis/{analysis.id}", headers=auth(user))).json()
    assert body["access"]["level"] == "preview"
    assert body["access"]["teaser"]["conflict_episodes"] == 1
    assert "sentiment" not in body["result"]


async def test_get_unlocked_returns_full(client, make_user, make_analysis):
    user = await make_user()
    analysis = await make_analysis(user, unlocked=True, result=FULL_RESULT)
    body = (await client.get(f"/api/v1/analysis/{analysis.id}", headers=auth(user))).json()
    assert body["access"] == {"level": "full"}
    assert body["result"] == FULL_RESULT


async def test_get_premium_flag_alone_does_not_unlock(client, make_user, make_analysis):
    user = await make_user(premium=True)
    analysis = await make_analysis(user, unlocked=False, result=FULL_RESULT)
    body = (await client.get(f"/api/v1/analysis/{analysis.id}", headers=auth(user))).json()
    assert body["access"]["level"] == "preview"


async def test_get_without_result_has_no_access(client, make_user, make_analysis):
    user = await make_user()
    analysis = await make_analysis(user, result=None)
    body = (await client.get(f"/api/v1/analysis/{analysis.id}", headers=auth(user))).json()
    assert body["access"] is None


async def test_list_includes_unlocked(client, make_user, make_analysis):
    user = await make_user()
    await make_analysis(user, unlocked=True)
    body = (await client.get("/api/v1/analysis/", headers=auth(user))).json()
    assert body[0]["unlocked"] is True
