import pytest
from sqlalchemy import select
from starlette.requests import Request

from app.api.v1.routes import auth as auth_routes
from app.api.v1.routes import upload as upload_routes
from app.core import ratelimit
from app.core.config import settings
from app.models.analysis import Analysis, AnalysisStatus
from app.models.user import User
from app.workers import pipeline
from tests.api.conftest import auth
from tests.api.test_uploads import CHAT, _Queue, _upload

pytestmark = pytest.mark.asyncio


def _request(headers: dict[str, str], peer: str = "10.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": (peer, 1234),
        }
    )


# ── Rate limit keys (pure) ────────────────────────────────────────────────────

def test_key_uses_user_when_token_is_valid():
    from app.api.v1.routes.auth import _create_token

    req = _request({"Authorization": f"Bearer {_create_token(7)}"})
    assert ratelimit.user_or_ip_key(req) == "user:7"


def test_key_falls_back_to_ip_on_bad_token():
    assert ratelimit.user_or_ip_key(_request({"Authorization": "Bearer junk"})) == "ip:10.0.0.1"
    assert ratelimit.user_or_ip_key(_request({})) == "ip:10.0.0.1"


def test_proxy_header_first_value_only_when_configured(monkeypatch):
    req = _request({"X-Forwarded-For": "1.2.3.4, 9.9.9.9"})
    assert ratelimit.ip_key(req) == "ip:10.0.0.1"  # not configured: header ignored
    monkeypatch.setattr(settings, "TRUSTED_PROXY_HEADER", "X-Forwarded-For")
    assert ratelimit.ip_key(req) == "ip:1.2.3.4"
    # Header configured but absent: fall back to the connection IP
    assert ratelimit.ip_key(_request({})) == "ip:10.0.0.1"


# ── Rate limit in the app ─────────────────────────────────────────────────────

async def test_eleventh_upload_is_429_not_500(client, monkeypatch):
    monkeypatch.setattr(upload_routes, "process_chat_upload", _Queue())
    codes = [(await _upload(client)).status_code for _ in range(11)]
    assert codes[:10] == [202] * 10
    assert codes[10] == 429


async def test_login_is_rate_limited_per_minute(client):
    form = {"username": "nobody@example.com", "password": "wrong-password"}
    codes = [
        (await client.post("/api/v1/auth/token", data=form)).status_code for _ in range(11)
    ]
    assert codes[:10] == [401] * 10
    assert codes[10] == 429


async def test_register_is_limited_to_five_per_hour(client):
    codes = []
    for i in range(6):
        resp = await client.post(
            "/api/v1/auth/register", json={"email": f"u{i}@example.com", "password": "password123"}
        )
        codes.append(resp.status_code)
    assert codes == [201] * 5 + [429]


# ── Optional auth ─────────────────────────────────────────────────────────────

async def test_invalid_token_on_upload_is_401(client, monkeypatch):
    monkeypatch.setattr(upload_routes, "process_chat_upload", _Queue())
    resp = await client.post(
        "/api/v1/upload/",
        files={"file": ("chat.txt", CHAT.encode(), "text/plain")},
        headers={"Authorization": "Bearer not-a-token"},
    )
    assert resp.status_code == 401


async def test_no_header_is_guest(client, monkeypatch):
    monkeypatch.setattr(upload_routes, "process_chat_upload", _Queue())
    assert (await _upload(client)).status_code == 202


# ── Upload size ───────────────────────────────────────────────────────────────

async def test_oversized_upload_is_413(client, monkeypatch):
    monkeypatch.setattr(upload_routes, "process_chat_upload", _Queue())
    monkeypatch.setattr(upload_routes, "MAX_FILE_SIZE", 1000)
    monkeypatch.setattr(upload_routes, "_READ_CHUNK", 256)
    resp = await client.post(
        "/api/v1/upload/",
        files={"file": ("chat.txt", b"x" * 5000, "text/plain")},
    )
    assert resp.status_code == 413


def test_max_file_size_is_20_mb():
    assert upload_routes.MAX_FILE_SIZE == 20 * 1024 * 1024


# ── Google link ───────────────────────────────────────────────────────────────

async def test_google_link_drops_old_password(client, monkeypatch):
    email = "victim@example.com"
    reg = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "old-password-1"}
    )
    assert reg.status_code == 201

    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setattr(
        auth_routes.google_id_token,
        "verify_oauth2_token",
        lambda *a, **k: {"sub": "g-123", "email": email, "email_verified": True},
    )
    ok = await client.post("/api/v1/auth/google", json={"credential": "x"})
    assert ok.status_code == 200

    old = await client.post(
        "/api/v1/auth/token", data={"username": email, "password": "old-password-1"}
    )
    assert old.status_code == 401


async def test_login_unknown_user_still_runs_bcrypt(client, monkeypatch):
    calls = []
    real = auth_routes._verify

    def spy(plain, hashed):
        calls.append(hashed)
        return real(plain, hashed)

    monkeypatch.setattr(auth_routes, "_verify", spy)
    resp = await client.post(
        "/api/v1/auth/token", data={"username": "ghost@example.com", "password": "whatever-1"}
    )
    assert resp.status_code == 401
    assert calls == [auth_routes._DUMMY_HASH]


# ── Incomplete report is re-locked ────────────────────────────────────────────

async def test_incomplete_report_refunds_and_relocks(
    make_user, make_analysis, db_session, monkeypatch
):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from tests.conftest import TEST_DATABASE_URL

    sync_engine = create_engine(TEST_DATABASE_URL.replace("+asyncpg", "+psycopg2"))
    monkeypatch.setattr("app.core.database.SyncSession", sessionmaker(sync_engine))

    user = await make_user(credits=0)
    user_id = user.id
    analysis = await make_analysis(user, unlocked=True, credits_spent=1, status=AnalysisStatus.processing)
    analysis_id = analysis.id

    pipeline._save_result(analysis_id, {"temporal": {}}, refund=True)
    sync_engine.dispose()

    await db_session.rollback()
    row = (await db_session.execute(select(Analysis).where(Analysis.id == analysis_id))).scalar_one()
    credits = (await db_session.execute(select(User.credits).where(User.id == user_id))).scalar_one()
    assert (row.unlocked, row.credits_spent, credits) == (False, 0, 1)
