import hashlib
import hmac
import json

import pytest
from sqlalchemy import select

from app.core.config import CreditPack, settings
from app.models.payment import Payment, PaymentStatus
from app.models.user import User

pytestmark = pytest.mark.asyncio

SECRET = "whsec_test"
URL = "/api/v1/payments/webhook"


@pytest.fixture(autouse=True)
def _payments_config(monkeypatch):
    monkeypatch.setattr(settings, "LEMONSQUEEZY_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(settings, "LEMONSQUEEZY_ALLOW_TEST_MODE", False)
    monkeypatch.setattr(
        settings,
        "CREDIT_PACKS",
        [
            CreditPack(
                key="triple", credits=3, price_cents=1000, currency="USD", variant_id=555,
                checkout_url="https://shop.lemonsqueezy.com/checkout/buy/abc",
            )
        ],
    )


def _event(event: str, user_id: int, order_id: str = "9", variant_id: int = 555, **meta) -> bytes:
    payload = {
        "meta": {"event_name": event, "custom_data": {"user_id": str(user_id)}, "test_mode": False, **meta},
        "data": {
            "type": "orders",
            "id": order_id,
            "attributes": {
                "status": "refunded" if event == "order_refunded" else "paid",
                "total": 1000,
                "currency": "USD",
                "first_order_item": {"variant_id": variant_id},
            },
        },
    }
    return json.dumps(payload).encode()


async def _post(client, body: bytes, secret: str = SECRET):
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return await client.post(URL, content=body, headers={"X-Signature": sig})


async def _state(db_session, user_id: int) -> tuple[int, list[Payment]]:
    await db_session.rollback()
    credits = (await db_session.execute(select(User.credits).where(User.id == user_id))).scalar_one()
    payments = (await db_session.execute(select(Payment))).scalars().all()
    return credits, list(payments)


async def test_bad_signature_is_401(client, make_user):
    user = await make_user()
    resp = await _post(client, _event("order_created", user.id), secret="wrong")
    assert resp.status_code == 401


async def test_missing_signature_is_401(client, make_user):
    user = await make_user()
    resp = await client.post(URL, content=_event("order_created", user.id))
    assert resp.status_code == 401


async def test_no_secret_is_503(client, monkeypatch):
    monkeypatch.setattr(settings, "LEMONSQUEEZY_WEBHOOK_SECRET", "")
    resp = await client.post(URL, content=b"{}")
    assert resp.status_code == 503


async def test_paid_order_adds_credits(client, make_user, db_session):
    user = await make_user(credits=1)
    resp = await _post(client, _event("order_created", user.id))
    assert resp.status_code == 200
    credits, payments = await _state(db_session, user.id)
    assert credits == 4
    assert [(p.status, p.credits, p.amount_cents) for p in payments] == [(PaymentStatus.paid, 3, 1000)]


async def test_repeated_order_does_not_duplicate(client, make_user, db_session):
    user_id = (await make_user()).id  # read now: the duplicate path rolls back and expires the ORM object
    body = _event("order_created", user_id)
    assert (await _post(client, body)).status_code == 200
    assert (await _post(client, body)).status_code == 200
    credits, payments = await _state(db_session, user_id)
    assert credits == 3
    assert len(payments) == 1


async def test_refund_subtracts_credits_once(client, make_user, db_session):
    user_id = (await make_user()).id
    await _post(client, _event("order_created", user_id))
    refund = _event("order_refunded", user_id)
    assert (await _post(client, refund)).status_code == 200
    assert (await _post(client, refund)).status_code == 200
    credits, payments = await _state(db_session, user_id)
    assert credits == 0
    assert payments[0].status == PaymentStatus.refunded


async def test_refund_never_goes_below_zero(client, make_user, db_session):
    user = await make_user()
    await _post(client, _event("order_created", user.id))
    await db_session.execute(User.__table__.update().where(User.id == user.id).values(credits=1))
    await db_session.commit()
    await _post(client, _event("order_refunded", user.id))
    credits, _ = await _state(db_session, user.id)
    assert credits == 0


async def test_unknown_variant_is_unmatched(client, make_user, db_session):
    user = await make_user()
    resp = await _post(client, _event("order_created", user.id, variant_id=1))
    assert resp.status_code == 200
    credits, payments = await _state(db_session, user.id)
    assert credits == 0
    assert [(p.status, p.credits) for p in payments] == [(PaymentStatus.unmatched, 0)]


async def test_unknown_user_is_unmatched(client, db_session):
    resp = await _post(client, _event("order_created", 99999))
    assert resp.status_code == 200
    await db_session.rollback()
    payments = (await db_session.execute(select(Payment))).scalars().all()
    assert [(p.status, p.user_id) for p in payments] == [(PaymentStatus.unmatched, None)]


async def test_test_mode_ignored_by_default(client, make_user, db_session):
    user = await make_user()
    resp = await _post(client, _event("order_created", user.id, test_mode=True))
    assert resp.status_code == 200
    credits, payments = await _state(db_session, user.id)
    assert credits == 0 and payments == []


async def test_foreign_event_is_ignored(client):
    body = json.dumps({"meta": {"event_name": "subscription_created"}, "data": {"type": "subscriptions", "id": "1"}}).encode()
    assert (await _post(client, body)).status_code == 200


async def test_refund_before_order_blocks_late_order(client, make_user, db_session):
    user_id = (await make_user()).id
    assert (await _post(client, _event("order_refunded", user_id))).status_code == 200
    assert (await _post(client, _event("order_created", user_id))).status_code == 200
    credits, payments = await _state(db_session, user_id)
    assert credits == 0
    assert [(p.status, p.credits) for p in payments] == [(PaymentStatus.refunded, 0)]
