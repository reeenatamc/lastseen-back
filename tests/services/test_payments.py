import hashlib
import hmac
from urllib.parse import parse_qs, urlsplit

from app.core.config import CreditPack
from app.services.payments import (
    build_checkout_url,
    credits_for_variant,
    parse_order_event,
    verify_signature,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

SECRET = "whsec_test"


def _sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _payload(event: str = "order_created", **overrides) -> dict:
    payload = {
        "meta": {"event_name": event, "custom_data": {"user_id": "123"}, "test_mode": False},
        "data": {
            "type": "orders",
            "id": "1",
            "attributes": {
                "status": "paid",
                "total": 400,
                "currency": "USD",
                "first_order_item": {"variant_id": 123456},
                "refunded": False,
            },
        },
    }
    payload.update(overrides)
    return payload


def _pack(**kw) -> CreditPack:
    base = dict(
        key="single", credits=1, price_cents=400, currency="USD", variant_id=123456,
        checkout_url="https://shop.lemonsqueezy.com/checkout/buy/abc",
    )
    base.update(kw)
    return CreditPack(**base)


# ── Signature ─────────────────────────────────────────────────────────────────

def test_signature_valid():
    body = b'{"a": 1}'
    assert verify_signature(body, _sign(body), SECRET)


def test_signature_invalid():
    body = b'{"a": 1}'
    assert not verify_signature(body, _sign(body, "other"), SECRET)
    assert not verify_signature(b"tampered", _sign(body), SECRET)


def test_signature_missing():
    assert not verify_signature(b"x", None, SECRET)
    assert not verify_signature(b"x", "", SECRET)


def test_signature_empty_secret():
    body = b"x"
    assert not verify_signature(body, _sign(body, ""), "")


# ── Parsing ───────────────────────────────────────────────────────────────────

def test_parse_order_created_paid():
    event = parse_order_event(_payload())
    assert event is not None
    assert event.event_name == "order_created"
    assert event.order_id == "1"
    assert event.user_id == 123
    assert event.variant_id == 123456
    assert event.status == "paid"
    assert event.amount_cents == 400
    assert event.currency == "USD"
    assert event.test_mode is False


def test_parse_order_refunded():
    payload = _payload("order_refunded")
    payload["data"]["attributes"]["status"] = "refunded"
    event = parse_order_event(payload)
    assert event.event_name == "order_refunded"
    assert event.status == "refunded"


def test_parse_foreign_payload():
    payload = _payload()
    payload["data"]["type"] = "subscriptions"
    assert parse_order_event(payload) is None
    assert parse_order_event({}) is None


def test_parse_missing_order_id():
    payload = _payload()
    del payload["data"]["id"]
    assert parse_order_event(payload) is None


def test_parse_user_id_missing_or_invalid():
    payload = _payload()
    payload["meta"]["custom_data"] = {}
    assert parse_order_event(payload).user_id is None
    payload["meta"]["custom_data"] = {"user_id": "abc"}
    assert parse_order_event(payload).user_id is None
    del payload["meta"]["custom_data"]
    assert parse_order_event(payload).user_id is None


def test_parse_test_mode():
    payload = _payload()
    payload["meta"]["test_mode"] = True
    assert parse_order_event(payload).test_mode is True


# ── Packs and checkout ────────────────────────────────────────────────────────

def test_credits_for_variant():
    packs = [_pack(), _pack(key="triple", credits=3, variant_id=999)]
    assert credits_for_variant(123456, packs) == 1
    assert credits_for_variant(999, packs) == 3
    assert credits_for_variant(1, packs) is None
    assert credits_for_variant(None, packs) is None


def test_checkout_url_encoding():
    url = build_checkout_url(_pack(), 42, "a+b@example.com")
    parts = urlsplit(url)
    assert parts.path == "/checkout/buy/abc"
    query = parse_qs(parts.query)
    assert query["checkout[custom][user_id]"] == ["42"]
    assert query["checkout[email]"] == ["a+b@example.com"]
    assert "+b@" not in parts.query  # plus and at-sign must be percent-encoded


def test_checkout_url_keeps_existing_params():
    pack = _pack(checkout_url="https://shop.lemonsqueezy.com/checkout/buy/abc?discount=0&embed=1")
    query = parse_qs(urlsplit(build_checkout_url(pack, 1, "x@y.com")).query)
    assert query["discount"] == ["0"]
    assert query["embed"] == ["1"]
    assert query["checkout[custom][user_id]"] == ["1"]
