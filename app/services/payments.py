"""Lemon Squeezy payments: signature check, event parsing and credit accounting."""
import hashlib
import hmac
import logging
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import CreditPack, settings
from app.models.payment import Payment, PaymentStatus
from app.models.user import User

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROVIDER = "lemonsqueezy"

# ── Pure helpers ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrderEvent:
    event_name: str
    order_id: str
    user_id: int | None
    variant_id: int | None
    status: str | None
    amount_cents: int | None
    currency: str | None
    test_mode: bool


def verify_signature(raw_body: bytes, signature: str | None, secret: str) -> bool:
    """Check the X-Signature header: hex HMAC-SHA256 of the raw body."""
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    # Compared as bytes: compare_digest rejects non-ASCII str, and the header is untrusted
    return hmac.compare_digest(expected.encode(), signature.strip().encode())


def _to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_order_event(payload: dict) -> OrderEvent | None:
    """Extract the fields we need from a webhook payload; None if not an order event."""
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta") or {}
    data = payload.get("data") or {}
    if not isinstance(meta, dict) or not isinstance(data, dict):
        return None
    if data.get("type") != "orders" or data.get("id") in (None, ""):
        return None

    attrs = data.get("attributes") or {}
    custom = meta.get("custom_data") or {}
    item = attrs.get("first_order_item") or {}
    if not all(isinstance(x, dict) for x in (attrs, custom, item)):
        return None

    return OrderEvent(
        event_name=str(meta.get("event_name") or ""),
        order_id=str(data["id"]),
        user_id=_to_int(custom.get("user_id")),
        variant_id=_to_int(item.get("variant_id")),
        status=attrs.get("status"),
        amount_cents=_to_int(attrs.get("total")),
        currency=attrs.get("currency"),
        test_mode=bool(meta.get("test_mode", False)),
    )


def credits_for_variant(variant_id: int | None, packs: list[CreditPack]) -> int | None:
    if variant_id is None:
        return None
    for pack in packs:
        if pack.variant_id == variant_id:
            return pack.credits
    return None


def build_checkout_url(pack: CreditPack, user_id: int, email: str) -> str:
    """Pack checkout URL with the buyer attached as custom data, keeping existing params."""
    parts = urlsplit(pack.checkout_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append(("checkout[custom][user_id]", str(user_id)))
    query.append(("checkout[email]", email))
    return urlunsplit(parts._replace(query=urlencode(query)))


# ── Event application (async, DB) ─────────────────────────────────────────────

async def apply_order_created(session: AsyncSession, event: OrderEvent) -> None:
    """Record a paid order and grant its credits in one transaction. Idempotent."""
    if event.status != "paid":
        return

    user: User | None = None
    if event.user_id is not None:
        user = await session.get(User, event.user_id)
    credits = credits_for_variant(event.variant_id, settings.CREDIT_PACKS) if user else None
    matched = user is not None and credits is not None

    # ON CONFLICT DO NOTHING: a repeated order id inserts nothing and grants nothing
    stmt = (
        pg_insert(Payment)
        .values(
            user_id=user.id if user else None,
            provider=PROVIDER,
            provider_order_id=event.order_id,
            credits=credits if matched else 0,
            amount_cents=event.amount_cents,
            currency=event.currency,
            status=PaymentStatus.paid if matched else PaymentStatus.unmatched,
        )
        .on_conflict_do_nothing(constraint="uq_payments_provider_order")
        .returning(Payment.id)
    )
    inserted = (await session.execute(stmt)).scalar_one_or_none()
    if inserted is None:
        await session.rollback()
        return

    if matched:
        await session.execute(
            update(User).where(User.id == user.id).values(credits=User.credits + credits)
        )
    else:
        logger.warning(
            "Unmatched order %s (user_found=%s, variant_known=%s)",
            event.order_id, user is not None, credits is not None,
        )
    await session.commit()


async def apply_order_refunded(session: AsyncSession, event: OrderEvent) -> None:
    """Mark a payment refunded and take its credits back, never below zero."""
    # FOR UPDATE serializes concurrent refund webhooks for the same order
    result = await session.execute(
        select(Payment)
        .where(Payment.provider == PROVIDER, Payment.provider_order_id == event.order_id)
        .with_for_update()
    )
    payment = result.scalar_one_or_none()
    if payment is None:
        # Refund arrived before its order_created (webhooks can be reordered). Record it
        # so the late order_created hits the unique constraint and credits nothing.
        await session.execute(
            pg_insert(Payment)
            .values(
                user_id=None,
                provider=PROVIDER,
                provider_order_id=event.order_id,
                credits=0,
                amount_cents=event.amount_cents,
                currency=event.currency,
                status=PaymentStatus.refunded,
            )
            .on_conflict_do_nothing(constraint="uq_payments_provider_order")
        )
        await session.commit()
        return
    if payment.status == PaymentStatus.refunded:
        await session.rollback()
        return

    if payment.status == PaymentStatus.paid and payment.user_id is not None and payment.credits:
        await session.execute(
            update(User)
            .where(User.id == payment.user_id)
            .values(credits=User.credits - payment.credits)
        )
        # GREATEST-style clamp: credits already spent cannot be recovered
        await session.execute(
            update(User).where(User.id == payment.user_id, User.credits < 0).values(credits=0)
        )
    payment.status = PaymentStatus.refunded
    await session.commit()


async def process_event(session: AsyncSession, event: OrderEvent) -> None:
    """Dispatch a parsed event, ignoring test-mode orders unless explicitly allowed."""
    if event.test_mode and not settings.LEMONSQUEEZY_ALLOW_TEST_MODE:
        return
    if event.event_name == "order_created":
        await apply_order_created(session, event)
    elif event.event_name == "order_refunded":
        await apply_order_refunded(session, event)
