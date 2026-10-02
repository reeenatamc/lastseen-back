import json
import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.dependencies import DB, CurrentUserId
from app.models.user import User
from app.services.payments import (
    build_checkout_url,
    parse_order_event,
    process_event,
    verify_signature,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Schemas ---

class PackOut(BaseModel):
    key: str
    credits: int
    price_cents: int
    currency: str


class CreditsOut(BaseModel):
    credits: int
    is_premium: bool


class CheckoutRequest(BaseModel):
    pack: str


class CheckoutOut(BaseModel):
    url: str


# --- Endpoints ---

@router.get("/packs", response_model=list[PackOut])
async def list_packs():
    """Public price list. Variant ids and checkout URLs stay server-side."""
    return [
        PackOut(key=p.key, credits=p.credits, price_cents=p.price_cents, currency=p.currency)
        for p in settings.CREDIT_PACKS
    ]


@router.get("/credits", response_model=CreditsOut)
async def get_credits(db: DB, user_id: CurrentUserId):
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return CreditsOut(credits=user.credits, is_premium=user.is_premium)


@router.post("/checkout", response_model=CheckoutOut)
async def create_checkout(body: CheckoutRequest, db: DB, user_id: CurrentUserId):
    if not settings.CREDIT_PACKS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="payments_not_configured"
        )
    pack = next((p for p in settings.CREDIT_PACKS if p.key == body.pack), None)
    if pack is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown_pack")
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return CheckoutOut(url=build_checkout_url(pack, user.id, user.email))


@router.post("/webhook", include_in_schema=False)
async def lemonsqueezy_webhook(request: Request, db: DB):
    """Lemon Squeezy retries on any non-2xx: unprocessable events get 200, DB failures get 500."""
    if not settings.LEMONSQUEEZY_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="payments_not_configured"
        )
    raw_body = await request.body()
    if not verify_signature(
        raw_body, request.headers.get("X-Signature"), settings.LEMONSQUEEZY_WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_signature")

    try:
        event = parse_order_event(json.loads(raw_body))
    except ValueError:
        return {"status": "ignored"}
    if event is None:
        return {"status": "ignored"}

    try:
        await process_event(db, event)
    except Exception:
        # Unexpected failure (e.g. DB down): 500 so the provider retries; processing is idempotent
        await db.rollback()
        logger.exception("Failed to process payment event %s", event.event_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="processing_failed"
        )
    return {"status": "ok"}
