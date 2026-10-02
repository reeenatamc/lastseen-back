from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PaymentStatus(str, Enum):
    paid = "paid"
    refunded = "refunded"
    unmatched = "unmatched"  # paid order we could not attribute to a user or pack


class Payment(Base):
    __tablename__ = "payments"
    # The unique pair makes webhook retries idempotent at the database level
    __table_args__ = (
        UniqueConstraint("provider", "provider_order_id", name="uq_payments_provider_order"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    # SET NULL: deleting a user must not erase the accounting record
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    provider: Mapped[str] = mapped_column(String(50))
    provider_order_id: Mapped[str] = mapped_column(String(100))
    credits: Mapped[int] = mapped_column(Integer)
    amount_cents: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str | None] = mapped_column(String(10))
    status: Mapped[PaymentStatus] = mapped_column(SAEnum(PaymentStatus))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
