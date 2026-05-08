from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Message(Base):
    """
    Raw message storage — not written by default.
    Enable in pipeline._persist() when message-level features are needed
    (search, data export, analytics resale). Requires explicit user consent.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    sender: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    is_media: Mapped[bool] = mapped_column(Boolean, default=False)
    media_type: Mapped[str | None] = mapped_column(String(100))

    analysis: Mapped["Analysis"] = relationship(back_populates="messages")  # noqa: F821
