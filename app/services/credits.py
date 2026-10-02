"""Credit accounting for analyses: spend on upload, refund on failure."""
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.user import User

# ── Spend ─────────────────────────────────────────────────────────────────────

async def spend_credit(session: AsyncSession, user_id: int) -> bool:
    """Take one credit atomically. Does not commit; returns False if none were left.

    A single conditional UPDATE means concurrent requests cannot spend the same
    credit twice.
    """
    result = await session.execute(
        update(User).where(User.id == user_id, User.credits > 0).values(credits=User.credits - 1)
    )
    return result.rowcount == 1


# ── Refund ────────────────────────────────────────────────────────────────────

def refund_analysis_credit(session: Session, analysis_id: int) -> int:
    """Return an analysis' spent credits to its owner (sync, for the worker).

    Does not commit: the caller's transaction covers it. Idempotent because
    `credits_spent` is zeroed in the same transaction. Returns credits refunded.
    """
    # FOR UPDATE serializes concurrent refunds of the same analysis
    row = session.execute(
        select(Analysis.user_id, Analysis.credits_spent)
        .where(Analysis.id == analysis_id)
        .with_for_update()
    ).one_or_none()
    if row is None or row.credits_spent <= 0:
        return 0
    session.execute(
        update(User).where(User.id == row.user_id).values(credits=User.credits + row.credits_spent)
    )
    session.execute(update(Analysis).where(Analysis.id == analysis_id).values(credits_spent=0))
    return row.credits_spent


async def refund_analysis_credit_async(session: AsyncSession, analysis_id: int) -> int:
    """Async twin of `refund_analysis_credit`, for the API process. Does not commit."""
    row = (
        await session.execute(
            select(Analysis.user_id, Analysis.credits_spent)
            .where(Analysis.id == analysis_id)
            .with_for_update()
        )
    ).one_or_none()
    if row is None or row.credits_spent <= 0:
        return 0
    await session.execute(
        update(User).where(User.id == row.user_id).values(credits=User.credits + row.credits_spent)
    )
    await session.execute(
        update(Analysis).where(Analysis.id == analysis_id).values(credits_spent=0)
    )
    return row.credits_spent
