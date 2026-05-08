from typing import TYPE_CHECKING, Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal

if TYPE_CHECKING:
    from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")
oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/token", auto_error=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def _decode_user_id(token: str) -> int:
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    sub = payload.get("sub")
    if sub is None:
        raise ValueError("missing sub")
    return int(sub)


async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> int:
    """Required auth — raises 401 if no valid token."""
    try:
        return _decode_user_id(token)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user_id(
    token: str | None = Depends(oauth2_scheme_optional),
) -> int | None:
    """Optional auth — returns None for guests, no error raised."""
    if token is None:
        return None
    try:
        return _decode_user_id(token)
    except (JWTError, ValueError):
        return None


async def get_current_user(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> "User":
    from app.models.user import User

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


# --- Annotated aliases — use these in route signatures instead of repeating Depends() ---

DB = Annotated[AsyncSession, Depends(get_db)]
CurrentUserId = Annotated[int, Depends(get_current_user_id)]
OptionalUserId = Annotated[int | None, Depends(get_optional_user_id)]
