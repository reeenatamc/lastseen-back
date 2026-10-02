from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordRequestForm
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from jose import jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from app.core.config import settings
from app.core.dependencies import DB, get_current_user
from app.core.ratelimit import GOOGLE_LIMIT, LOGIN_LIMITS, REGISTER_LIMIT, ip_key, limiter
from app.models.user import User

router = APIRouter()

FormData = Annotated[OAuth2PasswordRequestForm, Depends()]


# --- Schemas ---

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class GoogleLoginRequest(BaseModel):
    credential: str = Field(min_length=1, description="Google ID token (JWT) from GSI")


class UserOut(BaseModel):
    id: int
    email: str
    is_premium: bool

    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- Helpers ---

def _hash(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _verify(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# Verified when the e-mail is unknown so a missing account costs the same bcrypt
# time as a wrong password (no account enumeration by response time)
_DUMMY_HASH = _hash("timing-equalizer-not-a-real-password")


def _create_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


# --- Endpoints ---

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
@limiter.limit(REGISTER_LIMIT, key_func=ip_key)
async def register(request: Request, body: RegisterRequest, db: DB):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=body.email,
        hashed_password=await run_in_threadpool(_hash, body.password),
        credits=settings.FREE_CREDITS_ON_SIGNUP,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/token", response_model=TokenOut)
@limiter.limit(LOGIN_LIMITS[0], key_func=ip_key)
@limiter.limit(LOGIN_LIMITS[1], key_func=ip_key)
async def login(request: Request, form_data: FormData, db: DB):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    # Always run one bcrypt check, off the event loop, whether or not the account exists
    hashed = user.hashed_password if user and user.hashed_password else _DUMMY_HASH
    password_ok = await run_in_threadpool(_verify, form_data.password, hashed)
    if not user or not user.hashed_password or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    return {"access_token": _create_token(user.id)}


@router.post("/google", response_model=TokenOut)
@limiter.limit(GOOGLE_LIMIT, key_func=ip_key)
async def google_login(request: Request, body: GoogleLoginRequest, db: DB):
    """
    Verifies a Google ID token (from Google Identity Services on the frontend)
    and returns a LastSeen JWT. Auto-links to an existing email account when
    Google reports email_verified=True.
    """
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google login not configured",
        )

    try:
        # Blocking network call (fetches Google certs): off the event loop
        idinfo = await run_in_threadpool(
            google_id_token.verify_oauth2_token,
            body.credential,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token",
        )

    google_sub = idinfo["sub"]
    email = idinfo.get("email")
    email_verified = idinfo.get("email_verified", False)

    if not email or not email_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google account email is not verified",
        )

    # 1. Already linked: look up by google_sub
    result = await db.execute(select(User).where(User.google_sub == google_sub))
    user = result.scalar_one_or_none()

    # 2. Not linked yet: try to find an existing email account and link it
    if not user:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user:
            user.google_sub = google_sub
            # Google vouches for the e-mail owner; whoever registered a password for
            # this address first may be someone else, so that password stops working
            user.hashed_password = None
            await db.commit()
            await db.refresh(user)

    # 3. Brand new user: create Google-only account (no password)
    if not user:
        user = User(
            email=email,
            hashed_password=None,
            google_sub=google_sub,
            credits=settings.FREE_CREDITS_ON_SIGNUP,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    return {"access_token": _create_token(user.id)}


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user
