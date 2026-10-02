import hashlib
import hmac
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CreditPack(BaseModel):
    """One purchasable credit pack, as configured in the CREDIT_PACKS env JSON."""

    key: str = Field(min_length=1)
    credits: int = Field(gt=0)
    price_cents: int = Field(ge=0)
    currency: str
    variant_id: int  # Lemon Squeezy variant, used to map a paid order back to the pack
    checkout_url: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    PROJECT_NAME: str = "LastSeen"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str

    REDIS_URL: str = "redis://localhost:6380/0"
    CELERY_BROKER_URL: str = "redis://localhost:6380/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6380/0"

    ANTHROPIC_API_KEY: str | None = None
    NARRATIVE_MODEL: str = "claude-haiku-4-5"

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    # Cheap flash-lite sibling of GEMINI_MODEL: scoring thousands of short messages
    GEMINI_SENTIMENT_MODEL: str = "gemini-3.5-flash-lite"
    # auto = gemini when GEMINI_API_KEY is set, local models otherwise
    SENTIMENT_BACKEND: Literal["auto", "gemini", "local"] = "auto"

    GOOGLE_CLIENT_ID: str | None = None

    ADMIN_USERNAME: str
    ADMIN_PASSWORD_HASH: str

    SECRET_KEY: str
    # Pinned: a configurable JWT algorithm invites downgrade mistakes ("none", RS/HS confusion)
    ALGORITHM: Literal["HS256"] = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Empty secret = payments not configured (webhook answers 503)
    LEMONSQUEEZY_WEBHOOK_SECRET: str = ""
    # Test-mode orders are ignored unless this is on, so sandbox events never grant real credits
    LEMONSQUEEZY_ALLOW_TEST_MODE: bool = False
    CREDIT_PACKS: list[CreditPack] = []
    FREE_CREDITS_ON_SIGNUP: int = 0

    # production hides the API docs, enforces a strong SECRET_KEY and HTTPS-only admin cookies
    ENVIRONMENT: Literal["development", "production"] = "development"

    # Header carrying the real client IP ("CF-Connecting-IP" or "X-Forwarded-For").
    # Empty = use the connection IP. Only set it behind a reverse proxy you control
    # that overwrites this header: otherwise any client can spoof it and dodge rate limits.
    TRUSTED_PROXY_HEADER: str = ""
    # Where rate-limit counters live. Empty = REDIS_URL (shared by every process).
    RATE_LIMIT_STORAGE_URI: str = ""

    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]


# Below this length an HMAC key is cheap to brute-force offline
MIN_SECRET_KEY_LENGTH = 32


def validate_production_settings(cfg: "Settings") -> None:
    """Fail fast on unsafe production configuration. No-op in development."""
    if cfg.ENVIRONMENT != "production":
        return
    if len(cfg.SECRET_KEY) < MIN_SECRET_KEY_LENGTH:
        raise RuntimeError(
            f"SECRET_KEY must be at least {MIN_SECRET_KEY_LENGTH} characters in production"
        )


# Fixed label so the admin cookie key differs from the JWT key: a leaked cookie
# signature cannot be used to forge API tokens, and vice versa
_ADMIN_SESSION_LABEL = b"lastseen-admin-session-v1"


def derive_admin_session_key(secret_key: str) -> str:
    """Key for signing the /admin session cookie, derived from SECRET_KEY by HMAC."""
    return hmac.new(secret_key.encode(), _ADMIN_SESSION_LABEL, hashlib.sha256).hexdigest()


settings = Settings()
