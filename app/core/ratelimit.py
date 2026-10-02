"""Rate limiting: one Limiter shared by every route."""
from jose import JWTError
from slowapi import Limiter
from starlette.requests import Request

from app.core.config import settings
from app.core.dependencies import _decode_user_id

# ── Limits ────────────────────────────────────────────────────────────────────

# Uploads run the expensive pipeline; registered users and guests share the budget per key
UPLOAD_LIMIT = "10/hour"
# Login: the minute window stops bursts, the hour window stops slow credential stuffing
LOGIN_LIMITS = ("10/minute", "50/hour")
# Account creation is cheap to automate and each account can get free credits
REGISTER_LIMIT = "5/hour"
# Each call verifies a Google token over the network
GOOGLE_LIMIT = "20/minute"


# ── Keys (pure) ───────────────────────────────────────────────────────────────

def client_ip(headers: dict[str, str] | object, peer: str | None, proxy_header: str) -> str:
    """Client IP: first value of the trusted proxy header if configured, else the peer."""
    if proxy_header:
        value = headers.get(proxy_header, "")  # type: ignore[union-attr]
        first = value.split(",")[0].strip()
        if first:
            return first
    return peer or "unknown"


def _ip_of(request: Request) -> str:
    peer = request.client.host if request.client else None
    return client_ip(request.headers, peer, settings.TRUSTED_PROXY_HEADER)


def ip_key(request: Request) -> str:
    return f"ip:{_ip_of(request)}"


def user_or_ip_key(request: Request) -> str:
    """Key by user when the Authorization header holds a valid JWT, else by IP."""
    auth = request.headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token:
        try:
            return f"user:{_decode_user_id(token)}"
        except (JWTError, ValueError):
            pass
    return ip_key(request)


limiter = Limiter(
    key_func=user_or_ip_key,
    storage_uri=settings.RATE_LIMIT_STORAGE_URI or settings.REDIS_URL,
    # If Redis is unreachable keep serving with per-process counters instead of 500s
    in_memory_fallback_enabled=True,
)
