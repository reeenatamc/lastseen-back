import secrets

import bcrypt
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from slowapi.errors import RateLimitExceeded
from sqladmin import Admin
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request

import app.models  # noqa: F401 — registers all SQLAlchemy models
from app.admin.views import AnalysisAdmin, UserAdmin
from app.core.config import derive_admin_session_key, settings, validate_production_settings
from app.core.ratelimit import limiter
from app.core.database import engine
from app.api.v1 import router as api_v1_router


class AdminAuth(AuthenticationBackend):
    def __init__(self, secret_key: str, https_only: bool = False) -> None:
        super().__init__(secret_key)
        # Replace the default session middleware to control the cookie flags
        self.middlewares = [
            Middleware(SessionMiddleware, secret_key=secret_key, https_only=https_only),
        ]

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = form.get("username") or ""
        password = form.get("password") or ""
        username_ok = secrets.compare_digest(
            username.encode(), settings.ADMIN_USERNAME.encode()
        )
        try:
            # bcrypt is slow on purpose: keep it off the event loop
            password_ok = await run_in_threadpool(
                bcrypt.checkpw, password.encode(), settings.ADMIN_PASSWORD_HASH.encode()
            )
        except ValueError:
            password_ok = False
        if username_ok and password_ok:
            request.session["authenticated"] = True
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get("authenticated", False)


validate_production_settings(settings)
_is_production = settings.ENVIRONMENT == "production"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    # No public API docs in production
    openapi_url=None if _is_production else f"{settings.API_V1_STR}/openapi.json",
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix=settings.API_V1_STR)

admin = Admin(
    app,
    engine,
    authentication_backend=AdminAuth(
        secret_key=derive_admin_session_key(settings.SECRET_KEY),
        https_only=_is_production,
    ),
    title="LASTSEEN",
    base_url="/admin",
    templates_dir="app/admin/templates",
)
admin.add_view(UserAdmin)
admin.add_view(AnalysisAdmin)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
