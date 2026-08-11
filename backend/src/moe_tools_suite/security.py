from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from .domain import LoginRequest, SessionStatus
from .settings import Settings

SESSION_COOKIE = "moe_tools_session"
CSRF_HEADER = "x-csrf-token"
PUBLIC_API_PATHS = {
    "/api/session",
    "/api/session/login",
    "/api/openapi.json",
    "/api/docs",
}


@dataclass(frozen=True)
class SessionClaims:
    csrf_token: str
    expires_at: datetime


class SessionSigner:
    """Issue and verify compact stateless single-user session cookies."""

    def __init__(self, secret: str, ttl: timedelta) -> None:
        self._secret = hashlib.sha256(
            f"moe-tools-session-v1:{secret}".encode()
        ).digest()
        self._ttl = ttl

    def issue(self) -> tuple[str, SessionClaims]:
        expires_at = datetime.now(UTC) + self._ttl
        claims = SessionClaims(
            csrf_token=secrets.token_urlsafe(24),
            expires_at=expires_at,
        )
        payload = json.dumps(
            {
                "csrf": claims.csrf_token,
                "exp": int(expires_at.timestamp()),
                "nonce": secrets.token_urlsafe(16),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=")
        signature = hmac.new(self._secret, encoded, hashlib.sha256).digest()
        token = b".".join(
            (encoded, base64.urlsafe_b64encode(signature).rstrip(b"="))
        ).decode()
        return token, claims

    def verify(self, token: str | None) -> SessionClaims | None:
        if not token:
            return None
        try:
            encoded, raw_signature = token.encode().split(b".", maxsplit=1)
            signature = _urlsafe_decode(raw_signature)
            expected = hmac.new(self._secret, encoded, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                return None
            payload = json.loads(_urlsafe_decode(encoded))
            expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
            csrf_token = payload["csrf"]
            if not isinstance(csrf_token, str) or expires_at <= datetime.now(UTC):
                return None
            return SessionClaims(csrf_token=csrf_token, expires_at=expires_at)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None


class LoginRateLimiter:
    """Bound repeated login attempts by client address."""

    def __init__(self, attempts: int = 8, window_seconds: int = 60) -> None:
        self._attempts = attempts
        self._window = window_seconds
        self._events: defaultdict[str, deque[float]] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        events = self._events[client]
        while events and now - events[0] > self._window:
            events.popleft()
        if len(events) >= self._attempts:
            return False
        events.append(now)
        return True


class SessionMiddleware(BaseHTTPMiddleware):
    """Protect API routes and enforce CSRF on authenticated mutations."""

    def __init__(
        self,
        app,
        *,
        settings: Settings,
        signer: SessionSigner | None,
    ) -> None:
        super().__init__(app)
        self._settings = settings
        self._signer = signer

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not self._settings.require_auth:
            return await call_next(request)
        if not request.url.path.startswith("/api"):
            return await call_next(request)
        if request.url.path in PUBLIC_API_PATHS:
            return await call_next(request)

        claims = self._signer.verify(request.cookies.get(SESSION_COOKIE))
        if claims is None:
            return JSONResponse(
                {"detail": "authentication required"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        request.state.session_claims = claims
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            csrf_token = request.headers.get(CSRF_HEADER)
            if not csrf_token or not hmac.compare_digest(csrf_token, claims.csrf_token):
                return JSONResponse(
                    {"detail": "invalid CSRF token"},
                    status_code=status.HTTP_403_FORBIDDEN,
                )
        return await call_next(request)


def create_session_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/session")
    limiter = LoginRateLimiter()

    @router.get("", response_model=SessionStatus)
    def get_session(request: Request) -> SessionStatus:
        if not settings.require_auth:
            return SessionStatus(auth_required=False, authenticated=True)
        claims = request.app.state.session_signer.verify(
            request.cookies.get(SESSION_COOKIE)
        )
        return SessionStatus(
            auth_required=True,
            authenticated=claims is not None,
            csrf_token=claims.csrf_token if claims else None,
            expires_at=claims.expires_at if claims else None,
        )

    @router.post("/login", response_model=SessionStatus)
    def login(
        payload: LoginRequest, request: Request, response: Response
    ) -> SessionStatus:
        if not settings.require_auth:
            return SessionStatus(auth_required=False, authenticated=True)
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(status_code=429, detail="too many login attempts")
        expected = settings.auth_token.get_secret_value()
        if not hmac.compare_digest(payload.token, expected):
            raise HTTPException(status_code=401, detail="invalid access token")
        session_token, claims = request.app.state.session_signer.issue()
        response.set_cookie(
            SESSION_COOKIE,
            session_token,
            max_age=settings.session_ttl_hours * 3600,
            secure=settings.cookie_secure,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return SessionStatus(
            auth_required=True,
            authenticated=True,
            csrf_token=claims.csrf_token,
            expires_at=claims.expires_at,
        )

    @router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(response: Response) -> None:
        response.delete_cookie(
            SESSION_COOKIE,
            secure=settings.cookie_secure,
            httponly=True,
            samesite="strict",
            path="/",
        )

    return router


def create_session_signer(settings: Settings) -> SessionSigner | None:
    if not settings.require_auth:
        return None
    return SessionSigner(
        settings.auth_token.get_secret_value(),
        timedelta(hours=settings.session_ttl_hours),
    )


def _urlsafe_decode(value: bytes) -> bytes:
    padding = b"=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
