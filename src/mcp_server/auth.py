"""Bearer token authentication middleware for the MCP HTTP transport."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.config import settings


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validate Authorization: Bearer <token> on every HTTP request.

    Requests to the health/ping endpoint are exempted.

    Add to FastAPI / Starlette app:
        app.add_middleware(BearerAuthMiddleware)
    """

    EXEMPT_PATHS: set[str] = {"/health", "/ping", "/", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"detail": "Missing or invalid Authorization header."},
                status_code=401,
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token != settings.mcp_auth_token:
            return JSONResponse(
                {"detail": "Invalid bearer token."},
                status_code=403,
            )

        return await call_next(request)


def verify_token(token: str) -> bool:
    """Simple token verification helper.

    Args:
        token: Raw token string (without 'Bearer ' prefix).

    Returns:
        True if valid, False otherwise.
    """
    return token == settings.mcp_auth_token
