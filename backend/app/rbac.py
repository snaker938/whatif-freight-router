from __future__ import annotations

from typing import Literal

from fastapi import HTTPException, Request

from .settings import settings

Role = Literal["public", "user", "admin"]


def _token_from_request(request: Request) -> str | None:
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    token = request.headers.get("x-api-token", "").strip()
    return token or None


def _resolve_role(token: str | None) -> Role:
    if token is None:
        return "public"
    if token == settings.rbac_admin_token:
        return "admin"
    if token == settings.rbac_user_token:
        return "user"
    raise HTTPException(status_code=401, detail="invalid api token")


def require_role(request: Request, required: Role) -> None:
    if not settings.rbac_enabled or required == "public":
        return

    token = _token_from_request(request)
    if token is None:
        raise HTTPException(status_code=401, detail="missing api token")

    actual = _resolve_role(token)
    if required == "user":
        return
    if required == "admin" and actual != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
