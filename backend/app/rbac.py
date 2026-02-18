from __future__ import annotations

from typing import Literal

from fastapi import Request

Role = Literal["public", "user", "admin"]


def require_role(_request: Request, _required: Role) -> None:
    # RBAC is disabled; keep this shim to avoid changing endpoint signatures.
    return
