from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

from .settings import settings

SIGNATURE_ALGORITHM = "HMAC-SHA256"


def _canonical_payload_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_payload(payload: Any, *, secret: str | None = None) -> str:
    key = (secret or settings.manifest_signing_secret).encode("utf-8")
    msg = _canonical_payload_bytes(payload)
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_payload_signature(
    payload: Any,
    signature: str,
    *,
    secret: str | None = None,
) -> tuple[bool, str]:
    expected = sign_payload(payload, secret=secret)
    valid = hmac.compare_digest(expected, signature.strip().lower())
    return valid, expected


def build_signature_metadata(payload: Any) -> dict[str, str]:
    return {
        "algorithm": SIGNATURE_ALGORITHM,
        "signed_at": datetime.now(UTC).isoformat(),
        "signature": sign_payload(payload),
    }
