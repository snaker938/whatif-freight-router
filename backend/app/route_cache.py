from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._process_cache import ProcessGlobalCacheStore
from .settings import settings

ROUTE_CACHE_KEY_SCHEMA_VERSION = "route_cache_key_v1"

RouteCachePayload = tuple[list[dict[str, Any]], list[str], int] | tuple[
    list[dict[str, Any]],
    list[str],
    int,
    dict[str, Any],
]


def _normalize_cache_token(value: Any | None, *, default: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _normalize_cache_slug(value: Any | None, *, default: str = "unknown") -> str:
    text = _normalize_cache_token(value, default=default).lower()
    for token in (" ", "-", "/", ":", "."):
        text = text.replace(token, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or default


def _normalize_cache_bool_token(value: Any | None) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "supported"}:
        return "true"
    if text in {"0", "false", "no", "n", "unsupported"}:
        return "false"
    return "unknown"


def _normalize_cache_extra(extra: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not extra:
        return ()
    return tuple(
        sorted(
            (str(key).strip() or "unknown", _normalize_cache_token(value))
            for key, value in extra.items()
        )
    )


@dataclass(frozen=True)
class RouteCacheKeyState:
    artifact_kind: str
    run_id: str = "unknown"
    lane_id: str = "unknown"
    variant_id: str = "default"
    cache_mode: str = "cold"
    schema_version: str = ROUTE_CACHE_KEY_SCHEMA_VERSION
    support_flag: bool | None = None
    support_status: str | None = None
    fidelity_class: str | None = None
    terminal_type: str | None = None
    seed: int | None = None
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "run_id": self.run_id,
            "lane_id": self.lane_id,
            "variant_id": self.variant_id,
            "cache_mode": self.cache_mode,
            "schema_version": self.schema_version,
            "support_flag": self.support_flag,
            "support_status": self.support_status,
            "fidelity_class": self.fidelity_class,
            "terminal_type": self.terminal_type,
            "seed": self.seed,
            "extra": {key: value for key, value in self.extra},
        }

    def cache_key(self) -> str:
        return build_route_cache_key(
            artifact_kind=self.artifact_kind,
            run_id=self.run_id,
            lane_id=self.lane_id,
            variant_id=self.variant_id,
            cache_mode=self.cache_mode,
            schema_version=self.schema_version,
            support_flag=self.support_flag,
            support_status=self.support_status,
            fidelity_class=self.fidelity_class,
            terminal_type=self.terminal_type,
            seed=self.seed,
            extra=dict(self.extra),
        )


def build_route_cache_key_state(
    *,
    artifact_kind: str,
    run_id: str | None = None,
    lane_id: str | None = None,
    variant_id: str | None = None,
    cache_mode: str | None = None,
    schema_version: str | None = None,
    support_flag: bool | None = None,
    support_status: str | None = None,
    fidelity_class: str | None = None,
    terminal_type: str | None = None,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RouteCacheKeyState:
    return RouteCacheKeyState(
        artifact_kind=_normalize_cache_token(artifact_kind),
        run_id=_normalize_cache_token(run_id),
        lane_id=_normalize_cache_token(lane_id),
        variant_id=_normalize_cache_token(variant_id, default="default"),
        cache_mode=_normalize_cache_token(cache_mode, default="cold"),
        schema_version=_normalize_cache_token(schema_version, default=ROUTE_CACHE_KEY_SCHEMA_VERSION),
        support_flag=support_flag,
        support_status=_normalize_cache_slug(support_status) if support_status is not None else None,
        fidelity_class=_normalize_cache_slug(fidelity_class) if fidelity_class is not None else None,
        terminal_type=_normalize_cache_slug(terminal_type) if terminal_type is not None else None,
        seed=int(seed) if isinstance(seed, int) else None,
        extra=_normalize_cache_extra(extra),
    )


def build_route_cache_key(
    *,
    artifact_kind: str,
    run_id: str | None = None,
    lane_id: str | None = None,
    variant_id: str | None = None,
    cache_mode: str | None = None,
    schema_version: str | None = None,
    support_flag: bool | None = None,
    support_status: str | None = None,
    fidelity_class: str | None = None,
    terminal_type: str | None = None,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    state = build_route_cache_key_state(
        artifact_kind=artifact_kind,
        run_id=run_id,
        lane_id=lane_id,
        variant_id=variant_id,
        cache_mode=cache_mode,
        schema_version=schema_version,
        support_flag=support_flag,
        support_status=support_status,
        fidelity_class=fidelity_class,
        terminal_type=terminal_type,
        seed=seed,
        extra=extra,
    )
    parts = [
        f"schema={state.schema_version}",
        f"artifact={state.artifact_kind}",
        f"run={state.run_id}",
        f"lane={state.lane_id}",
        f"variant={state.variant_id}",
        f"mode={state.cache_mode}",
        f"support={_normalize_cache_bool_token(state.support_flag)}",
        f"support_status={_normalize_cache_token(state.support_status)}",
        f"fidelity={_normalize_cache_token(state.fidelity_class)}",
        f"terminal={_normalize_cache_token(state.terminal_type)}",
        f"seed={state.seed if state.seed is not None else 'unknown'}",
    ]
    parts.extend(f"extra.{key}={value}" for key, value in state.extra)
    return "|".join(parts)


class RouteCacheStore(ProcessGlobalCacheStore[RouteCachePayload]):
    pass


ROUTE_CACHE = RouteCacheStore(
    ttl_s=settings.route_cache_ttl_s,
    max_entries=settings.route_cache_max_entries,
    max_estimated_bytes=settings.route_cache_max_estimated_bytes,
)
HOT_RERUN_ROUTE_CACHE_CHECKPOINT = RouteCacheStore(
    ttl_s=settings.route_cache_ttl_s,
    max_entries=settings.route_cache_max_entries,
    max_estimated_bytes=settings.route_cache_max_estimated_bytes,
)


def get_cached_routes(
    key: str,
) -> RouteCachePayload | None:
    return ROUTE_CACHE.get(key)


def set_cached_routes(
    key: str,
    value: RouteCachePayload,
) -> bool:
    return ROUTE_CACHE.set(key, value)


def clear_route_cache() -> int:
    return ROUTE_CACHE.clear()


def checkpoint_route_cache() -> int:
    return HOT_RERUN_ROUTE_CACHE_CHECKPOINT.import_items(ROUTE_CACHE.export_items(), clear_first=False)


def restore_checkpointed_route_cache(*, clear_first: bool = False) -> int:
    return ROUTE_CACHE.import_items(
        HOT_RERUN_ROUTE_CACHE_CHECKPOINT.export_items(),
        clear_first=clear_first,
    )


def clear_route_cache_checkpoint() -> int:
    return HOT_RERUN_ROUTE_CACHE_CHECKPOINT.clear()


def route_cache_stats() -> dict[str, int]:
    return ROUTE_CACHE.snapshot()


def route_cache_checkpoint_stats() -> dict[str, int]:
    return HOT_RERUN_ROUTE_CACHE_CHECKPOINT.snapshot()
