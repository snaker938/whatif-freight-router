from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import ExperimentBundle, ExperimentBundleInput, ScenarioCompareRequest
from .settings import settings


def _experiments_dir() -> Path:
    path = Path(settings.out_dir) / "experiments"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path() -> Path:
    return _experiments_dir() / "index.json"


def _bundle_path(experiment_id: str) -> Path:
    return _experiments_dir() / f"{experiment_id}.json"


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _clean_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("experiment name cannot be empty")
    return cleaned


def _load_index_ids() -> list[str]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    ids = raw.get("ids")
    if not isinstance(ids, list):
        return []
    out: list[str] = []
    for item in ids:
        if not isinstance(item, str):
            continue
        try:
            out.append(str(uuid.UUID(item)))
        except ValueError:
            continue
    return out


def _save_index_ids(ids: list[str]) -> Path:
    path = _index_path()
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    path.write_text(json.dumps({"ids": deduped}, indent=2), encoding="utf-8")
    return path


def _load_bundle(experiment_id: str) -> ExperimentBundle:
    valid_id = str(uuid.UUID(experiment_id))
    path = _bundle_path(valid_id)
    if not path.exists():
        raise KeyError("experiment not found")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError("experiment file is invalid JSON") from e
    if not isinstance(raw, dict):
        raise ValueError("experiment file payload is invalid")
    return ExperimentBundle.model_validate(raw)


def list_experiments() -> list[ExperimentBundle]:
    out: list[ExperimentBundle] = []
    for experiment_id in _load_index_ids():
        try:
            out.append(_load_bundle(experiment_id))
        except Exception:
            continue
    out.sort(key=lambda item: item.updated_at, reverse=True)
    return out


def create_experiment(payload: ExperimentBundleInput) -> tuple[ExperimentBundle, Path]:
    experiment_id = str(uuid.uuid4())
    now = _utc_now_iso()
    bundle = ExperimentBundle(
        id=experiment_id,
        name=_clean_name(payload.name),
        description=(payload.description.strip() if payload.description else None),
        request=ScenarioCompareRequest.model_validate(payload.request.model_dump(mode="json")),
        created_at=now,
        updated_at=now,
    )

    path = _bundle_path(experiment_id)
    path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    ids = _load_index_ids()
    ids.insert(0, experiment_id)
    _save_index_ids(ids)
    return bundle, path


def get_experiment(experiment_id: str) -> ExperimentBundle:
    return _load_bundle(experiment_id)


def update_experiment(experiment_id: str, payload: ExperimentBundleInput) -> tuple[ExperimentBundle, Path]:
    existing = _load_bundle(experiment_id)
    updated = ExperimentBundle(
        id=existing.id,
        name=_clean_name(payload.name),
        description=(payload.description.strip() if payload.description else None),
        request=ScenarioCompareRequest.model_validate(payload.request.model_dump(mode="json")),
        created_at=existing.created_at,
        updated_at=_utc_now_iso(),
    )
    path = _bundle_path(existing.id)
    path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2), encoding="utf-8")
    ids = _load_index_ids()
    if existing.id not in ids:
        ids.insert(0, existing.id)
        _save_index_ids(ids)
    return updated, path


def delete_experiment(experiment_id: str) -> tuple[str, Path]:
    valid_id = str(uuid.UUID(experiment_id))
    path = _bundle_path(valid_id)
    if not path.exists():
        raise KeyError("experiment not found")
    path.unlink()
    ids = [item for item in _load_index_ids() if item != valid_id]
    index_path = _save_index_ids(ids)
    return valid_id, index_path
