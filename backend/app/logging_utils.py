from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pythonjsonlogger import jsonlogger

from .settings import settings


def _parse_level(name: str) -> int:
    level = logging.getLevelName(name.upper())
    return level if isinstance(level, int) else logging.INFO


def get_logger() -> logging.Logger:
    logger = logging.getLogger("freight_router")

    # Prevent duplicate handlers (common with reloaders)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(_parse_level(settings.log_level))
    logger.propagate = False

    out_dir = Path(settings.out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = jsonlogger.JsonFormatter()

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / "api.log.jsonl", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    return logger


LOGGER = get_logger()


def log_event(event: str, **fields: Any) -> None:
    # Structured: event is message + a top-level key
    LOGGER.info(event, extra={"event": event, **fields})
