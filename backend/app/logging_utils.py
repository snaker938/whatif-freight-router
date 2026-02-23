from __future__ import annotations

import logging
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from pythonjsonlogger import jsonlogger

from .settings import settings


def _parse_level(name: str) -> int:
    level = logging.getLevelName(name.upper())
    return level if isinstance(level, int) else logging.INFO


def _resolve_log_dir(configured_out_dir: str) -> Path | None:
    candidates = (
        Path(configured_out_dir) / "logs",
        Path.cwd() / "out" / "logs",
        Path(gettempdir()) / "whatif-freight-router" / "logs",
    )
    for log_dir in candidates:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            probe = log_dir / ".writetest"
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return log_dir
        except OSError:
            continue
    return None


def get_logger() -> logging.Logger:
    logger = logging.getLogger("freight_router")

    # Prevent duplicate handlers (common with reloaders)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(_parse_level(settings.log_level))
    logger.propagate = False

    formatter = jsonlogger.JsonFormatter()

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_dir = _resolve_log_dir(settings.out_dir)
    if log_dir is not None:
        try:
            fh = logging.FileHandler(log_dir / "api.log.jsonl", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except OSError:
            pass

    logger._configured = True  # type: ignore[attr-defined]
    return logger


LOGGER: logging.Logger | None = None


def log_event(event: str, **fields: Any) -> None:
    global LOGGER
    if LOGGER is None:
        LOGGER = get_logger()
    # Structured: event is message + a top-level key
    LOGGER.info(event, extra={"event": event, **fields})
