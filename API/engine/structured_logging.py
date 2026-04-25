"""Structured JSON logging helpers for engine workflows."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_LEVEL = "INFO"


def _timestamp() -> str:
    """Return an ISO8601 timestamp with timezone information."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _json_ready(value: Any) -> Any:
    """Convert common Python objects to JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value, key=str)]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_ready(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def build_event_record(
    event: str,
    *,
    level: str = DEFAULT_LOG_LEVEL,
    **fields: Any,
) -> dict[str, Any]:
    """Build a structured log record with required common fields."""
    record: dict[str, Any] = {
        "timestamp": _timestamp(),
        "level": level.upper(),
        "event": event,
    }
    record.update(_json_ready(fields))
    return record


def write_json(path: str | Path, record: dict[str, Any]) -> None:
    """Write one JSON object to a file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(record), file, ensure_ascii=False, sort_keys=False)
        file.write("\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a JSON Lines file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        json.dump(_json_ready(record), file, ensure_ascii=False, sort_keys=False)
        file.write("\n")


class JsonlFormatter(logging.Formatter):
    """Format stdlib log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", None) or "log"
        fields = getattr(record, "event_fields", {}) or {}
        payload = build_event_record(event, level=record.levelname, **fields)
        message = record.getMessage()
        if event == "log" and message:
            payload["message"] = message
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=False)


def _log_level(config: dict[str, Any]) -> int:
    logging_config = config.get("logging", {}) or {}
    level_name = str(logging_config.get("level", DEFAULT_LOG_LEVEL)).upper()
    level = getattr(logging, level_name, logging.INFO)
    return level if isinstance(level, int) else logging.INFO


def resolve_run_dir(config: dict[str, Any], workflow: str = "run") -> Path:
    """Resolve or create the run directory used for structured log artifacts."""
    logging_config = config.setdefault("logging", {})
    runtime_config = config.setdefault("runtime", {})

    configured_run_dir = runtime_config.get("run_dir") or logging_config.get("run_dir")
    if configured_run_dir:
        run_dir = Path(str(configured_run_dir))
    else:
        logs_config = config.get("logs", {}) or {}
        base_log_dir = Path(
            str(logging_config.get("log_dir", logs_config.get("log_dir", "runs")))
        )
        run_id = datetime.now().strftime(f"{workflow}-%Y%m%d-%H%M%S")
        run_dir = base_log_dir / run_id

    runtime_config["run_dir"] = str(run_dir)
    logging_config["run_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_json_logging(config: dict[str, Any], workflow: str = "run") -> Path:
    """Configure stdlib logging to emit JSON lines into the run directory."""
    run_dir = resolve_run_dir(config, workflow)
    debug_path = run_dir / "debug.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(_log_level(config))
    formatter = JsonlFormatter()

    file_handler = logging.FileHandler(debug_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging_config = config.get("logging", {}) or {}
    if bool(logging_config.get("console", False)):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    setattr(root_logger, "_quantization_json_run_dir", str(run_dir))
    return run_dir


def ensure_json_logging(config: dict[str, Any], workflow: str = "run") -> Path:
    """Configure JSON logging if it is not already active for this run."""
    run_dir = resolve_run_dir(config, workflow)
    root_logger = logging.getLogger()
    if getattr(root_logger, "_quantization_json_run_dir", None) != str(run_dir):
        configure_json_logging(config, workflow)
    return run_dir


def log_event(
    event: str,
    *,
    level: int | str = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured event through the configured stdlib logger."""
    log_level = getattr(logging, str(level).upper(), level)
    if not isinstance(log_level, int):
        log_level = logging.INFO
    logging.getLogger("quantization").log(
        log_level,
        "",
        extra={"event": event, "event_fields": _json_ready(fields)},
    )
