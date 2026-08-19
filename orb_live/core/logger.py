"""
core/logger.py — structlog JSON-lines setup.

Call configure_logging() once at process startup (e.g. in main.py or the
strategy runner).  All subsequent `structlog.get_logger()` calls will emit
machine-readable JSON lines to stdout and, optionally, a log file.
"""

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import structlog


#: Written to when no explicit log_file is given. One file per session date;
#: these are the only durable record of entry rejections, margin rates and
#: fill timing, so they are on by default rather than opt-in.
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"


def default_log_file(for_date: Optional[date] = None) -> Path:
    d = for_date or date.today()
    return DEFAULT_LOG_DIR / f"orb_live_{d.isoformat()}.jsonl"


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_to_file: bool = True,
    to_file: bool = True,
) -> None:
    """
    Set up structlog + standard logging.

    - Console: human-readable coloured output (dev) or plain JSON (prod).
    - File: JSON lines, one event per line. On by default — pass to_file=False
      to suppress (tests). Defaults to data/logs/orb_live_<date>.jsonl.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is None and to_file:
        log_file = default_log_file()

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(fh)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
        force=True,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
