"""
utils/logger.py
───────────────
Centralised logging using Loguru.

All modules import `get_logger` and call:
    logger = get_logger(__name__)
"""

import sys
from pathlib import Path

from loguru import logger as _logger


def get_logger(name: str = "ai_writer"):
    """Return a named Loguru logger bound with context.

    Args:
        name: Dotted module name, e.g. ``services.groq_service``.

    Returns:
        A Loguru logger instance with ``name`` bound.
    """
    _configure_root()
    return _logger.bind(module=name)


# ── internal ────────────────────────────────────────────────────────────────

_configured = False


def _configure_root() -> None:
    """One-time setup for the root Loguru logger."""
    global _configured
    if _configured:
        return

    _logger.remove()

    # Console sink ─ colourised, human-readable
    _logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[module]}</cyan> | "
            "<level>{message}</level>"
        ),
        level="DEBUG",
        colorize=True,
    )

    # File sink ─ rotating, machine-readable
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    _logger.add(
        log_dir / "ai_writer_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[module]} | {message}",
        enqueue=True,
    )

    _configured = True
