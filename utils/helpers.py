"""
utils/helpers.py
────────────────
Shared utilities: config loading, simple LRU cache wrapper,
text helpers, and timing utilities.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)

# Load .env on import
load_dotenv()


# ── Configuration ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    """Load and cache the YAML configuration file.

    Args:
        path: Relative path to the config YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"Config file not found at {path}, using defaults.")
        return {}
    with config_path.open("r") as fh:
        data = yaml.safe_load(fh)
    logger.info(f"Config loaded from {path}")
    return data or {}


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Retrieve a nested config value using dot notation.

    Args:
        key_path: Dot-separated path, e.g. ``"inference.temperature"``.
        default: Value to return if the key is not found.

    Returns:
        The config value or *default*.
    """
    config = load_config()
    keys = key_path.split(".")
    node: Any = config
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, None)
        if node is None:
            return default
    return node


# ── Simple In-Memory Cache ────────────────────────────────────────────────────

class SimpleCache:
    """Thread-safe, TTL-based in-memory key-value cache."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, *args: Any) -> str:
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, *args: Any) -> Any | None:
        key = self._make_key(*args)
        if key in self._store:
            value, ts = self._store[key]
            if time.time() - ts < self._ttl:
                return value
            del self._store[key]
        return None

    def set(self, *args_and_value: Any) -> None:
        *args, value = args_and_value
        key = self._make_key(*args)
        if len(self._store) >= self._max_size:
            # Evict oldest entry
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, time.time())

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# Global cache instance
_cache = SimpleCache(
    ttl_seconds=get_config_value("cache.ttl_seconds", 300),
    max_size=get_config_value("cache.max_size", 100),
)


def get_cache() -> SimpleCache:
    """Return the global cache instance."""
    return _cache


# ── Text Utilities ────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    """Return the number of words in *text*."""
    return len(text.split()) if text.strip() else 0


def char_count(text: str) -> int:
    """Return the number of non-whitespace characters."""
    return len(text.replace(" ", ""))


def sentence_count(text: str) -> int:
    """Approximate sentence count by splitting on terminal punctuation."""
    import re
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def reading_time_minutes(text: str, wpm: int = 200) -> float:
    """Estimate reading time in minutes."""
    return round(word_count(text) / wpm, 1)


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate text to *max_chars*, preserving whole words."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] + "…" if last_space > 0 else truncated


def sanitize_text(text: str) -> str:
    """Strip leading/trailing whitespace and normalise internal spaces."""
    import re
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ── API Key Helpers ───────────────────────────────────────────────────────────

def get_groq_api_key() -> str | None:
    """Return the Groq API key from environment."""
    return os.getenv("GROQ_API_KEY")


def get_openai_api_key() -> str | None:
    """Return the OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")


def validate_api_key(key: str | None, prefix: str = "gsk_") -> bool:
    """Basic API key format validation.

    Args:
        key: The API key string.
        prefix: Expected key prefix.

    Returns:
        True if the key looks valid.
    """
    if not key:
        return False
    return len(key) > 20


# ── Timing ────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager for measuring elapsed time in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# ── JSON Parsing ──────────────────────────────────────────────────────────────

def safe_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse JSON from a potentially noisy LLM response.

    Strips common markdown fences before parsing.

    Args:
        text: Raw LLM output.

    Returns:
        Parsed dict or None on failure.
    """
    import re

    # Strip ```json ... ``` fences
    clean = re.sub(r"```json\s*", "", text)
    clean = re.sub(r"```\s*", "", clean)
    clean = clean.strip()

    # Extract first JSON object
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        clean = match.group(0)

    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse failed: {exc} | raw={text[:200]}")
        return None
