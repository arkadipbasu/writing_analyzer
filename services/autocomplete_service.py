"""
services/autocomplete_service.py
─────────────────────────────────
Real-time sentence autocomplete and next-word prediction.

Uses Groq for generation with a local cache to avoid re-querying
identical prefixes within the TTL window.
"""

from __future__ import annotations

from utils.helpers import get_cache, get_config_value, safe_parse_json, truncate_text
from utils.logger import get_logger
from utils.prompts import (
    AUTOCOMPLETE_SYSTEM,
    AUTOCOMPLETE_USER,
    NEXT_WORD_SYSTEM,
    NEXT_WORD_USER,
)
from services.groq_service import GroqService

logger = get_logger(__name__)

# Prediction modes surfaced in the UI
PREDICTION_MODES = ["word", "3words", "sentence"]


class AutocompleteService:
    """Provides sentence completion and next-word prediction.

    Args:
        groq_service: Shared :class:`GroqService` instance.
    """

    def __init__(self, groq_service: GroqService) -> None:
        self._svc = groq_service
        self._cache = get_cache()
        self._min_chars: int = get_config_value("features.autocomplete.min_chars", 10)
        self._autocomplete_tokens: int = get_config_value(
            "inference.autocomplete_max_tokens", 60
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def complete(self, text: str) -> str:
        """Return the predicted completion for *text*.

        Args:
            text: The text the user has typed so far.

        Returns:
            Completion string (words after the cursor), or empty string.
        """
        text = text.strip()
        if len(text) < self._min_chars:
            return ""

        cached = self._cache.get("autocomplete", text)
        if cached is not None:
            logger.debug("Autocomplete cache hit")
            return cached

        resp = self._svc.complete(
            system=AUTOCOMPLETE_SYSTEM,
            user=AUTOCOMPLETE_USER.format(text=truncate_text(text, 800)),
            temperature=0.2,
            max_tokens=self._autocomplete_tokens,
        )

        if not resp.success or not resp.content:
            return ""

        completion = resp.content.strip()
        # Guard: strip any accidental repetition of the input
        if completion.lower().startswith(text[-20:].lower().strip()):
            completion = completion[len(text) :].strip()

        self._cache.set("autocomplete", text, completion)
        return completion

    def predict_next_words(self, text: str, mode: str = "sentence") -> dict:
        """Predict the next word(s) after *text*.

        Args:
            text: Current text.
            mode: One of ``"word"``, ``"3words"``, ``"sentence"``.

        Returns:
            Dict with ``predictions`` list and ``confidence`` list.
        """
        text = text.strip()
        if len(text) < self._min_chars:
            return {"predictions": [], "confidence": []}

        cached = self._cache.get("next_word", text, mode)
        if cached is not None:
            return cached

        resp = self._svc.complete(
            system=NEXT_WORD_SYSTEM,
            user=NEXT_WORD_USER.format(text=truncate_text(text, 800), mode=mode),
            temperature=0.3,
            max_tokens=80,
            expect_json=True,
        )

        result: dict = {"predictions": [], "confidence": []}
        if resp.parsed_json:
            result = {
                "predictions": resp.parsed_json.get("predictions", [])[:5],
                "confidence": resp.parsed_json.get("confidence", []),
            }

        self._cache.set("next_word", text, mode, result)
        return result
