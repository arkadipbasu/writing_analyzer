"""
services/sentiment_service.py
──────────────────────────────
Real-time sentiment analysis and intent detection services.
"""

from __future__ import annotations

from utils.helpers import get_cache, get_config_value, truncate_text
from utils.logger import get_logger
from utils.prompts import (
    INTENT_SYSTEM, INTENT_USER,
    SENTIMENT_SYSTEM, SENTIMENT_USER,
)
from services.groq_service import GroqService

logger = get_logger(__name__)


class SentimentService:
    """Classifies text sentiment as Positive / Neutral / Negative."""

    def __init__(self, groq_service: GroqService) -> None:
        self._svc = groq_service
        self._cache = get_cache()
        self._min_chars: int = get_config_value("features.sentiment.min_chars", 10)

    def analyze(self, text: str) -> dict:
        """Run sentiment analysis on *text*.

        Returns:
            Dict with keys: ``sentiment``, ``confidence``, ``scores``, ``tone``.
        """
        text = text.strip()
        if len(text) < self._min_chars:
            return _empty_sentiment()

        cached = self._cache.get("sentiment", text)
        if cached is not None:
            return cached

        resp = self._svc.complete(
            system=SENTIMENT_SYSTEM,
            user=SENTIMENT_USER.format(text=truncate_text(text, 1000)),
            temperature=0.1,
            max_tokens=150,
            expect_json=True,
        )

        result = _empty_sentiment()
        if resp.parsed_json:
            result = {
                "sentiment": resp.parsed_json.get("sentiment", "Neutral"),
                "confidence": float(resp.parsed_json.get("confidence", 0.0)),
                "scores": resp.parsed_json.get("scores", {"positive": 0.33, "neutral": 0.34, "negative": 0.33}),
                "tone": resp.parsed_json.get("tone", ""),
                "success": True,
            }

        self._cache.set("sentiment", text, result)
        return result


class IntentService:
    """Detects the primary intent of the user's text."""

    INTENT_ICONS = {
        "Question": "❓",
        "Greeting": "👋",
        "Complaint": "😤",
        "Request": "🙏",
        "Technical": "⚙️",
        "Travel": "✈️",
        "Medical": "🏥",
        "Finance": "💰",
        "Programming": "💻",
        "General": "💬",
    }

    def __init__(self, groq_service: GroqService) -> None:
        self._svc = groq_service
        self._cache = get_cache()
        self._min_chars: int = get_config_value("features.intent.min_chars", 10)

    def detect(self, text: str) -> dict:
        """Detect the primary intent in *text*.

        Returns:
            Dict with keys: ``intent``, ``confidence``, ``reasoning``, ``icon``.
        """
        text = text.strip()
        if len(text) < self._min_chars:
            return _empty_intent()

        cached = self._cache.get("intent", text)
        if cached is not None:
            return cached

        resp = self._svc.complete(
            system=INTENT_SYSTEM,
            user=INTENT_USER.format(text=truncate_text(text, 1000)),
            temperature=0.1,
            max_tokens=120,
            expect_json=True,
        )

        result = _empty_intent()
        if resp.parsed_json:
            intent = resp.parsed_json.get("intent", "General")
            result = {
                "intent": intent,
                "confidence": float(resp.parsed_json.get("confidence", 0.0)),
                "reasoning": resp.parsed_json.get("reasoning", ""),
                "icon": self.INTENT_ICONS.get(intent, "💬"),
                "success": True,
            }

        self._cache.set("intent", text, result)
        return result


# ── Defaults ──────────────────────────────────────────────────────────────────

def _empty_sentiment() -> dict:
    return {
        "sentiment": "Neutral",
        "confidence": 0.0,
        "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
        "tone": "",
        "success": False,
    }


def _empty_intent() -> dict:
    return {
        "intent": "General",
        "confidence": 0.0,
        "reasoning": "",
        "icon": "💬",
        "success": False,
    }
