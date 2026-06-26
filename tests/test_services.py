"""
tests/test_services.py
───────────────────────
Unit tests for all AI services.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.groq_service import GroqService, LLMResponse
from services.autocomplete_service import AutocompleteService
from services.sentiment_service import SentimentService, IntentService
from services.grammar_service import GrammarService
from services.rewrite_service import RewriteService
from utils.helpers import (
    char_count,
    reading_time_minutes,
    safe_parse_json,
    sanitize_text,
    sentence_count,
    truncate_text,
    word_count,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_groq_service():
    """Return a GroqService with the _post method mocked."""
    svc = GroqService.__new__(GroqService)
    svc.groq_key = "gsk_test"
    svc.openai_key = None
    svc.model = "llama-3.3-70b-versatile"
    svc.temperature = 0.3
    svc.max_tokens = 256
    svc._timeout = 30
    svc._max_retries = 1
    svc._retry_delay = 0.0
    from services.groq_service import UsageStats
    svc.stats = UsageStats()
    return svc


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, success=True, model="test", provider="groq",
                       prompt_tokens=10, completion_tokens=20, total_tokens=30, latency_ms=50.0)


# ── Helper Tests ─────────────────────────────────────────────────────────────

class TestHelpers:
    def test_word_count_empty(self):
        assert word_count("") == 0

    def test_word_count_normal(self):
        assert word_count("hello world foo") == 3

    def test_char_count(self):
        assert char_count("hello world") == 10  # no spaces

    def test_sentence_count(self):
        assert sentence_count("Hello. World! How are you?") == 3

    def test_reading_time(self):
        # 200 words → 1 min
        text = " ".join(["word"] * 200)
        assert reading_time_minutes(text) == 1.0

    def test_truncate_text_short(self):
        assert truncate_text("hello", 100) == "hello"

    def test_truncate_text_long(self):
        text = "a " * 300
        result = truncate_text(text, 10)
        assert len(result) <= 12  # allow for "…"

    def test_sanitize_text(self):
        assert sanitize_text("  hello   world  ") == "hello world"

    def test_safe_parse_json_clean(self):
        raw = '{"key": "value", "number": 42}'
        result = safe_parse_json(raw)
        assert result == {"key": "value", "number": 42}

    def test_safe_parse_json_fenced(self):
        raw = '```json\n{"key": "value"}\n```'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_safe_parse_json_invalid(self):
        assert safe_parse_json("not json at all") is None


# ── GroqService Tests ─────────────────────────────────────────────────────────

class TestGroqService:
    def test_complete_success(self, mock_groq_service):
        mock_groq_service._post = MagicMock(
            return_value=_make_llm_response("Hello there!")
        )
        resp = mock_groq_service.complete(system="You are helpful.", user="Hi")
        assert resp.success
        assert resp.content == "Hello there!"

    def test_complete_json_parse(self, mock_groq_service):
        mock_groq_service._post = MagicMock(
            return_value=_make_llm_response('{"sentiment": "Positive", "confidence": 0.9}')
        )
        resp = mock_groq_service.complete(system="sys", user="user", expect_json=True)
        assert resp.parsed_json is not None
        assert resp.parsed_json["sentiment"] == "Positive"

    def test_complete_no_keys_returns_failure(self):
        svc = GroqService.__new__(GroqService)
        svc.groq_key = None
        svc.openai_key = None
        svc.model = "llama-3.3-70b-versatile"
        svc.temperature = 0.3
        svc.max_tokens = 256
        svc._timeout = 10
        svc._max_retries = 1
        svc._retry_delay = 0.0
        from services.groq_service import UsageStats
        svc.stats = UsageStats()

        resp = svc.complete(system="sys", user="user")
        assert not resp.success
        assert "API" in resp.error or "key" in resp.error.lower()


# ── AutocompleteService Tests ─────────────────────────────────────────────────

class TestAutocompleteService:
    def test_complete_short_text_returns_empty(self, mock_groq_service):
        svc = AutocompleteService(mock_groq_service)
        result = svc.complete("Hi")
        assert result == ""

    def test_complete_returns_string(self, mock_groq_service):
        mock_groq_service.complete = MagicMock(
            return_value=_make_llm_response("and we should proceed with caution.")
        )
        svc = AutocompleteService(mock_groq_service)
        result = svc.complete("The quick brown fox jumps over the lazy dog")
        assert isinstance(result, str)


# ── SentimentService Tests ────────────────────────────────────────────────────

class TestSentimentService:
    def test_analyze_success(self, mock_groq_service):
        payload = '{"sentiment": "Positive", "confidence": 0.92, "scores": {"positive": 0.92, "neutral": 0.05, "negative": 0.03}, "tone": "happy"}'
        mock_groq_service.complete = MagicMock(
            return_value=LLMResponse(
                content=payload, success=True,
                parsed_json={"sentiment": "Positive", "confidence": 0.92,
                             "scores": {"positive": 0.92, "neutral": 0.05, "negative": 0.03}, "tone": "happy"},
            )
        )
        svc = SentimentService(mock_groq_service)
        result = svc.analyze("I love this product, it's amazing!")
        assert result["sentiment"] in ("Positive", "Neutral", "Negative")

    def test_analyze_short_text(self, mock_groq_service):
        svc = SentimentService(mock_groq_service)
        result = svc.analyze("Hi")
        assert result["sentiment"] == "Neutral"
        assert not result["success"]


# ── IntentService Tests ───────────────────────────────────────────────────────

class TestIntentService:
    def test_detect_returns_known_intent(self, mock_groq_service):
        mock_groq_service.complete = MagicMock(
            return_value=LLMResponse(
                content='{"intent": "Question", "confidence": 0.95, "reasoning": "Asks a question"}',
                success=True,
                parsed_json={"intent": "Question", "confidence": 0.95, "reasoning": "Asks a question"},
            )
        )
        svc = IntentService(mock_groq_service)
        result = svc.detect("What is the capital of France?")
        assert result["intent"] in IntentService.INTENT_ICONS
        assert result["icon"] != ""

    def test_detect_short_text_returns_default(self, mock_groq_service):
        svc = IntentService(mock_groq_service)
        result = svc.detect("Hey")
        assert result["intent"] == "General"


# ── RewriteService Tests ──────────────────────────────────────────────────────

class TestRewriteService:
    def test_rewrite_empty_text(self, mock_groq_service):
        svc = RewriteService(mock_groq_service)
        result = svc.rewrite("", "Formal")
        assert not result["success"]

    def test_summarize_empty_text(self, mock_groq_service):
        svc = RewriteService(mock_groq_service)
        result = svc.summarize("")
        assert not result["success"]

    def test_paraphrase_returns_text(self, mock_groq_service):
        mock_groq_service.complete = MagicMock(
            return_value=LLMResponse(
                content='{"paraphrased": "The dog is quick.", "alternatives": [], "meaning_preserved": true}',
                success=True,
                parsed_json={"paraphrased": "The dog is quick.", "alternatives": [], "meaning_preserved": True},
            )
        )
        svc = RewriteService(mock_groq_service)
        result = svc.paraphrase("The quick dog.")
        assert result.get("success")
        assert "paraphrased" in result

    def test_detect_language(self, mock_groq_service):
        mock_groq_service.complete = MagicMock(
            return_value=LLMResponse(
                content='{"language": "English", "language_code": "en", "confidence": 0.99, "script": "Latin"}',
                success=True,
                parsed_json={"language": "English", "language_code": "en", "confidence": 0.99, "script": "Latin"},
            )
        )
        svc = RewriteService(mock_groq_service)
        result = svc.detect_language("Hello world, this is a test.")
        assert result.get("language") == "English"
