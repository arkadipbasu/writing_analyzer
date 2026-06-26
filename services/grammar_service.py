"""
services/grammar_service.py
────────────────────────────
Grammar, spelling, and punctuation checking.

Primary: language_tool_python (offline, rule-based, fast)
Fallback: Groq LLM for nuanced suggestions
"""

from __future__ import annotations

from utils.helpers import get_cache, get_config_value, truncate_text
from utils.logger import get_logger
from utils.prompts import GRAMMAR_SYSTEM, GRAMMAR_USER
from services.groq_service import GroqService

logger = get_logger(__name__)


class GrammarService:
    """Checks grammar, spelling, and punctuation in user text.

    Attempts to use LanguageTool locally; falls back to the LLM.
    """

    def __init__(self, groq_service: GroqService) -> None:
        self._svc = groq_service
        self._cache = get_cache()
        self._use_lt: bool = get_config_value("features.grammar.use_language_tool", True)
        self._lt = self._init_language_tool()

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(self, text: str) -> dict:
        """Check *text* for grammar issues.

        Returns:
            Dict with ``issues`` list, ``corrected_text``, ``overall_quality``.
        """
        text = text.strip()
        if not text:
            return _empty_grammar()

        cached = self._cache.get("grammar", text)
        if cached is not None:
            return cached

        # Try LanguageTool first (fast, offline)
        if self._lt:
            result = self._check_with_lt(text)
            if result.get("success"):
                self._cache.set("grammar", text, result)
                return result

        # Fall back to LLM
        result = self._check_with_llm(text)
        self._cache.set("grammar", text, result)
        return result

    # ── Private ────────────────────────────────────────────────────────────────

    def _init_language_tool(self):
        """Initialise LanguageTool; return None if unavailable."""
        if not self._use_lt:
            return None
        try:
            import language_tool_python  # type: ignore
            lt = language_tool_python.LanguageTool("en-US")
            logger.info("LanguageTool initialised successfully.")
            return lt
        except Exception as exc:
            logger.warning(f"LanguageTool unavailable: {exc}. Using LLM fallback.")
            return None

    def _check_with_lt(self, text: str) -> dict:
        """Run LanguageTool grammar check."""
        try:
            matches = self._lt.check(text)
            issues = []
            for m in matches:
                if m.replacements:
                    issues.append(
                        {
                            "type": _lt_category(m.ruleId),
                            "original": text[m.offset : m.offset + m.errorLength],
                            "suggestion": m.replacements[0] if m.replacements else "",
                            "explanation": m.message,
                            "position": m.offset,
                            "length": m.errorLength,
                        }
                    )

            corrected = self._lt.correct(text)
            quality = _quality_from_issue_count(len(issues), len(text.split()))

            return {
                "issues": issues,
                "corrected_text": corrected,
                "overall_quality": quality,
                "issue_count": len(issues),
                "source": "language_tool",
                "success": True,
            }
        except Exception as exc:
            logger.error(f"LanguageTool check failed: {exc}")
            return {"success": False}

    def _check_with_llm(self, text: str) -> dict:
        """Use the LLM to detect grammar issues."""
        resp = self._svc.complete(
            system=GRAMMAR_SYSTEM,
            user=GRAMMAR_USER.format(text=truncate_text(text, 1500)),
            temperature=0.1,
            max_tokens=600,
            expect_json=True,
        )

        if not resp.parsed_json:
            return _empty_grammar()

        data = resp.parsed_json
        return {
            "issues": data.get("issues", []),
            "corrected_text": data.get("corrected_text", text),
            "overall_quality": data.get("overall_quality", "good"),
            "issue_count": len(data.get("issues", [])),
            "source": "llm",
            "success": True,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lt_category(rule_id: str) -> str:
    rule_id = rule_id.upper()
    if "SPELL" in rule_id or "TYPO" in rule_id:
        return "spelling"
    if "PUNCT" in rule_id or "COMMA" in rule_id:
        return "punctuation"
    return "grammar"


def _quality_from_issue_count(issues: int, words: int) -> str:
    if words == 0:
        return "good"
    ratio = issues / max(words, 1)
    if ratio == 0:
        return "excellent"
    if ratio < 0.02:
        return "good"
    if ratio < 0.05:
        return "fair"
    return "poor"


def _empty_grammar() -> dict:
    return {
        "issues": [],
        "corrected_text": "",
        "overall_quality": "good",
        "issue_count": 0,
        "source": "none",
        "success": False,
    }
