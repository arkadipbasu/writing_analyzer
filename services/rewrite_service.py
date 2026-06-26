"""
services/rewrite_service.py
────────────────────────────
Text transformation services: rewrite, summarize, paraphrase,
synonym suggestion, keyword extraction, NER, language detection,
and translation.
"""

from __future__ import annotations

from utils.helpers import get_cache, get_config_value, truncate_text
from utils.logger import get_logger
from utils.prompts import (
    KEYWORD_SYSTEM, KEYWORD_USER,
    LANG_DETECT_SYSTEM, LANG_DETECT_USER,
    NER_SYSTEM, NER_USER,
    PARAPHRASE_SYSTEM, PARAPHRASE_USER,
    REWRITE_SYSTEM, REWRITE_STYLES, REWRITE_USER,
    SUMMARIZE_SYSTEM, SUMMARIZE_USER,
    SYNONYM_SYSTEM, SYNONYM_USER,
    TRANSLATE_SYSTEM, TRANSLATE_USER,
)
from services.groq_service import GroqService

logger = get_logger(__name__)


class RewriteService:
    """All text-transformation AI features."""

    def __init__(self, groq_service: GroqService) -> None:
        self._svc = groq_service
        self._cache = get_cache()

    # ── Rewrite ────────────────────────────────────────────────────────────────

    def rewrite(self, text: str, style: str) -> dict:
        """Rewrite *text* in the given *style*.

        Args:
            text: Selected text to rewrite.
            style: One of the keys in :data:`REWRITE_STYLES`.

        Returns:
            Dict with ``rewritten``, ``changes_made``, ``style_notes``.
        """
        text = text.strip()
        if not text:
            return {"rewritten": "", "changes_made": [], "style_notes": "", "success": False}

        style_desc = REWRITE_STYLES.get(style, "clear and professional")
        resp = self._svc.complete(
            system=REWRITE_SYSTEM,
            user=REWRITE_USER.format(text=truncate_text(text, 2000), style=style, style_description=style_desc),
            temperature=0.5,
            max_tokens=800,
            expect_json=True,
        )

        if resp.parsed_json:
            return {**resp.parsed_json, "success": True}
        # If JSON parse fails, treat the raw content as the rewrite
        return {"rewritten": resp.content or text, "changes_made": [], "style_notes": "", "success": resp.success}

    # ── Summarize ─────────────────────────────────────────────────────────────

    def summarize(self, text: str, mode: str = "medium") -> dict:
        """Summarize *text*.

        Args:
            text: Source text.
            mode: ``"short"``, ``"medium"``, or ``"detailed"``.

        Returns:
            Dict with ``summary``, ``key_points``, ``compression_ratio``.
        """
        text = text.strip()
        if not text:
            return _empty_summary()

        cached = self._cache.get("summarize", text, mode)
        if cached:
            return cached

        resp = self._svc.complete(
            system=SUMMARIZE_SYSTEM,
            user=SUMMARIZE_USER.format(text=truncate_text(text, 3000), mode=mode),
            temperature=0.3,
            max_tokens=600,
            expect_json=True,
        )

        result = _empty_summary()
        if resp.parsed_json:
            data = resp.parsed_json
            result = {
                "summary": data.get("summary", ""),
                "key_points": data.get("key_points", []),
                "word_count_original": data.get("word_count_original", len(text.split())),
                "word_count_summary": data.get("word_count_summary", 0),
                "compression_ratio": data.get("compression_ratio", 0.0),
                "success": True,
            }
        elif resp.content:
            result = {"summary": resp.content, "key_points": [], "word_count_original": len(text.split()), "word_count_summary": 0, "compression_ratio": 0.0, "success": True}

        self._cache.set("summarize", text, mode, result)
        return result

    # ── Paraphrase ────────────────────────────────────────────────────────────

    def paraphrase(self, text: str) -> dict:
        """Paraphrase *text* without changing its meaning.

        Returns:
            Dict with ``paraphrased``, ``alternatives``.
        """
        text = text.strip()
        if not text:
            return {"paraphrased": "", "alternatives": [], "success": False}

        resp = self._svc.complete(
            system=PARAPHRASE_SYSTEM,
            user=PARAPHRASE_USER.format(text=truncate_text(text, 2000)),
            temperature=0.4,
            max_tokens=600,
            expect_json=True,
        )

        if resp.parsed_json:
            return {**resp.parsed_json, "success": True}
        return {"paraphrased": resp.content or text, "alternatives": [], "success": resp.success}

    # ── Synonyms ──────────────────────────────────────────────────────────────

    def get_synonyms(self, word: str, context: str = "") -> dict:
        """Fetch synonym suggestions for *word*.

        Returns:
            Dict with synonym categories.
        """
        word = word.strip()
        if not word:
            return _empty_synonyms()

        cached = self._cache.get("synonyms", word, context[:100])
        if cached:
            return cached

        resp = self._svc.complete(
            system=SYNONYM_SYSTEM,
            user=SYNONYM_USER.format(word=word, context=truncate_text(context, 500)),
            temperature=0.3,
            max_tokens=250,
            expect_json=True,
        )

        result = _empty_synonyms()
        if resp.parsed_json:
            result = {**_empty_synonyms(), **resp.parsed_json, "success": True}

        self._cache.set("synonyms", word, context[:100], result)
        return result

    # ── Keywords ──────────────────────────────────────────────────────────────

    def extract_keywords(self, text: str) -> dict:
        """Extract keywords and topics from *text*.

        Returns:
            Dict with ``keywords`` list and ``topics`` list.
        """
        text = text.strip()
        if not text:
            return {"keywords": [], "topics": [], "success": False}

        cached = self._cache.get("keywords", text)
        if cached:
            return cached

        max_kw = get_config_value("features.keywords.max_keywords", 10)
        resp = self._svc.complete(
            system=KEYWORD_SYSTEM,
            user=KEYWORD_USER.format(text=truncate_text(text, 2000), max_keywords=max_kw),
            temperature=0.2,
            max_tokens=300,
            expect_json=True,
        )

        result = {"keywords": [], "topics": [], "success": False}
        if resp.parsed_json:
            result = {
                "keywords": resp.parsed_json.get("keywords", [])[:max_kw],
                "topics": resp.parsed_json.get("topics", []),
                "success": True,
            }

        self._cache.set("keywords", text, result)
        return result

    # ── NER ───────────────────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> dict:
        """Extract named entities from *text*.

        Returns:
            Dict with ``entities`` list and ``entity_count`` summary.
        """
        text = text.strip()
        if not text:
            return {"entities": [], "entity_count": {}, "success": False}

        cached = self._cache.get("ner", text)
        if cached:
            return cached

        resp = self._svc.complete(
            system=NER_SYSTEM,
            user=NER_USER.format(text=truncate_text(text, 2000)),
            temperature=0.1,
            max_tokens=600,
            expect_json=True,
        )

        result = {"entities": [], "entity_count": {}, "success": False}
        if resp.parsed_json:
            result = {
                "entities": resp.parsed_json.get("entities", []),
                "entity_count": resp.parsed_json.get("entity_count", {}),
                "success": True,
            }

        self._cache.set("ner", text, result)
        return result

    # ── Language Detection ────────────────────────────────────────────────────

    def detect_language(self, text: str) -> dict:
        """Detect the language of *text*.

        Returns:
            Dict with ``language``, ``language_code``, ``confidence``.
        """
        text = text.strip()
        if not text or len(text) < 5:
            return {"language": "Unknown", "language_code": "", "confidence": 0.0, "script": "", "success": False}

        cached = self._cache.get("lang_detect", text[:200])
        if cached:
            return cached

        resp = self._svc.complete(
            system=LANG_DETECT_SYSTEM,
            user=LANG_DETECT_USER.format(text=text[:500]),
            temperature=0.1,
            max_tokens=80,
            expect_json=True,
        )

        result = {"language": "Unknown", "language_code": "", "confidence": 0.0, "script": "", "success": False}
        if resp.parsed_json:
            result = {**resp.parsed_json, "success": True}

        self._cache.set("lang_detect", text[:200], result)
        return result

    # ── Translation ───────────────────────────────────────────────────────────

    def translate(self, text: str, target_language: str, source_language: str = "Auto") -> dict:
        """Translate *text* into *target_language*.

        Returns:
            Dict with ``translated``, ``confidence``, ``notes``.
        """
        text = text.strip()
        if not text:
            return {"translated": "", "confidence": 0.0, "notes": "", "success": False}

        resp = self._svc.complete(
            system=TRANSLATE_SYSTEM,
            user=TRANSLATE_USER.format(
                text=truncate_text(text, 2000),
                target_language=target_language,
                source_language=source_language,
            ),
            temperature=0.2,
            max_tokens=800,
            expect_json=True,
        )

        if resp.parsed_json:
            return {**resp.parsed_json, "success": True}
        return {"translated": resp.content or "", "confidence": 0.0, "notes": "", "success": resp.success}


# ── Defaults ──────────────────────────────────────────────────────────────────

def _empty_summary() -> dict:
    return {
        "summary": "",
        "key_points": [],
        "word_count_original": 0,
        "word_count_summary": 0,
        "compression_ratio": 0.0,
        "success": False,
    }


def _empty_synonyms() -> dict:
    return {
        "original": "",
        "synonyms": [],
        "simpler": [],
        "formal": [],
        "business": [],
        "academic": [],
        "context_aware": [],
        "success": False,
    }
