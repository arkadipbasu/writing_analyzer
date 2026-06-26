"""
utils/prompts.py
────────────────
Centralised prompt templates for every AI feature.

All prompts produce deterministic, structured JSON responses so that
downstream services can parse them reliably without brittle string
parsing.
"""

from __future__ import annotations


# ── Autocomplete ────────────────────────────────────────────────────────────

AUTOCOMPLETE_SYSTEM = """You are an intelligent writing assistant.
Complete the user's partial sentence naturally and fluently.
Return ONLY the completion text (the words that come after the user's input).
Do NOT repeat the user's input. Keep the completion concise and relevant.
Maximum 20 words."""

AUTOCOMPLETE_USER = """Complete this partial text naturally:
"{text}"

Return only the completion, nothing else."""


# ── Next Word Prediction ─────────────────────────────────────────────────────

NEXT_WORD_SYSTEM = """You are a next-word prediction engine.
Always respond with valid JSON only. No markdown, no explanation."""

NEXT_WORD_USER = """Given this text, predict the next word(s).
Text: "{text}"
Mode: {mode}

Return JSON:
{{
  "predictions": ["word1", "word2", "word3"],
  "confidence": [0.9, 0.7, 0.5]
}}"""


# ── Intent Detection ─────────────────────────────────────────────────────────

INTENT_SYSTEM = """You are a text intent classifier.
Always respond with valid JSON only. No markdown, no explanation."""

INTENT_USER = """Classify the intent of this text.
Text: "{text}"

Possible intents: Question, Greeting, Complaint, Request, Technical,
Travel, Medical, Finance, Programming, General

Return JSON:
{{
  "intent": "detected_intent",
  "confidence": 0.95,
  "reasoning": "brief explanation"
}}"""


# ── Sentiment Analysis ────────────────────────────────────────────────────────

SENTIMENT_SYSTEM = """You are a sentiment analysis engine.
Always respond with valid JSON only. No markdown, no explanation."""

SENTIMENT_USER = """Analyze the sentiment of this text.
Text: "{text}"

Return JSON:
{{
  "sentiment": "Positive|Neutral|Negative",
  "confidence": 0.92,
  "scores": {{
    "positive": 0.85,
    "neutral": 0.10,
    "negative": 0.05
  }},
  "tone": "brief tone description"
}}"""


# ── Synonym Suggestions ───────────────────────────────────────────────────────

SYNONYM_SYSTEM = """You are a vocabulary enhancement assistant.
Always respond with valid JSON only. No markdown, no explanation."""

SYNONYM_USER = """Provide synonym suggestions for the word "{word}" in this context:
Context: "{context}"

Return JSON:
{{
  "original": "{word}",
  "synonyms": ["word1", "word2", "word3"],
  "simpler": ["simpler1", "simpler2"],
  "formal": ["formal1", "formal2"],
  "business": ["business1", "business2"],
  "academic": ["academic1", "academic2"],
  "context_aware": ["contextual1", "contextual2"]
}}"""


# ── Grammar Check ────────────────────────────────────────────────────────────

GRAMMAR_SYSTEM = """You are a grammar and style checker.
Always respond with valid JSON only. No markdown, no explanation."""

GRAMMAR_USER = """Check this text for grammar, spelling, and punctuation errors.
Text: "{text}"

Return JSON:
{{
  "issues": [
    {{
      "type": "grammar|spelling|punctuation",
      "original": "incorrect text",
      "suggestion": "corrected text",
      "explanation": "brief reason",
      "position": 0
    }}
  ],
  "overall_quality": "poor|fair|good|excellent",
  "corrected_text": "full corrected version"
}}"""


# ── Rewrite Assistant ────────────────────────────────────────────────────────

REWRITE_SYSTEM = """You are a professional writing assistant.
Always respond with valid JSON only. No markdown, no explanation."""

REWRITE_USER = """Rewrite the following text in {style} style.
Text: "{text}"

Style: {style}
Style description: {style_description}

Return JSON:
{{
  "rewritten": "the rewritten text",
  "changes_made": ["change1", "change2"],
  "style_notes": "brief note about stylistic choices"
}}"""

REWRITE_STYLES = {
    "Formal": "professional and authoritative, suitable for official documents",
    "Casual": "relaxed and conversational, like talking to a friend",
    "Professional": "business-appropriate, clear, and polished",
    "Technical": "precise, structured, with domain-specific terminology",
    "Academic": "scholarly, with formal academic conventions and citations style",
    "Creative": "imaginative, expressive, with vivid language and metaphors",
}


# ── Summarization ────────────────────────────────────────────────────────────

SUMMARIZE_SYSTEM = """You are a text summarization expert.
Always respond with valid JSON only. No markdown, no explanation."""

SUMMARIZE_USER = """Summarize the following text.
Text: "{text}"
Mode: {mode}

Mode descriptions:
- short: 1-2 sentences, key point only
- medium: 3-5 sentences, main ideas
- detailed: full paragraph, comprehensive

Return JSON:
{{
  "summary": "the summary text",
  "key_points": ["point1", "point2", "point3"],
  "word_count_original": 0,
  "word_count_summary": 0,
  "compression_ratio": 0.0
}}"""


# ── Paraphrasing ─────────────────────────────────────────────────────────────

PARAPHRASE_SYSTEM = """You are a paraphrasing assistant.
Always respond with valid JSON only. No markdown, no explanation."""

PARAPHRASE_USER = """Paraphrase the following text while preserving its meaning.
Text: "{text}"

Return JSON:
{{
  "paraphrased": "the paraphrased text",
  "alternatives": ["alternative1", "alternative2"],
  "meaning_preserved": true
}}"""


# ── Keyword Extraction ────────────────────────────────────────────────────────

KEYWORD_SYSTEM = """You are a keyword extraction engine.
Always respond with valid JSON only. No markdown, no explanation."""

KEYWORD_USER = """Extract the most important keywords and keyphrases from this text.
Text: "{text}"
Max keywords: {max_keywords}

Return JSON:
{{
  "keywords": [
    {{"term": "keyword", "score": 0.95, "frequency": 2}},
    {{"term": "keyphrase", "score": 0.87, "frequency": 1}}
  ],
  "topics": ["topic1", "topic2"]
}}"""


# ── Named Entity Recognition ──────────────────────────────────────────────────

NER_SYSTEM = """You are a named entity recognition engine.
Always respond with valid JSON only. No markdown, no explanation."""

NER_USER = """Extract all named entities from this text.
Text: "{text}"

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, EMAIL, PHONE, CURRENCY, URL

Return JSON:
{{
  "entities": [
    {{
      "text": "entity text",
      "type": "PERSON",
      "start": 0,
      "end": 10,
      "confidence": 0.98
    }}
  ],
  "entity_count": {{
    "PERSON": 1,
    "ORGANIZATION": 0,
    "LOCATION": 0,
    "DATE": 0,
    "EMAIL": 0,
    "PHONE": 0,
    "CURRENCY": 0,
    "URL": 0
  }}
}}"""


# ── Language Detection ────────────────────────────────────────────────────────

LANG_DETECT_SYSTEM = """You are a language detection engine.
Always respond with valid JSON only. No markdown, no explanation."""

LANG_DETECT_USER = """Detect the language of this text.
Text: "{text}"

Return JSON:
{{
  "language": "English",
  "language_code": "en",
  "confidence": 0.99,
  "script": "Latin"
}}"""


# ── Translation ───────────────────────────────────────────────────────────────

TRANSLATE_SYSTEM = """You are a professional translator.
Always respond with valid JSON only. No markdown, no explanation."""

TRANSLATE_USER = """Translate the following text to {target_language}.
Text: "{text}"
Source language: {source_language}

Return JSON:
{{
  "translated": "translated text",
  "source_language": "{source_language}",
  "target_language": "{target_language}",
  "confidence": 0.97,
  "notes": "any translation notes"
}}"""


# ── Available Languages ───────────────────────────────────────────────────────

AVAILABLE_LANGUAGES = [
    "Arabic", "Chinese (Simplified)", "Chinese (Traditional)", "Dutch",
    "English", "French", "German", "Hindi", "Indonesian", "Italian",
    "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Spanish",
    "Swedish", "Thai", "Turkish", "Ukrainian", "Vietnamese",
]
