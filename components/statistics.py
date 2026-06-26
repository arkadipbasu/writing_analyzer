"""
components/statistics.py
─────────────────────────
Live analysis panel: sentiment, intent, language detection,
text statistics, and NER summary rendered in the right column.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from utils.helpers import char_count, reading_time_minutes, sentence_count, word_count
from utils.logger import get_logger

logger = get_logger(__name__)


def render_analysis_panel(
    text: str,
    sentiment_result: dict,
    intent_result: dict,
    lang_result: dict,
    settings: dict,
) -> None:
    """Render the live analysis right-side panel.

    Args:
        text: Current editor text.
        sentiment_result: Output from SentimentService.
        intent_result: Output from IntentService.
        lang_result: Output from language detection.
        settings: Active UI settings.
    """
    st.markdown("### 📊 Live Analysis")

    # ── Text Statistics ────────────────────────────────────────────────────────
    with st.container():
        st.markdown("#### 📐 Text Stats")
        col1, col2 = st.columns(2)
        wc = word_count(text)
        cc = char_count(text)
        sc = sentence_count(text)
        rt = reading_time_minutes(text)
        with col1:
            st.metric("Words", wc)
            st.metric("Sentences", sc)
        with col2:
            st.metric("Characters", cc)
            st.metric("Read time", f"{rt} min")

    st.divider()

    # ── Sentiment ─────────────────────────────────────────────────────────────
    if settings.get("feat_sentiment") and sentiment_result.get("success"):
        _render_sentiment(sentiment_result)
        st.divider()

    # ── Intent ────────────────────────────────────────────────────────────────
    if settings.get("feat_intent") and intent_result.get("success"):
        _render_intent(intent_result)
        st.divider()

    # ── Language ──────────────────────────────────────────────────────────────
    if settings.get("feat_lang") and lang_result.get("success"):
        _render_language(lang_result)
        st.divider()

    # ── Placeholder when analysis hasn't run yet ───────────────────────────────
    total_results = sum([
        sentiment_result.get("success", False),
        intent_result.get("success", False),
        lang_result.get("success", False),
    ])
    if total_results == 0 and len(text.strip()) > 0:
        st.info("Keep typing… analysis will appear shortly.")
    elif not text.strip():
        st.markdown(
            "<div style='text-align:center;padding:30px 10px;color:#555;'>"
            "<div style='font-size:2rem;'>✍️</div>"
            "<div style='margin-top:8px;font-size:.85rem;'>Start typing to see live analysis</div>"
            "</div>",
            unsafe_allow_html=True,
        )


# ── Private Renderers ─────────────────────────────────────────────────────────

def _render_sentiment(result: dict) -> None:
    """Render sentiment gauge and scores."""
    st.markdown("#### 🎭 Sentiment")
    sentiment = result.get("sentiment", "Neutral")
    confidence = result.get("confidence", 0.0)
    scores = result.get("scores", {})
    tone = result.get("tone", "")

    COLOUR_MAP = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}
    ICON_MAP = {"Positive": "😊", "Neutral": "😐", "Negative": "😟"}
    colour = COLOUR_MAP.get(sentiment, "#6b7280")
    icon = ICON_MAP.get(sentiment, "😐")

    st.markdown(
        f"<div style='text-align:center;padding:10px;border-radius:8px;"
        f"background:{colour}22;border:1px solid {colour}44;'>"
        f"<div style='font-size:1.8rem;'>{icon}</div>"
        f"<div style='font-size:1rem;font-weight:700;color:{colour};margin-top:4px;'>{sentiment}</div>"
        f"<div style='font-size:.78rem;color:#888;'>{confidence:.0%} confidence</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if scores:
        _render_score_bars(scores, {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"})

    if tone:
        st.caption(f"Tone: {tone}")


def _render_intent(result: dict) -> None:
    """Render intent chip and confidence."""
    st.markdown("#### 🎯 Intent")
    intent = result.get("intent", "General")
    confidence = result.get("confidence", 0.0)
    icon = result.get("icon", "💬")
    reasoning = result.get("reasoning", "")

    st.markdown(
        f"<div style='padding:8px 12px;border-radius:8px;"
        f"background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.3);'>"
        f"<span style='font-size:1.2rem;'>{icon}</span> "
        f"<span style='font-weight:700;font-size:.95rem;'>{intent}</span>"
        f"<span style='float:right;font-size:.8rem;color:#888;margin-top:2px;'>{confidence:.0%}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if reasoning:
        st.caption(reasoning)


def _render_language(result: dict) -> None:
    """Render detected language badge."""
    st.markdown("#### 🌐 Language")
    lang = result.get("language", "Unknown")
    code = result.get("language_code", "")
    confidence = result.get("confidence", 0.0)
    script = result.get("script", "")

    flag_hint = f"({code.upper()})" if code else ""
    st.markdown(
        f"<div style='padding:8px 12px;border-radius:8px;"
        f"background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);'>"
        f"<span style='font-weight:700;'>{lang}</span> {flag_hint}"
        f"<span style='float:right;font-size:.8rem;color:#888;'>{confidence:.0%}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if script:
        st.caption(f"Script: {script}")


def _render_score_bars(scores: dict, colour_map: dict) -> None:
    """Render a set of labelled score bars."""
    for label, value in scores.items():
        pct = int(float(value) * 100)
        colour = colour_map.get(label.lower(), "#6b7280")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
            f"<span style='min-width:60px;font-size:.78rem;color:#aaa;text-transform:capitalize;'>{label}</span>"
            f"<div style='flex:1;background:#1e1e2e;border-radius:4px;height:6px;'>"
            f"<div style='width:{pct}%;background:{colour};border-radius:4px;height:6px;transition:width .3s;'></div>"
            f"</div><span style='font-size:.75rem;color:#666;min-width:30px;text-align:right;'>{pct}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
