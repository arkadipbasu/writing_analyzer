"""
components/sidebar.py
──────────────────────
Streamlit sidebar: API configuration, model settings, feature toggles,
and session statistics.
"""

from __future__ import annotations

import streamlit as st

from services.groq_service import GroqService
from utils.helpers import get_config_value
from utils.logger import get_logger
from utils.prompts import AVAILABLE_LANGUAGES, REWRITE_STYLES

logger = get_logger(__name__)

GROQ_MODELS = get_config_value("models.groq.available", [])
GROQ_MODEL_IDS = [m["id"] for m in GROQ_MODELS]
GROQ_MODEL_NAMES = {m["id"]: m["name"] for m in GROQ_MODELS}


def render_sidebar(groq_service: GroqService) -> dict:
    """Render the sidebar and return the current settings dict.

    Args:
        groq_service: The active :class:`GroqService` instance.

    Returns:
        Dict of current UI settings.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:12px 0 8px 0;">
              <span style="font-size:1.7rem;">✍️</span>
              <div style="font-size:1.1rem;font-weight:700;letter-spacing:.03em;margin-top:4px;">
                AI Writing Assistant
              </div>
              <div style="font-size:.75rem;color:#888;margin-top:2px;">v1.0.0 · Groq + OpenAI</div>
              <div style="font-size:.75rem;color:#4e90bf;margin-top:2px;">arkadipbasu.github.io</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── API Status ─────────────────────────────────────────────────────────
        with st.expander("🔑 API Configuration", expanded=True):
            _render_api_status(groq_service)

        st.divider()

        # ── Model & Inference ─────────────────────────────────────────────────
        with st.expander("🤖 Model Settings", expanded=True):
            settings = _render_model_settings(groq_service)

        st.divider()

        # ── Feature Toggles ────────────────────────────────────────────────────
        with st.expander("🎛️ Features", expanded=False):
            feature_settings = _render_feature_toggles()

        st.divider()

        # ── Theme ──────────────────────────────────────────────────────────────
        with st.expander("🎨 Appearance", expanded=False):
            theme = _render_theme()

        st.divider()

        # ── Session Statistics ─────────────────────────────────────────────────
        with st.expander("📊 Session Statistics", expanded=False):
            _render_stats(groq_service)

        # Footer
        st.markdown(
            "<div style='text-align:center;font-size:.7rem;color:#555;padding-top:10px;'>"
            "Powered by Groq · Built with Streamlit"
            "</div>",
            unsafe_allow_html=True,
        )

    return {**settings, **feature_settings, "theme": theme}


# ── Private Renderers ─────────────────────────────────────────────────────────

def _render_api_status(groq_service: GroqService) -> None:
    """Show API key status for Groq and OpenAI."""
    groq_ok = bool(groq_service.groq_key)
    oai_ok = bool(groq_service.openai_key)

    def _badge(ok: bool, label: str) -> str:
        colour, icon = ("#22c55e", "✓") if ok else ("#ef4444", "✗")
        status = "Connected" if ok else "Not set"
        return (
            f"<div style='display:flex;align-items:center;gap:6px;margin:4px 0;'>"
            f"<span style='color:{colour};font-weight:700;font-size:.9rem;'>{icon}</span>"
            f"<span style='font-size:.85rem;'>{label}</span>"
            f"<span style='margin-left:auto;font-size:.75rem;color:{colour};'>{status}</span>"
            f"</div>"
        )

    st.markdown(_badge(groq_ok, "Groq API"), unsafe_allow_html=True)
    st.markdown(_badge(oai_ok, "OpenAI (fallback)"), unsafe_allow_html=True)

    if not groq_ok:
        st.error(
            "GROQ_API_KEY is not configured. Please add it to `.streamlit/secrets.toml`."
        )


def _render_model_settings(groq_service: GroqService) -> dict:
    """Render model selector, temperature, max tokens."""
    default_idx = (
        GROQ_MODEL_IDS.index(groq_service.model)
        if groq_service.model in GROQ_MODEL_IDS
        else 0
    )

    selected_model = st.selectbox(
        "Model",
        options=GROQ_MODEL_IDS,
        format_func=lambda m: GROQ_MODEL_NAMES.get(m, m),
        index=default_idx,
        key="model_selector",
    )
    if selected_model != groq_service.model:
        groq_service.set_model(selected_model)

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(get_config_value("inference.temperature", 0.3)),
        step=0.05,
        key="temperature_slider",
        help="Lower = more focused; Higher = more creative",
    )
    groq_service.set_temperature(temperature)

    max_tokens = st.slider(
        "Max Tokens",
        min_value=64,
        max_value=2048,
        value=int(get_config_value("inference.max_tokens", 256)),
        step=64,
        key="max_tokens_slider",
    )
    groq_service.set_max_tokens(max_tokens)

    prediction_mode = st.selectbox(
        "Next Word Prediction",
        options=["word", "3words", "sentence"],
        format_func=lambda x: {
            "word": "Next Word",
            "3words": "Next 3 Words",
            "sentence": "Complete Sentence",
        }[x],
        key="prediction_mode",
    )

    return {
        "model": selected_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prediction_mode": prediction_mode,
    }


def _render_feature_toggles() -> dict:
    """Render toggles for individual AI features."""
    st.markdown(
        "<div style='font-size:.82rem;color:#aaa;margin-bottom:6px;'>Toggle features on/off</div>",
        unsafe_allow_html=True,
    )

    autocomplete = st.toggle("Autocomplete", value=True, key="feat_autocomplete")
    sentiment = st.toggle("Sentiment Analysis", value=True, key="feat_sentiment")
    intent = st.toggle("Intent Detection", value=True, key="feat_intent")
    grammar = st.toggle("Grammar Check", value=True, key="feat_grammar")
    keywords = st.toggle("Keyword Extraction", value=True, key="feat_keywords")
    ner = st.toggle("Named Entity Recognition", value=False, key="feat_ner")
    lang_detect = st.toggle("Language Detection", value=True, key="feat_lang")

    return {
        "feat_autocomplete": autocomplete,
        "feat_sentiment": sentiment,
        "feat_intent": intent,
        "feat_grammar": grammar,
        "feat_keywords": keywords,
        "feat_ner": ner,
        "feat_lang": lang_detect,
    }


def _render_theme() -> str:
    theme = st.radio(
        "UI Theme",
        options=["Dark", "Light"],
        index=0,
        horizontal=True,
        key="ui_theme",
    )
    return theme.lower()


def _render_stats(groq_service: GroqService) -> None:
    """Display session-level usage statistics."""
    stats = groq_service.stats

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Requests", stats.total_requests)
        st.metric("Tokens Used", f"{stats.total_tokens:,}")
    with col2:
        st.metric("Avg Latency", f"{stats.avg_latency_ms:.0f} ms")
        st.metric("Errors", stats.errors)

    if stats.total_requests > 0:
        groq_pct = int(stats.groq_requests / stats.total_requests * 100)
        st.progress(groq_pct / 100, text=f"Groq: {groq_pct}% | OpenAI: {100-groq_pct}%")

    if st.button("🗑️ Clear Cache", use_container_width=True):
        from utils.helpers import get_cache

        get_cache().clear()
        st.toast("Cache cleared.")
