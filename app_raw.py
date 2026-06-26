"""
app.py
───────
AI Writing Assistant — main Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
import streamlit as st

# ── Page config must be FIRST Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="AI Writing Assistant",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Services ──────────────────────────────────────────────────────────────────
from services.groq_service import GroqService
from services.autocomplete_service import AutocompleteService
from services.sentiment_service import SentimentService, IntentService
from services.grammar_service import GrammarService
from services.rewrite_service import RewriteService

# ── Components ────────────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.editor import render_editor
from components.statistics import render_analysis_panel

# ── Utilities ─────────────────────────────────────────────────────────────────
from utils.helpers import get_config_value
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Global CSS ────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Base layout ──────────────────────────────────────────── */
        .main .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

        /* ── Editor textarea ──────────────────────────────────────── */
        textarea[data-testid="stTextArea"] {
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
            font-size: 0.95rem !important;
            line-height: 1.65 !important;
            border: 1.5px solid rgba(99,102,241,.35) !important;
            border-radius: 10px !important;
            background: rgba(15,15,25,.6) !important;
            padding: 14px !important;
            resize: vertical !important;
            transition: border-color .2s !important;
        }
        textarea[data-testid="stTextArea"]:focus {
            border-color: rgba(99,102,241,.8) !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important;
        }

        /* ── Sidebar ──────────────────────────────────────────────── */
        [data-testid="stSidebar"] { background: rgba(10,10,20,.9) !important; }
        [data-testid="stSidebar"] .stExpander { border: 1px solid rgba(255,255,255,.07) !important; border-radius: 8px !important; }

        /* ── Metric cards ─────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: rgba(99,102,241,.07);
            border: 1px solid rgba(99,102,241,.18);
            border-radius: 8px;
            padding: 10px 14px;
        }

        /* ── Tabs ─────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; font-size: .82rem; padding: 6px 12px; }

        /* ── Buttons ──────────────────────────────────────────────── */
        .stButton > button {
            border-radius: 7px !important;
            font-weight: 600 !important;
            font-size: .84rem !important;
            transition: all .18s !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            border: none !important;
        }

        /* ── Code chips ───────────────────────────────────────────── */
        code { border-radius: 5px !important; }

        /* ── Scrollbar ────────────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(99,102,241,.4); border-radius: 3px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Session State Initialisation ──────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "editor_text": "",
        "editor_changed": False,
        "autocomplete_suggestion": "",
        "sentiment_result": {},
        "intent_result": {},
        "lang_result": {},
        "grammar_result": None,
        "last_analysis_text": "",
        "last_autocomplete_text": "",
        "analysis_debounce_ts": 0.0,
        "autocomplete_debounce_ts": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Service Initialisation (cached) ──────────────────────────────────────────

@st.cache_resource
def _init_services() -> dict:
    logger.info("Initialising AI services…")
    groq_svc = GroqService()
    return {
        "groq": groq_svc,
        "autocomplete": AutocompleteService(groq_svc),
        "sentiment": SentimentService(groq_svc),
        "intent": IntentService(groq_svc),
        "grammar": GrammarService(groq_svc),
        "rewrite": RewriteService(groq_svc),
    }


# ── Main Application ──────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    _init_session()

    services = _init_services()
    groq_svc: GroqService = services["groq"]
    autocomplete_svc: AutocompleteService = services["autocomplete"]
    sentiment_svc: SentimentService = services["sentiment"]
    intent_svc: IntentService = services["intent"]
    grammar_svc: GrammarService = services["grammar"]
    rewrite_svc: RewriteService = services["rewrite"]

    # ── Sidebar ────────────────────────────────────────────────────────────────
    settings = render_sidebar(groq_svc)

    # ── Main layout: editor (left) + analysis panel (right) ───────────────────
    editor_col, analysis_col = st.columns([3, 1], gap="medium")

    with editor_col:
        current_text = render_editor(rewrite_svc, autocomplete_svc, settings)

    with analysis_col:
        render_analysis_panel(
            text=current_text,
            sentiment_result=st.session_state["sentiment_result"],
            intent_result=st.session_state["intent_result"],
            lang_result=st.session_state["lang_result"],
            settings=settings,
        )

    # ── Debounced background analysis ─────────────────────────────────────────
    _run_background_analysis(
        text=current_text,
        settings=settings,
        autocomplete_svc=autocomplete_svc,
        sentiment_svc=sentiment_svc,
        intent_svc=intent_svc,
        grammar_svc=grammar_svc,
        rewrite_svc=rewrite_svc,
    )

    # ── Grammar check button (below editor) ───────────────────────────────────
    _render_grammar_button(current_text, grammar_svc, settings)


def _run_background_analysis(
    text: str,
    settings: dict,
    autocomplete_svc: AutocompleteService,
    sentiment_svc: SentimentService,
    intent_svc: IntentService,
    grammar_svc: GrammarService,
    rewrite_svc: RewriteService,
) -> None:
    """Run analysis when text changes, with debouncing to avoid flooding the API."""
    now = time.time()
    debounce_ms = float(get_config_value("features.autocomplete.debounce_ms", 400)) / 1000.0
    text_changed = text != st.session_state["last_analysis_text"]

    if not text_changed:
        return
    if (now - st.session_state["analysis_debounce_ts"]) < debounce_ms:
        return

    st.session_state["analysis_debounce_ts"] = now
    st.session_state["last_analysis_text"] = text

    if len(text.strip()) < 5:
        st.session_state["sentiment_result"] = {}
        st.session_state["intent_result"] = {}
        st.session_state["lang_result"] = {}
        st.session_state["autocomplete_suggestion"] = ""
        return

    # Autocomplete
    if settings.get("feat_autocomplete"):
        ac_text = text
        if ac_text != st.session_state["last_autocomplete_text"]:
            completion = autocomplete_svc.complete(ac_text)
            st.session_state["autocomplete_suggestion"] = completion
            st.session_state["last_autocomplete_text"] = ac_text

    # Sentiment
    if settings.get("feat_sentiment"):
        st.session_state["sentiment_result"] = sentiment_svc.analyze(text)

    # Intent
    if settings.get("feat_intent"):
        st.session_state["intent_result"] = intent_svc.detect(text)

    # Language detection
    if settings.get("feat_lang"):
        st.session_state["lang_result"] = rewrite_svc.detect_language(text)

    # Force re-render so analysis panel updates
    st.rerun()


def _render_grammar_button(text: str, grammar_svc: GrammarService, settings: dict) -> None:
    """Render grammar check button below the editor."""
    if not settings.get("feat_grammar") or not text.strip():
        return

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        if st.button("🔍 Check Grammar", key="run_grammar", type="secondary"):
            with st.spinner("Checking grammar…"):
                result = grammar_svc.check(text)
            st.session_state["grammar_result"] = result
            st.rerun()
    with col_clear:
        if st.session_state.get("grammar_result"):
            if st.button("Clear grammar results", key="clear_grammar"):
                st.session_state["grammar_result"] = None
                st.rerun()


if __name__ == "__main__":
    main()
