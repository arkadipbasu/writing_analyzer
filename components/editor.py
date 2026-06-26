"""
components/editor.py
─────────────────────
Main text editor panel with autocomplete suggestion display,
text action toolbar (rewrite / summarize / paraphrase / translate),
synonym popup logic, and grammar issue highlights.
"""

from __future__ import annotations

import streamlit as st

from services.autocomplete_service import AutocompleteService
from services.groq_service import GroqService
from services.rewrite_service import RewriteService
from utils.helpers import char_count, reading_time_minutes, sentence_count, word_count
from utils.logger import get_logger
from utils.prompts import AVAILABLE_LANGUAGES, REWRITE_STYLES

logger = get_logger(__name__)


def render_editor(
    rewrite_svc: RewriteService,
    autocomplete_svc: AutocompleteService,
    settings: dict,
) -> str:
    """Render the main writing editor and return the current text.

    Args:
        rewrite_svc: Rewrite/transform service.
        autocomplete_svc: Autocomplete service.
        settings: Active UI settings from the sidebar.

    Returns:
        Current editor text content.
    """
    # ── Editor header ──────────────────────────────────────────────────────────
    col_title, col_wc = st.columns([3, 1])
    with col_title:
        st.markdown("### ✍️ Writing Editor")
    with col_wc:
        text = st.session_state.get("editor_text", "")
        wc = word_count(text)
        st.markdown(
            f"<div style='text-align:right;padding-top:8px;font-size:.82rem;color:#888;'>"
            f"{wc} words · {char_count(text)} chars · {reading_time_minutes(text)} min read"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Autocomplete suggestion banner ─────────────────────────────────────────
    if settings.get("feat_autocomplete") and st.session_state.get("autocomplete_suggestion"):
        suggestion = st.session_state["autocomplete_suggestion"]
        ghost_col, accept_col, dismiss_col = st.columns([5, 1, 1])
        with ghost_col:
            st.markdown(
                f"<div style='padding:6px 10px;border-radius:6px;background:rgba(99,102,241,.12);"
                f"border:1px solid rgba(99,102,241,.3);font-size:.88rem;color:#a5b4fc;'>"
                f"💡 <em>{suggestion[:120]}{'…' if len(suggestion) > 120 else ''}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with accept_col:
            if st.button("Accept ↵", key="accept_suggestion", use_container_width=True):
                current = st.session_state.get("editor_text", "")
                st.session_state["editor_text"] = current.rstrip() + " " + suggestion
                st.session_state["autocomplete_suggestion"] = ""
                st.rerun()
        with dismiss_col:
            if st.button("✕", key="dismiss_suggestion", use_container_width=True):
                st.session_state["autocomplete_suggestion"] = ""
                st.rerun()

    # ── Main text area ─────────────────────────────────────────────────────────
    editor_text = st.text_area(
        label="editor",
        value=st.session_state.get("editor_text", ""),
        height=int(get_editor_height()),
        placeholder="Start typing your content here… AI suggestions will appear as you type.",
        label_visibility="collapsed",
        key="editor_textarea",
    )

    # Sync state
    if editor_text != st.session_state.get("editor_text", ""):
        st.session_state["editor_text"] = editor_text
        st.session_state["editor_changed"] = True
    else:
        st.session_state["editor_changed"] = False

    # ── Action toolbar ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='margin-top:4px;margin-bottom:8px;font-size:.78rem;color:#666;'>"
        "Select text → use the tools below to transform it</div>",
        unsafe_allow_html=True,
    )
    _render_action_toolbar(editor_text, rewrite_svc, settings)

    # ── Grammar issues ─────────────────────────────────────────────────────────
    if st.session_state.get("grammar_result"):
        _render_grammar_panel(editor_text)

    return editor_text


def _render_action_toolbar(text: str, rewrite_svc: RewriteService, settings: dict) -> None:
    """Horizontal toolbar with transform actions."""
    tabs = st.tabs(["🔄 Rewrite", "📝 Summarize", "🔀 Paraphrase", "🌐 Translate", "🔍 Synonyms", "📌 Keywords", "👤 Entities"])

    with tabs[0]:
        _render_rewrite(text, rewrite_svc)

    with tabs[1]:
        _render_summarize(text, rewrite_svc)

    with tabs[2]:
        _render_paraphrase(text, rewrite_svc)

    with tabs[3]:
        _render_translate(text, rewrite_svc)

    with tabs[4]:
        _render_synonyms(text, rewrite_svc)

    with tabs[5]:
        _render_keywords(text, rewrite_svc)

    with tabs[6]:
        _render_ner(text, rewrite_svc)


def _render_rewrite(text: str, svc: RewriteService) -> None:
    if not text.strip():
        st.info("Type or paste some text in the editor first.")
        return

    col_style, col_btn = st.columns([2, 1])
    with col_style:
        style = st.selectbox("Style", list(REWRITE_STYLES.keys()), key="rewrite_style_sel", label_visibility="collapsed")
    with col_btn:
        run = st.button("Rewrite", key="btn_rewrite", use_container_width=True, type="primary")

    if run:
        with st.spinner(f"Rewriting in {style} style…"):
            result = svc.rewrite(text, style)
        if result.get("success"):
            st.success("Rewritten text:")
            st.text_area("Rewrite output", value=result["rewritten"], height=140, key="rewrite_out")
            if result.get("changes_made"):
                st.markdown("**Changes:** " + " · ".join(result["changes_made"]))
            if st.button("Use this version", key="use_rewrite"):
                st.session_state["editor_text"] = result["rewritten"]
                st.rerun()
        else:
            st.error("Rewrite failed. Please try again.")


def _render_summarize(text: str, svc: RewriteService) -> None:
    if not text.strip():
        st.info("Type or paste some text in the editor first.")
        return

    col_mode, col_btn = st.columns([2, 1])
    with col_mode:
        mode = st.selectbox("Mode", ["short", "medium", "detailed"], key="sum_mode", label_visibility="collapsed",
                            format_func=lambda x: x.capitalize())
    with col_btn:
        run = st.button("Summarize", key="btn_summarize", use_container_width=True, type="primary")

    if run:
        with st.spinner("Summarizing…"):
            result = svc.summarize(text, mode)
        if result.get("success") and result.get("summary"):
            st.success(result["summary"])
            if result.get("key_points"):
                st.markdown("**Key Points:**")
                for pt in result["key_points"]:
                    st.markdown(f"- {pt}")
            ratio = result.get("compression_ratio", 0)
            if ratio:
                st.caption(f"Compressed to {ratio:.0%} of original")
        else:
            st.error("Summarization failed.")


def _render_paraphrase(text: str, svc: RewriteService) -> None:
    if not text.strip():
        st.info("Type or paste some text in the editor first.")
        return

    if st.button("Paraphrase", key="btn_paraphrase", type="primary"):
        with st.spinner("Paraphrasing…"):
            result = svc.paraphrase(text)
        if result.get("success"):
            st.success(result["paraphrased"])
            if result.get("alternatives"):
                st.markdown("**Alternatives:**")
                for alt in result["alternatives"]:
                    st.markdown(f"> {alt}")
            if st.button("Use this version", key="use_paraphrase"):
                st.session_state["editor_text"] = result["paraphrased"]
                st.rerun()
        else:
            st.error("Paraphrasing failed.")


def _render_translate(text: str, svc: RewriteService) -> None:
    if not text.strip():
        st.info("Type or paste some text in the editor first.")
        return

    col_lang, col_btn = st.columns([2, 1])
    with col_lang:
        lang = st.selectbox("Target language", AVAILABLE_LANGUAGES, key="translate_lang", label_visibility="collapsed")
    with col_btn:
        run = st.button("Translate", key="btn_translate", use_container_width=True, type="primary")

    if run:
        with st.spinner(f"Translating to {lang}…"):
            result = svc.translate(text, lang)
        if result.get("success") and result.get("translated"):
            st.success(result["translated"])
            if result.get("notes"):
                st.caption(f"Note: {result['notes']}")
        else:
            st.error("Translation failed.")


def _render_synonyms(text: str, svc: RewriteService) -> None:
    col_word, col_btn = st.columns([2, 1])
    with col_word:
        word = st.text_input("Word to look up", placeholder="Enter a word…", key="syn_word", label_visibility="collapsed")
    with col_btn:
        run = st.button("Get Synonyms", key="btn_synonyms", use_container_width=True, type="primary")

    if run and word:
        with st.spinner("Fetching synonyms…"):
            result = svc.get_synonyms(word, context=text)

        if result.get("success"):
            cats = [
                ("Synonyms", result.get("synonyms", [])),
                ("Simpler", result.get("simpler", [])),
                ("Formal", result.get("formal", [])),
                ("Business", result.get("business", [])),
                ("Academic", result.get("academic", [])),
                ("Context-aware", result.get("context_aware", [])),
            ]
            for label, words in cats:
                if words:
                    chips = " ".join(
                        f"<code style='cursor:pointer;margin:2px;padding:3px 7px;border-radius:12px;"
                        f"background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.3);'>{w}</code>"
                        for w in words
                    )
                    st.markdown(f"**{label}:** {chips}", unsafe_allow_html=True)
        else:
            st.error("Could not fetch synonyms.")
    elif run:
        st.warning("Please enter a word.")


def _render_keywords(text: str, svc: RewriteService) -> None:
    if not text.strip():
        st.info("Type or paste some text in the editor first.")
        return

    if st.button("Extract Keywords", key="btn_keywords", type="primary"):
        with st.spinner("Extracting keywords…"):
            result = svc.extract_keywords(text)

        if result.get("success") and result.get("keywords"):
            kws = result["keywords"]
            # Sort by score desc
            kws_sorted = sorted(kws, key=lambda k: k.get("score", 0), reverse=True)
            for kw in kws_sorted:
                score = kw.get("score", 0)
                bar_width = int(score * 100)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin:3px 0;'>"
                    f"<span style='min-width:140px;font-size:.85rem;'>{kw['term']}</span>"
                    f"<div style='flex:1;background:#1e1e2e;border-radius:4px;height:8px;'>"
                    f"<div style='width:{bar_width}%;background:#6366f1;border-radius:4px;height:8px;'></div>"
                    f"</div><span style='font-size:.75rem;color:#888;'>{score:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if result.get("topics"):
                st.markdown("**Topics:** " + " · ".join(result["topics"]))
        else:
            st.error("Keyword extraction failed.")


def _render_ner(text: str, svc: RewriteService) -> None:
    if not text.strip():
        st.info("Type or paste some text in the editor first.")
        return

    if st.button("Extract Entities", key="btn_ner", type="primary"):
        with st.spinner("Extracting entities…"):
            result = svc.extract_entities(text)

        if result.get("success") and result.get("entities"):
            ENTITY_COLOURS = {
                "PERSON": "#f59e0b",
                "ORGANIZATION": "#6366f1",
                "LOCATION": "#10b981",
                "DATE": "#3b82f6",
                "EMAIL": "#ec4899",
                "PHONE": "#8b5cf6",
                "CURRENCY": "#14b8a6",
                "URL": "#f97316",
            }
            for ent in result["entities"]:
                etype = ent.get("type", "OTHER")
                colour = ENTITY_COLOURS.get(etype, "#6b7280")
                st.markdown(
                    f"<span style='background:{colour}22;border:1px solid {colour}55;"
                    f"border-radius:4px;padding:2px 8px;margin:2px;display:inline-block;"
                    f"font-size:.82rem;'>"
                    f"<b style='color:{colour};'>{etype}</b> {ent.get('text', '')}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("")
            count = result.get("entity_count", {})
            if count:
                totals = {k: v for k, v in count.items() if v > 0}
                if totals:
                    st.caption(" · ".join(f"{k}: {v}" for k, v in totals.items()))
        elif result.get("success"):
            st.info("No named entities found in the text.")
        else:
            st.error("Entity extraction failed.")


def _render_grammar_panel(text: str) -> None:
    """Display grammar issues inline."""
    result = st.session_state.get("grammar_result", {})
    issues = result.get("issues", [])
    quality = result.get("overall_quality", "good")

    QUALITY_COLOUR = {"excellent": "#22c55e", "good": "#84cc16", "fair": "#f59e0b", "poor": "#ef4444"}
    colour = QUALITY_COLOUR.get(quality, "#6b7280")

    with st.expander(f"📋 Grammar Check — {len(issues)} issue(s) — Quality: **{quality.capitalize()}**", expanded=bool(issues)):
        if issues:
            for i, issue in enumerate(issues[:10]):
                otype = issue.get("type", "grammar")
                icon = {"grammar": "⚠️", "spelling": "🔤", "punctuation": "🔣"}.get(otype, "⚠️")
                col_issue, col_fix = st.columns([3, 1])
                with col_issue:
                    st.markdown(
                        f"{icon} **{otype.capitalize()}**: "
                        f"`{issue.get('original', '')}` → `{issue.get('suggestion', '')}`  \n"
                        f"<span style='font-size:.78rem;color:#888;'>{issue.get('explanation', '')}</span>",
                        unsafe_allow_html=True,
                    )
                with col_fix:
                    if st.button("Fix", key=f"fix_issue_{i}", use_container_width=True):
                        original = issue.get("original", "")
                        suggestion = issue.get("suggestion", "")
                        if original and suggestion:
                            current = st.session_state.get("editor_text", "")
                            st.session_state["editor_text"] = current.replace(original, suggestion, 1)
                            st.rerun()
        else:
            st.success(f"✓ No issues found! Quality: {quality.capitalize()}")

        if result.get("corrected_text"):
            if st.button("Apply all corrections", key="apply_all_grammar"):
                st.session_state["editor_text"] = result["corrected_text"]
                st.session_state["grammar_result"] = None
                st.rerun()


def get_editor_height() -> int:
    return int(get_config_value("ui.editor_height", 400))


def get_config_value(key: str, default=None):
    from utils.helpers import get_config_value as _gcv
    return _gcv(key, default)
