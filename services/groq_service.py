"""
services/groq_service.py
────────────────────────
Unified LLM client that targets Groq as primary provider and falls back to
OpenAI when Groq is unavailable or returns an error.

Features
--------
- Retry logic with exponential back-off
- Rate-limit awareness
- Response-time tracking
- Token-usage aggregation
- Streaming support (optional)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generator
import streamlit as st
import requests
from pydantic import BaseModel, Field

from utils.helpers import (
    get_config_value,
    get_groq_api_key,
    get_openai_api_key,
    safe_parse_json,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────

class LLMResponse(BaseModel):
    """Structured response from any LLM provider."""

    content: str = ""
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    parsed_json: dict[str, Any] | None = None


@dataclass
class UsageStats:
    """Running totals for token usage across a session."""

    total_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    groq_requests: int = 0
    openai_requests: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_requests, 1)


# ── Groq Service ──────────────────────────────────────────────────────────────

class GroqService:
    """
    Production-grade LLM service wrapping Groq (primary) and OpenAI (fallback).

    Usage
    -----
    ```python
    svc = GroqService()
    resp = svc.complete(system="You are helpful.", user="Hello!")
    print(resp.content)
    ```
    """

    GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
    OPENAI_BASE = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.groq_key = st.secrets.get("GROQ_API_KEY")
        self.openai_key = st.secrets.get("OPENAI_API_KEY")
        self.model = model or get_config_value("models.groq.default", "llama-3.3-70b-versatile")
        self.temperature = temperature if temperature is not None else get_config_value("inference.temperature", 0.3)
        self.max_tokens = max_tokens or get_config_value("inference.max_tokens", 256)
        self._timeout = get_config_value("api.groq.timeout", 30)
        self._max_retries = get_config_value("api.groq.max_retries", 3)
        self._retry_delay = get_config_value("api.groq.retry_delay", 1.0)
        self.stats = UsageStats()

    # ── Public API ─────────────────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        expect_json: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            system: System prompt.
            user: User message.
            temperature: Override instance temperature.
            max_tokens: Override instance max_tokens.
            expect_json: If True, attempt to parse the response as JSON.

        Returns:
            :class:`LLMResponse` with content and metadata.
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Try Groq first
        if self.groq_key:
            resp = self._call_with_retry(
                url=self.GROQ_BASE,
                api_key=self.groq_key,
                payload=payload,
                provider="groq",
            )
            if resp.success:
                if expect_json:
                    resp.parsed_json = safe_parse_json(resp.content)
                self._update_stats(resp, provider="groq")
                return resp
            logger.warning("Groq failed, attempting OpenAI fallback.")

        # Fallback to OpenAI
        if self.openai_key:
            # Use a compatible OpenAI model
            oai_payload = {**payload, "model": get_config_value("models.openai.default", "gpt-4o-mini")}
            resp = self._call_with_retry(
                url=self.OPENAI_BASE,
                api_key=self.openai_key,
                payload=oai_payload,
                provider="openai",
            )
            if resp.success:
                if expect_json:
                    resp.parsed_json = safe_parse_json(resp.content)
                self._update_stats(resp, provider="openai")
                return resp

        # Both failed
        self.stats.errors += 1
        return LLMResponse(
            success=False,
            error="Both Groq and OpenAI API calls failed. Check your API keys and network.",
        )

    def validate_groq_key(self) -> bool:
        """Ping Groq with a minimal request to validate the API key."""
        if not self.groq_key:
            return False
        try:
            resp = self._post(
                url=self.GROQ_BASE,
                api_key=self.groq_key,
                payload={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
            return resp.success
        except Exception:
            return False

    def validate_openai_key(self) -> bool:
        """Ping OpenAI with a minimal request to validate the API key."""
        if not self.openai_key:
            return False
        try:
            resp = self._post(
                url=self.OPENAI_BASE,
                api_key=self.openai_key,
                payload={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
            return resp.success
        except Exception:
            return False

    def set_model(self, model: str) -> None:
        """Hot-swap the active model."""
        self.model = model
        logger.info(f"Model switched to {model}")

    def set_temperature(self, temperature: float) -> None:
        """Update the sampling temperature."""
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int) -> None:
        """Update max completion tokens."""
        self.max_tokens = max_tokens

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _call_with_retry(
        self,
        url: str,
        api_key: str,
        payload: dict[str, Any],
        provider: str,
    ) -> LLMResponse:
        """Execute an API call with exponential back-off retries."""
        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._post(url=url, api_key=api_key, payload=payload)
                if resp.success:
                    resp.provider = provider
                    return resp
                last_error = resp.error

                # Don't retry on auth errors
                if "401" in last_error or "403" in last_error or "invalid_api_key" in last_error.lower():
                    break

                # Exponential back-off
                if attempt < self._max_retries:
                    sleep_time = self._retry_delay * (2 ** (attempt - 1))
                    logger.warning(f"{provider} attempt {attempt} failed. Retrying in {sleep_time}s…")
                    time.sleep(sleep_time)

            except Exception as exc:
                last_error = str(exc)
                logger.error(f"{provider} attempt {attempt} raised: {exc}")
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)

        return LLMResponse(success=False, error=last_error, provider=provider)

    def _post(
        self,
        url: str,
        api_key: str,
        payload: dict[str, Any],
    ) -> LLMResponse:
        """Execute a single HTTP POST to the completions endpoint."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        t0 = time.perf_counter()
        try:
            raw = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            if raw.status_code == 429:
                return LLMResponse(
                    success=False,
                    error=f"Rate limited (429). Retry-After: {raw.headers.get('Retry-After', 'unknown')}s",
                    latency_ms=latency_ms,
                )

            if raw.status_code not in (200, 201):
                err_body = raw.text[:300]
                return LLMResponse(
                    success=False,
                    error=f"HTTP {raw.status_code}: {err_body}",
                    latency_ms=latency_ms,
                )

            data = raw.json()
            choice = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return LLMResponse(
                content=choice.strip(),
                model=data.get("model", payload.get("model", "")),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=latency_ms,
                success=True,
            )

        except requests.Timeout:
            latency_ms = (time.perf_counter() - t0) * 1000
            return LLMResponse(success=False, error="Request timed out.", latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return LLMResponse(success=False, error=str(exc), latency_ms=latency_ms)

    def _update_stats(self, resp: LLMResponse, provider: str) -> None:
        self.stats.total_requests += 1
        self.stats.total_tokens += resp.total_tokens
        self.stats.total_prompt_tokens += resp.prompt_tokens
        self.stats.total_completion_tokens += resp.completion_tokens
        self.stats.total_latency_ms += resp.latency_ms
        if provider == "groq":
            self.stats.groq_requests += 1
        else:
            self.stats.openai_requests += 1
