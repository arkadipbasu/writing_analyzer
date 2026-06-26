# ✍️ AI Writing Assistant

A production-grade, AI-powered writing assistant built with **Python**, **Streamlit**, and the **Groq API** (with OpenAI fallback). Inspired by Google Gboard and Microsoft SwiftKey — for the desktop browser.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **Real-Time Autocomplete** | Sentence-level ghost-text completion as you type |
| **Next Word Prediction** | Word / 3-word / sentence prediction modes |
| **Intent Detection** | 10 intent classes with confidence score |
| **Sentiment Analysis** | Positive / Neutral / Negative + score breakdown |
| **Synonym Suggestions** | 6 vocabulary categories (formal, academic, business…) |
| **Grammar Checker** | LanguageTool (offline) + LLM fallback with one-click fixes |
| **Rewrite Assistant** | 6 styles: Formal, Casual, Professional, Technical, Academic, Creative |
| **Summarization** | Short / Medium / Detailed modes |
| **Paraphrasing** | Meaning-preserving rewrite with alternatives |
| **Keyword Extraction** | Ranked keywords + topic detection |
| **Named Entity Recognition** | 8 entity types with colour-coded display |
| **Language Detection** | Auto-detects source language |
| **Translation** | 21 target languages |

---

## 🏗️ Architecture

```
ai_writer/
├── app.py                      # Streamlit entry point
├── config/
│   └── config.yaml             # All configuration (no hardcoded values)
├── services/
│   ├── groq_service.py         # LLM client: Groq + OpenAI fallback
│   ├── autocomplete_service.py # Sentence completion & next-word prediction
│   ├── sentiment_service.py    # Sentiment + Intent detection
│   ├── grammar_service.py      # Grammar checking (LT + LLM)
│   └── rewrite_service.py      # Rewrite, summarize, paraphrase, NER, translate…
├── components/
│   ├── sidebar.py              # Settings, API status, statistics panel
│   ├── editor.py               # Main text editor + action toolbar
│   └── statistics.py          # Live analysis right panel
├── utils/
│   ├── logger.py               # Loguru-based logging
│   ├── prompts.py              # All LLM prompt templates
│   └── helpers.py              # Config, cache, text utilities
├── tests/
│   └── test_services.py        # pytest test suite
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.12+
- A [Groq API key](https://console.groq.com) (free tier available)
- Optional: [OpenAI API key](https://platform.openai.com) for fallback

### Steps

```bash
# 1. Clone or extract the project
cd ai_writer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model (optional, for enhanced NER)
python -m spacy download en_core_web_sm

# 5. Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 6. Launch
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔑 Configuration

### `.env`
```env
GROQ_API_KEY=gsk_your_key_here
OPENAI_API_KEY=sk_your_key_here   # optional fallback
```

### `config/config.yaml`

Key settings you may want to tweak:

| Path | Default | Description |
|---|---|---|
| `models.groq.default` | `llama-3.3-70b-versatile` | Default inference model |
| `inference.temperature` | `0.3` | Sampling temperature |
| `inference.max_tokens` | `256` | Max completion tokens |
| `features.autocomplete.debounce_ms` | `400` | Debounce delay (ms) |
| `cache.ttl_seconds` | `300` | Response cache TTL |

---

## 🤖 Supported Models

| Model ID | Name | Speed |
|---|---|---|
| `llama-3.3-70b-versatile` | LLaMA 3.3 70B | Standard |
| `llama-3.1-8b-instant` | LLaMA 3.1 8B Instant | Fast |
| `qwen/qwen3-32b` | Qwen3 32B | Standard |
| `deepseek-r1-distill-llama-70b` | DeepSeek R1 70B | Standard |

Switch models in real-time via the sidebar without restarting.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker (Optional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t ai-writer .
docker run -p 8501:8501 --env-file .env ai-writer
```

---

## 🔒 Security

- API keys stored **only** in `.env` — never committed to version control
- `.env` is in `.gitignore`
- All user input is sanitized before sending to the API
- No secrets in `config.yaml` or source code

---

## 📐 Design Principles

- **SOLID** — each service has a single responsibility
- **Type hints** throughout — fully typed with `pydantic` models
- **Configuration-driven** — change behaviour in `config.yaml`, not code
- **Cache-first** — identical requests are served from cache within TTL
- **Graceful degradation** — Groq failure → OpenAI fallback → clear error message
- **Retry with back-off** — transient failures are retried automatically

---

## 📝 License

MIT
