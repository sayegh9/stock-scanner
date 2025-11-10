# Enhanced U.S. Stock Analysis Web App (v3.1)

> Streaming AI-powered research dashboard tailored for U.S. equities.

## Overview

The v3.1 web experience delivers a Flask + SSE front end that streams analysis updates from `EnhancedWebStockAnalyzer`. The analyzer scores each symbol across technical, fundamental, and sentiment pillars while optionally invoking LLM providers for narrative output. This directory packages the web server, analyzer logic, and configuration template.

### Key Features

- 🇺🇸 **U.S.-first market support** – A-share and Hong Kong pipelines are disabled by default.
- 🌊 **Live SSE streaming** – watch technical calculations, sentiment pulls, and AI commentary arrive in real time.
- 🤖 **Multi-LLM adapters** – switch between OpenAI, Anthropic, and Zhipu endpoints from `config.json`.
- 🧮 **Weighted scoring** – blend technical, fundamental, and sentiment grades into a 0–100 composite.
- 🗂️ **Batch workflows** – submit multiple tickers and receive independent streamed results.
- 📈 **Modern U.S. data feeds** – downloads prices and fundamentals from Yahoo Finance via `yfinance`, with automatic Stooq and akshare fallbacks.

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy and edit the config**
   ```bash
   cp config.sample.json config.json
   ```
   - Provide at least one AI provider API key under `api_keys`.
   - Review the `markets.us_stock` block (the only enabled market by default).
   - Enable `web_auth` and set a password if you want to gate the dashboard.

3. **Launch the streaming server**
   ```bash
   python enhanced_flask_server.py
   ```
   Then browse to `http://localhost:5000`.

## Configuration Notes

The bundled sample (`config.sample.json`) documents every option inline. Highlights:

| Section | Purpose |
| --- | --- |
| `ai` | Default provider, model names, API base overrides, sampling params |
| `analysis_weights` | Contribution of technical/fundamental/sentiment scores |
| `analysis_params` | Sliding windows, news caps, and AI input limits |
| `streaming` | Toggle SSE behaviour and response pacing |
| `markets` | Enable/disable supported markets – only `us_stock` ships enabled |
| `web_auth` | Optional password wall for the web UI |

## Data Providers

- Daily OHLCV candles and key valuation metrics are sourced from [yfinance](https://github.com/ranaroussi/yfinance) to avoid China-only endpoints.
- If Yahoo Finance is temporarily unavailable (for example due to rate limiting), the analyzer automatically falls back to [Stooq](https://stooq.com/).
- When `akshare` is installed the analyzer performs a final retry using the legacy akshare pathway before surfacing an error.

## Running Batch Jobs via API

- `POST /api/analyze_stream` – stream a single symbol
- `POST /api/batch_analyze_stream` – stream multiple symbols (max 10)
- `GET /api/system_info` – inspect worker pool and configured markets

All endpoints return english payloads tuned for U.S. tickers.

## Roadmap & Customisation

- Swap in your preferred U.S. price/fundamental/news data providers inside `enhanced_web_stock_analyzer.py`.
- Adjust the HTML templates within `enhanced_flask_server.py` to suit your branding.
- Extend the scoring weights or AI prompts to focus on sectors, ESG profiles, or other custom heuristics.

---

Questions or improvements? Open an issue or tailor the analyzer for your coverage universe.
