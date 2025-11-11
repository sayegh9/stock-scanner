# Enhanced U.S. Stock Analysis Web App (v3.1)

> Streaming AI-powered research dashboard tailored for U.S. equities.

## Overview

The v3.1 web experience delivers a Flask + SSE front end that streams analysis updates from `EnhancedWebStockAnalyzer`. The analyzer scores each symbol across technical, fundamental, and sentiment pillars while optionally invoking LLM providers for narrative output. This directory packages the web server, analyzer logic, and configuration template.

### Key Features

- 🇺🇸 **U.S.-first market support** – A-share and Hong Kong pipelines are disabled by default.
- 🌊 **Live SSE streaming** – watch technical calculations, sentiment pulls, and AI commentary arrive in real time.
- 🎨 **Material Design 3 UI** – refreshed cards, buttons, and typography built with Material You tokens for cohesive visuals.
- 🤖 **Multi-LLM adapters** – switch between OpenAI, Anthropic, and Zhipu endpoints from `config.json`.
- 🧮 **Weighted scoring** – blend technical, fundamental, and sentiment grades into a 0–100 composite.
- 🗂️ **Batch workflows** – submit multiple tickers and receive independent streamed results.
- 📈 **Modern U.S. data feeds** – downloads prices and fundamentals from Yahoo Finance via `yfinance`, with automatic Stooq and akshare fallbacks plus a direct Yahoo quoteSummary bridge when the SDK yields sparse data.
- 🛟 **Data quality call-outs** – the dashboard surfaces missing fundamentals, empty news feeds, or provider warnings so you immediately know when an input needs attention.
- 💬 **Analyst follow-up chat** – ask real-time questions about the streamed report; responses reuse the full scorecard and data-quality context.
- ⚡ **Parallelised data pulls** – price, fundamentals, and news downloads run concurrently to minimise waiting time before scores appear.

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
   - Without a valid AI key the dashboard falls back to a concise rule-based summary instead of the in-depth AI report.
   - To enable real-time news and sentiment coverage, add API keys for [Finnhub](https://finnhub.io/) and optionally [NewsData.io](https://newsdata.io/). Finnhub is used first, with NewsData.io acting as an automatic fallback when rate limits or outages occur.
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

- Daily OHLCV candles and key valuation metrics are sourced from [yfinance](https://github.com/ranaroussi/yfinance) (including `fast_info` and financial statement fallbacks) to avoid China-only endpoints.
- If Yahoo Finance is temporarily unavailable (for example due to rate limiting), the analyzer automatically falls back to [Stooq](https://stooq.com/).
- When `akshare` is installed the analyzer performs a final retry using the legacy akshare pathway before surfacing an error.
- Company news is fetched from [Finnhub](https://finnhub.io/) with an automatic fallback to [NewsData.io](https://newsdata.io/) so the sentiment engine always has diverse coverage.
- Sentiment scoring uses [vaderSentiment](https://github.com/cjhutto/vaderSentiment) when available and falls back to keyword heuristics if the package is missing—install it to unlock richer tone detection.

## Analyst Chat Sidebar

- Run a single streaming analysis to populate the dashboard.
- The new **Analyst follow-up chat** panel (right-hand column) becomes active once a report is available.
- Each question is answered with awareness of the latest scores, price snapshot, fundamental coverage, sentiment breakdown, and any data-quality warnings.
- Chat responses stream token-by-token when LLM keys are configured; without keys the system provides a concise rule-based answer using the structured output.
- Clearing the chat history keeps the current report context so you can continue the conversation without rerunning the analysis.

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
