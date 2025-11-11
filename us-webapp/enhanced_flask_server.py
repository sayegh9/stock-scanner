"""Flask web server for the streaming U.S. stock analysis experience."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import wraps
from queue import Empty, Queue
from typing import Dict, Optional

import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)
from flask_cors import CORS

try:
    from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer
except ImportError:  # pragma: no cover - dependency guard
    print("Unable to import enhanced_web_stock_analyzer.py", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)
CORS(app)
app.config["JSON_SORT_KEYS"] = False
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
app.secret_key = secrets.token_hex(32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

analyzer: Optional[EnhancedWebStockAnalyzer] = None
analysis_tasks: Dict[str, Dict] = {}
analysis_results: Dict[str, Dict] = {}
analysis_lock = threading.Lock()
sse_clients: Dict[str, Queue] = {}
sse_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
class SSEManager:
    """Track connected SSE clients and deliver JSON payloads."""

    def add_client(self, client_id: str) -> Queue:
        queue: Queue = Queue()
        with sse_lock:
            sse_clients[client_id] = queue
        logger.info("SSE client connected: %s", client_id)
        return queue

    def remove_client(self, client_id: str) -> None:
        with sse_lock:
            sse_clients.pop(client_id, None)
        logger.info("SSE client disconnected: %s", client_id)

    def send(self, client_id: str, event: str, data) -> None:
        with sse_lock:
            queue = sse_clients.get(client_id)
        if not queue:
            return
        payload = {
            "event": event,
            "data": clean_data_for_json(data),
            "timestamp": datetime.utcnow().isoformat(),
        }
        queue.put(payload)

    def broadcast(self, event: str, data) -> None:
        payload = {
            "event": event,
            "data": clean_data_for_json(data),
            "timestamp": datetime.utcnow().isoformat(),
        }
        with sse_lock:
            for queue in sse_clients.values():
                queue.put(payload)


sse_manager = SSEManager()


def clean_data_for_json(obj):
    """Convert numpy/pandas objects into JSON-serialisable structures."""

    import pandas as pd

    if isinstance(obj, dict):
        return {key: clean_data_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_data_for_json(item) for item in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, pd.Series):
        return clean_data_for_json(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return clean_data_for_json(obj.to_dict(orient="records"))
    return str(obj)


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------
def _auth_config():
    if not analyzer:
        return False, {}
    config = analyzer.config.get("web_auth", {})
    return config.get("enabled", False), config


def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        enabled, config = _auth_config()
        if not enabled:
            return func(*args, **kwargs)

        authenticated = session.get("authenticated")
        login_time = session.get("login_time")
        timeout = config.get("session_timeout", 3600)
        if authenticated and login_time:
            if datetime.utcnow() - datetime.fromisoformat(login_time) < timedelta(seconds=timeout):
                return func(*args, **kwargs)
            session.clear()
        return redirect(url_for("login"))

    return wrapper


LOGIN_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign in · U.S. Stock Analyzer</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #0b1120; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .card { background: white; padding: 40px; border-radius: 16px; box-shadow: 0 30px 60px rgba(15, 23, 42, 0.25); max-width: 360px; width: 100%; }
        h1 { margin: 0 0 8px 0; font-size: 26px; color: #111827; }
        p { margin: 0 0 24px 0; color: #4b5563; }
        label { font-weight: 600; font-size: 14px; display: block; margin-bottom: 8px; color: #1f2937; }
        input[type=password] { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #d1d5db; font-size: 14px; }
        button { width: 100%; padding: 12px; border: none; border-radius: 8px; background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; font-weight: 600; cursor: pointer; margin-top: 16px; }
        button:hover { transform: translateY(-1px); }
        .error { background: #fee2e2; color: #991b1b; padding: 12px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; }
        .meta { margin-top: 24px; font-size: 12px; color: #6b7280; text-align: center; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🔐 Secure access</h1>
        <p>Enhanced streaming analysis for U.S. equities.</p>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <label for="password">Access password</label>
            <input id="password" name="password" type="password" required placeholder="Enter the shared password">
            <button type="submit">Sign in</button>
        </form>
        <div class="meta">
            Sessions expire after {{ timeout_minutes }} minutes.
        </div>
    </div>
    <script>document.getElementById('password').focus();</script>
</body>
</html>"""


MAIN_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modern Stock Analysis System · SSE Streaming</title>
    <style>
        :root {
            color-scheme: light;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 0;
            background: #eef2ff;
            color: #0f172a;
        }
        a { color: inherit; text-decoration: none; }
        .hero {
            background: linear-gradient(135deg, #312e81 0%, #1d4ed8 45%, #9333ea 100%);
            color: white;
            padding: 48px 0 56px;
        }
        .hero-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 32px;
            display: flex;
            gap: 32px;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: rgba(255, 255, 255, 0.16);
            border-radius: 999px;
            padding: 8px 14px;
        }
        .hero h1 {
            margin: 16px 0 12px;
            font-size: 36px;
            font-weight: 700;
        }
        .hero p {
            margin: 0 0 20px;
            font-size: 17px;
            max-width: 560px;
            color: rgba(255, 255, 255, 0.82);
        }
        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.25);
            font-size: 13px;
        }
        .hero-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4ade80;
            display: inline-block;
        }
        .hero-actions {
            display: flex;
            flex-direction: column;
            gap: 16px;
            min-width: 240px;
        }
        .hero-button {
            align-self: flex-end;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            font-weight: 600;
            padding: 10px 18px;
            border-radius: 999px;
            transition: background 0.2s ease, transform 0.2s ease;
        }
        .hero-button:hover {
            background: rgba(255, 255, 255, 0.25);
            transform: translateY(-1px);
        }
        .hero-stat {
            background: rgba(15, 23, 42, 0.25);
            border-radius: 16px;
            padding: 16px 18px;
        }
        .hero-stat span {
            display: block;
            font-size: 12px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.6);
        }
        .hero-stat strong {
            display: block;
            margin-top: 6px;
            font-size: 18px;
            font-weight: 700;
            color: white;
        }
        main.layout {
            max-width: 1200px;
            margin: -32px auto 64px;
            padding: 0 32px;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 28px;
        }
        .column {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        .card {
            background: white;
            border-radius: 24px;
            box-shadow: 0 30px 60px rgba(15, 23, 42, 0.18);
            padding: 28px;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        .card-header h2 {
            margin: 0;
            font-size: 22px;
            font-weight: 700;
        }
        .card-header p {
            margin: 6px 0 0;
            font-size: 14px;
            color: #6b7280;
        }
        .status-pill {
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            background: #e0e7ff;
            color: #3730a3;
        }
        .status-pill.offline {
            background: #fee2e2;
            color: #b91c1c;
        }
        .status-pill.live {
            background: rgba(74, 222, 128, 0.18);
            color: #047857;
        }
        .status-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
            margin-bottom: 24px;
        }
        .status-item {
            display: flex;
            gap: 12px;
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
        }
        .status-item.ok {
            border-color: rgba(74, 222, 128, 0.35);
            background: #f0fdf4;
        }
        .status-item.warn {
            border-color: rgba(251, 191, 36, 0.45);
            background: #fffbeb;
        }
        .status-icon {
            width: 28px;
            height: 28px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 600;
        }
        .status-item.ok .status-icon {
            background: #4ade80;
            color: #064e3b;
        }
        .status-item.warn .status-icon {
            background: #fbbf24;
            color: #78350f;
        }
        .status-title {
            font-weight: 600;
            font-size: 15px;
        }
        .status-subtitle {
            font-size: 13px;
            color: #64748b;
        }
        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 8px;
        }
        input[type="text"], textarea {
            width: 100%;
            border-radius: 14px;
            border: 1px solid #cbd5f5;
            background: #f8fafc;
            padding: 14px;
            font-size: 15px;
            transition: border 0.2s ease, box-shadow 0.2s ease;
            color: #0f172a;
        }
        input[type="text"]:focus, textarea:focus {
            border-color: #6366f1;
            outline: none;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        .input-row {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
        }
        .primary-button {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 14px 20px;
            border-radius: 14px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            font-size: 15px;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: white;
            box-shadow: 0 12px 24px rgba(79, 70, 229, 0.35);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .primary-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 20px 32px rgba(79, 70, 229, 0.35);
        }
        .ghost-button {
            padding: 10px 16px;
            border-radius: 12px;
            border: 1px solid #cbd5f5;
            background: #f8fafc;
            color: #1f2937;
            font-weight: 600;
            cursor: pointer;
        }
        .ghost-button:hover {
            background: #e2e8f0;
        }
        .hint {
            font-size: 13px;
            color: #64748b;
            margin-top: -6px;
            margin-bottom: 18px;
        }
        .divider {
            margin: 26px 0 20px;
            border-top: 1px dashed #cbd5f5;
        }
        .status-banner {
            margin-top: 20px;
            padding: 14px 18px;
            border-radius: 14px;
            background: #e0f2fe;
            color: #0c4a6e;
            font-weight: 600;
            font-size: 14px;
        }
        .log-panel {
            background: #0f172a;
            border-radius: 18px;
            padding: 18px;
            height: 240px;
            overflow-y: auto;
            font-family: 'Source Code Pro', 'Consolas', monospace;
            color: #e2e8f0;
            font-size: 13px;
        }
        .log-entry { margin-bottom: 6px; }
        .log-entry.info { color: #94a3b8; }
        .log-entry.warn { color: #f59e0b; }
        .log-entry.error { color: #fca5a5; }
        .log-entry.warning { color: #facc15; }
        .score-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .score-card {
            border-radius: 20px;
            padding: 18px 20px;
            color: white;
            display: flex;
            flex-direction: column;
            gap: 6px;
            box-shadow: 0 18px 28px rgba(15, 23, 42, 0.15);
        }
        .score-card .label {
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .score-card .value {
            font-size: 34px;
            font-weight: 700;
        }
        .score-card .caption {
            font-size: 13px;
            opacity: 0.8;
        }
        .score-card.technical { background: linear-gradient(135deg, #0ea5e9, #2563eb); }
        .score-card.fundamental { background: linear-gradient(135deg, #14b8a6, #0f766e); }
        .score-card.sentiment { background: linear-gradient(135deg, #f59e0b, #d946ef); }
        .score-card.composite { background: linear-gradient(135deg, #6366f1, #312e81); }
        .result-shell {
            border-radius: 20px;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            padding: 22px;
            margin-bottom: 24px;
        }
        .result-title {
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 12px;
            color: #1f2937;
        }
        .result-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px 16px;
            margin-bottom: 16px;
            font-size: 13px;
            color: #475569;
        }
        .result-grid {
            display: grid;
            gap: 14px;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        }
        .result-item {
            background: #f8fafc;
            border-radius: 14px;
            padding: 14px;
            border: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .result-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
        }
        .result-value {
            font-size: 16px;
            font-weight: 600;
            color: #0f172a;
        }
        .result-note {
            font-size: 12px;
            color: #64748b;
        }
        .result-item.warn {
            background: #fff7ed;
            border-color: #fb923c;
        }
        .result-item.warn .result-value {
            color: #c2410c;
        }
        .result-item.warn .result-note {
            color: #b45309;
        }
        .result-item.info {
            background: #eef2ff;
            border-color: #a5b4fc;
        }
        .result-item.info .result-value {
            color: #3730a3;
        }
        .result-body {
            margin-top: 18px;
            background: rgba(99, 102, 241, 0.06);
            border-left: 4px solid #6366f1;
            border-radius: 12px;
            padding: 16px 18px;
            font-size: 14px;
            color: #1f2937;
            min-height: 110px;
        }
        .result-notes {
            margin-top: 16px;
            background: #fff7ed;
            border: 1px solid #f97316;
            border-radius: 12px;
            padding: 14px 16px;
            color: #9a3412;
            font-size: 13px;
            display: none;
        }
        .result-notes h4 {
            margin: 0 0 8px;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .result-notes ul {
            margin: 0;
            padding-left: 18px;
        }
        .result-notes li {
            margin-bottom: 4px;
            line-height: 1.4;
        }
        .ai-section { display: none; }
        .ai-section h3 {
            margin: 0 0 12px;
            font-size: 18px;
            font-weight: 700;
            color: #1f2937;
        }
        .ai-stream {
            background: #fef3c7;
            border: 1px solid #fcd34d;
            border-radius: 18px;
            padding: 20px;
            font-size: 14px;
            line-height: 1.6;
            color: #78350f;
            white-space: pre-wrap;
        }
        .meta-panel {
            margin-top: 24px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
        }
        .meta-card {
            border-radius: 16px;
            padding: 16px 18px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .meta-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
        }
        .meta-value {
            display: block;
            margin-top: 8px;
            font-size: 16px;
            font-weight: 600;
            color: #1e293b;
        }
        @media (max-width: 1024px) {
            main.layout {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 720px) {
            .hero-content {
                padding: 0 20px;
            }
            main.layout {
                padding: 0 20px;
            }
            .input-row {
                flex-direction: column;
            }
            .hero-actions {
                width: 100%;
                flex-direction: row;
                flex-wrap: wrap;
                align-items: center;
            }
            .hero-button { align-self: stretch; text-align: center; }
        }
    </style>
</head>
<body>
    {% set ui = ui_context or {} %}
    {% set streaming = ui.get('streaming', {}) %}
    {% set ai = ui.get('ai', {}) %}
    {% set markets = ui.get('markets', []) %}
    <div class="hero">
        <div class="hero-content">
            <div>
                <div class="hero-eyebrow">Modern Stock Analysis System</div>
                <h1>SSE Streaming Version</h1>
                <p>English-language dashboard delivering technical, fundamental, sentiment, and AI intelligence for U.S. equities in real time.</p>
                <div class="hero-meta">
                    <span class="hero-chip"><span class="hero-dot"></span>Streaming {{ 'enabled' if streaming.get('enabled') else 'disabled' }} · {{ '%.2f' % streaming.get('delay', 0.0) }}s cadence</span>
                    <span class="hero-chip">Primary market · {{ ui.get('market_summary') or 'Configure markets in config.json' }}</span>
                </div>
            </div>
            <div class="hero-actions">
                {% if auth_enabled %}
                <a class="hero-button" href="/logout">Sign out</a>
                {% endif %}
                <div class="hero-stat">
                    <span>Session ID</span>
                    <strong id="sessionId">--</strong>
                </div>
                <div class="hero-stat">
                    <span>AI provider</span>
                    <strong>{{ ai.get('model') if ai.get('has_keys') else 'API key required' }}</strong>
                </div>
            </div>
        </div>
    </div>
    <main class="layout">
        <section class="column">
            <div class="card">
                <div class="card-header">
                    <div>
                        <h2>Stock Control</h2>
                        <p>Launch live streaming runs or queue batch requests for up to ten tickers.</p>
                    </div>
                    <span class="status-pill {{ 'live' if streaming.get('enabled') else 'offline' }}">{{ 'Live streaming' if streaming.get('enabled') else 'Offline mode' }}</span>
                </div>
                <div class="status-grid">
                    <div class="status-item {{ 'ok' if streaming.get('enabled') else 'warn' }}">
                        <div class="status-icon">{{ '✓' if streaming.get('enabled') else '!' }}</div>
                        <div>
                            <div class="status-title">Streaming {{ 'enabled' if streaming.get('enabled') else 'disabled' }}</div>
                            <div class="status-subtitle">Server-Sent Events · {{ '%.2f' % streaming.get('delay', 0.0) }}s cadence{% if streaming.get('show_thinking') %} · AI thinking visible{% endif %}</div>
                        </div>
                    </div>
                    <div class="status-item {{ 'ok' if ai.get('has_keys') else 'warn' }}">
                        <div class="status-icon">{{ '✓' if ai.get('has_keys') else '!' }}</div>
                        <div>
                            <div class="status-title">AI model preference · {{ ai.get('preference', 'openai')|upper }}</div>
                            <div class="status-subtitle">{{ ai.get('model') if ai.get('has_keys') else 'Add an API key to unlock the AI deep-dive report.' }}</div>
                        </div>
                    </div>
                    <div class="status-item ok">
                        <div class="status-icon">✓</div>
                        <div>
                            <div class="status-title">Market coverage</div>
                            <div class="status-subtitle">{% if markets %}{% for market in markets %}{{ market.name }} · {{ market.currency }} · {{ market.timezone }}{% if not loop.last %} | {% endif %}{% endfor %}{% else %}Enable at least one market in config.json{% endif %}</div>
                        </div>
                    </div>
                    {% if ui.get('weights') %}
                    <div class="status-item ok">
                        <div class="status-icon">✓</div>
                        <div>
                            <div class="status-title">Scoring focus</div>
                            <div class="status-subtitle">{{ ui.get('weights') }}</div>
                        </div>
                    </div>
                    {% endif %}
                    {% if ui.get('cache') %}
                    <div class="status-item ok">
                        <div class="status-icon">✓</div>
                        <div>
                            <div class="status-title">Cache windows</div>
                            <div class="status-subtitle">{{ ui.get('cache') }}</div>
                        </div>
                    </div>
                    {% endif %}
                </div>
                <label for="singleSymbol">Ticker symbol</label>
                <div class="input-row">
                    <input id="singleSymbol" type="text" placeholder="e.g. AAPL" autocomplete="off">
                    <button id="analyzeBtn" type="button" class="primary-button">🚀 Stream analysis</button>
                </div>
                <div class="hint">Use standard U.S. ticker symbols (1–7 characters).</div>
                <div class="divider"></div>
                <label for="batchSymbols">Batch queue (comma separated)</label>
                <textarea id="batchSymbols" placeholder="AAPL, MSFT, NVDA"></textarea>
                <div class="hint">Batch analysis processes up to 10 tickers sequentially.</div>
                <button id="batchBtn" type="button" class="primary-button">📦 Start batch run</button>
                <div class="status-banner" id="systemStatus">Ready</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <div>
                        <h2>Activity log</h2>
                        <p>Real-time server updates, errors, and streaming events.</p>
                    </div>
                    <button id="resetBtn" type="button" class="ghost-button">Reset dashboard</button>
                </div>
                <div class="log-panel" id="logStream"></div>
            </div>
        </section>
        <section class="column">
            <div class="card">
                <div class="card-header">
                    <div>
                        <h2>Analysis results</h2>
                        <p>Scores refresh as each analytical stage completes.</p>
                    </div>
                    <span class="status-pill {{ 'live' if streaming.get('enabled') else 'offline' }}">{{ 'Streaming mode' if streaming.get('enabled') else 'Awaiting data' }}</span>
                </div>
                <div class="score-grid">
                    <div class="score-card technical">
                        <span class="label">Technical</span>
                        <span class="value" id="technicalScore">--</span>
                        <span class="caption">Trend &amp; momentum</span>
                    </div>
                    <div class="score-card fundamental">
                        <span class="label">Fundamental</span>
                        <span class="value" id="fundamentalScore">--</span>
                        <span class="caption">Quality &amp; valuation</span>
                    </div>
                    <div class="score-card sentiment">
                        <span class="label">Sentiment</span>
                        <span class="value" id="sentimentScore">--</span>
                        <span class="caption">News &amp; tone</span>
                    </div>
                    <div class="score-card composite">
                        <span class="label">Composite</span>
                        <span class="value" id="compositeScore">--</span>
                        <span class="caption">Weighted outlook</span>
                    </div>
                </div>
                <div class="result-shell">
                    <div class="result-title">Market snapshot</div>
                    <div class="result-meta" id="resultMeta">
                        <span>Waiting for ticker…</span>
                    </div>
                    <div class="result-grid" id="resultHighlights"></div>
                    <div class="result-body" id="resultPanel">Waiting for results…</div>
                    <div class="result-notes" id="resultNotes"></div>
                </div>
                <div class="ai-section" id="aiSection">
                    <h3>AI Deep Analysis</h3>
                    <div class="ai-stream" id="aiStream"></div>
                </div>
                <div class="meta-panel">
                    <div class="meta-card">
                        <span class="meta-label">Technical lookback</span>
                        <span class="meta-value">{{ ui.get('analysis', {}).get('technical_days') or 'Configurable' }} days</span>
                    </div>
                    <div class="meta-card">
                        <span class="meta-label">News limit</span>
                        <span class="meta-value">{{ ui.get('analysis', {}).get('news_limit') or 'Configurable' }} articles</span>
                    </div>
                    {% if ai.get('configured_keys') %}
                    <div class="meta-card">
                        <span class="meta-label">Active AI keys</span>
                        <span class="meta-value">{{ ai.get('configured_keys')|join(', ') }}</span>
                    </div>
                    {% endif %}
                </div>
            </div>
        </section>
    </main>
<script>
(function () {
    var DEFAULT_MARKET = 'us_stock';
    var clientId = generateClientId();
    var eventSource = null;
    var currentReport = null;
    var lastHeartbeat = new Date().getTime();

    function generateClientId() {
        try {
            var cryptoObj = (typeof window !== 'undefined' && window.crypto) ? window.crypto : null;
            if (cryptoObj && typeof cryptoObj.randomUUID === 'function') {
                return cryptoObj.randomUUID();
            }
            if (cryptoObj && typeof cryptoObj.getRandomValues === 'function' && typeof Uint8Array !== 'undefined') {
                var buffer = new Uint8Array(16);
                cryptoObj.getRandomValues(buffer);
                buffer[6] = (buffer[6] & 15) | 64;
                buffer[8] = (buffer[8] & 63) | 128;
                var hex = [];
                for (var i = 0; i < buffer.length; i += 1) {
                    var value = buffer[i].toString(16);
                    if (value.length < 2) {
                        value = '0' + value;
                    }
                    hex.push(value);
                }
                return hex.slice(0, 4).join('') + '-' +
                    hex.slice(4, 6).join('') + '-' +
                    hex.slice(6, 8).join('') + '-' +
                    hex.slice(8, 10).join('') + '-' +
                    hex.slice(10, 16).join('');
            }
        } catch (error) {
            // ignore and fall back
        }
        return 'client-' + new Date().getTime() + '-' + Math.floor(Math.random() * 1e9);
    }

    function isWhitespace(character) {
        return character === ' ' || character === '\n' || character === '\r' ||
            character === '\t' || character === '\f' || character === '\v';
    }

    function trim(value) {
        var start = 0;
        var end = value.length;
        while (start < end && isWhitespace(value.charAt(start))) {
            start += 1;
        }
        while (end > start && isWhitespace(value.charAt(end - 1))) {
            end -= 1;
        }
        return value.substring(start, end);
    }

    function setSessionId() {
        var sessionTarget = document.getElementById('sessionId');
        if (sessionTarget) {
            sessionTarget.textContent = clientId;
        }
    }

    function addLog(message, level) {
        var logPanel = document.getElementById('logStream');
        if (!logPanel) {
            return;
        }
        var entry = document.createElement('div');
        entry.className = 'log-entry ' + (level || 'info');
        entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + message;
        logPanel.appendChild(entry);
        logPanel.scrollTop = logPanel.scrollHeight;
    }

    function setStatus(text) {
        var statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.textContent = text;
        }
    }

    function formatScore(value) {
        if (value === undefined || value === null) {
            return 'N/A';
        }
        if (typeof value === 'number') {
            if (isNaN(value)) {
                return 'N/A';
            }
            return value.toFixed(1);
        }
        if (typeof value === 'string') {
            var cleaned = trim(value);
            if (!cleaned) {
                return 'N/A';
            }
            if (cleaned.toUpperCase() === 'N/A') {
                return 'N/A';
            }
            var numberValue = Number(cleaned);
            if (!isNaN(numberValue)) {
                return numberValue.toFixed(1);
            }
            return cleaned;
        }
        return 'N/A';
    }

    function updateScores(scores) {
        document.getElementById('technicalScore').textContent = formatScore(scores.technical);
        document.getElementById('fundamentalScore').textContent = formatScore(scores.fundamental);
        document.getElementById('sentimentScore').textContent = formatScore(scores.sentiment);
        document.getElementById('compositeScore').textContent = formatScore(scores.comprehensive);
    }

    function formatInteger(value) {
        if (typeof value === 'number' && !isNaN(value)) {
            return Math.round(value).toLocaleString();
        }
        return 'N/A';
    }

    function hasNumber(value) {
        return typeof value === 'number' && !isNaN(value);
    }

    function formatPriceValue(value, currency) {
        if (hasNumber(value)) {
            var priceText = value.toFixed(2);
            if (currency) {
                priceText += ' ' + currency;
            }
            return priceText;
        }
        return 'N/A';
    }

    function formatChangeValue(value) {
        if (hasNumber(value)) {
            var prefix = value > 0 ? '+' : '';
            return prefix + value.toFixed(2) + '%';
        }
        return 'N/A';
    }

    function buildHighlight(label, value, severity, note) {
        var classes = 'result-item';
        if (severity) {
            classes += ' ' + severity;
        }
        var noteHtml = note ? '<div class="result-note">' + note + '</div>' : '';
        return '<div class="' + classes + '"><div class="result-label">' + label + '</div><div class="result-value">' + value + '</div>' + noteHtml + '</div>';
    }

    function showResult(report) {
        currentReport = report;
        var summaryPanel = document.getElementById('resultPanel');
        var metaPanel = document.getElementById('resultMeta');
        var highlightPanel = document.getElementById('resultHighlights');
        var notesPanel = document.getElementById('resultNotes');
        if (!summaryPanel) {
            return;
        }

        var market = report && report.market_info ? report.market_info : {};
        var stockName = report && report.stock_name ? report.stock_name : 'Unknown';
        var stockCode = report && report.stock_code ? report.stock_code : '';
        var currency = market.currency || 'USD';
        var priceInfo = report && report.price_info ? report.price_info : {};
        var recommendation = report && report.recommendation ? report.recommendation : 'N/A';
        var scores = report && report.scores ? report.scores : {};
        var dataQuality = report && report.data_quality ? report.data_quality : {};
        var newsSources = Array.isArray(dataQuality.news_sources) ? dataQuality.news_sources : [];
        var fundamentalProviders = Array.isArray(dataQuality.fundamental_providers) ? dataQuality.fundamental_providers : [];

        if (metaPanel) {
            var metaParts = [];
            metaParts.push('<span><strong>Market:</strong> ' + (market.name || 'U.S. equities') + '</span>');
            metaParts.push('<span><strong>Currency:</strong> ' + currency + '</span>');
            if (market.timezone) {
                metaParts.push('<span><strong>Timezone:</strong> ' + market.timezone + '</span>');
            }
            metaParts.push('<span><strong>Generated:</strong> ' + (report.analysis_date || new Date().toLocaleString()) + '</span>');
            metaPanel.innerHTML = metaParts.join('');
        }

        if (highlightPanel) {
            var highlightBlocks = [];
            var priceText = formatPriceValue(priceInfo.current_price, currency);
            var priceSeverity = priceText === 'N/A' ? 'warn' : '';
            var priceNote = priceSeverity ? 'Price snapshot unavailable.' : '';
            highlightBlocks.push(buildHighlight('Last close', priceText, priceSeverity, priceNote));

            var changeText = formatChangeValue(priceInfo.price_change);
            var changeSeverity = changeText === 'N/A' ? 'warn' : '';
            var changeNote = changeSeverity ? 'Daily change could not be calculated.' : '';
            highlightBlocks.push(buildHighlight('Daily change', changeText, changeSeverity, changeNote));

            if (dataQuality.financial_indicators_count !== undefined) {
                var indicatorCount = dataQuality.financial_indicators_count;
                var indicatorSeverity = hasNumber(indicatorCount) && indicatorCount > 0 ? '' : 'warn';
                var indicatorText = hasNumber(indicatorCount) && indicatorCount > 0 ? formatInteger(indicatorCount) : 'None';
                var indicatorNote = indicatorSeverity ? 'No fundamentals returned from configured providers.' : '';
                highlightBlocks.push(buildHighlight('Financial indicators', indicatorText, indicatorSeverity, indicatorNote));
            }

            if (dataQuality.total_news_count !== undefined) {
                var newsCount = dataQuality.total_news_count;
                var newsSeverity = hasNumber(newsCount) && newsCount > 0 ? '' : 'warn';
                var newsText = hasNumber(newsCount) && newsCount > 0 ? formatInteger(newsCount) : 'None';
                var newsNote = newsSeverity ? 'No recent news items were analysed.' : '';
                highlightBlocks.push(buildHighlight('News coverage', newsText, newsSeverity, newsNote));
            }

            if (Array.isArray(dataQuality.news_sources) && dataQuality.news_sources.length) {
                highlightBlocks.push(
                    buildHighlight('News sources', dataQuality.news_sources.join(', '))
                );
            }

            if (dataQuality.analysis_completeness) {
                var complete = dataQuality.analysis_completeness === 'complete';
                highlightBlocks.push(
                    buildHighlight(
                        'Data coverage',
                        complete ? 'Complete coverage' : 'Partial coverage',
                        complete ? '' : 'warn',
                        complete ? '' : 'Review the alerts below to address missing inputs.'
                    )
                );
            }

            if (dataQuality.fundamental_source) {
                var sourceSeverity = dataQuality.fundamental_source === 'unavailable' ? 'warn' : '';
                var sourceNote = sourceSeverity ? 'Fundamental data provider did not return values.' : '';
                highlightBlocks.push(
                    buildHighlight('Fundamental source', dataQuality.fundamental_source, sourceSeverity, sourceNote)
                );
            }

            if (Array.isArray(dataQuality.fundamental_providers) && dataQuality.fundamental_providers.length) {
                highlightBlocks.push(
                    buildHighlight('Providers attempted', dataQuality.fundamental_providers.join(', '), 'info')
                );
            }

            if (dataQuality.sentiment_analyzer) {
                var analyzerKey = dataQuality.sentiment_analyzer.toString().toLowerCase();
                var analyzerSeverity = analyzerKey === 'vader' ? '' : 'warn';
                var analyzerLabel = analyzerKey === 'vader' ? 'VADER sentiment' : (analyzerKey === 'keyword' ? 'Keyword fallback' : 'No sentiment engine');
                var analyzerNote = analyzerSeverity ? 'Install vaderSentiment for richer tone detection.' : '';
                highlightBlocks.push(
                    buildHighlight('Sentiment engine', analyzerLabel, analyzerSeverity, analyzerNote)
                );
            }

            highlightPanel.innerHTML = highlightBlocks.join('');
        }

        var compositeScore = formatScore(scores.comprehensive);
        var technicalScore = formatScore(scores.technical);
        var fundamentalScore = formatScore(scores.fundamental);
        var sentimentScore = formatScore(scores.sentiment);

        var summaryHtml = '';
        summaryHtml += '<p><strong>' + stockName + (stockCode ? ' (' + stockCode + ')' : '') + '</strong> · Recommendation: ' + recommendation + '</p>';
        summaryHtml += '<p><strong>Scorecard:</strong> Technical ' + technicalScore + ' · Fundamental ' + fundamentalScore + ' · Sentiment ' + sentimentScore + ' · Composite ' + compositeScore + '</p>';
        summaryHtml += '<p><strong>Analysis date:</strong> ' + (report.analysis_date || new Date().toLocaleString()) + '</p>';
        if (dataQuality.analysis_completeness) {
            summaryHtml += '<p><strong>Data coverage:</strong> ' + (dataQuality.analysis_completeness === 'complete' ? 'Full fundamental and sentiment inputs' : 'Partial inputs – review alerts below') + '</p>';
        }
        if (dataQuality.fundamental_source) {
            summaryHtml += '<p><strong>Fundamentals source:</strong> ' + dataQuality.fundamental_source + '</p>';
        }
        if (fundamentalProviders.length) {
            summaryHtml += '<p><strong>Providers attempted:</strong> ' + fundamentalProviders.join(', ') + '</p>';
        }
        if (newsSources.length) {
            summaryHtml += '<p><strong>News sources:</strong> ' + newsSources.join(', ') + '</p>';
        }
        if (dataQuality.sentiment_analyzer) {
            var analyzerKeySummary = dataQuality.sentiment_analyzer.toString().toLowerCase();
            var analyzerLabelSummary = analyzerKeySummary === 'vader' ? 'VADER sentiment' : (analyzerKeySummary === 'keyword' ? 'Keyword fallback' : 'No sentiment engine');
            summaryHtml += '<p><strong>Sentiment engine:</strong> ' + analyzerLabelSummary + '</p>';
        }

        summaryPanel.innerHTML = summaryHtml;

        if (notesPanel) {
            var messages = dataQuality.messages || [];
            if (messages.length) {
                var listHtml = '<h4>Data quality alerts</h4><ul>';
                for (var i = 0; i < messages.length; i += 1) {
                    listHtml += '<li>' + messages[i] + '</li>';
                }
                listHtml += '</ul>';
                notesPanel.innerHTML = listHtml;
                notesPanel.style.display = 'block';
                addLog('Data quality warnings: ' + messages.join(' | '), 'warn');
            } else {
                notesPanel.innerHTML = '';
                notesPanel.style.display = 'none';
            }
        }

        var aiPanel = document.getElementById('aiStream');
        var aiSection = document.getElementById('aiSection');
        if (aiPanel && aiSection) {
            if (report && report.ai_analysis) {
                if (!trim(aiPanel.textContent || '')) {
                    aiPanel.textContent = report.ai_analysis;
                }
                aiSection.style.display = 'block';
            } else if (!trim(aiPanel.textContent || '')) {
                aiPanel.textContent = '';
                aiSection.style.display = 'none';
            }
        }
    }

    function appendAI(content) {
        var panel = document.getElementById('aiStream');
        var section = document.getElementById('aiSection');
        if (!panel) {
            return;
        }
        if (section && section.style.display !== 'block') {
            section.style.display = 'block';
        }
        panel.textContent = panel.textContent + content;
    }

    function resetAI() {
        var panel = document.getElementById('aiStream');
        var section = document.getElementById('aiSection');
        if (!panel) {
            return;
        }
        panel.textContent = '';
        if (section) {
            section.style.display = 'none';
        }
    }

    function resetDashboard() {
        var single = document.getElementById('singleSymbol');
        var batch = document.getElementById('batchSymbols');
        var logPanel = document.getElementById('logStream');
        var resultPanel = document.getElementById('resultPanel');
        var metaPanel = document.getElementById('resultMeta');
        var highlightPanel = document.getElementById('resultHighlights');
        if (single) {
            single.value = '';
        }
        if (batch) {
            batch.value = '';
        }
        if (logPanel) {
            logPanel.textContent = '';
        }
        if (resultPanel) {
            resultPanel.textContent = 'Waiting for results…';
        }
        if (metaPanel) {
            metaPanel.innerHTML = '<span>Waiting for ticker…</span>';
        }
        if (highlightPanel) {
            highlightPanel.innerHTML = '';
        }
        if (notesPanel) {
            notesPanel.innerHTML = '';
            notesPanel.style.display = 'none';
        }
        updateScores({});
        resetAI();
        currentReport = null;
        setStatus('Ready');
    }

    function sendJsonRequest(url, payload) {
        var body = JSON.stringify(payload || {});
        if (typeof window !== 'undefined' && window.fetch) {
            return window.fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: body
            }).then(function (response) {
                return response.json().catch(function () { return {}; }).then(function (data) {
                    return { ok: response.ok, status: response.status, data: data };
                });
            });
        }

        return new Promise(function (resolve, reject) {
            try {
                var xhr = new XMLHttpRequest();
                xhr.open('POST', url, true);
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.onreadystatechange = function () {
                    if (xhr.readyState === 4) {
                        var data = {};
                        try {
                            data = JSON.parse(xhr.responseText || '{}');
                        } catch (error) {
                            data = {};
                        }
                        var ok = xhr.status >= 200 && xhr.status < 300;
                        resolve({ ok: ok, status: xhr.status, data: data });
                    }
                };
                xhr.onerror = function () {
                    reject(new Error('Network request failed'));
                };
                xhr.send(body);
            } catch (error) {
                reject(error);
            }
        });
    }

    function connectSSE() {
        if (typeof EventSource === 'undefined') {
            return false;
        }
        try {
            if (eventSource) {
                eventSource.close();
            }
            var url = '/api/sse?client_id=' + encodeURIComponent(clientId);
            eventSource = new EventSource(url);
        } catch (error) {
            addLog('Unable to initialise streaming channel: ' + error, 'error');
            return false;
        }

        eventSource.onmessage = function (event) {
            lastHeartbeat = new Date().getTime();
            var payload = {};
            try {
                payload = JSON.parse(event.data || '{}');
            } catch (error) {
                addLog('Received malformed event payload.', 'error');
                return;
            }
            var type = payload.event;
            var data = payload.data || {};

            if (type === 'heartbeat') {
                return;
            }
            if (type === 'connected') {
                addLog('SSE channel ready.');
                return;
            }
            if (type === 'log') {
                addLog(data.message || '', data.type);
                return;
            }
            if (type === 'progress') {
                setStatus(data.message || 'Processing…');
                return;
            }
            if (type === 'scores_update') {
                updateScores(data.scores || {});
                return;
            }
            if (type === 'final_result') {
                showResult(data);
                setStatus('Analysis complete');
                return;
            }
            if (type === 'ai_stream') {
                appendAI(data.content || '');
                return;
            }
            if (type === 'analysis_complete') {
                addLog(data.message || 'Completed');
                return;
            }
            if (type === 'error') {
                addLog(data.error || 'Unknown error', 'error');
                setStatus('Error');
            }
        };

        eventSource.onerror = function () {
            setStatus('Connection lost. Reconnecting…');
            setTimeout(connectSSE, 2000);
        };
        return true;
    }

    function normaliseSymbol(value) {
        return trim(String(value || '')).toUpperCase();
    }

    function startSingleAnalysis(event) {
        if (event && event.preventDefault) {
            event.preventDefault();
        }
        var symbolInput = document.getElementById('singleSymbol');
        var symbol = symbolInput ? normaliseSymbol(symbolInput.value) : '';
        if (!symbol) {
            addLog('Please enter a ticker symbol.', 'warning');
            return;
        }
        resetAI();
        setStatus('Streaming analysis for ' + symbol + '…');
        addLog('Submitting ' + symbol + ' to the analyzer.');

        sendJsonRequest('/api/analyze_stream', {
            stock_code: symbol,
            client_id: clientId,
            target_market: DEFAULT_MARKET,
            enable_streaming: true
        }).then(function (result) {
            var ok = result.ok;
            var status = result.status;
            var data = result.data || {};
            if (!ok || !data.success) {
                addLog((data && data.error) ? data.error : 'Request failed (' + status + ')', 'error');
                setStatus('Error');
            }
        }).catch(function (error) {
            addLog('Network error: ' + error, 'error');
            setStatus('Error');
        });
    }

    function startBatchAnalysis(event) {
        if (event && event.preventDefault) {
            event.preventDefault();
        }
        var raw = document.getElementById('batchSymbols');
        var entries = [];
        if (raw && raw.value) {
            var parts = raw.value.split(',');
            for (var i = 0; i < parts.length; i += 1) {
                var cleaned = normaliseSymbol(parts[i]);
                if (cleaned) {
                    entries.push(cleaned);
                }
            }
        }

        if (!entries.length) {
            addLog('Please provide at least one ticker.', 'warning');
            return;
        }
        if (entries.length > 10) {
            addLog('Batch analysis supports up to 10 tickers.', 'warning');
            return;
        }

        resetAI();
        setStatus('Running batch analysis for ' + entries.length + ' tickers…');
        addLog('Submitting batch: ' + entries.join(', '));

        sendJsonRequest('/api/batch_analyze_stream', {
            stock_codes: entries,
            client_id: clientId,
            enable_streaming: true
        }).then(function (result) {
            var ok = result.ok;
            var status = result.status;
            var data = result.data || {};
            if (!ok || !data.success) {
                addLog((data && data.error) ? data.error : 'Batch request failed (' + status + ')', 'error');
                setStatus('Error');
            }
        }).catch(function (error) {
            addLog('Network error: ' + error, 'error');
            setStatus('Error');
        });
    }

    function bindControls() {
        var analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn && analyzeBtn.addEventListener) {
            analyzeBtn.addEventListener('click', startSingleAnalysis);
        }
        var batchBtn = document.getElementById('batchBtn');
        if (batchBtn && batchBtn.addEventListener) {
            batchBtn.addEventListener('click', startBatchAnalysis);
        }
        var resetBtn = document.getElementById('resetBtn');
        if (resetBtn && resetBtn.addEventListener) {
            resetBtn.addEventListener('click', resetDashboard);
        }
    }

    function initialiseDashboard() {
        bindControls();
        setSessionId();
        var streamingActive = connectSSE();
        if (!streamingActive) {
            addLog('Live streaming is unavailable in this browser. Requests will still run, but updates will appear after completion.', 'warning');
            setStatus('Streaming unavailable');
            return;
        }
        setInterval(function () {
            if (new Date().getTime() - lastHeartbeat > 60000) {
                addLog('No heartbeat from server. Reconnecting…', 'warning');
                connectSSE();
            }
        }, 15000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialiseDashboard);
    } else {
        initialiseDashboard();
    }
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
class StreamingAnalyzer:
    """Bridge between analyzer events and SSE clients."""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id

    def log(self, message: str, level: str = "info") -> None:
        sse_manager.send(self.client_id, "log", {"message": message, "type": level})

    def progress(self, message: str) -> None:
        sse_manager.send(self.client_id, "progress", {"message": message})

    def scores(self, scores: Dict[str, float]) -> None:
        sse_manager.send(self.client_id, "scores_update", {"scores": scores})

    def final_result(self, report: Dict) -> None:
        sse_manager.send(self.client_id, "final_result", report)

    def ai_stream(self, content: str) -> None:
        sse_manager.send(self.client_id, "ai_stream", {"content": content})

    def complete(self, message: str) -> None:
        sse_manager.send(self.client_id, "analysis_complete", {"message": message})

    def error(self, error_message: str) -> None:
        sse_manager.send(self.client_id, "error", {"error": error_message})


async def run_async(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


def perform_streaming_analysis(stock_code: str, client_id: str) -> Dict:
    streamer = StreamingAnalyzer(client_id)
    streamer.log(f"Starting streaming analysis for {stock_code}")
    try:
        streamer.progress("Fetching price history…")
        report = analyzer.analyze_stock(
            stock_code,
            enable_streaming=True,
            stream_callback=streamer.ai_stream,
        )
        streamer.scores(report.get("scores", {}))
        streamer.final_result(report)
        streamer.complete(f"Analysis completed for {stock_code}")
        return report
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Streaming analysis failed for %s", stock_code)
        error_message = str(exc)
        streamer.error(error_message)
        streamer.log(f"Analysis failed for {stock_code}: {error_message}", "error")
        raise


def perform_batch_analysis(stock_codes: list[str], client_id: str) -> Dict:
    streamer = StreamingAnalyzer(client_id)
    streamer.log(f"Starting batch run for {len(stock_codes)} tickers")
    reports = []
    for index, symbol in enumerate(stock_codes, start=1):
        try:
            streamer.progress(f"{index}/{len(stock_codes)} · analysing {symbol}")
            report = analyzer.analyze_stock(symbol, enable_streaming=False)
            reports.append(report)
            streamer.log(f"✓ {symbol} completed")
        except Exception as exc:  # pragma: no cover
            streamer.log(f"✖ {symbol} failed: {exc}", "error")
    if reports:
        aggregate = {
            "technical": sum(r["scores"]["technical"] for r in reports) / len(reports),
            "fundamental": sum(r["scores"]["fundamental"] for r in reports) / len(reports),
            "sentiment": sum(r["scores"]["sentiment"] for r in reports) / len(reports),
            "comprehensive": sum(r["scores"]["comprehensive"] for r in reports) / len(reports),
        }
        streamer.scores(aggregate)
        streamer.final_result({"reports": reports, "aggregate_scores": aggregate})
        streamer.complete("Batch run complete")
    else:
        streamer.error("Batch run failed: no successful symbols")
    return {"reports": reports}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    enabled, config = _auth_config()
    if not enabled:
        return redirect(url_for("index"))

    if request.method == "POST":
        password = request.form.get("password", "")
        expected = config.get("password", "")
        if password and hashlib.sha256(password.encode()).hexdigest() == hashlib.sha256(expected.encode()).hexdigest():
            session["authenticated"] = True
            session["login_time"] = datetime.utcnow().isoformat()
            return redirect(url_for("index"))
        return render_template_string(
            LOGIN_TEMPLATE,
            error="Incorrect password",
            timeout_minutes=int(config.get("session_timeout", 3600) / 60),
        )

    return render_template_string(
        LOGIN_TEMPLATE,
        timeout_minutes=int(config.get("session_timeout", 3600) / 60),
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/")
@require_auth
def index():
    enabled, _ = _auth_config()
    ui_context = analyzer.get_ui_context() if analyzer else {}
    return render_template_string(
        MAIN_TEMPLATE, auth_enabled=enabled, ui_context=ui_context
    )


@app.route("/api/sse")
@require_auth
def sse_stream():
    client_id = request.args.get("client_id")
    if not client_id:
        return "Missing client_id", 400

    queue = sse_manager.add_client(client_id)

    def stream():
        try:
            yield f"data: {json.dumps({'event': 'connected', 'data': {'client_id': client_id}})}\n\n"
            while True:
                try:
                    message = queue.get(timeout=25)
                    yield f"data: {json.dumps(message)}\n\n"
                except Empty:
                    heartbeat = {"event": "heartbeat", "data": {"timestamp": datetime.utcnow().isoformat()}}
                    yield f"data: {json.dumps(heartbeat)}\n\n"
        finally:
            sse_manager.remove_client(client_id)

    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.route("/api/analyze_stream", methods=["POST"])
@require_auth
def analyze_stream():
    if not analyzer:
        return jsonify({"success": False, "error": "Analyzer not initialised"}), 500

    payload = request.get_json(force=True)
    stock_code = payload.get("stock_code", "").strip()
    client_id = payload.get("client_id")
    if not stock_code:
        return jsonify({"success": False, "error": "Ticker cannot be empty"}), 400
    if not client_id:
        return jsonify({"success": False, "error": "Missing client ID"}), 400

    is_valid, message = analyzer.validate_stock_code(stock_code)
    if not is_valid:
        return jsonify({"success": False, "error": message}), 400

    with analysis_lock:
        if stock_code in analysis_tasks:
            return jsonify({"success": False, "error": "Analysis already running for this ticker"}), 409
        task_id = str(uuid.uuid4())
        analysis_tasks[stock_code] = {"task_id": task_id, "client_id": client_id, "start": time.time()}

    def task():
        try:
            result = perform_streaming_analysis(stock_code, client_id)
            with analysis_lock:
                analysis_results[stock_code] = result
        finally:
            with analysis_lock:
                analysis_tasks.pop(stock_code, None)

    executor.submit(task)
    return jsonify({"success": True, "task_id": task_id})


@app.route("/api/batch_analyze_stream", methods=["POST"])
@require_auth
def batch_analyze_stream():
    if not analyzer:
        return jsonify({"success": False, "error": "Analyzer not initialised"}), 500

    payload = request.get_json(force=True)
    stock_codes = payload.get("stock_codes", [])
    client_id = payload.get("client_id")

    if not stock_codes:
        return jsonify({"success": False, "error": "Ticker list cannot be empty"}), 400
    if len(stock_codes) > 10:
        return jsonify({"success": False, "error": "Batch analysis supports up to 10 tickers"}), 400
    if not client_id:
        return jsonify({"success": False, "error": "Missing client ID"}), 400

    invalid = []
    for code in stock_codes:
        ok, message = analyzer.validate_stock_code(code)
        if not ok:
            invalid.append(f"{code}: {message}")
    if invalid:
        return jsonify({"success": False, "error": "; ".join(invalid)}), 400

    executor.submit(lambda: perform_batch_analysis(stock_codes, client_id))
    return jsonify({"success": True})


@app.route("/api/status")
@require_auth
def status():
    with analysis_lock:
        active = [
            {
                "stock_code": key,
                "task_id": value["task_id"],
                "runtime": time.time() - value["start"],
            }
            for key, value in analysis_tasks.items()
        ]
    return jsonify({
        "success": True,
        "active_tasks": active,
        "max_workers": executor._max_workers,
    })


@app.route("/api/system_info")
@require_auth
def system_info():
    markets = analyzer.get_supported_markets() if analyzer else []
    return jsonify({
        "success": True,
        "analyzer": "EnhancedWebStockAnalyzer",
        "markets": markets,
        "streaming": analyzer.streaming_config if analyzer else {},
        "analysis_weights": analyzer.analysis_weights if analyzer else {},
    })


@app.route("/api/validate_stock", methods=["POST"])
@require_auth
def validate_stock():
    payload = request.get_json(force=True)
    code = payload.get("stock_code", "")
    valid, message = analyzer.validate_stock_code(code)
    normalized, market = (None, None)
    if valid:
        normalized, market = analyzer.normalize_stock_code(code)
    return jsonify({
        "success": valid,
        "message": message,
        "normalized": normalized,
        "market": market,
        "market_info": analyzer.market_config.get(market, {}) if market else {},
    })


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------
def bootstrap_analyzer():
    global analyzer
    logger.info("Initialising EnhancedWebStockAnalyzer…")
    analyzer = EnhancedWebStockAnalyzer()
    logger.info("Analyzer ready. Enabled markets: %s", analyzer.get_supported_markets())


def create_app():
    if not getattr(app, "analyzer_bootstrapped", False):
        bootstrap_analyzer()
        app.analyzer_bootstrapped = True
    return app


if __name__ == "__main__":  # pragma: no cover - development convenience
    bootstrap_analyzer()
    app.run(host="0.0.0.0", port=5000, debug=False)
