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
client_reports: Dict[str, Dict] = {}
analysis_lock = threading.Lock()
config_lock = threading.Lock()
client_reports_lock = threading.Lock()
sse_clients: Dict[str, Queue] = {}
sse_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=6)


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


def record_analysis_result(report: Dict, client_id: str, original_code: str) -> None:
    """Persist the latest analysis for reuse in follow-up chats."""

    if not report:
        return

    stock_code = report.get("stock_code") or original_code
    if stock_code:
        normalised = stock_code.upper()
    else:
        normalised = original_code.upper()

    with analysis_lock:
        analysis_results[normalised] = report
        analysis_results[original_code.upper()] = report

    if client_id:
        with client_reports_lock:
            client_reports[client_id] = report


def resolve_report_for_chat(client_id: Optional[str], stock_code: Optional[str]) -> Optional[Dict]:
    """Fetch the most relevant report for conversational follow-ups."""

    candidate = None
    if stock_code:
        with analysis_lock:
            candidate = analysis_results.get(stock_code.upper())
        if candidate:
            return candidate

    if client_id:
        with client_reports_lock:
            candidate = client_reports.get(client_id)
    return candidate


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
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@400;500;600&display=swap');
        body { display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; font-family:'Roboto Flex','Roboto',sans-serif; background:linear-gradient(145deg,#ede7f6 0%,#f7f2fa 60%,#fff 100%); color:#1d1b20; }
        .card { background:#fef7ff; border-radius:28px; padding:40px 44px; box-shadow:0 24px 48px rgba(103,80,164,0.2); border:1px solid #dad2e9; max-width:420px; width:100%; }
        h1 { margin:0 0 8px; font-size:1.8rem; font-weight:600; color:#1d1b20; }
        p { margin:0 0 28px; color:#49454f; line-height:1.5; }
        label { font-weight:600; font-size:0.95rem; display:block; margin-bottom:10px; color:#1d1b20; }
        input[type=password] { width:100%; padding:14px 16px; border-radius:16px; border:none; background:#ece6f0; box-shadow:inset 0 0 0 1px #cac4d0; font-size:1rem; transition:box-shadow 0.2s ease, background 0.2s ease; }
        input[type=password]:focus { outline:none; background:#e8def8; box-shadow:inset 0 0 0 2px #6750a4; }
        button { width:100%; padding:14px 18px; border:none; border-radius:999px; background:#6750a4; color:white; font-weight:600; letter-spacing:0.04em; cursor:pointer; margin-top:24px; box-shadow:0 12px 28px rgba(103,80,164,0.3); transition:transform 0.2s ease, box-shadow 0.2s ease; }
        button:hover { transform:translateY(-1px); box-shadow:0 16px 32px rgba(103,80,164,0.28); }
        .error { background:rgba(179,38,30,0.12); color:#b3261e; padding:12px 16px; border-radius:16px; border:1px solid rgba(179,38,30,0.18); margin-bottom:16px; font-size:0.95rem; }
        .meta { margin-top:28px; font-size:0.85rem; color:#625b71; text-align:center; }
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
            --md-sys-color-primary: #1f3a64;
            --md-sys-color-on-primary: #ffffff;
            --md-sys-color-primary-container: #dce5f7;
            --md-sys-color-on-primary-container: #13294a;
            --md-sys-color-secondary: #27476f;
            --md-sys-color-on-secondary: #ffffff;
            --md-sys-color-secondary-container: #e1e7f2;
            --md-sys-color-surface: #f6f8fc;
            --md-sys-color-surface-container-low: #ffffff;
            --md-sys-color-surface-container: #ffffff;
            --md-sys-color-surface-container-high: #eff2f8;
            --md-sys-color-surface-container-highest: #e7ebf4;
            --md-sys-color-outline: #d6dce7;
            --md-sys-color-outline-variant: #e5e8f0;
            --md-sys-color-on-surface: #1c2635;
            --md-sys-color-on-surface-variant: #576072;
            --md-sys-color-error: #c53d3d;
            --md-sys-color-success: #1b8a5a;
            --md-sys-color-warning: #d17b24;
            --md-sys-color-info: #4a4f69;
        }

        *, *::before, *::after {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--md-sys-color-surface);
            color: var(--md-sys-color-on-surface);
            min-height: 100vh;
        }

        a {
            color: inherit;
        }

        main {
            max-width: 1320px;
            margin: 0 auto;
            padding: 20px 24px 32px;
        }

        .hero {
            background: var(--md-sys-color-primary);
            border-radius: 14px;
            border: 1px solid rgba(19, 41, 74, 0.32);
            color: var(--md-sys-color-on-primary);
            padding: 18px 22px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin: 0 auto 18px;
            max-width: 1320px;
        }

        .hero h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.015em;
        }

        .hero p {
            margin: 0;
            max-width: 540px;
            font-size: 0.95rem;
            line-height: 1.45;
            color: rgba(255, 255, 255, 0.82);
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.78rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .hero-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #4fd17d;
            display: inline-block;
        }

        .hero-content {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 18px;
            flex-wrap: wrap;
        }

        .hero-content > div:first-child {
            flex: 1 1 320px;
        }

        .hero-actions {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .hero-button {
            align-self: flex-start;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 8px 14px;
            background: rgba(12, 29, 58, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.24);
            color: var(--md-sys-color-on-primary);
            font-weight: 600;
            letter-spacing: 0.03em;
            text-decoration: none;
            transition: background 0.2s ease;
        }

        .hero-button:hover {
            background: rgba(12, 29, 58, 0.4);
        }

        .hero-stat {
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 9px 12px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.14);
            min-width: 128px;
        }

        .hero-stat span {
            font-size: 0.72rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            opacity: 0.85;
        }

        .hero-stat strong {
            font-size: 1.02rem;
            font-weight: 600;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(260px, 1fr) minmax(320px, 1fr) minmax(260px, 0.9fr);
            gap: 18px;
            align-items: start;
        }

        .column {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .card {
            background: var(--md-sys-color-surface-container);
            border-radius: 14px;
            border: 1px solid var(--md-sys-color-outline);
            box-shadow: none;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
        }

        .card-header h2 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        .card-header p {
            margin: 4px 0 0;
            color: var(--md-sys-color-on-surface-variant);
            font-size: 0.95rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            border: 1px solid transparent;
        }

        .status-pill.live {
            background: rgba(51, 178, 73, 0.16);
            color: var(--md-sys-color-success);
            border-color: rgba(51, 178, 73, 0.32);
        }

        .status-pill.offline {
            background: rgba(179, 38, 30, 0.12);
            color: var(--md-sys-color-error);
            border-color: rgba(179, 38, 30, 0.18);
        }

        .status-pill.pending {
            background: rgba(255, 191, 0, 0.14);
            color: var(--md-sys-color-warning);
            border-color: rgba(255, 191, 0, 0.24);
        }

        .status-pill.disabled {
            background: rgba(73, 69, 79, 0.12);
            color: rgba(29, 27, 32, 0.52);
            border-color: rgba(73, 69, 79, 0.18);
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }

        .status-item {
            display: flex;
            gap: 10px;
            align-items: flex-start;
            padding: 12px;
            border-radius: 12px;
            background: var(--md-sys-color-surface-container-high);
            border: 1px solid var(--md-sys-color-outline-variant);
        }

        .status-item.ok {
            border-color: rgba(27, 138, 90, 0.3);
            background: rgba(27, 138, 90, 0.12);
        }

        .status-item.warn {
            border-color: rgba(209, 123, 36, 0.35);
            background: rgba(209, 123, 36, 0.12);
        }

        .config-card {
            padding: 0;
            background: transparent;
            border: none;
            box-shadow: none;
            gap: 0;
        }

        .config-panel {
            border: 1px solid var(--md-sys-color-outline-variant);
            border-radius: 16px;
            background: var(--md-sys-color-surface-container-high);
            overflow: hidden;
        }

        .config-panel summary {
            list-style: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            cursor: pointer;
            padding: 14px 16px;
            font-weight: 600;
            font-size: 0.92rem;
        }

        .config-panel summary::-webkit-details-marker {
            display: none;
        }

        .config-summary-meta {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .config-summary-meta span {
            font-weight: 500;
        }

        .config-summary-meta small {
            font-weight: 400;
            color: var(--md-sys-color-on-surface-variant);
        }

        .config-chevron {
            transition: transform 0.2s ease;
            font-size: 1.1rem;
        }

        .config-panel[open] .config-chevron {
            transform: rotate(180deg);
        }

        .config-form {
            padding: 0 16px 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .config-group {
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid var(--md-sys-color-outline-variant);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .config-group h3 {
            margin: 0;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--md-sys-color-on-surface-variant);
        }

        .config-field-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
        }

        .config-field-grid.three {
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        }

        .config-group label {
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--md-sys-color-on-surface-variant);
        }

        .config-group input[type="text"],
        .config-group input[type="number"],
        .config-group select {
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--md-sys-color-outline-variant);
            background: var(--md-sys-color-surface-container-highest);
            padding: 12px 14px;
            font-size: 0.95rem;
            color: var(--md-sys-color-on-surface);
        }

        .config-group input:focus,
        .config-group select:focus {
            outline: none;
            border-color: var(--md-sys-color-primary);
            box-shadow: 0 0 0 2px rgba(27, 78, 216, 0.16);
            background: #ffffff;
        }

        .config-market-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .config-market {
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }

        .config-market input {
            margin-top: 6px;
        }

        .config-market-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .config-market-info strong {
            font-size: 0.95rem;
        }

        .config-market-info span {
            font-size: 0.8rem;
            color: var(--md-sys-color-on-surface-variant);
        }

        .config-actions {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }

        .config-status {
            font-size: 0.85rem;
            color: var(--md-sys-color-on-surface-variant);
        }

        .config-status.pending {
            color: #0b3d91;
        }

        .config-status.success {
            color: #1b7f5c;
        }

        .config-status.error {
            color: #d13438;
        }

        .config-status.info {
            color: var(--md-sys-color-on-surface-variant);
        }

        .config-buttons {
            display: flex;
            gap: 8px;
        }

        .config-empty {
            font-size: 0.85rem;
            color: var(--md-sys-color-on-surface-variant);
        }

        .status-icon {
            width: 30px;
            height: 30px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            background: rgba(31, 58, 100, 0.12);
            color: var(--md-sys-color-primary);
            font-size: 0.85rem;
        }

        label {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--md-sys-color-on-surface);
        }

        input[type="text"], textarea {
            width: 100%;
            border: none;
            border-radius: 14px;
            padding: 14px 16px;
            background: var(--md-sys-color-surface-container-highest);
            color: var(--md-sys-color-on-surface);
            font-size: 0.97rem;
            box-shadow: inset 0 0 0 1px var(--md-sys-color-outline-variant);
            transition: box-shadow 0.2s ease, background 0.2s ease;
        }

        input[type="text"]:focus, textarea:focus {
            outline: none;
            background: var(--md-sys-color-surface-container-low);
            box-shadow: inset 0 0 0 2px var(--md-sys-color-primary);
        }

        textarea {
            min-height: 110px;
            resize: vertical;
        }

        .input-row {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .primary-button {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: none;
            border-radius: 12px;
            padding: 10px 16px;
            font-weight: 600;
            letter-spacing: 0.03em;
            background: var(--md-sys-color-primary);
            color: var(--md-sys-color-on-primary);
            cursor: pointer;
            box-shadow: none;
            transition: background 0.2s ease, transform 0.2s ease;
        }

        .primary-button:hover:not(:disabled) {
            background: #244879;
        }

        .primary-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .ghost-button {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 12px;
            border: 1px solid var(--md-sys-color-outline);
            padding: 9px 14px;
            font-weight: 600;
            letter-spacing: 0.03em;
            background: transparent;
            color: var(--md-sys-color-on-surface);
            cursor: pointer;
            transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
        }

        .ghost-button:hover:not(:disabled) {
            background: var(--md-sys-color-surface-container-high);
            border-color: var(--md-sys-color-outline-variant);
            color: var(--md-sys-color-primary);
        }

        .divider {
            height: 1px;
            background: var(--md-sys-color-outline-variant);
        }

        .hint {
            color: var(--md-sys-color-on-surface-variant);
            font-size: 0.85rem;
        }

        .status-banner {
            background: var(--md-sys-color-surface-container);
            border-radius: 14px;
            padding: 12px 16px;
            border: 1px solid var(--md-sys-color-outline);
            font-weight: 600;
            color: var(--md-sys-color-on-surface-variant);
        }

        .log-panel {
            background: var(--md-sys-color-surface-container);
            border-radius: 12px;
            padding: 14px;
            border: 1px solid var(--md-sys-color-outline-variant);
            max-height: 280px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', 'SFMono-Regular', 'Menlo', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
        }

        .log-entry {
            margin: 0 0 6px;
        }

        .log-entry.warn {
            color: var(--md-sys-color-warning);
        }

        .log-entry.error {
            color: var(--md-sys-color-error);
        }

        .score-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }

        .score-card {
            border-radius: 12px;
            padding: 14px;
            background: var(--md-sys-color-surface-container);
            border: 1px solid var(--md-sys-color-outline-variant);
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .score-card .label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--md-sys-color-on-surface-variant);
        }

        .score-card .value {
            font-size: 1.7rem;
            font-weight: 700;
        }

        .score-card .caption {
            color: var(--md-sys-color-on-surface-variant);
            font-size: 0.85rem;
        }

        .score-card.technical {
            border-color: rgba(31, 58, 100, 0.35);
            background: rgba(31, 58, 100, 0.08);
        }

        .score-card.technical .value {
            color: #1f3a64;
        }

        .score-card.fundamental {
            border-color: rgba(27, 138, 90, 0.35);
            background: rgba(27, 138, 90, 0.08);
        }

        .score-card.fundamental .value {
            color: #1b8a5a;
        }

        .score-card.sentiment {
            border-color: rgba(209, 123, 36, 0.35);
            background: rgba(209, 123, 36, 0.08);
        }

        .score-card.sentiment .value {
            color: #d17b24;
        }

        .score-card.composite {
            border-color: rgba(91, 75, 138, 0.32);
            background: rgba(91, 75, 138, 0.08);
        }

        .score-card.composite .value {
            color: #5b4b8a;
        }

        .result-shell {
            background: var(--md-sys-color-surface-container);
            border-radius: 12px;
            border: 1px solid var(--md-sys-color-outline-variant);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .result-title {
            font-weight: 600;
            font-size: 1rem;
        }

        .result-meta {
            display: flex;
            flex-direction: column;
            gap: 6px;
            color: var(--md-sys-color-on-surface-variant);
            font-size: 0.9rem;
        }

        .result-grid {
            display: grid;
            gap: 10px;
        }

        .result-item {
            border-radius: 12px;
            padding: 12px;
            border: 1px solid var(--md-sys-color-outline-variant);
            background: var(--md-sys-color-surface-container);
            box-shadow: none;
            border-left: 3px solid var(--md-sys-color-outline-variant);
        }

        .result-item.warn {
            border-color: rgba(209, 123, 36, 0.3);
            background: rgba(209, 123, 36, 0.08);
            border-left-color: rgba(209, 123, 36, 0.55);
        }

        .result-item.info {
            border-color: rgba(31, 58, 100, 0.25);
            border-left-color: rgba(31, 58, 100, 0.5);
            background: rgba(31, 58, 100, 0.06);
        }

        .result-heading {
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 8px;
        }

        .result-value {
            font-size: 1.05rem;
            font-weight: 500;
        }

        .result-note {
            font-size: 0.85rem;
            color: var(--md-sys-color-on-surface-variant);
        }

        .result-body {
            background: var(--md-sys-color-surface-container-highest);
            border-radius: 12px;
            padding: 16px;
            line-height: 1.55;
            font-size: 0.95rem;
            color: var(--md-sys-color-on-surface);
            white-space: pre-wrap;
        }

        .result-notes {
            border-radius: 12px;
            padding: 14px;
            border: 1px solid rgba(197, 61, 61, 0.35);
            background: rgba(197, 61, 61, 0.1);
            color: var(--md-sys-color-error);
            display: none;
        }

        .meta-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
        }

        .meta-card {
            border-radius: 12px;
            padding: 12px;
            background: var(--md-sys-color-surface-container);
            border: 1px solid var(--md-sys-color-outline-variant);
        }

        .meta-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--md-sys-color-on-surface-variant);
        }

        .meta-value {
            margin-top: 6px;
            font-weight: 600;
            font-size: 1rem;
        }

        .ai-section {
            display: none;
            flex-direction: column;
            gap: 12px;
        }

        .ai-section h3 {
            margin: 0;
            font-size: 1.15rem;
            font-weight: 600;
        }

        .ai-stream {
            background: var(--md-sys-color-secondary-container);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(76, 92, 124, 0.24);
            color: var(--md-sys-color-on-secondary);
            white-space: pre-wrap;
            line-height: 1.55;
        }

        .chat-card {
            display: flex;
            flex-direction: column;
            height: 100%;
            gap: 14px;
        }

        .chat-transcript {
            flex: 1;
            border-radius: 12px;
            background: var(--md-sys-color-surface-container-highest);
            border: 1px solid var(--md-sys-color-outline-variant);
            padding: 16px;
            color: var(--md-sys-color-on-surface);
            font-size: 0.95rem;
            overflow-y: auto;
            max-height: 480px;
        }

        .chat-messages {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .chat-message {
            display: flex;
            flex-direction: column;
            gap: 6px;
            border-radius: 12px;
            padding: 12px 14px;
            line-height: 1.5;
        }

        .chat-message.user {
            align-self: flex-end;
            background: var(--md-sys-color-primary-container);
            color: var(--md-sys-color-on-primary-container);
        }

        .chat-message.assistant {
            align-self: flex-start;
            background: var(--md-sys-color-surface-container);
            border: 1px solid var(--md-sys-color-outline-variant);
        }

        .chat-form {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .chat-form textarea {
            min-height: 100px;
        }

        .chat-actions {
            display: flex;
            gap: 12px;
        }

        .chat-form.chat-disabled textarea {
            opacity: 0.5;
        }

        .chat-form.chat-disabled .primary-button,
        .chat-form.chat-disabled .ghost-button {
            cursor: not-allowed;
            opacity: 0.6;
        }

        @media (max-width: 1180px) {
            main {
                padding: 16px;
            }

            .layout {
                grid-template-columns: 1fr;
                gap: 16px;
            }

            .hero {
                border-radius: 16px;
                margin-bottom: 16px;
            }

            .column:nth-child(3) {
                order: 3;
            }
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
        <section class="column chat-column">
            <div class="card config-card">
                <details class="config-panel" id="configDetails">
                    <summary>
                        <div class="config-summary-meta">
                            <span>Module configuration</span>
                            <small>AI provider, weights, and runtime parameters</small>
                        </div>
                        <span class="config-chevron">▾</span>
                    </summary>
                    <form id="configForm" class="config-form">
                        <div class="config-group">
                            <h3>AI provider</h3>
                            <label for="configProvider">Preferred provider</label>
                            <select id="configProvider">
                                <option value="openai">OpenAI</option>
                                <option value="anthropic">Anthropic</option>
                                <option value="zhipu">Zhipu AI</option>
                            </select>
                            <div class="config-field-grid">
                                <div>
                                    <label for="configModelOpenAI">OpenAI model</label>
                                    <input type="text" id="configModelOpenAI" placeholder="e.g. gpt-4o-mini">
                                </div>
                                <div>
                                    <label for="configModelAnthropic">Anthropic model</label>
                                    <input type="text" id="configModelAnthropic" placeholder="e.g. claude-3-haiku">
                                </div>
                                <div>
                                    <label for="configModelZhipu">Zhipu model</label>
                                    <input type="text" id="configModelZhipu" placeholder="e.g. chatglm_turbo">
                                </div>
                            </div>
                        </div>
                        <div class="config-group">
                            <h3>Analysis weighting</h3>
                            <div class="config-field-grid three">
                                <div>
                                    <label for="weightTechnical">Technical (%)</label>
                                    <input type="number" id="weightTechnical" min="0" max="100" step="1" placeholder="40">
                                </div>
                                <div>
                                    <label for="weightFundamental">Fundamental (%)</label>
                                    <input type="number" id="weightFundamental" min="0" max="100" step="1" placeholder="40">
                                </div>
                                <div>
                                    <label for="weightSentiment">Sentiment (%)</label>
                                    <input type="number" id="weightSentiment" min="0" max="100" step="1" placeholder="20">
                                </div>
                            </div>
                            <small class="hint">Values normalise automatically if they do not sum to 100.</small>
                        </div>
                        <div class="config-group">
                            <h3>Analysis parameters</h3>
                            <div class="config-field-grid">
                                <div>
                                    <label for="paramTechnicalWindow">Technical lookback (days)</label>
                                    <input type="number" id="paramTechnicalWindow" min="30" max="365" step="1" placeholder="180">
                                </div>
                                <div>
                                    <label for="paramNewsLimit">News limit (articles)</label>
                                    <input type="number" id="paramNewsLimit" min="10" max="200" step="5" placeholder="100">
                                </div>
                                <div>
                                    <label for="paramFinancialIndicators">Financial indicators</label>
                                    <input type="number" id="paramFinancialIndicators" min="5" max="40" step="1" placeholder="25">
                                </div>
                            </div>
                        </div>
                        <div class="config-group">
                            <h3>Markets</h3>
                            <div class="config-market-list" id="configMarketList">
                                <div class="config-empty">Loading markets…</div>
                            </div>
                        </div>
                        <div class="config-actions">
                            <span class="config-status" id="configStatus"></span>
                            <div class="config-buttons">
                                <button type="button" class="ghost-button" id="configResetBtn">Reset</button>
                                <button type="submit" class="primary-button" id="configSaveBtn">💾 Save settings</button>
                            </div>
                        </div>
                    </form>
                </details>
            </div>
            <div class="card chat-card">
                <div class="card-header">
                    <div>
                        <h2>Analyst follow-up chat</h2>
                        <p>Interactive Q&amp;A about the structured report.</p>
                    </div>
                    <span class="status-pill disabled" id="chatStatus">Awaiting report</span>
                </div>
                <div class="chat-transcript" id="chatTranscript">
                    <div class="chat-placeholder" id="chatPlaceholder">Run a streaming analysis to unlock Q&amp;A.</div>
                    <div class="chat-messages" id="chatMessages"></div>
                </div>
                <form id="chatForm" class="chat-form">
                    <label for="chatInput">Ask a question</label>
                    <textarea id="chatInput" placeholder="e.g. What risks should I monitor next quarter?" rows="3"></textarea>
                    <div class="chat-actions">
                        <button type="submit" class="primary-button chat-send" id="chatSend">💬 Send question</button>
                        <button type="button" class="ghost-button" id="chatClear">Clear chat</button>
                    </div>
                </form>
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
    var chatHistory = [];
    var activeChatReplies = {};
    var chatAvailable = false;
    var MAX_CHAT_HISTORY = 12;
    var configCache = null;

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

    function setConfigStatus(message, tone) {
        var status = document.getElementById('configStatus');
        if (!status) {
            return;
        }
        status.textContent = message || '';
        status.className = 'config-status' + (tone ? ' ' + tone : '');
    }

    function populateConfigForm(config) {
        configCache = config || {};
        var ai = (configCache.ai) || {};
        var models = ai.models || {};

        var providerSelect = document.getElementById('configProvider');
        if (providerSelect) {
            providerSelect.value = ai.model_preference || 'openai';
        }

        var openaiInput = document.getElementById('configModelOpenAI');
        if (openaiInput) {
            openaiInput.value = models.openai || '';
        }
        var anthropicInput = document.getElementById('configModelAnthropic');
        if (anthropicInput) {
            anthropicInput.value = models.anthropic || '';
        }
        var zhipuInput = document.getElementById('configModelZhipu');
        if (zhipuInput) {
            zhipuInput.value = models.zhipu || '';
        }

        var weights = configCache.analysis_weights || {};
        var tech = document.getElementById('weightTechnical');
        if (tech) {
            tech.value = (typeof weights.technical === 'number' && !isNaN(weights.technical)) ? Math.round(weights.technical * 100) : '';
        }
        var fundamental = document.getElementById('weightFundamental');
        if (fundamental) {
            fundamental.value = (typeof weights.fundamental === 'number' && !isNaN(weights.fundamental)) ? Math.round(weights.fundamental * 100) : '';
        }
        var sentiment = document.getElementById('weightSentiment');
        if (sentiment) {
            sentiment.value = (typeof weights.sentiment === 'number' && !isNaN(weights.sentiment)) ? Math.round(weights.sentiment * 100) : '';
        }

        var params = configCache.analysis_params || {};
        var technicalWindow = document.getElementById('paramTechnicalWindow');
        if (technicalWindow) {
            technicalWindow.value = params.technical_period_days || '';
        }
        var newsLimit = document.getElementById('paramNewsLimit');
        if (newsLimit) {
            newsLimit.value = params.max_news_count || '';
        }
        var indicatorCount = document.getElementById('paramFinancialIndicators');
        if (indicatorCount) {
            indicatorCount.value = params.financial_indicators_count || '';
        }

        var marketList = document.getElementById('configMarketList');
        if (marketList) {
            marketList.innerHTML = '';
            var markets = configCache.markets || {};
            var keys = Object.keys(markets);
            if (!keys.length) {
                var empty = document.createElement('div');
                empty.className = 'config-empty';
                empty.textContent = 'No markets configured.';
                marketList.appendChild(empty);
            } else {
                keys.sort();
                keys.forEach(function (code) {
                    var meta = markets[code] || {};
                    var wrapper = document.createElement('label');
                    wrapper.className = 'config-market';

                    var checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.id = 'market-' + code;
                    checkbox.dataset.market = code;
                    checkbox.dataset.currency = meta.currency || '';
                    checkbox.dataset.timezone = meta.timezone || '';
                    checkbox.dataset.tradingHours = meta.trading_hours || '';
                    checkbox.checked = !!meta.enabled;
                    wrapper.appendChild(checkbox);

                    var info = document.createElement('div');
                    info.className = 'config-market-info';
                    var strong = document.createElement('strong');
                    strong.textContent = meta.name || code;
                    info.appendChild(strong);
                    var details = [];
                    if (meta.currency) { details.push(meta.currency); }
                    if (meta.timezone) { details.push(meta.timezone); }
                    if (meta.trading_hours) { details.push(meta.trading_hours); }
                    if (details.length) {
                        var span = document.createElement('span');
                        span.textContent = details.join(' · ');
                        info.appendChild(span);
                    }
                    wrapper.appendChild(info);
                    marketList.appendChild(wrapper);
                });
            }
        }
    }

    function extractConfigPayload() {
        var payload = { ai: { models: {} }, analysis_weights: {}, analysis_params: {}, markets: {} };

        var providerSelect = document.getElementById('configProvider');
        if (providerSelect) {
            payload.ai.model_preference = providerSelect.value || 'openai';
        }

        var openaiInput = document.getElementById('configModelOpenAI');
        if (openaiInput) {
            payload.ai.models.openai = trim(String(openaiInput.value || ''));
        }
        var anthropicInput = document.getElementById('configModelAnthropic');
        if (anthropicInput) {
            payload.ai.models.anthropic = trim(String(anthropicInput.value || ''));
        }
        var zhipuInput = document.getElementById('configModelZhipu');
        if (zhipuInput) {
            payload.ai.models.zhipu = trim(String(zhipuInput.value || ''));
        }

        var tech = document.getElementById('weightTechnical');
        if (tech && tech.value !== '') {
            var techVal = parseFloat(tech.value);
            if (!isNaN(techVal)) {
                payload.analysis_weights.technical = techVal / 100;
            }
        }
        var fund = document.getElementById('weightFundamental');
        if (fund && fund.value !== '') {
            var fundVal = parseFloat(fund.value);
            if (!isNaN(fundVal)) {
                payload.analysis_weights.fundamental = fundVal / 100;
            }
        }
        var sent = document.getElementById('weightSentiment');
        if (sent && sent.value !== '') {
            var sentVal = parseFloat(sent.value);
            if (!isNaN(sentVal)) {
                payload.analysis_weights.sentiment = sentVal / 100;
            }
        }

        var technicalWindow = document.getElementById('paramTechnicalWindow');
        if (technicalWindow && technicalWindow.value !== '') {
            var daysVal = parseInt(technicalWindow.value, 10);
            if (!isNaN(daysVal)) {
                payload.analysis_params.technical_period_days = daysVal;
            }
        }
        var newsLimit = document.getElementById('paramNewsLimit');
        if (newsLimit && newsLimit.value !== '') {
            var newsVal = parseInt(newsLimit.value, 10);
            if (!isNaN(newsVal)) {
                payload.analysis_params.max_news_count = newsVal;
            }
        }
        var indicatorCount = document.getElementById('paramFinancialIndicators');
        if (indicatorCount && indicatorCount.value !== '') {
            var indVal = parseInt(indicatorCount.value, 10);
            if (!isNaN(indVal)) {
                payload.analysis_params.financial_indicators_count = indVal;
            }
        }

        var baseMarkets = (configCache && configCache.markets) || {};
        Object.keys(baseMarkets).forEach(function (code) {
            var checkbox = document.getElementById('market-' + code);
            var meta = baseMarkets[code] || {};
            payload.markets[code] = {
                enabled: checkbox ? checkbox.checked : !!meta.enabled,
                name: meta.name || code,
                currency: meta.currency || (checkbox ? checkbox.dataset.currency || '' : ''),
                timezone: meta.timezone || (checkbox ? checkbox.dataset.timezone || '' : ''),
                trading_hours: meta.trading_hours || (checkbox ? checkbox.dataset.tradingHours || '' : ''),
            };
        });

        return payload;
    }

    function loadConfigSettings() {
        setConfigStatus('Loading settings…', 'pending');
        sendJsonRequest('/api/config', null, 'GET').then(function (result) {
            var data = result.data || {};
            if (!result.ok || !data.success) {
                setConfigStatus(data.error || 'Unable to load configuration.', 'error');
                return;
            }
            populateConfigForm(data.config || {});
            setConfigStatus('Settings loaded', 'info');
        }).catch(function (error) {
            setConfigStatus('Load failed: ' + error, 'error');
        });
    }

    function saveConfigSettings(event) {
        if (event && event.preventDefault) {
            event.preventDefault();
        }
        setConfigStatus('Saving…', 'pending');
        var payload = extractConfigPayload();
        sendJsonRequest('/api/config', payload, 'POST').then(function (result) {
            var data = result.data || {};
            if (!result.ok || !data.success) {
                setConfigStatus(data.error || 'Unable to save settings.', 'error');
                return;
            }
            populateConfigForm(data.config || {});
            setConfigStatus('Settings saved', 'success');
        }).catch(function (error) {
            setConfigStatus('Save failed: ' + error, 'error');
        });
    }

    function resetConfigForm(event) {
        if (event && event.preventDefault) {
            event.preventDefault();
        }
        populateConfigForm(configCache || {});
        setConfigStatus('Settings reset', 'info');
    }

    function bindConfigControls() {
        var form = document.getElementById('configForm');
        if (form && form.addEventListener) {
            form.addEventListener('submit', saveConfigSettings);
        }
        var resetBtn = document.getElementById('configResetBtn');
        if (resetBtn && resetBtn.addEventListener) {
            resetBtn.addEventListener('click', resetConfigForm);
        }
        var details = document.getElementById('configDetails');
        if (details && details.addEventListener) {
            details.addEventListener('toggle', function () {
                if (details.open && !configCache) {
                    loadConfigSettings();
                }
            });
        }
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

    function setChatStatus(text, mode) {
        var pill = document.getElementById('chatStatus');
        if (!pill) {
            return;
        }
        pill.textContent = text;
        pill.classList.remove('live', 'offline', 'pending', 'disabled');
        if (mode) {
            pill.classList.add(mode);
        }
    }

    function setChatAvailability(available) {
        chatAvailable = !!(available && currentReport);
        var sendBtn = document.getElementById('chatSend');
        var input = document.getElementById('chatInput');
        var form = document.getElementById('chatForm');
        if (sendBtn) {
            sendBtn.disabled = !chatAvailable;
        }
        if (input) {
            input.disabled = !chatAvailable;
        }
        if (form) {
            if (chatAvailable) {
                form.classList.remove('chat-disabled');
            } else {
                form.classList.add('chat-disabled');
            }
        }
    }

    function showChatPlaceholder(text) {
        var placeholder = document.getElementById('chatPlaceholder');
        if (!placeholder) {
            return;
        }
        if (text) {
            placeholder.textContent = text;
            placeholder.style.display = 'block';
        } else {
            placeholder.textContent = '';
            placeholder.style.display = 'none';
        }
    }

    function hideChatPlaceholder() {
        showChatPlaceholder('');
    }

    function scrollChatToBottom() {
        var transcript = document.getElementById('chatTranscript');
        if (transcript) {
            transcript.scrollTop = transcript.scrollHeight;
        }
    }

    function addChatMessage(role, content, chatId) {
        var container = document.getElementById('chatMessages');
        if (!container) {
            return null;
        }
        hideChatPlaceholder();
        var messageWrapper = document.createElement('div');
        messageWrapper.className = 'chat-message ' + role;
        if (chatId) {
            messageWrapper.dataset.chatId = chatId;
        }
        var meta = document.createElement('div');
        meta.className = 'chat-meta';
        meta.textContent = role === 'user' ? 'You' : 'Analyst AI';
        var bubble = document.createElement('div');
        bubble.className = 'chat-bubble ' + role;
        bubble.textContent = content || '';
        messageWrapper.appendChild(meta);
        messageWrapper.appendChild(bubble);
        container.appendChild(messageWrapper);
        scrollChatToBottom();
        return bubble;
    }

    function appendChatChunk(chatId, chunk) {
        if (!chatId) {
            return null;
        }
        var bubble = activeChatReplies[chatId];
        if (!bubble || !bubble.parentNode) {
            bubble = addChatMessage('assistant', '', chatId);
            if (bubble) {
                activeChatReplies[chatId] = bubble;
            }
        }
        if (!bubble) {
            return null;
        }
        if (chunk) {
            bubble.textContent = (bubble.textContent || '') + chunk;
        }
        scrollChatToBottom();
        return bubble;
    }

    function pushChatHistory(role, content) {
        if (!content) {
            return;
        }
        chatHistory.push({ role: role, content: content });
        if (chatHistory.length > MAX_CHAT_HISTORY) {
            chatHistory = chatHistory.slice(-MAX_CHAT_HISTORY);
        }
    }

    function resetChat(keepAvailability) {
        chatHistory = [];
        activeChatReplies = {};
        var container = document.getElementById('chatMessages');
        if (container) {
            container.innerHTML = '';
        }
        if (keepAvailability) {
            hideChatPlaceholder();
        } else {
            showChatPlaceholder('Run a streaming analysis to unlock Q&A.');
            setChatAvailability(false);
            setChatStatus('Awaiting report', 'disabled');
        }
    }

    function prepareChatForReport(report) {
        resetChat(true);
        var name = '';
        if (report && report.stock_name) {
            name = report.stock_name;
        } else if (report && report.stock_code) {
            name = report.stock_code;
        }
        showChatPlaceholder(
            name
                ? 'Chat ready. Ask a question about ' + name + '.'
                : 'Chat ready. Ask a question about this ticker.'
        );
        setChatAvailability(true);
        setChatStatus('Ready for questions', 'live');
    }

    function handleChatStreamEvent(data) {
        if (!data || !data.chat_id) {
            return;
        }
        appendChatChunk(data.chat_id, data.content || '');
        setChatStatus('Streaming answer…', 'pending');
    }

    function handleChatCompleteEvent(data) {
        if (!data || !data.chat_id) {
            return;
        }
        var response = data.response || '';
        var bubble = appendChatChunk(data.chat_id, '');
        if (bubble && response) {
            bubble.textContent = response;
        } else if (bubble && !bubble.textContent) {
            bubble.textContent = 'No additional commentary was generated.';
        }
        if (response) {
            pushChatHistory('assistant', response);
        }
        delete activeChatReplies[data.chat_id];
        setChatStatus('Ready for questions', 'live');
    }

    function handleChatErrorEvent(data) {
        var error = (data && data.error) ? data.error : 'Unable to generate chat response.';
        if (data && data.chat_id) {
            var bubble = appendChatChunk(data.chat_id, '');
            if (bubble) {
                bubble.textContent = '⚠️ ' + error;
            }
            delete activeChatReplies[data.chat_id];
        } else {
            addChatMessage('assistant', '⚠️ ' + error);
        }
        addLog('Chat error: ' + error, 'error');
        setChatStatus('Ready for questions', chatAvailable ? 'live' : 'disabled');
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
        var summaryPanel = document.getElementById('resultPanel');
        var metaPanel = document.getElementById('resultMeta');
        var highlightPanel = document.getElementById('resultHighlights');
        var notesPanel = document.getElementById('resultNotes');
        if (!summaryPanel) {
            return;
        }

        if (report && report.stock_code) {
            currentReport = report;
            prepareChatForReport(report);
        } else {
            currentReport = null;
            resetChat(false);
            showChatPlaceholder('Chat is available after running a single streaming analysis.');
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
                var indicatorText = hasNumber(indicatorCount) && indicatorCount > 0 ? formatInteger(indicatorCount) : 'Not available';
                var indicatorNote = indicatorSeverity ? 'Fundamental metrics were not returned by the active providers.' : '';
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
                var coverageKey = dataQuality.analysis_completeness;
                var coverageLabel = 'Partial coverage';
                var coverageSeverity = '';
                var coverageNote = '';
                if (coverageKey === 'complete') {
                    coverageLabel = 'Complete coverage';
                } else if (coverageKey === 'minimal') {
                    coverageLabel = 'Limited coverage';
                    coverageSeverity = 'warn';
                    coverageNote = 'Fundamental and sentiment datasets are both missing.';
                } else {
                    coverageSeverity = 'info';
                    coverageNote = 'Review the alerts below to address missing inputs.';
                }
                highlightBlocks.push(
                    buildHighlight('Data coverage', coverageLabel, coverageSeverity, coverageNote)
                );
            }

            if (dataQuality.fundamental_source) {
                var sourceSeverity = dataQuality.fundamental_status === 'ok' ? '' : 'warn';
                var sourceNote = '';
                if (sourceSeverity === 'warn') {
                    sourceNote = 'Fundamental data provider did not return values.';
                }
                highlightBlocks.push(
                    buildHighlight('Fundamental source', dataQuality.fundamental_source, sourceSeverity, sourceNote)
                );
            }

            if (Array.isArray(dataQuality.fundamental_providers) && dataQuality.fundamental_providers.length) {
                highlightBlocks.push(
                    buildHighlight('Providers attempted', dataQuality.fundamental_providers.join(', '), 'info')
                );
            }

            if (dataQuality.sentiment_label) {
                var sentimentSeverity = '';
                if (dataQuality.sentiment_status === 'warn') {
                    sentimentSeverity = 'warn';
                } else if (dataQuality.sentiment_status === 'info') {
                    sentimentSeverity = 'info';
                }
                var sentimentNote = '';
                if (dataQuality.sentiment_status === 'warn') {
                    sentimentNote = 'No sentiment feed is currently available.';
                } else if (dataQuality.sentiment_status === 'info' && dataQuality.sentiment_analyzer && dataQuality.sentiment_analyzer.toString().toLowerCase() === 'keyword') {
                    sentimentNote = 'Install vaderSentiment for richer tone detection.';
                }
                highlightBlocks.push(
                    buildHighlight('Sentiment engine', dataQuality.sentiment_label, sentimentSeverity, sentimentNote)
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
            if (dataQuality.analysis_completeness === 'complete') {
                summaryHtml += '<p><strong>Data coverage:</strong> Full fundamental and sentiment inputs</p>';
            } else if (dataQuality.analysis_completeness === 'minimal') {
                summaryHtml += '<p><strong>Data coverage:</strong> No fundamental or sentiment data available</p>';
            } else {
                summaryHtml += '<p><strong>Data coverage:</strong> Partial inputs – review alerts below</p>';
            }
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
        if (dataQuality.sentiment_label) {
            summaryHtml += '<p><strong>Sentiment engine:</strong> ' + dataQuality.sentiment_label + '</p>';
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
        var notesPanel = document.getElementById('resultNotes');
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
        resetChat(false);
        currentReport = null;
        setStatus('Ready');
    }

    function sendJsonRequest(url, payload, method) {
        var verb = (method || 'POST').toUpperCase();
        var hasBody = !(verb === 'GET' || verb === 'HEAD');
        var body = hasBody ? JSON.stringify(payload || {}) : null;

        if (typeof window !== 'undefined' && window.fetch) {
            var options = {
                method: verb,
                headers: { 'Content-Type': 'application/json' }
            };
            if (hasBody) {
                options.body = body;
            }
            return window.fetch(url, options).then(function (response) {
                return response
                    .json()
                    .catch(function () { return {}; })
                    .then(function (data) {
                        return { ok: response.ok, status: response.status, data: data };
                    });
            });
        }

        return new Promise(function (resolve, reject) {
            try {
                var xhr = new XMLHttpRequest();
                xhr.open(verb, url, true);
                if (hasBody) {
                    xhr.setRequestHeader('Content-Type', 'application/json');
                }
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
                if (hasBody && body !== null) {
                    xhr.send(body);
                } else {
                    xhr.send();
                }
            } catch (error) {
                reject(error);
            }
        });
    }

    function startChatRequest(message, historySnapshot) {
        var payload = {
            client_id: clientId,
            message: message,
            stock_code: currentReport && currentReport.stock_code ? currentReport.stock_code : null,
            conversation: Array.isArray(historySnapshot) ? historySnapshot : [],
        };

        sendJsonRequest('/api/chat', payload).then(function (result) {
            if (!result || !result.ok) {
                var errorText = (result && result.data && result.data.error) ? result.data.error : 'Unable to start chat.';
                addLog('Chat request rejected: ' + errorText, 'error');
                addChatMessage('assistant', '⚠️ ' + errorText);
                setChatStatus('Ready for questions', chatAvailable ? 'live' : 'disabled');
                return;
            }

            var data = result.data || {};
            if (!data.chat_id) {
                addLog('Chat response missing identifier.', 'error');
                addChatMessage('assistant', '⚠️ Chat session could not be created.');
                setChatStatus('Ready for questions', chatAvailable ? 'live' : 'disabled');
                return;
            }

            pushChatHistory('user', message);
            var bubble = appendChatChunk(data.chat_id, '');
            if (bubble && !bubble.textContent) {
                bubble.textContent = '';
            }
        }).catch(function (error) {
            addLog('Chat network error: ' + error, 'error');
            addChatMessage('assistant', '⚠️ Unable to contact chat service.');
            setChatStatus('Ready for questions', chatAvailable ? 'live' : 'disabled');
        });
    }

    function handleChatSubmit(event) {
        if (event && event.preventDefault) {
            event.preventDefault();
        }
        if (!chatAvailable || !currentReport) {
            addLog('Run an analysis before starting a chat.', 'warning');
            return;
        }
        var input = document.getElementById('chatInput');
        var message = input ? trim(input.value || '') : '';
        if (!message) {
            addLog('Please enter a chat question.', 'warning');
            return;
        }
        addChatMessage('user', message);
        if (input) {
            input.value = '';
        }
        setChatStatus('Waiting for response…', 'pending');
        startChatRequest(message, chatHistory.slice(-MAX_CHAT_HISTORY));
    }

    function handleChatClear(event) {
        if (event && event.preventDefault) {
            event.preventDefault();
        }
        if (!chatAvailable) {
            resetChat(false);
            return;
        }
        resetChat(true);
        setChatAvailability(true);
        showChatPlaceholder('Chat cleared. Ask another question.');
        setChatStatus('Ready for questions', 'live');
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
            if (type === 'chat_stream') {
                handleChatStreamEvent(data);
                return;
            }
            if (type === 'chat_complete') {
                handleChatCompleteEvent(data);
                return;
            }
            if (type === 'chat_error') {
                handleChatErrorEvent(data);
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
        var chatForm = document.getElementById('chatForm');
        if (chatForm && chatForm.addEventListener) {
            chatForm.addEventListener('submit', handleChatSubmit);
        }
        var chatClear = document.getElementById('chatClear');
        if (chatClear && chatClear.addEventListener) {
            chatClear.addEventListener('click', handleChatClear);
        }
        bindConfigControls();
    }

    function initialiseDashboard() {
        bindControls();
        setSessionId();
        resetChat(false);
        loadConfigSettings();
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


class ChatStreamer:
    """Stream chat responses back to the requesting client."""

    def __init__(self, client_id: str, chat_id: str) -> None:
        self.client_id = client_id
        self.chat_id = chat_id

    def stream(self, content: str) -> None:
        sse_manager.send(
            self.client_id,
            "chat_stream",
            {"chat_id": self.chat_id, "content": content},
        )

    def complete(self, response: str) -> None:
        sse_manager.send(
            self.client_id,
            "chat_complete",
            {"chat_id": self.chat_id, "response": response},
        )

    def error(self, error_message: str) -> None:
        sse_manager.send(
            self.client_id,
            "chat_error",
            {"chat_id": self.chat_id, "error": error_message},
        )


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
        record_analysis_result(report, client_id, stock_code)
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
            perform_streaming_analysis(stock_code, client_id)
        finally:
            with analysis_lock:
                analysis_tasks.pop(stock_code, None)

    executor.submit(task)
    return jsonify({"success": True, "task_id": task_id})


@app.route("/api/chat", methods=["POST"])
@require_auth
def chat_follow_up():
    if not analyzer:
        return jsonify({"success": False, "error": "Analyzer not initialised"}), 500

    payload = request.get_json(force=True)
    message = (payload.get("message") or "").strip()
    client_id = payload.get("client_id")
    stock_code = payload.get("stock_code")
    conversation = payload.get("conversation") or []

    if not client_id:
        return jsonify({"success": False, "error": "Missing client ID"}), 400
    if not message:
        return jsonify({"success": False, "error": "Question cannot be empty"}), 400

    report = resolve_report_for_chat(client_id, stock_code)
    if not report:
        return jsonify({"success": False, "error": "Run an analysis before starting a chat."}), 400

    chat_id = str(uuid.uuid4())
    streamer = ChatStreamer(client_id, chat_id)

    def task():
        try:
            response = analyzer.generate_chat_followup(
                report,
                conversation if isinstance(conversation, list) else [],
                message,
                enable_streaming=True,
                stream_callback=streamer.stream,
            )
            streamer.complete(response)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Chat follow-up failed")
            streamer.error(str(exc))

    executor.submit(task)
    return jsonify({"success": True, "chat_id": chat_id})


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


@app.route("/api/config", methods=["GET", "POST"])
@require_auth
def configure():
    if not analyzer:
        return jsonify({"success": False, "error": "Analyzer not initialised"}), 500

    if request.method == "GET":
        return jsonify({"success": True, "config": analyzer.get_editable_config()})

    payload = request.get_json(force=True) or {}
    try:
        with config_lock:
            updated = analyzer.update_runtime_config(payload)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({"success": True, "config": updated})


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
