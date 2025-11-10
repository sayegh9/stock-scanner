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
    <title>Enhanced U.S. Stock Analyzer</title>
    <style>
        :root { color-scheme: light dark; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; margin: 0; padding: 0; color: #0b1120; }
        .wrapper { max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }
        header { margin-bottom: 24px; color: white; }
        header h1 { margin: 0 0 8px 0; font-size: 32px; font-weight: 700; }
        header p { margin: 0; max-width: 640px; color: rgba(255,255,255,0.75); }
        .grid { display: grid; grid-template-columns: 360px 1fr; gap: 24px; }
        .panel { background: white; border-radius: 18px; box-shadow: 0 30px 60px rgba(15, 23, 42, 0.25); padding: 24px; }
        h2 { margin-top: 0; font-size: 20px; }
        label { display: block; font-weight: 600; font-size: 14px; margin-bottom: 8px; color: #1f2937; }
        input, textarea { width: 100%; padding: 12px; border-radius: 10px; border: 1px solid #d1d5db; font-size: 14px; }
        textarea { min-height: 120px; }
        button { display: inline-flex; align-items: center; gap: 8px; padding: 12px 20px; border-radius: 10px; border: none; font-weight: 600; cursor: pointer; background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; }
        button.secondary { background: #f3f4f6; color: #1f2937; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }
        .status { margin-top: 16px; padding: 12px; border-radius: 10px; font-size: 14px; background: #e0f2fe; color: #0c4a6e; }
        .log { background: #0b1120; color: #e2e8f0; font-family: 'Consolas', monospace; font-size: 13px; border-radius: 12px; padding: 16px; height: 220px; overflow-y: auto; }
        .log-entry { margin-bottom: 4px; }
        .scores { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
        .score-card { background: #f3f4f6; border-radius: 12px; padding: 16px; text-align: center; }
        .score-card .value { font-size: 28px; font-weight: 700; }
        .result { background: #f9fafb; border-radius: 12px; padding: 20px; min-height: 260px; font-size: 14px; overflow-y: auto; }
        .ai-stream { font-family: 'Segoe UI', sans-serif; white-space: pre-wrap; background: #fff; border-radius: 12px; padding: 16px; border: 1px solid #e5e7eb; margin-top: 16px; }
        .badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(37, 99, 235, 0.15); color: #1d4ed8; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
        @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="wrapper">
        <header>
            <div class="badge">🇺🇸 Streaming U.S. equity intelligence</div>
            <h1>Enhanced U.S. Stock Analyzer</h1>
            <p>Submit a U.S. ticker to stream technical, fundamental, and sentiment insights in real time. Batch jobs analyse up to 10 symbols sequentially.</p>
        </header>
        <div class="grid">
            <section class="panel">
                <h2>Run analysis</h2>
                <label for="singleSymbol">Ticker symbol</label>
                <input id="singleSymbol" placeholder="e.g. AAPL" autocomplete="off">
                <div class="actions">
                    <button id="analyzeBtn" type="button">🚀 Stream analysis</button>
                    <button id="resetBtn" class="secondary" type="button">Clear</button>
                </div>
                <hr style="margin: 24px 0; border: none; border-top: 1px solid #e5e7eb;">
                <h2>Batch workflow</h2>
                <label for="batchSymbols">Enter up to 10 tickers separated by commas</label>
                <textarea id="batchSymbols" placeholder="AAPL, MSFT, NVDA"></textarea>
                <div class="actions">
                    <button id="batchBtn" type="button">📦 Start batch</button>
                </div>
                <div class="status" id="systemStatus">Ready</div>
                <div style="margin-top: 16px; font-size: 12px; color: #6b7280;">
                    <strong>Session:</strong> <span id="sessionId"></span><br>
                    <strong>Streaming:</strong> Server-Sent Events
                </div>
            </section>
            <section class="panel">
                <h2>Live feed</h2>
                <div class="scores">
                    <div class="score-card"><div>Technical</div><div class="value" id="technicalScore">--</div></div>
                    <div class="score-card"><div>Fundamental</div><div class="value" id="fundamentalScore">--</div></div>
                    <div class="score-card"><div>Sentiment</div><div class="value" id="sentimentScore">--</div></div>
                    <div class="score-card"><div>Composite</div><div class="value" id="compositeScore">--</div></div>
                </div>
                <div class="log" id="logStream"></div>
                <div class="result" id="resultPanel">Waiting for results…</div>
                <div class="ai-stream" id="aiStream" style="display:none;"></div>
            </section>
        </div>
    </div>
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
        if (value === undefined || value === null || value === '') {
            return '--';
        }
        var numberValue = Number(value);
        if (isNaN(numberValue)) {
            return '--';
        }
        return numberValue.toFixed(1);
    }

    function updateScores(scores) {
        document.getElementById('technicalScore').textContent = formatScore(scores.technical);
        document.getElementById('fundamentalScore').textContent = formatScore(scores.fundamental);
        document.getElementById('sentimentScore').textContent = formatScore(scores.sentiment);
        document.getElementById('compositeScore').textContent = formatScore(scores.comprehensive);
    }

    function safeNumber(value) {
        if (typeof value === 'number' && !isNaN(value)) {
            return value.toFixed(2);
        }
        return 'N/A';
    }

    function showResult(report) {
        currentReport = report;
        var container = document.getElementById('resultPanel');
        if (!container) {
            return;
        }
        var market = report && report.market_info ? report.market_info : {};
        var stockName = report && report.stock_name ? report.stock_name : 'Unknown';
        var stockCode = report && report.stock_code ? report.stock_code : '';
        var currency = market.currency || 'USD';
        var priceInfo = report && report.price_info ? report.price_info : {};
        var recommendation = report && report.recommendation ? report.recommendation : 'N/A';
        var scores = report && report.scores ? report.scores : {};

        var html = '';
        html += '<h3 style="margin-top:0;">' + stockName + (stockCode ? ' (' + stockCode + ')' : '') + '</h3>';
        html += '<p><strong>Market:</strong> ' + (market.name || 'U.S. equities') + ' · ' + currency + '</p>';
        html += '<p><strong>Current price:</strong> ' + safeNumber(priceInfo.current_price) + ' ' + currency +
            ' · <strong>Change:</strong> ' + safeNumber(priceInfo.price_change) + '%</p>';
        html += '<p><strong>Recommendation:</strong> ' + recommendation + '</p>';
        html += '<pre style="background:#fff;border-radius:10px;padding:12px;overflow:auto;">' +
            JSON.stringify(scores, null, 2) + '</pre>';
        container.innerHTML = html;
    }

    function appendAI(content) {
        var panel = document.getElementById('aiStream');
        if (!panel) {
            return;
        }
        if (panel.style.display !== 'block') {
            panel.style.display = 'block';
        }
        panel.textContent = panel.textContent + content;
    }

    function resetAI() {
        var panel = document.getElementById('aiStream');
        if (!panel) {
            return;
        }
        panel.style.display = 'none';
        panel.textContent = '';
    }

    function resetDashboard() {
        var single = document.getElementById('singleSymbol');
        var batch = document.getElementById('batchSymbols');
        var logPanel = document.getElementById('logStream');
        var resultPanel = document.getElementById('resultPanel');
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
    return render_template_string(MAIN_TEMPLATE, auth_enabled=enabled)


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
