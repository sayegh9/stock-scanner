"""
Flask web server for the streaming U.S. stock analysis experience.
Provides Server-Sent Events (SSE) to stream progress updates and results in real time.
Configured for U.S. equities by default.
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory, session, redirect, url_for, Response
from flask_cors import CORS
import logging
import json
import threading
import time
from datetime import datetime, timedelta
import os
import sys
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import wraps
import hashlib
import secrets
import uuid
from queue import Queue, Empty

# Import the analyzer
try:
    from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer
except ImportError:
    print("❌ Unable to import enhanced_web_stock_analyzer.py")
    print("Please ensure enhanced_web_stock_analyzer.py is located in the same directory")
    sys.exit(1)

# Create the Flask application
app = Flask(__name__)
CORS(app)  # allow cross-origin requests

# High-concurrency optimisations
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['JSON_SORT_KEYS'] = False

# Generate a random SECRET_KEY
app.secret_key = secrets.token_hex(32)

# Global state
analyzer = None
analysis_tasks = {}  # stores analysis task status
task_results = {}   # stores task results
task_lock = threading.Lock()
sse_clients = {}    # stores SSE client connections
sse_lock = threading.Lock()

# Thread pool for concurrent analysis
executor = ThreadPoolExecutor(max_workers=4)

# Log configuration – console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SSEManager:
    """Manages SSE connections"""
    
    def __init__(self):
        self.clients = {}
        self.lock = threading.Lock()
    
    def add_client(self, client_id, queue):
        """Register a new SSE client"""
        with self.lock:
            self.clients[client_id] = queue
            logger.info(f"SSE client connected: {client_id}")

    def remove_client(self, client_id):
        """Remove an SSE client"""
        with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"SSE client disconnected: {client_id}")

    def send_to_client(self, client_id, event_type, data):
        """Send a message to a specific client"""
        with self.lock:
            if client_id in self.clients:
                try:
                    # Clean data to ensure JSON serialisation
                    cleaned_data = clean_data_for_json(data)
                    message = {
                        'event': event_type,
                        'data': cleaned_data,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.clients[client_id].put(message, block=False)
                    return True
                except Exception as e:
                    logger.error(f"Failed to send SSE message: {e}")
                    return False
            return False

    def broadcast(self, event_type, data):
        """Broadcast a message to every client"""
        with self.lock:
            # Clean data to ensure JSON serialisation
            cleaned_data = clean_data_for_json(data)
            message = {
                'event': event_type,
                'data': cleaned_data,
                'timestamp': datetime.now().isoformat()
            }
            
            dead_clients = []
            for client_id, queue in self.clients.items():
                try:
                    queue.put(message, block=False)
                except Exception as e:
                    logger.error(f"SSE broadcast failed for client {client_id}: {e}")
                    dead_clients.append(client_id)

            # 清理死连接
            for client_id in dead_clients:
                del self.clients[client_id]

# 全局SSE管理器
sse_manager = SSEManager()

def clean_data_for_json(obj):
    """Clean NaN/Infinity/date values so payloads serialise to JSON correctly"""
    import pandas as pd
    from datetime import datetime, date, time
    
    if isinstance(obj, dict):
        return {key: clean_data_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_data_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [clean_data_for_json(item) for item in obj]
    elif isinstance(obj, (int, float)):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return None
        else:
            return obj
    elif isinstance(obj, np.ndarray):
        return clean_data_for_json(obj.tolist())
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        else:
            return obj.item()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, time):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.NaT.__class__):
        return None
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'to_dict'):  # DataFrame or Series
        try:
            return clean_data_for_json(obj.to_dict())
        except:
            return str(obj)
    elif hasattr(obj, 'item'):  # numpy scalar
        try:
            return clean_data_for_json(obj.item())
        except:
            return str(obj)
    elif obj is None:
        return None
    elif isinstance(obj, (str, bool)):
        return obj
    else:
        # Convert unsupported types to strings
        try:
            # Try to serialise directly first
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

def check_auth_config():
    """检查鉴权配置"""
    if not analyzer:
        return False, {}
    
    web_auth_config = analyzer.config.get('web_auth', {})
    return web_auth_config.get('enabled', False), web_auth_config

def require_auth(f):
    """鉴权装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_enabled, auth_config = check_auth_config()
        
        if not auth_enabled:
            return f(*args, **kwargs)
        
        # 检查session中是否已认证
        if session.get('authenticated'):
            # 检查session是否过期
            login_time = session.get('login_time')
            if login_time:
                session_timeout = auth_config.get('session_timeout', 3600)
                if (datetime.now() - datetime.fromisoformat(login_time)).total_seconds() < session_timeout:
                    return f(*args, **kwargs)
                else:
                    session.pop('authenticated', None)
                    session.pop('login_time', None)
        
        return redirect(url_for('login'))
    
    return decorated_function

# 登录页面HTML模板（保持不变）
LOGIN_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - U.S. Stock Analysis System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #333;
        }

        .login-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            max-width: 400px;
            width: 100%;
            text-align: center;
        }

        .login-header {
            margin-bottom: 30px;
        }

        .login-header h1 {
            font-size: 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .login-header p {
            color: #6c757d;
            font-size: 14px;
        }

        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }

        .form-control {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .form-control:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .btn {
            width: 100%;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-bottom: 20px;
        }

        .btn:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
            transform: translateY(-2px);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }

        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }

        .success-message {
            background: #d4edda;
            color: #155724;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }

        .login-footer {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            color: #6c757d;
            font-size: 12px;
        }

        @media (max-width: 640px) {
            .login-container {
                margin: 20px;
                padding: 30px 20px;
            }
            
            .login-header h1 {
                font-size: 24px;
            }
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-header">
            <h1>🇺🇸 System Login</h1>
            <p>Enhanced v3.1 Streaming U.S. Stock Analysis System</p>
        </div>

        {% if error %}
        <div class="error-message">
            {{ error }}
        </div>
        {% endif %}

        {% if success %}
        <div class="success-message">
            {{ success }}
        </div>
        {% endif %}

        <form method="POST">
            <div class="form-group">
                <label for="password">Access Password</label>
                <input type="password" id="password" name="password"
                       class="form-control" placeholder="Enter the access password" required>
            </div>
            
            <button type="submit" class="btn">
                🚀 Sign In
            </button>
        </form>

        <div class="login-footer">
            <p>🔒 Password authentication is required.</p>
            <p>🛡️ Sessions expire after {{ session_timeout }} minutes.</p>
            <p>🇺🇸 Optimised for U.S. equities.</p>
            <p>🌊 Live SSE streaming updates.</p>
        </div>
    </div>

    <script>
        document.getElementById('password').focus();
        
        document.querySelector('form').addEventListener('submit', function() {
            const btn = document.querySelector('.btn');
            btn.disabled = true;
            btn.textContent = '🔄 Signing in...';
            
            setTimeout(() => {
                btn.disabled = false;
                btn.textContent = '🚀 Sign In';
            }, 3000);
        });
    </script>
</body>
</html>"""

# Main page HTML template - SSE streaming with multi-market support
MAIN_TEMPLATE_MULTI_MARKET = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>U.S. Stock Analysis System - Enhanced v3.1</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        .header h1 {
            font-size: 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 16px;
        }

        .version-info {
            color: #6c757d;
            font-size: 14px;
        }

        .header-buttons {
            display: flex;
            gap: 8px;
        }

        .config-btn, .logout-btn {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 8px 16px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            text-decoration: none;
            color: #495057;
            font-size: 14px;
        }

        .logout-btn {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            border-color: #dc3545;
            color: white;
        }

        .config-btn:hover {
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            transform: translateY(-2px);
        }

        .logout-btn:hover {
            background: linear-gradient(135deg, #c82333 0%, #a71e2a 100%);
            transform: translateY(-2px);
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            min-height: 600px;
        }

        .left-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        .right-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        .tabs {
            display: flex;
            border-bottom: 2px solid #e9ecef;
            margin-bottom: 20px;
        }

        .tab {
            padding: 12px 24px;
            background: #f8f9fa;
            border: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-weight: 600;
            margin-right: 4px;
            transition: all 0.3s ease;
        }

        .tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }

        .form-control {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .form-control:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .textarea {
            min-height: 120px;
            resize: vertical;
        }

        .market-selector {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .market-option {
            padding: 8px 16px;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            background: white;
        }

        .market-option.selected {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }

        .market-option:hover {
            transform: translateY(-1px);
        }

        .stock-input-group {
            display: flex;
            gap: 8px;
            align-items: flex-end;
        }

        .stock-input-group .form-control {
            flex: 1;
        }

        .market-detect-btn {
            padding: 12px 16px;
            background: linear-gradient(135deg, #ff9800 0%, #ffcc80 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .market-detect-btn:hover {
            background: linear-gradient(135deg, #f57c00 0%, #ffb74d 100%);
            transform: translateY(-1px);
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .checkbox-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: #667eea;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
            transform: translateY(-2px);
        }

        .btn-success {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            color: white;
        }

        .btn-success:hover {
            background: linear-gradient(135deg, #4e9a2a 0%, #96d4b5 100%);
            transform: translateY(-2px);
        }

        .btn-secondary {
            background: #f8f9fa;
            color: #495057;
            border: 2px solid #e9ecef;
        }

        .btn-secondary:hover {
            background: #e9ecef;
            border-color: #adb5bd;
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }

        .progress-bar {
            width: 100%;
            height: 12px;
            background-color: #e9ecef;
            border-radius: 6px;
            overflow: hidden;
            margin: 16px 0;
            display: none;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            width: 0%;
        }

        .log-container {
            margin-top: 20px;
        }

        .log-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .log-header h3 {
            color: #495057;
            font-size: 16px;
        }

        .log-display {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 16px;
            max-height: 250px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.4;
        }

        .log-entry {
            margin-bottom: 4px;
            padding: 2px 0;
        }

        .log-info { color: #3498db; }
        .log-success { color: #27ae60; font-weight: bold; }
        .log-warning { color: #f39c12; font-weight: bold; }
        .log-error { color: #e74c3c; font-weight: bold; }
        .log-header-type { color: #667eea; font-weight: bold; font-size: 14px; }
        .log-progress { color: #9b59b6; font-weight: bold; }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .results-header h2 {
            color: #2c3e50;
            font-size: 20px;
        }

        .sse-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #6c757d;
        }

        .sse-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #dc3545;
            transition: background-color 0.3s;
        }

        .sse-indicator.connected {
            background: #28a745;
        }

        .score-cards {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
            display: none;
        }

        .score-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            color: white;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: all 0.3s ease;
        }

        .score-card.excellent { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); }
        .score-card.good { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .score-card.average { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        .score-card.poor { background: linear-gradient(135deg, #ff4b2b 0%, #ff416c 100%); }

        .score-card.updating {
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }

        .score-card h4 {
            font-size: 12px;
            margin-bottom: 8px;
            opacity: 0.9;
        }

        .score-card .score {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 4px;
        }

        .score-card .max-score {
            font-size: 10px;
            opacity: 0.8;
        }

        .data-quality {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
            display: none;
        }

        .quality-indicator {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }

        .quality-indicator .value {
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 4px;
        }

        .quality-indicator .label {
            font-size: 10px;
            color: #6c757d;
        }

        .results-content {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 20px;
            min-height: 400px;
            overflow-y: auto;
        }

        .loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 300px;
            color: #6c757d;
        }

        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #e9ecef;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 16px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .empty-state {
            text-align: center;
            color: #6c757d;
            padding: 60px 20px;
        }

        .empty-state h3 {
            margin-bottom: 8px;
        }

        .status-indicator {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }

        .status-ready { background: #d4edda; color: #155724; }
        .status-analyzing { background: #d1ecf1; color: #0c5460; }
        .status-error { background: #f8d7da; color: #721c24; }

        .market-info-display {
            background: rgba(255, 255, 255, 0.8);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 12px;
            color: #6c757d;
            display: none;
        }

        .ai-analysis-content {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .ai-analysis-content h1,
        .ai-analysis-content h2,
        .ai-analysis-content h3,
        .ai-analysis-content h4,
        .ai-analysis-content h5,
        .ai-analysis-content h6 {
            color: #2c3e50;
            margin-top: 16px;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .ai-analysis-content h1 { font-size: 1.5em; }
        .ai-analysis-content h2 { font-size: 1.3em; }
        .ai-analysis-content h3 { font-size: 1.1em; }

        .ai-analysis-content p {
            margin: 8px 0;
            line-height: 1.6;
        }

        .ai-analysis-content ul,
        .ai-analysis-content ol {
            margin: 8px 0;
            padding-left: 20px;
        }

        .ai-analysis-content li {
            margin: 4px 0;
            line-height: 1.5;
        }

        .ai-analysis-content strong {
            color: #1976d2;
            font-weight: 600;
        }

        .ai-analysis-content em {
            color: #f57c00;
            font-style: italic;
        }

        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
                gap: 16px;
            }
            
            .score-cards {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .data-quality {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 640px) {
            .container {
                padding: 16px;
            }
            
            .header {
                padding: 16px;
            }
            
            .header h1 {
                font-size: 24px;
            }
            
            .header-info {
                flex-direction: column;
                gap: 12px;
                align-items: flex-start;
            }
            
            .score-cards {
                grid-template-columns: 1fr;
            }
            
            .data-quality {
                grid-template-columns: 1fr;
            }

            .market-selector {
                justify-content: center;
            }

            .stock-input-group {
                flex-direction: column;
                gap: 12px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🌍 Global Stock Analysis System - Multi-Market Edition</h1>
            <div class="header-info">
                <div class="version-info">
                    Enhanced v3.1-Multi-Market | EnhancedWebStockAnalyzer | A-Share/Hong Kong/U.S. {% if auth_enabled %}| 🔐 Authenticated{% endif %}
                    <span id="systemStatus" class="status-indicator status-ready">System Ready</span>
                    <div class="market-indicators" id="marketIndicators">
                        <div class="market-badge a">A-Share</div>
                        <div class="market-badge hk">Hong Kong</div>
                        <div class="market-badge us">U.S.</div>
                    </div>
                </div>
                <div class="header-buttons">
                    <button class="config-btn" onclick="showConfig()">⚙️ Market Configuration</button>
                    {% if auth_enabled %}
                    <a href="{{ url_for('logout') }}" class="logout-btn">🚪 Log Out</a>
                    {% endif %}
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <!-- Left Panel - Input and Controls -->
            <div class="left-panel">
                <!-- Tabs -->
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('single')">📈 Single Stock Analysis</button>
                    <button class="tab" onclick="switchTab('batch')">📊 Batch Analysis</button>
                </div>

                <!-- Single Stock Analysis -->
                <div id="singleTab" class="tab-content active">
                    <div class="form-group">
                        <label for="stockCode">Stock Code</label>
                        
                        <!-- Market Selector -->
                        <div class="market-selector">
                            <div class="market-option selected" data-market="auto">🤖 Auto Detect</div>
                            <div class="market-option" data-market="a_stock">🇨🇳 A-Share</div>
                            <div class="market-option" data-market="hk_stock">🇭🇰 Hong Kong</div>
                            <div class="market-option" data-market="us_stock">🇺🇸 U.S.</div>
                        </div>

                        <div class="stock-input-group">
                            <input type="text" id="stockCode" class="form-control" 
                                   placeholder="Enter a stock code (e.g., 000001, 00700, AAPL)">
                            <button class="market-detect-btn" onclick="detectMarket()">🔍 Detect</button>
                        </div>

                        <div id="marketInfoDisplay" class="market-info-display">
                            <div id="marketInfo"></div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="enableStreaming" checked>
                            <label for="enableStreaming">Enable AI streaming output</label>
                        </div>
                    </div>
                    
                    <button id="analyzeBtn" class="btn btn-primary" onclick="analyzeSingleStock()">
                        🔍 Start Global Deep Analysis
                    </button>
                    
                    <div id="singleProgress" class="progress-bar">
                        <div class="progress-bar-fill"></div>
                    </div>
                </div>

                <!-- Batch Analysis -->
                <div id="batchTab" class="tab-content">
                    <div class="form-group">
                        <label for="stockList">Global stock code list</label>
                        <textarea id="stockList" class="form-control textarea" 
                                  placeholder="Enter multiple stock codes, one per line. Mixed markets are supported.&#10;For example:&#10;000001 (A-Share)&#10;00700 (Hong Kong)&#10;AAPL (U.S.)&#10;600036 (A-Share)&#10;00388 (Hong Kong)&#10;TSLA (U.S.)"></textarea>
                    </div>
                    
                    <button id="batchAnalyzeBtn" class="btn btn-success" onclick="analyzeBatchStocks()">
                        🌍 Batch Global Deep Analysis
                    </button>
                    
                    <div id="batchProgress" class="progress-bar">
                        <div class="progress-bar-fill"></div>
                    </div>
                    
                    <div id="currentStock" style="display: none; margin-top: 12px; color: #6c757d; font-size: 12px; font-style: italic;"></div>
                </div>

                <!-- Log Container -->
                <div class="log-container">
                    <div class="log-header">
                        <h3>📋 Analysis Log</h3>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <div class="sse-status">
                                <div id="sseIndicator" class="sse-indicator"></div>
                                <span id="sseStatus">SSE Disconnected</span>
                            </div>
                            <button class="btn btn-secondary" onclick="clearLog()" style="padding: 4px 12px; font-size: 12px;">
                                🗑️ Clear
                            </button>
                        </div>
                    </div>
                    <div id="logDisplay" class="log-display">
                        <div class="log-entry log-info">📋 Global stock analysis system ready. Supports A-Share/Hong Kong/U.S. markets...</div>
                    </div>
                </div>
            </div>

            <!-- Right Panel - Results -->
            <div class="right-panel">
                <div class="results-header">
                    <h2>📋 Analysis Results</h2>
                    <button id="exportBtn" class="btn btn-secondary" onclick="exportReport()" style="display: none;">
                        📤 Export Report
                    </button>
                </div>

                <!-- Score Cards -->
                <div id="scoreCards" class="score-cards">
                    <div class="score-card" id="comprehensiveCard">
                        <h4>Overall Score</h4>
                        <div class="score">--</div>
                        <div class="max-score">/100</div>
                    </div>
                    <div class="score-card" id="technicalCard">
                        <h4>Technical Analysis</h4>
                        <div class="score">--</div>
                        <div class="max-score">/100</div>
                    </div>
                    <div class="score-card" id="fundamentalCard">
                        <h4>Fundamentals</h4>
                        <div class="score">--</div>
                        <div class="max-score">/100</div>
                    </div>
                    <div class="score-card" id="sentimentCard">
                        <h4>Market Sentiment</h4>
                        <div class="score">--</div>
                        <div class="max-score">/100</div>
                    </div>
                </div>

                <!-- Data Quality Indicators -->
                <div id="dataQuality" class="data-quality">
                    <div class="quality-indicator">
                        <div id="financialCount" class="value">--</div>
                        <div class="label">Financial Indicators</div>
                    </div>
                    <div class="quality-indicator">
                        <div id="newsCount" class="value">--</div>
                        <div class="label">News Data</div>
                    </div>
                    <div class="quality-indicator">
                        <div id="completeness" class="value">--</div>
                        <div class="label">Completeness</div>
                    </div>
                    <div class="quality-indicator">
                        <div id="marketType" class="value">--</div>
                        <div class="label">Market Type</div>
                    </div>
                </div>

                <!-- Results Content -->
                <div id="resultsContent" class="results-content">
                    <div class="empty-state">
                        <h3>🌍 Awaiting Global Analysis</h3>
                        <p>Please enter stock codes on the left to begin analysis.</p>
                        <p style="margin-top: 8px; font-size: 12px; color: #9ba2ab;">
                            💫 Supports A-Share (6 digits) | Hong Kong (5 digits) | U.S. (letter codes)<br>
                            🌊 SSE real-time streaming | 🤖 AI deep analysis
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Include marked.js for markdown parsing -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
    
    <script>
        // Global variables
        let currentAnalysis = null;
        let isAnalyzing = false;
        let sseConnection = null;
        let currentClientId = null;
        let selectedMarket = 'auto';
        const API_BASE = '';  // Flask server base URL
        
        // Market configuration
        const MARKET_CONFIG = {
            'a_stock': {
                name: 'A-Share',
                currency: 'CNY',
                flag: '🇨🇳',
                pattern: /^\d{6}$/,
                example: '000001, 600036, 300019'
            },
            'hk_stock': {
                name: 'Hong Kong',
                currency: 'HKD',
                flag: '🇭🇰',
                pattern: /^\d{5}$|^HK\d{5}$/i,
                example: '00700, 00388, 01024'
            },
            'us_stock': {
                name: 'U.S.',
                currency: 'USD',
                flag: '🇺🇸',
                pattern: /^[A-Z]{1,5}$/,
                example: 'AAPL, TSLA, GOOGL'
            }
        };
        
        // 配置marked.js
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true,
                sanitize: false,
                smartLists: true,
                smartypants: true
            });
        }

        // Market selector handling
        document.querySelectorAll('.market-option').forEach(option => {
            option.addEventListener('click', function() {
                document.querySelectorAll('.market-option').forEach(opt => opt.classList.remove('selected'));
                this.classList.add('selected');
                selectedMarket = this.dataset.market;
                updateMarketInfo();
            });
        });

        function updateMarketInfo() {
            const stockCode = document.getElementById('stockCode').value.trim();
            const marketInfo = document.getElementById('marketInfo');
            const marketInfoDisplay = document.getElementById('marketInfoDisplay');
            
            if (selectedMarket === 'auto') {
                if (stockCode) {
                    const detectedMarket = detectStockMarket(stockCode);
                    if (detectedMarket) {
                        const config = MARKET_CONFIG[detectedMarket];
                        marketInfo.innerHTML = `
                            🤖 自动识别为: ${config.flag} ${config.name} | 货币: ${config.currency} | 示例: ${config.example}
                        `;
                        marketInfoDisplay.style.display = 'block';
                    } else {
                        marketInfoDisplay.style.display = 'none';
                    }
                } else {
                    marketInfoDisplay.style.display = 'none';
                }
            } else {
                const config = MARKET_CONFIG[selectedMarket];
                marketInfo.innerHTML = `
                    ${config.flag} ${config.name} | 货币: ${config.currency} | 示例: ${config.example}
                `;
                marketInfoDisplay.style.display = 'block';
            }
        }

        function detectStockMarket(stockCode) {
            stockCode = stockCode.trim().toUpperCase();
            
            if (MARKET_CONFIG.a_stock.pattern.test(stockCode)) {
                return 'a_stock';
            } else if (MARKET_CONFIG.hk_stock.pattern.test(stockCode)) {
                return 'hk_stock';
            } else if (MARKET_CONFIG.us_stock.pattern.test(stockCode)) {
                return 'us_stock';
            }
            return null;
        }

        function detectMarket() {
            const stockCode = document.getElementById('stockCode').value.trim();
            if (!stockCode) {
                addLog('Please enter a stock symbol first', 'warning');
                return;
            }
            
            const market = detectStockMarket(stockCode);
            if (market) {
                const config = MARKET_CONFIG[market];
                addLog(`🔍 检测到股票代码 ${stockCode} 属于 ${config.flag} ${config.name} 市场`, 'success');
                
                // 自动选择对应市场
                document.querySelectorAll('.market-option').forEach(opt => opt.classList.remove('selected'));
                document.querySelector(`[data-market="${market}"]`).classList.add('selected');
                selectedMarket = market;
                updateMarketInfo();
            } else {
                addLog(`❌ 无法识别股票代码 ${stockCode} 的市场类型`, 'error');
            }
        }

        // Stock code input event
        document.getElementById('stockCode').addEventListener('input', function() {
            if (selectedMarket === 'auto') {
                updateMarketInfo();
            }
        });

        // SSE连接管理
        function initSSE() {
            if (sseConnection) {
                sseConnection.close();
            }

            currentClientId = generateClientId();
            const sseUrl = `${API_BASE}/api/sse?client_id=${currentClientId}`;
            
            addLog('🌊 Establishing SSE connection...', 'info');
            
            sseConnection = new EventSource(sseUrl);
            
            sseConnection.onopen = function(event) {
                addLog('✅ SSE connection established', 'success');
                updateSSEStatus(true);
            };
            
            sseConnection.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleSSEMessage(data);
                } catch (e) {
                    console.error('SSE消息解析失败:', e);
                }
            };
            
            sseConnection.onerror = function(event) {
                addLog('❌ SSE connection error', 'error');
                updateSSEStatus(false);
                
                // 自动重连
                setTimeout(() => {
                    if (!sseConnection || sseConnection.readyState === EventSource.CLOSED) {
                        addLog('🔄 Attempting SSE reconnection...', 'warning');
                        initSSE();
                    }
                }, 3000);
            };
            
            sseConnection.onclose = function(event) {
                addLog('🔌 SSE connection closed', 'warning');
                updateSSEStatus(false);
            };
        }

        function generateClientId() {
            return 'client_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        }

        function updateSSEStatus(connected) {
            const indicator = document.getElementById('sseIndicator');
            const status = document.getElementById('sseStatus');
            
            if (connected) {
                indicator.classList.add('connected');
                status.textContent = 'SSE已连接';
            } else {
                indicator.classList.remove('connected');
                status.textContent = 'SSE断开';
            }
        }

        function handleSSEMessage(data) {
            const eventType = data.event;
            const eventData = data.data;
            
            switch (eventType) {
                case 'log':
                    addLog(eventData.message, eventData.type || 'info');
                    break;
                    
                case 'progress':
                    updateProgress(eventData.element_id, eventData.percent);
                    if (eventData.message) {
                        addLog(eventData.message, 'progress');
                    }
                    if (eventData.current_stock) {
                        document.getElementById('currentStock').textContent = 
                            `正在分析: ${eventData.current_stock}`;
                        document.getElementById('currentStock').style.display = 'block';
                    }
                    break;
                    
                case 'scores_update':
                    updateScoreCards(eventData.scores);
                    if (eventData.animate) {
                        animateScoreCards();
                    }
                    break;
                    
                case 'data_quality_update':
                    updateDataQuality(eventData);
                    break;
                    
                case 'partial_result':
                    displayPartialResults(eventData);
                    break;
                    
                case 'final_result':
                    displayResults(eventData);
                    currentAnalysis = eventData;
                    break;
                    
                case 'batch_result':
                    displayBatchResults(eventData);
                    currentAnalysis = eventData;
                    break;
                    
                case 'analysis_complete':
                    onAnalysisComplete(eventData);
                    break;
                    
                case 'analysis_error':
                    onAnalysisError(eventData);
                    break;
                    
                case 'ai_stream':
                    handleAIStream(eventData);
                    break;
                    
                case 'error':
                    addLog(`⚠️ SSE错误: ${eventData.error || '未知错误'}`, 'warning');
                    break;
                    
                case 'heartbeat':
                    // 心跳，不需要处理
                    break;
                    
                default:
                    console.log('未知SSE事件:', eventType, eventData);
            }
        }

        function handleAIStream(data) {
            // 获取或创建AI流式显示区域
            let aiStreamDiv = document.getElementById('aiStreamContent');
            if (!aiStreamDiv) {
                // 在结果区域中查找AI分析部分
                const resultsContent = document.getElementById('resultsContent');
                const aiSection = resultsContent.querySelector('.ai-analysis-content');
                
                if (aiSection) {
                    // 如果找到了AI分析部分，创建流式内容区域
                    aiStreamDiv = document.createElement('div');
                    aiStreamDiv.id = 'aiStreamContent';
                    aiStreamDiv.style.cssText = `
                        border: 2px solid #ff9800;
                        border-radius: 8px;
                        padding: 16px;
                        margin: 16px 0;
                        background: rgba(255, 152, 0, 0.1);
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        line-height: 1.6;
                        min-height: 100px;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    `;
                    
                    // 添加流式标题
                    const streamTitle = document.createElement('h3');
                    streamTitle.textContent = '🤖 AI 全球深度分析 - 实时生成中...';
                    streamTitle.style.cssText = 'color: #f57c00; margin-bottom: 12px; font-size: 16px;';
                    
                    const streamContainer = document.createElement('div');
                    streamContainer.appendChild(streamTitle);
                    streamContainer.appendChild(aiStreamDiv);
                    
                    // 插入到结果区域
                    resultsContent.appendChild(streamContainer);
                } else {
                    // 如果没有找到结果区域，创建临时显示区域
                    const resultsContent = document.getElementById('resultsContent');
                    resultsContent.innerHTML = `
                        <div style="line-height: 1.6;">
                            <h2 style="color: #2c3e50; border-bottom: 2px solid #e9ecef; padding-bottom: 12px; margin-bottom: 20px;">
                                📈 全球股票实时分析进行中...
                                <span style="font-size: 12px; color: #28a745; font-weight: normal;">🌊 AI流式生成中</span>
                            </h2>
                            
                            <div style="background: #fff3e0; padding: 20px; border-radius: 8px; border-left: 4px solid #ff9800;">
                                <h3 style="color: #f57c00; margin-bottom: 12px;">🤖 AI 全球深度分析 - 实时生成中...</h3>
                                <div id="aiStreamContent" style="color: #5d4037; font-size: 14px; line-height: 1.7; white-space: pre-wrap; word-wrap: break-word;"></div>
                            </div>
                        </div>
                    `;
                    aiStreamDiv = document.getElementById('aiStreamContent');
                }
            }
            
            // 添加AI流式内容
            if (aiStreamDiv && data.content) {
                aiStreamDiv.textContent += data.content;
                
                // 自动滚动到底部
                aiStreamDiv.scrollTop = aiStreamDiv.scrollHeight;
                
                // 如果容器可见，也滚动到底部
                const resultsContent = document.getElementById('resultsContent');
                if (resultsContent) {
                    resultsContent.scrollTop = resultsContent.scrollHeight;
                }
            }
        }

        function animateScoreCards() {
            const cards = document.querySelectorAll('.score-card');
            cards.forEach(card => {
                card.classList.add('updating');
                setTimeout(() => {
                    card.classList.remove('updating');
                }, 1500);
            });
        }

        // Tab switching
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
            
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById(tabName + 'Tab').classList.add('active');
        }

        // Log functions
        function addLog(message, type = 'info') {
            const logDisplay = document.getElementById('logDisplay');
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${type}`;
            
            const timestamp = new Date().toLocaleTimeString();
            let icon = '📋';
            
            switch(type) {
                case 'success': icon = '✅'; break;
                case 'warning': icon = '⚠️'; break;
                case 'error': icon = '❌'; break;
                case 'header': icon = '🎯'; break;
                case 'progress': icon = '🔄'; break;
            }
            
            logEntry.innerHTML = `<span style="color: #999;">[${timestamp}]</span> ${icon} ${message}`;
            logDisplay.appendChild(logEntry);
            logDisplay.scrollTop = logDisplay.scrollHeight;
        }

        function clearLog() {
            document.getElementById('logDisplay').innerHTML = 
                '<div class="log-entry log-info">📋 日志已清空</div>';
        }

        // Progress bar functions
        function showProgress(elementId, show = true) {
            const progressBar = document.getElementById(elementId);
            progressBar.style.display = show ? 'block' : 'none';
            if (!show) {
                progressBar.querySelector('.progress-bar-fill').style.width = '0%';
            }
        }

        function updateProgress(elementId, percent) {
            const fill = document.getElementById(elementId).querySelector('.progress-bar-fill');
            fill.style.width = percent + '%';
        }

        // Score card functions
        function updateScoreCards(scores) {
            const cards = {
                comprehensive: document.getElementById('comprehensiveCard'),
                technical: document.getElementById('technicalCard'),
                fundamental: document.getElementById('fundamentalCard'),
                sentiment: document.getElementById('sentimentCard')
            };

            Object.keys(scores).forEach(key => {
                const card = cards[key];
                if (card) {
                    const score = scores[key];
                    card.querySelector('.score').textContent = score.toFixed(1);
                    
                    card.className = 'score-card';
                    if (score >= 80) card.classList.add('excellent');
                    else if (score >= 60) card.classList.add('good');
                    else if (score >= 40) card.classList.add('average');
                    else card.classList.add('poor');
                }
            });

            document.getElementById('scoreCards').style.display = 'grid';
        }

        function updateDataQuality(data) {
            document.getElementById('financialCount').textContent = 
                data.financial_indicators_count || 0;
            document.getElementById('newsCount').textContent = 
                data.total_news_count || 0;
            document.getElementById('completeness').textContent = 
                (data.analysis_completeness || '部分').substring(0, 2);
            document.getElementById('marketType').textContent = 
                (data.market_coverage || 'Unknown').substring(0, 3);
            
            document.getElementById('dataQuality').style.display = 'grid';
        }

        // Results display
        function showLoading() {
            document.getElementById('resultsContent').innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <p>正在进行全球深度分析...</p>
                    <p style="font-size: 12px; color: #9ba2ab;">🌍 多市场实时流式推送中</p>
                </div>
            `;
        }

        function displayPartialResults(data) {
            // 显示部分结果，比如基本信息
            const resultsContent = document.getElementById('resultsContent');
            
            if (data.type === 'basic_info') {
                const marketInfo = data.market ? MARKET_CONFIG[data.market] || {} : {};
                
                resultsContent.innerHTML = `
                    <div style="line-height: 1.6;">
                        <h2 style="color: #2c3e50; border-bottom: 2px solid #e9ecef; padding-bottom: 12px; margin-bottom: 20px;">
                            📈 ${data.stock_name || data.stock_code} 全球分析报告
                            <span style="font-size: 12px; color: #6c757d; font-weight: normal;">🌊 实时流式分析中...</span>
                        </h2>
                        
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
                            <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                                <h4 style="color: #495057; margin-bottom: 8px;">基本信息</h4>
                                <p><strong>股票代码:</strong> ${data.stock_code}</p>
                                <p><strong>交易市场:</strong> ${marketInfo.flag || '🌍'} ${marketInfo.name || '未知'}</p>
                                <p><strong>计价货币:</strong> ${marketInfo.currency || 'Unknown'}</p>
                                <p><strong>当前价格:</strong> ${(data.current_price || 0).toFixed(2)}</p>
                                <p><strong>涨跌幅:</strong> ${(data.price_change || 0).toFixed(2)}%</p>
                            </div>
                            
                            <div style="background: #e3f2fd; padding: 16px; border-radius: 8px;">
                                <h4 style="color: #495057; margin-bottom: 8px;">分析进度</h4>
                                <p>🔄 正在获取技术指标...</p>
                                <p>⏳ 正在分析财务数据...</p>
                                <p>🌊 正在处理新闻情绪...</p>
                                <p>🤖 正在准备AI分析...</p>
                            </div>
                        </div>
                        
                        <div style="background: #fff3e0; padding: 20px; border-radius: 8px; border-left: 4px solid #ff9800;">
                            <h3 style="color: #f57c00; margin-bottom: 12px;">🤖 AI 全球深度分析进行中</h3>
                            <div style="color: #5d4037; font-size: 14px; line-height: 1.7;">
                                正在收集全球市场数据并进行AI智能分析，请稍候...
                            </div>
                        </div>
                    </div>
                `;
            }
        }

        function displayResults(report) {
            const resultsContent = document.getElementById('resultsContent');
            
            // 检查是否有AI流式内容正在显示
            const existingAIStream = document.getElementById('aiStreamContent');
            let aiAnalysisHtml = '';
            
            if (existingAIStream && existingAIStream.textContent.trim()) {
                // 如果有流式内容，使用流式内容并标记为完成
                const streamTitle = existingAIStream.parentElement.querySelector('h3');
                if (streamTitle) {
                    streamTitle.innerHTML = '🤖 AI 全球深度分析 <span style="color: #28a745; font-size: 12px;">✅ 生成完成</span>';
                }
                
                // 将流式内容转换为markdown格式
                const streamContent = existingAIStream.textContent;
                if (typeof marked !== 'undefined') {
                    aiAnalysisHtml = marked.parse(streamContent);
                } else {
                    aiAnalysisHtml = simpleMarkdownParse(streamContent);
                }
                
                // 更新AI分析区域
                existingAIStream.innerHTML = aiAnalysisHtml;
                existingAIStream.classList.add('ai-analysis-content');
                existingAIStream.style.whiteSpace = 'normal';
                
                // 保留现有的完整结果，只更新其他部分
                updateNonAIContent(report);
                return;
            }
            
            // 处理AI分析的markdown内容（如果没有流式内容）
            if (report.ai_analysis) {
                if (typeof marked !== 'undefined') {
                    aiAnalysisHtml = marked.parse(report.ai_analysis);
                } else {
                    aiAnalysisHtml = simpleMarkdownParse(report.ai_analysis);
                }
            } else {
                aiAnalysisHtml = '<p>分析数据准备中...</p>';
            }

            // 获取市场信息
            const marketInfo = MARKET_CONFIG[report.market] || {};
            
            const html = `
                <div style="line-height: 1.6;">
                    <h2 style="color: #2c3e50; border-bottom: 2px solid #e9ecef; padding-bottom: 12px; margin-bottom: 20px;">
                        📈 ${report.stock_name || report.stock_code} 全球分析报告
                        <span style="font-size: 12px; color: #28a745; font-weight: normal;">✅ 流式分析完成</span>
                    </h2>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
                        <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                            <h4 style="color: #495057; margin-bottom: 8px;">基本信息</h4>
                            <p><strong>股票代码:</strong> ${report.stock_code}</p>
                            <p><strong>交易市场:</strong> ${marketInfo.flag || '🌍'} ${marketInfo.name || report.market}</p>
                            <p><strong>计价货币:</strong> ${marketInfo.currency || 'Unknown'}</p>
                            <p><strong>当前价格:</strong> ${(report.price_info?.current_price || 0).toFixed(2)}</p>
                            <p><strong>涨跌幅:</strong> ${(report.price_info?.price_change || 0).toFixed(2)}%</p>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                            <h4 style="color: #495057; margin-bottom: 8px;">技术指标</h4>
                            <p><strong>RSI:</strong> ${(report.technical_analysis?.rsi || 0).toFixed(1)}</p>
                            <p><strong>趋势:</strong> ${report.technical_analysis?.ma_trend || '未知'}</p>
                            <p><strong>MACD:</strong> ${report.technical_analysis?.macd_signal || '未知'}</p>
                            <p><strong>成交量:</strong> ${report.technical_analysis?.volume_status || '未知'}</p>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                            <h4 style="color: #495057; margin-bottom: 8px;">市场情绪</h4>
                            <p><strong>情绪趋势:</strong> ${report.sentiment_analysis?.sentiment_trend || '中性'}</p>
                            <p><strong>新闻数量:</strong> ${report.sentiment_analysis?.total_analyzed || 0} 条</p>
                            <p><strong>置信度:</strong> ${((report.sentiment_analysis?.confidence_score || 0) * 100).toFixed(1)}%</p>
                        </div>
                    </div>
                    
                    <div style="background: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2196f3; margin-bottom: 24px;">
                        <h3 style="color: #1976d2; margin-bottom: 12px;">🎯 投资建议</h3>
                        <p style="font-size: 18px; font-weight: 600; color: #1976d2;">${report.recommendation || '数据不足'}</p>
                    </div>
                    
                    <div style="background: #fff3e0; padding: 20px; border-radius: 8px; border-left: 4px solid #ff9800;">
                        <h3 style="color: #f57c00; margin-bottom: 12px;">🤖 AI 全球深度分析</h3>
                        <div style="color: #5d4037; font-size: 14px; line-height: 1.7;" class="ai-analysis-content">
                            ${aiAnalysisHtml}
                        </div>
                    </div>
                </div>
            `;
            
            resultsContent.innerHTML = html;
            document.getElementById('exportBtn').style.display = 'inline-flex';
        }

        function updateNonAIContent(report) {
            // 更新非AI分析的其他内容
            const resultsContent = document.getElementById('resultsContent');
            
            // 获取市场信息
            const marketInfo = MARKET_CONFIG[report.market] || {};
            
            // 更新标题
            const title = resultsContent.querySelector('h2');
            if (title) {
                title.innerHTML = `📈 ${report.stock_name || report.stock_code} 全球分析报告 <span style="font-size: 12px; color: #28a745; font-weight: normal;">✅ 流式分析完成</span>`;
            }
            
            // 更新基本信息
            const basicInfoDiv = resultsContent.querySelector('div[style*="grid-template-columns"]');
            if (basicInfoDiv) {
                basicInfoDiv.innerHTML = `
                    <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                        <h4 style="color: #495057; margin-bottom: 8px;">基本信息</h4>
                        <p><strong>股票代码:</strong> ${report.stock_code}</p>
                        <p><strong>交易市场:</strong> ${marketInfo.flag || '🌍'} ${marketInfo.name || report.market}</p>
                        <p><strong>计价货币:</strong> ${marketInfo.currency || 'Unknown'}</p>
                        <p><strong>当前价格:</strong> ${(report.price_info?.current_price || 0).toFixed(2)}</p>
                        <p><strong>涨跌幅:</strong> ${(report.price_info?.price_change || 0).toFixed(2)}%</p>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                        <h4 style="color: #495057; margin-bottom: 8px;">技术指标</h4>
                        <p><strong>RSI:</strong> ${(report.technical_analysis?.rsi || 0).toFixed(1)}</p>
                        <p><strong>趋势:</strong> ${report.technical_analysis?.ma_trend || '未知'}</p>
                        <p><strong>MACD:</strong> ${report.technical_analysis?.macd_signal || '未知'}</p>
                        <p><strong>成交量:</strong> ${report.technical_analysis?.volume_status || '未知'}</p>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                        <h4 style="color: #495057; margin-bottom: 8px;">市场情绪</h4>
                        <p><strong>情绪趋势:</strong> ${report.sentiment_analysis?.sentiment_trend || '中性'}</p>
                        <p><strong>新闻数量:</strong> ${report.sentiment_analysis?.total_analyzed || 0} 条</p>
                        <p><strong>置信度:</strong> ${((report.sentiment_analysis?.confidence_score || 0) * 100).toFixed(1)}%</p>
                    </div>
                `;
            }
            
            // 更新投资建议
            const recommendationDiv = resultsContent.querySelector('div[style*="background: #e3f2fd"]');
            if (recommendationDiv) {
                const recommendationText = recommendationDiv.querySelector('p');
                if (recommendationText) {
                    recommendationText.textContent = report.recommendation || '数据不足';
                }
            }
            
            document.getElementById('exportBtn').style.display = 'inline-flex';
        }

        // 简单的markdown解析器（备用方案）
        function simpleMarkdownParse(text) {
            if (!text) return '';
            
            return text
                .replace(/^### (.*$)/gim, '<h3 style="color: #2c3e50; margin: 16px 0 8px 0;">$1</h3>')
                .replace(/^## (.*$)/gim, '<h2 style="color: #2c3e50; margin: 20px 0 10px 0;">$1</h2>')
                .replace(/^# (.*$)/gim, '<h1 style="color: #2c3e50; margin: 24px 0 12px 0;">$1</h1>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code style="background: #f1f3f4; padding: 2px 4px; border-radius: 3px; font-family: monospace;">$1</code>')
                .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color: #1976d2;">$1</a>')
                .replace(/^[\-\*\+] (.*$)/gim, '<li style="margin: 4px 0;">$1</li>')
                .replace(/\n\n/g, '</p><p>')
                .replace(/\n/g, '<br>');
        }

        function displayBatchResults(reports) {
            if (!reports || reports.length === 0) {
            addLog('Batch analysis returned no results', 'warning');
                return;
            }

            const avgScores = {
                comprehensive: reports.reduce((sum, r) => sum + r.scores.comprehensive, 0) / reports.length,
                technical: reports.reduce((sum, r) => sum + r.scores.technical, 0) / reports.length,
                fundamental: reports.reduce((sum, r) => sum + r.scores.fundamental, 0) / reports.length,
                sentiment: reports.reduce((sum, r) => sum + r.scores.sentiment, 0) / reports.length
            };

            updateScoreCards(avgScores);

            const avgFinancial = reports.reduce((sum, r) => sum + (r.data_quality?.financial_indicators_count || 0), 0) / reports.length;
            const avgNews = reports.reduce((sum, r) => sum + (r.sentiment_analysis?.total_analyzed || 0), 0) / reports.length;
            
            document.getElementById('financialCount').textContent = Math.round(avgFinancial);
            document.getElementById('newsCount').textContent = Math.round(avgNews);
            document.getElementById('completeness').textContent = '批量';
            document.getElementById('marketType').textContent = '混合';
            document.getElementById('dataQuality').style.display = 'grid';

            const resultsContent = document.getElementById('resultsContent');
            
            let tableRows = reports
                .sort((a, b) => b.scores.comprehensive - a.scores.comprehensive)
                .map((report, index) => {
                    const marketInfo = MARKET_CONFIG[report.market] || {};
                    return `
                        <tr style="border-bottom: 1px solid #e9ecef;">
                            <td style="padding: 12px; font-weight: 600;">${index + 1}</td>
                            <td style="padding: 12px;">${report.stock_code}</td>
                            <td style="padding: 12px;">${report.stock_name || report.stock_code}</td>
                            <td style="padding: 12px;">${marketInfo.flag || '🌍'} ${marketInfo.name || report.market}</td>
                            <td style="padding: 12px; font-weight: 600; color: ${report.scores.comprehensive >= 70 ? '#27ae60' : report.scores.comprehensive >= 50 ? '#667eea' : '#e74c3c'};">
                                ${report.scores.comprehensive.toFixed(1)}
                            </td>
                            <td style="padding: 12px;">${report.scores.technical.toFixed(1)}</td>
                            <td style="padding: 12px;">${report.scores.fundamental.toFixed(1)}</td>
                            <td style="padding: 12px;">${report.scores.sentiment.toFixed(1)}</td>
                            <td style="padding: 12px; font-weight: 600;">${report.recommendation}</td>
                        </tr>
                    `;
                }).join('');

            const html = `
                <div style="line-height: 1.6;">
                    <h2 style="color: #2c3e50; border-bottom: 2px solid #e9ecef; padding-bottom: 12px; margin-bottom: 20px;">
                        🌍 全球批量分析报告 (${reports.length} 只股票)
                        <span style="font-size: 12px; color: #28a745; font-weight: normal;">✅ 流式分析完成</span>
                    </h2>
                    
                    <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="color: #495057; margin-bottom: 12px;">📋 全球分析汇总</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;">
                            <div><strong>分析数量:</strong> ${reports.length} 只</div>
                            <div><strong>平均得分:</strong> ${avgScores.comprehensive.toFixed(1)}</div>
                            <div><strong>优秀股票:</strong> ${reports.filter(r => r.scores.comprehensive >= 80).length} 只</div>
                            <div><strong>良好股票:</strong> ${reports.filter(r => r.scores.comprehensive >= 60).length} 只</div>
                            <div><strong>A股数量:</strong> ${reports.filter(r => r.market === 'a_stock').length} 只</div>
                            <div><strong>港股数量:</strong> ${reports.filter(r => r.market === 'hk_stock').length} 只</div>
                            <div><strong>美股数量:</strong> ${reports.filter(r => r.market === 'us_stock').length} 只</div>
                        </div>
                    </div>
                    
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <thead>
                                <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                    <th style="padding: 16px; text-align: left;">排名</th>
                                    <th style="padding: 16px; text-align: left;">代码</th>
                                    <th style="padding: 16px; text-align: left;">名称</th>
                                    <th style="padding: 16px; text-align: left;">市场</th>
                                    <th style="padding: 16px; text-align: left;">综合得分</th>
                                    <th style="padding: 16px; text-align: left;">技术面</th>
                                    <th style="padding: 16px; text-align: left;">基本面</th>
                                    <th style="padding: 16px; text-align: left;">情绪面</th>
                                    <th style="padding: 16px; text-align: left;">投资建议</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${tableRows}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
            
            resultsContent.innerHTML = html;
            document.getElementById('exportBtn').style.display = 'inline-flex';
        }

        function onAnalysisComplete(data) {
            isAnalyzing = false;
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('batchAnalyzeBtn').disabled = false;
            document.getElementById('systemStatus').className = 'status-indicator status-ready';
            document.getElementById('systemStatus').textContent = '系统就绪';
            showProgress('singleProgress', false);
            showProgress('batchProgress', false);
            document.getElementById('currentStock').style.display = 'none';
            
            addLog('✅ U.S. analysis complete', 'success');
        }

        function onAnalysisError(data) {
            isAnalyzing = false;
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('batchAnalyzeBtn').disabled = false;
            document.getElementById('systemStatus').className = 'status-indicator status-error';
            document.getElementById('systemStatus').textContent = '分析失败';
            showProgress('singleProgress', false);
            showProgress('batchProgress', false);
            document.getElementById('currentStock').style.display = 'none';
            
            document.getElementById('resultsContent').innerHTML = `
                <div class="empty-state">
                    <h3>❌ 分析失败</h3>
                    <p>${data.error || '未知错误'}</p>
                </div>
            `;
            
            addLog(`❌ 分析失败: ${data.error}`, 'error');
        }

        // Analysis functions with SSE support
        async function analyzeSingleStock() {
            const stockCode = document.getElementById('stockCode').value.trim();
            if (!stockCode) {
        addLog('Please enter a stock symbol', 'warning');
                return;
            }

            if (isAnalyzing) {
            addLog('Analysis already in progress—please wait', 'warning');
                return;
            }

            // 市场检测和验证
            let targetMarket = selectedMarket;
            if (selectedMarket === 'auto') {
                targetMarket = detectStockMarket(stockCode);
                if (!targetMarket) {
            addLog('❌ Unable to detect the market from the ticker format—please select manually', 'error');
                    return;
                }
                addLog(`🤖 自动识别为 ${MARKET_CONFIG[targetMarket].name} 市场`, 'info');
            }

            isAnalyzing = true;
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('systemStatus').className = 'status-indicator status-analyzing';
            document.getElementById('systemStatus').textContent = '全球分析中';

            addLog(`🚀 开始全球流式分析股票: ${stockCode} (${MARKET_CONFIG[targetMarket].name})`, 'header');
            showLoading();
            showProgress('singleProgress');

            try {
                const response = await fetch(`${API_BASE}/api/analyze_stream`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        stock_code: stockCode,
                        enable_streaming: document.getElementById('enableStreaming').checked,
                        client_id: currentClientId,
                        target_market: targetMarket
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const result = await response.json();
                
                if (!result.success) {
                    throw new Error(result.error || '分析失败');
                }

            } catch (error) {
                onAnalysisError({error: error.message});
            }
        }

        async function analyzeBatchStocks() {
            const stockListText = document.getElementById('stockList').value.trim();
            if (!stockListText) {
        addLog('Please provide a list of stock symbols', 'warning');
                return;
            }

            if (isAnalyzing) {
                addLog('Analysis already in progress—please wait', 'warning');
                return;
            }

            const stockList = stockListText.split('\n').map(s => s.trim()).filter(s => s);
            if (stockList.length === 0) {
            addLog('The stock symbol list is empty', 'warning');
                return;
            }

            // 验证所有股票代码
            const invalidCodes = [];
            for (const code of stockList) {
                if (!detectStockMarket(code)) {
                    invalidCodes.push(code);
                }
            }

            if (invalidCodes.length > 0) {
                addLog(`❌ 以下股票代码格式无法识别: ${invalidCodes.join(', ')}`, 'error');
                return;
            }

            isAnalyzing = true;
            document.getElementById('batchAnalyzeBtn').disabled = true;
            document.getElementById('systemStatus').className = 'status-indicator status-analyzing';
            document.getElementById('systemStatus').textContent = '全球批量分析中';

            addLog(`🌍 开始全球流式批量分析 ${stockList.length} 只股票`, 'header');
            showLoading();
            showProgress('batchProgress');
            document.getElementById('currentStock').style.display = 'block';

            try {
                const response = await fetch(`${API_BASE}/api/batch_analyze_stream`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        stock_codes: stockList,
                        client_id: currentClientId
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const result = await response.json();
                
                if (!result.success) {
                    throw new Error(result.error || '批量分析失败');
                }

            } catch (error) {
                onAnalysisError({error: error.message});
            }
        }

        // Configuration
        function showConfig() {
            addLog('⚙️ 打开全球市场配置对话框', 'info');
            
            fetch(`${API_BASE}/api/system_info`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const apis = data.data.configured_apis || [];
                        const versions = data.data.api_versions || {};
                        const primary = data.data.primary_api || 'openai';
                        const markets = data.data.supported_markets || [];
                        
                        let configInfo = `🔧 Enhanced v3.1-Multi-Market 全球股票配置状态

🎯 当前系统状态：
✅ 分析器：EnhancedWebStockAnalyzer (全球多市场版)
✅ 高并发：${data.data.max_workers}个工作线程
✅ 活跃任务：${data.data.active_tasks}个
🌊 流式推送：SSE Server-Sent Events
🌍 支持市场：${markets.length}个

🌍 全球市场配置：`;

                        if (markets.length > 0) {
                            configInfo += `
✅ 已启用市场：`;
                            markets.forEach(market => {
                                configInfo += `
  • ${market.flag || '🌍'} ${market.name} (${market.currency})`;
                            });
                        } else {
                            configInfo += `
⚠️ 未检测到已启用的市场`;
                        }

                        configInfo += `

🤖 AI API配置状态：`;

                        if (apis.length > 0) {
                            configInfo += `
✅ 已配置API：${apis.join(', ')}
🎯 主要API：${primary}

API版本详情：`;
                            apis.forEach(api => {
                                const version = versions[api] || '未知';
                                const status = version.includes('未安装') ? '❌' : '✅';
                                configInfo += `
${status} ${api}: ${version}`;
                            });
                            
                            configInfo += `

🚀 AI全球分析功能：完全可用
✅ 多市场深度财务分析
✅ 全球技术面精准解读  
✅ 跨市场情绪挖掘
✅ 国际化投资策略
✅ 汇率风险评估
🌊 实时流式推送`;
                        } else {
                            configInfo += `
⚠️ 未配置任何AI API密钥
🔧 当前使用：高级规则分析模式`;
                        }

                        configInfo += `

📋 配置方法：
1. 编辑项目目录下的 config.json 文件
2. 在 api_keys 部分填入您的API密钥
3. 在 markets 部分配置启用的市场
4. 重启服务器生效

🌟 推荐配置：
• OpenAI GPT-4o-mini (性价比首选)
• Claude-3-haiku (分析质量优秀)
• 智谱AI ChatGLM (国内网络稳定)

💡 全球市场特性：
• 🇨🇳 A股: 6位数字代码 (000001, 600036, 300019)
• 🇭🇰 港股: 5位数字代码 (00700, 00388, 01024)
• 🇺🇸 美股: 字母代码 (AAPL, TSLA, GOOGL)

🌊 SSE流式特性:
• Server-Sent Events: 支持
• 实时进度推送: 启用
• 动态结果更新: 启用
• 多市场并发: 支持
• 断线重连: 自动
• 心跳检测: 启用

📁 相关文件：
• 配置文件：config.json
• 分析器：enhanced_web_stock_analyzer.py (多市场版)
• 服务器：enhanced_flask_web_server.py (全球版)`;

                        alert(configInfo);
                    }
                })
                .catch(error => {
                    const fallbackInfo = `🔧 Enhanced v3.1-Multi-Market 全球股票配置管理

❌ 无法获取当前配置状态，请检查服务器连接

📋 基本配置方法：
1. 在项目目录创建或编辑 config.json
2. 填入AI API密钥
3. 配置支持的市场
4. 重启服务器

🌍 支持市场：A股、港股、美股
🌊 新特性：支持SSE实时流式推送

💡 如需帮助，请查看控制台日志`;
                    alert(fallbackInfo);
                });
        }

        // Export report
        function exportReport() {
            if (!currentAnalysis) {
                addLog('⚠️ 没有可导出的报告', 'warning');
                return;
            }

            try {
                addLog('📤 开始导出全球分析报告...', 'info');
                
                const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
                let content, filename, reportType;

                if (Array.isArray(currentAnalysis)) {
                    const marketCounts = {};
                    currentAnalysis.forEach(report => {
                        const market = MARKET_CONFIG[report.market]?.name || report.market;
                        marketCounts[market] = (marketCounts[market] || 0) + 1;
                    });
                    const marketSummary = Object.entries(marketCounts).map(([market, count]) => `${market}:${count}`).join(',');
                    
                    reportType = `全球批量分析(${currentAnalysis.length}只股票-${marketSummary})`;
                    filename = `global_batch_analysis_${timestamp}.md`;
                    content = generateBatchMarkdown(currentAnalysis);
                } else {
                    const marketInfo = MARKET_CONFIG[currentAnalysis.market] || {};
                    reportType = `单个股票(${currentAnalysis.stock_code}-${marketInfo.name || currentAnalysis.market})`;
                    filename = `global_stock_analysis_${currentAnalysis.stock_code}_${timestamp}.md`;
                    content = generateSingleMarkdown(currentAnalysis);
                }

                const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                a.click();
                URL.revokeObjectURL(url);

                addLog(`✅ ${reportType}报告导出成功: ${filename}`, 'success');
                
                const fileSize = (content.length / 1024).toFixed(1);
                setTimeout(() => {
                    alert(`全球股票分析报告已导出！\\n\\n📄 文件名：${filename}\\n📊 报告类型：${reportType}\\n📏 文件大小：${fileSize} KB\\n🌍 分析范围：全球多市场\\n🌊 分析方式：SSE实时流式推送\\n🔧 分析器：Enhanced v3.1-Multi-Market`);
                }, 100);

            } catch (error) {
                const errorMsg = `导出失败：${error.message}`;
                addLog(`❌ ${errorMsg}`, 'error');
                alert(errorMsg);
            }
        }

        function generateSingleMarkdown(report) {
            const aiAnalysis = report.ai_analysis || '分析数据准备中...';
            const marketInfo = MARKET_CONFIG[report.market] || {};
            
            return `# 📈 全球股票分析报告 (Enhanced v3.1-Multi-Market)

## 🏢 基本信息
| 项目 | 值 |
|------|-----|
| **股票代码** | ${report.stock_code} |
| **原始代码** | ${report.original_code || report.stock_code} |
| **股票名称** | ${report.stock_name} |
| **交易市场** | ${marketInfo.flag || '🌍'} ${marketInfo.name || report.market} |
| **计价货币** | ${marketInfo.currency || 'Unknown'} |
| **分析时间** | ${report.analysis_date} |
| **当前价格** | ${report.price_info.current_price.toFixed(2)} ${marketInfo.currency || ''} |
| **价格变动** | ${report.price_info.price_change.toFixed(2)}% |

## 📊 综合评分

### 🎯 总体评分：${report.scores.comprehensive.toFixed(1)}/100

| 维度 | 得分 | 评级 |
|------|------|------|
| **技术分析** | ${report.scores.technical.toFixed(1)}/100 | ${getScoreRating(report.scores.technical)} |
| **基本面分析** | ${report.scores.fundamental.toFixed(1)}/100 | ${getScoreRating(report.scores.fundamental)} |
| **情绪分析** | ${report.scores.sentiment.toFixed(1)}/100 | ${getScoreRating(report.scores.sentiment)} |

## 🎯 投资建议

### ${report.recommendation}

## 🌍 市场分析

**市场特征:** ${marketInfo.name || report.market}
**交易货币:** ${marketInfo.currency || 'Unknown'}
**数据完整度:** ${report.data_quality.market_coverage}

## 🤖 AI全球分析

${aiAnalysis}

---
*报告生成时间：${new Date().toLocaleString('zh-CN')}*  
*分析器版本：Enhanced v3.1-Multi-Market*  
*分析器类：EnhancedWebStockAnalyzer (全球多市场版)*  
*推送方式：Server-Sent Events 实时流式*  
*支持市场：A股/港股/美股*  
*数据来源：全球多维度综合分析*
`;
        }

        function generateBatchMarkdown(reports) {
            let content = `# 🌍 全球批量股票分析报告 - Enhanced v3.1-Multi-Market

**分析时间：** ${new Date().toLocaleString('zh-CN')}
**分析数量：** ${reports.length} 只股票
**分析器版本：** Enhanced v3.1-Multi-Market
**分析器类：** EnhancedWebStockAnalyzer (全球多市场版)
**推送方式：** Server-Sent Events 实时流式
**支持市场：** A股/港股/美股

## 📋 全球分析汇总

### 市场分布
`;
            
            // 统计市场分布
            const marketStats = {};
            reports.forEach(report => {
                const marketInfo = MARKET_CONFIG[report.market] || {};
                const marketName = marketInfo.name || report.market;
                marketStats[marketName] = (marketStats[marketName] || 0) + 1;
            });
            
            Object.entries(marketStats).forEach(([market, count]) => {
                content += `- **${market}：** ${count} 只\n`;
            });

            content += `\n| 排名 | 股票代码 | 股票名称 | 市场 | 综合得分 | 技术面 | 基本面 | 情绪面 | 投资建议 |
|------|----------|----------|------|----------|--------|--------|--------|----------|
`;

            reports.sort((a, b) => b.scores.comprehensive - a.scores.comprehensive)
                   .forEach((report, index) => {
                const marketInfo = MARKET_CONFIG[report.market] || {};
                content += `| ${index + 1} | ${report.stock_code} | ${report.stock_name} | ${marketInfo.flag || '🌍'} ${marketInfo.name || report.market} | ${report.scores.comprehensive.toFixed(1)} | ${report.scores.technical.toFixed(1)} | ${report.scores.fundamental.toFixed(1)} | ${report.scores.sentiment.toFixed(1)} | ${report.recommendation} |\n`;
            });

            content += `\n## 📈 详细分析\n\n`;
            
            reports.forEach(report => {
                content += generateSingleMarkdown(report);
                content += '\n---\n\n';
            });

            return content;
        }

        function getScoreRating(score) {
            if (score >= 80) return 'Excellent';
            if (score >= 60) return 'Good';
            if (score >= 40) return 'Average';
            return 'Weak';
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            addLog('🇺🇸 U.S. stock analysis system ready', 'success');
            addLog('📋 Enhanced v3.1 | EnhancedWebStockAnalyzer', 'info');
            addLog('🌊 SSE streaming enabled for live progress', 'info');
            addLog('🔥 High-concurrency pipeline: thread pool + async tasks', 'info');
            addLog('🤖 AI analysis: OpenAI/Claude/Zhipu switching supported', 'info');
            addLog('🇺🇸 Market focus: U.S. equities coverage', 'info');
            addLog('🔐 Security: password authentication + session management', 'info');
            addLog('💡 Ticker examples: AAPL, TSLA, MSFT', 'info');
            
            // 初始化SSE连接
            initSSE();
            
            // 检查服务器连接和系统信息
            fetch(`${API_BASE}/api/system_info`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        addLog('✅ Backend connection successful', 'success');
                        addLog(`🔧 Active tasks: ${data.data.active_tasks}`, 'info');
                        addLog(`🧵 Worker threads: ${data.data.max_workers}`, 'info');
                        
                        // 显示支持的市场
                        if (data.data.supported_markets && data.data.supported_markets.length > 0) {
                            const markets = data.data.supported_markets.map(m => m.name).join(', ');
                            addLog(`🇺🇸 Supported market configuration: ${markets}`, 'success');
                        }
                        
                        if (data.data.api_configured) {
                            const apis = data.data.configured_apis || [];
                            const versions = data.data.api_versions || {};
                            const primary = data.data.primary_api || 'openai';
                            
                            addLog(`🤖 AI APIs configured: ${apis.join(', ')}`, 'success');
                            addLog(`🎯 Primary API: ${primary}`, 'info');
                            
                            apis.forEach(api => {
                                const version = versions[api] || '';
                                if (version) {
                                    addLog(`   - ${api}: ${version}`, 'info');
                                }
                            });
                            
                            addLog('🚀 Full AI-driven deep analysis available', 'success');
                        } else {
                            addLog('⚠️ No AI API configured—falling back to rule-based analysis', 'warning');
                            addLog('💡 Add AI API keys for the best U.S. equity experience', 'info');
                        }
                    }
                })
                .catch(error => {
                    addLog('❌ Backend connection failed—please check the server', 'error');
                });
        });

        // 页面卸载时关闭SSE连接
        window.addEventListener('beforeunload', function() {
            if (sseConnection) {
                sseConnection.close();
            }
        });
    </script>
</body>
</html>"""

def init_analyzer():
    """初始化分析器"""
    global analyzer
    try:
        logger.info("正在初始化EnhancedWebStockAnalyzer...")
        analyzer = EnhancedWebStockAnalyzer()
        logger.info("✅ EnhancedWebStockAnalyzer初始化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 分析器初始化失败: {e}")
        return False

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    auth_enabled, auth_config = check_auth_config()
    
    if not auth_enabled:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        password = request.form.get('password', '')
        config_password = auth_config.get('password', '')
        
        if not config_password:
            return render_template_string(LOGIN_TEMPLATE, 
                error="系统未设置访问密码，请联系管理员配置", 
                session_timeout=auth_config.get('session_timeout', 3600) // 60
            )
        
        if password == config_password:
            session['authenticated'] = True
            session['login_time'] = datetime.now().isoformat()
            logger.info("用户登录成功")
            return redirect(url_for('index'))
        else:
            logger.warning("用户登录失败：密码错误")
            return render_template_string(LOGIN_TEMPLATE, 
                error="密码错误，请重试", 
                session_timeout=auth_config.get('session_timeout', 3600) // 60
            )
    
    return render_template_string(LOGIN_TEMPLATE, 
        session_timeout=auth_config.get('session_timeout', 3600) // 60
    )

@app.route('/logout')
def logout():
    """退出登录"""
    session.pop('authenticated', None)
    session.pop('login_time', None)
    logger.info("用户退出登录")
    return redirect(url_for('login'))

@app.route('/')
@require_auth
def index():
    """主页"""
    auth_enabled, _ = check_auth_config()
    return render_template_string(MAIN_TEMPLATE_MULTI_MARKET, auth_enabled=auth_enabled)

@app.route('/api/sse')
@require_auth
def sse_stream():
    """SSE流接口"""
    client_id = request.args.get('client_id')
    if not client_id:
        return "Missing client_id", 400
    
    def event_stream():
        # 创建客户端队列
        client_queue = Queue()
        sse_manager.add_client(client_id, client_queue)
        
        try:
            # 发送连接确认
            yield f"data: {json.dumps({'event': 'connected', 'data': {'client_id': client_id}})}\n\n"
            
            while True:
                try:
                    # 获取消息（带超时，防止长时间阻塞）
                    message = client_queue.get(timeout=30)
                    
                    # 确保消息可以JSON序列化
                    try:
                        json_data = json.dumps(message, ensure_ascii=False)
                        yield f"data: {json_data}\n\n"
                    except (TypeError, ValueError) as e:
                        logger.error(f"SSE消息序列化失败: {e}, 消息类型: {type(message)}")
                        # 发送错误消息
                        error_message = {
                            'event': 'error',
                            'data': {'error': f'消息序列化失败: {str(e)}'},
                            'timestamp': datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(error_message)}\n\n"
                        
                except Empty:
                    # 发送心跳
                    yield f"data: {json.dumps({'event': 'heartbeat', 'data': {'timestamp': datetime.now().isoformat()}})}\n\n"
                except GeneratorExit:
                    break
                except Exception as e:
                    logger.error(f"SSE流处理错误: {e}")
                    try:
                        error_message = {
                            'event': 'error',
                            'data': {'error': f'流处理错误: {str(e)}'},
                            'timestamp': datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(error_message)}\n\n"
                    except:
                        pass
                    break
                    
        except Exception as e:
            logger.error(f"SSE流错误: {e}")
        finally:
            sse_manager.remove_client(client_id)
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )

class StreamingAnalyzer:
    """流式分析器"""
    
    def __init__(self, client_id):
        self.client_id = client_id
    
    def send_log(self, message, log_type='info'):
        """发送日志消息"""
        sse_manager.send_to_client(self.client_id, 'log', {
            'message': message,
            'type': log_type
        })
    
    def send_progress(self, element_id, percent, message=None, current_stock=None):
        """发送进度更新"""
        sse_manager.send_to_client(self.client_id, 'progress', {
            'element_id': element_id,
            'percent': percent,
            'message': message,
            'current_stock': current_stock
        })
    
    def send_scores(self, scores, animate=True):
        """发送评分更新"""
        sse_manager.send_to_client(self.client_id, 'scores_update', {
            'scores': scores,
            'animate': animate
        })
    
    def send_data_quality(self, data_quality):
        """发送数据质量指标"""
        sse_manager.send_to_client(self.client_id, 'data_quality_update', data_quality)
    
    def send_partial_result(self, data):
        """发送部分结果"""
        cleaned_data = clean_data_for_json(data)
        sse_manager.send_to_client(self.client_id, 'partial_result', cleaned_data)
    
    def send_final_result(self, result):
        """发送最终结果"""
        cleaned_result = clean_data_for_json(result)
        sse_manager.send_to_client(self.client_id, 'final_result', cleaned_result)
    
    def send_batch_result(self, results):
        """发送批量结果"""
        cleaned_results = clean_data_for_json(results)
        sse_manager.send_to_client(self.client_id, 'batch_result', cleaned_results)
    
    def send_completion(self, message=None):
        """发送完成信号"""
        sse_manager.send_to_client(self.client_id, 'analysis_complete', {
            'message': message or '分析完成'
        })
    
    def send_error(self, error_message):
        """发送错误信息"""
        sse_manager.send_to_client(self.client_id, 'analysis_error', {
            'error': error_message
        })
    
    def send_ai_stream(self, content):
        """发送AI流式内容"""
        sse_manager.send_to_client(self.client_id, 'ai_stream', {
            'content': content
        })

def analyze_stock_streaming(stock_code, enable_streaming, client_id, target_market=None):
    """流式股票分析（支持多市场）"""
    streamer = StreamingAnalyzer(client_id)
    
    try:
        # 标准化代码并检测市场
        normalized_code, detected_market = analyzer.normalize_stock_code(stock_code)
        if target_market:
            market = target_market
        else:
            market = detected_market
        
        streamer.send_log(f"🚀 开始全球流式分析股票: {normalized_code} ({market.upper()})", 'header')
        streamer.send_progress('singleProgress', 5, f"正在获取{market.upper()}股票基本信息...")
        
        # 获取股票名称
        stock_name = analyzer.get_stock_name(normalized_code)
        streamer.send_log(f"✓ 股票名称: {stock_name} ({market.upper()})", 'success')
        
        # 发送基本信息
        streamer.send_partial_result({
            'type': 'basic_info',
            'stock_code': normalized_code,
            'stock_name': stock_name,
            'market': market,
            'current_price': 0,
            'price_change': 0
        })
        
        # 1. 获取价格数据和技术分析
        streamer.send_progress('singleProgress', 15, f"正在获取{market.upper()}价格数据...")
        streamer.send_log(f"正在获取{market.upper()}历史价格数据...", 'info')
        
        price_data = analyzer.get_stock_data(normalized_code)
        if price_data.empty:
            raise ValueError(f"无法获取股票 {market.upper()} {normalized_code} 的价格数据")
        
        price_info = analyzer.get_price_info(price_data)
        market_config = analyzer.market_config.get(market, {})
        currency = market_config.get('currency', 'CNY')
        streamer.send_log(f"✓ 当前价格: {price_info['current_price']:.2f} {currency}", 'success')
        
        # 更新基本信息
        streamer.send_partial_result({
            'type': 'basic_info',
            'stock_code': normalized_code,
            'stock_name': stock_name,
            'market': market,
            'current_price': price_info['current_price'],
            'price_change': price_info['price_change']
        })
        
        streamer.send_progress('singleProgress', 25, f"正在计算{market.upper()}技术指标...")
        technical_analysis = analyzer.calculate_technical_indicators(price_data)
        technical_score = analyzer.calculate_technical_score(technical_analysis)
        streamer.send_log(f"✓ {market.upper()}技术分析完成，得分: {technical_score:.1f}", 'success')
        
        # 发送技术面得分
        streamer.send_scores({
            'technical': technical_score,
            'fundamental': 50,
            'sentiment': 50,
            'comprehensive': 50
        })
        
        # 2. 获取基本面数据
        streamer.send_progress('singleProgress', 45, f"正在分析{market.upper()}财务指标...")
        streamer.send_log(f"正在获取{market.upper()}财务指标...", 'info')
        
        fundamental_data = analyzer.get_comprehensive_fundamental_data(normalized_code)
        fundamental_score = analyzer.calculate_fundamental_score(fundamental_data)
        streamer.send_log(f"✓ {market.upper()}基本面分析完成，得分: {fundamental_score:.1f}", 'success')
        
        # 发送基本面得分
        streamer.send_scores({
            'technical': technical_score,
            'fundamental': fundamental_score,
            'sentiment': 50,
            'comprehensive': (technical_score + fundamental_score) / 2
        })
        
        # 3. 获取新闻和情绪分析
        streamer.send_progress('singleProgress', 65, f"正在分析{market.upper()}市场情绪...")
        streamer.send_log(f"正在获取{market.upper()}新闻数据和分析市场情绪...", 'info')
        
        comprehensive_news_data = analyzer.get_comprehensive_news_data(normalized_code, days=30)
        sentiment_analysis = analyzer.calculate_advanced_sentiment_analysis(comprehensive_news_data)
        sentiment_score = analyzer.calculate_sentiment_score(sentiment_analysis)
        streamer.send_log(f"✓ {market.upper()}情绪分析完成，得分: {sentiment_score:.1f}", 'success')
        
        # 合并新闻数据到情绪分析结果中
        sentiment_analysis.update(comprehensive_news_data)
        
        # 4. 计算综合得分
        scores = {
            'technical': technical_score,
            'fundamental': fundamental_score,
            'sentiment': sentiment_score,
            'comprehensive': analyzer.calculate_comprehensive_score({
                'technical': technical_score,
                'fundamental': fundamental_score,
                'sentiment': sentiment_score
            })
        }
        
        # 发送最终得分
        streamer.send_scores(scores, animate=True)
        
        # 发送数据质量指标
        data_quality = {
            'financial_indicators_count': len(fundamental_data.get('financial_indicators', {})),
            'total_news_count': sentiment_analysis.get('total_analyzed', 0),
            'analysis_completeness': '完整' if len(fundamental_data.get('financial_indicators', {})) >= 10 else '部分',
            'market_coverage': market.upper()
        }
        streamer.send_data_quality(data_quality)
        
        # 5. 生成投资建议
        streamer.send_progress('singleProgress', 80, f"正在生成{market.upper()}投资建议...")
        recommendation = analyzer.generate_recommendation(scores, market)
        streamer.send_log(f"✓ {market.upper()}投资建议: {recommendation}", 'success')
        
        # 6. AI增强分析（流式）
        streamer.send_progress('singleProgress', 90, f"正在进行{market.upper()}AI深度分析...")
        streamer.send_log(f"🤖 正在调用AI进行{market.upper()}深度分析...", 'info')
        
        # 设置AI流式内容处理
        ai_content_buffer = ""
        
        def ai_stream_callback(content):
            """AI流式内容回调"""
            nonlocal ai_content_buffer
            ai_content_buffer += content
            # 实时发送AI流式内容
            streamer.send_ai_stream(content)
        
        # 执行AI分析，支持流式输出
        ai_analysis = analyzer.generate_ai_analysis({
            'stock_code': normalized_code,
            'stock_name': stock_name,
            'price_info': price_info,
            'technical_analysis': technical_analysis,
            'fundamental_data': fundamental_data,
            'sentiment_analysis': sentiment_analysis,
            'scores': scores,
            'market': market
        }, enable_streaming, ai_stream_callback)
        
        # 如果AI分析返回了完整内容，使用返回的内容，否则使用缓冲的内容
        if not ai_analysis and ai_content_buffer:
            ai_analysis = ai_content_buffer
        
        streamer.send_log(f"✅ {market.upper()}AI深度分析完成", 'success')
        
        # 7. 生成最终报告
        streamer.send_progress('singleProgress', 100, "全球分析完成")
        
        report = {
            'stock_code': normalized_code,
            'original_code': stock_code,
            'stock_name': stock_name,
            'market': market,
            'market_info': market_config,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'price_info': price_info,
            'technical_analysis': technical_analysis,
            'fundamental_data': fundamental_data,
            'comprehensive_news_data': comprehensive_news_data,
            'sentiment_analysis': sentiment_analysis,
            'scores': scores,
            'analysis_weights': analyzer.analysis_weights,
            'recommendation': recommendation,
            'ai_analysis': ai_analysis,
            'data_quality': data_quality
        }
        
        # 发送最终结果
        streamer.send_final_result(report)
        streamer.send_completion(f"✅ {market.upper()} {normalized_code} 全球流式分析完成，综合得分: {scores['comprehensive']:.1f}")
        
        return report
        
    except Exception as e:
        error_msg = f"全球流式分析失败: {str(e)}"
        streamer.send_error(error_msg)
        streamer.send_log(f"❌ {error_msg}", 'error')
        raise

def analyze_batch_streaming(stock_codes, client_id):
    """流式批量股票分析（支持多市场）"""
    streamer = StreamingAnalyzer(client_id)
    
    try:
        total_stocks = len(stock_codes)
        streamer.send_log(f"🌍 开始全球流式批量分析 {total_stocks} 只股票", 'header')
        
        results = []
        failed_stocks = []
        market_counts = {}
        
        for i, stock_code in enumerate(stock_codes):
            try:
                progress = int((i / total_stocks) * 100)
                
                # 检测市场
                normalized_code, market = analyzer.normalize_stock_code(stock_code)
                market_counts[market] = market_counts.get(market, 0) + 1
                
                streamer.send_progress('batchProgress', progress, 
                    f"正在分析第 {i+1}/{total_stocks} 只股票", 
                    f"{normalized_code} ({market.upper()})")
                
                streamer.send_log(f"🔍 开始分析 {normalized_code} ({market.upper()}) ({i+1}/{total_stocks})", 'info')
                
                # 分析单只股票（简化版，不发送中间进度）
                report = analyzer.analyze_stock(normalized_code, False)
                results.append(report)
                
                streamer.send_log(f"✓ {normalized_code} ({market.upper()}) 分析完成 (得分: {report['scores']['comprehensive']:.1f})", 'success')
                
            except Exception as e:
                failed_stocks.append(stock_code)
                streamer.send_log(f"❌ {stock_code} 分析失败: {e}", 'error')
        
        # 计算平均得分并发送
        if results:
            avg_scores = {
                'comprehensive': sum(r['scores']['comprehensive'] for r in results) / len(results),
                'technical': sum(r['scores']['technical'] for r in results) / len(results),
                'fundamental': sum(r['scores']['fundamental'] for r in results) / len(results),
                'sentiment': sum(r['scores']['sentiment'] for r in results) / len(results)
            }
            streamer.send_scores(avg_scores, animate=True)
            
            # 发送数据质量指标
            avg_financial = sum(r['data_quality']['financial_indicators_count'] for r in results) / len(results)
            avg_news = sum(r['sentiment_analysis']['total_analyzed'] for r in results) / len(results)
            
            streamer.send_data_quality({
                'financial_indicators_count': round(avg_financial),
                'total_news_count': round(avg_news),
                'analysis_completeness': '批量',
                'market_coverage': '全球'
            })
        
        streamer.send_progress('batchProgress', 100, f"全球批量分析完成")
        
        # 发送批量结果
        streamer.send_batch_result(results)
        
        success_count = len(results)
        market_summary = ", ".join([f"{market.upper()}:{count}" for market, count in market_counts.items()])
        message = f"🎉 全球批量分析完成！成功分析 {success_count}/{total_stocks} 只股票 ({market_summary})"
        if failed_stocks:
            message += f"，失败: {', '.join(failed_stocks)}"
        
        streamer.send_completion(message)
        
        return results
        
    except Exception as e:
        error_msg = f"全球批量流式分析失败: {str(e)}"
        streamer.send_error(error_msg)
        streamer.send_log(f"❌ {error_msg}", 'error')
        raise

@app.route('/api/analyze_stream', methods=['POST'])
@require_auth
def analyze_stock_stream():
    """单只股票流式分析接口（支持多市场）"""
    try:
        if not analyzer:
            return jsonify({
                'success': False,
                'error': 'Analyzer not initialised'
            }), 500
        
        data = request.json
        stock_code = data.get('stock_code', '').strip()
        enable_streaming = data.get('enable_streaming', False)
        client_id = data.get('client_id')
        target_market = data.get('target_market')
        
        if not stock_code:
            return jsonify({
                'success': False,
                'error': 'Stock symbol cannot be empty'
            }), 400
        
        if not client_id:
            return jsonify({
                'success': False,
                'error': 'Missing client ID'
            }), 400
        
        # 验证股票代码格式
        is_valid, validation_message = analyzer.validate_stock_code(stock_code)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'股票代码格式错误: {validation_message}'
            }), 400
        
        # 检查是否有相同的分析正在进行
        with task_lock:
            if stock_code in analysis_tasks:
                return jsonify({
                    'success': False,
                    'error': f'股票 {stock_code} 正在分析中，请稍候'
                }), 429
            
            analysis_tasks[stock_code] = {
                'start_time': datetime.now(),
                'status': 'analyzing',
                'client_id': client_id,
                'market': target_market
            }
        
        logger.info(f"开始全球流式分析股票: {stock_code}, 目标市场: {target_market}, 客户端: {client_id}")
        
        # 异步执行分析
        def run_analysis():
            try:
                global currentAnalysis
                report = analyze_stock_streaming(stock_code, enable_streaming, client_id, target_market)
                currentAnalysis = report
                logger.info(f"全球股票流式分析完成: {stock_code}")
            except Exception as e:
                logger.error(f"全球股票流式分析失败: {stock_code}, 错误: {e}")
            finally:
                with task_lock:
                    analysis_tasks.pop(stock_code, None)
        
        # 在线程池中执行
        executor.submit(run_analysis)
        
        return jsonify({
            'success': True,
            'message': f'股票 {stock_code} 全球流式分析已启动',
            'client_id': client_id,
            'target_market': target_market
        })
        
    except Exception as e:
        logger.error(f"启动全球股票流式分析失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch_analyze_stream', methods=['POST'])
@require_auth
def batch_analyze_stream():
    """批量股票流式分析接口（支持多市场）"""
    try:
        if not analyzer:
            return jsonify({
                'success': False,
                'error': 'Analyzer not initialised'
            }), 500
        
        data = request.json
        stock_codes = data.get('stock_codes', [])
        client_id = data.get('client_id')
        
        if not stock_codes:
            return jsonify({
                'success': False,
                'error': 'Stock symbol list cannot be empty'
            }), 400
        
        if not client_id:
            return jsonify({
                'success': False,
                'error': 'Missing client ID'
            }), 400
        
        # 限制批量分析数量
        if len(stock_codes) > 10:
            return jsonify({
                'success': False,
                'error': 'Batch analysis supports up to 10 symbols'
            }), 400
        
        # 验证所有股票代码
        invalid_codes = []
        market_distribution = {}
        
        for code in stock_codes:
            is_valid, validation_message = analyzer.validate_stock_code(code)
            if not is_valid:
                invalid_codes.append(f"{code}: {validation_message}")
            else:
                _, market = analyzer.normalize_stock_code(code)
                market_distribution[market] = market_distribution.get(market, 0) + 1
        
        if invalid_codes:
            return jsonify({
                'success': False,
                'error': f'以下股票代码格式错误: {"; ".join(invalid_codes)}'
            }), 400
        
        logger.info(f"开始全球流式批量分析 {len(stock_codes)} 只股票, 市场分布: {market_distribution}, 客户端: {client_id}")
        
        # 异步执行批量分析
        def run_batch_analysis():
            try:
                global currentAnalysis
                results = analyze_batch_streaming(stock_codes, client_id)
                currentAnalysis = results
                logger.info(f"全球批量流式分析完成，成功分析 {len(results)}/{len(stock_codes)} 只股票")
            except Exception as e:
                logger.error(f"全球批量流式分析失败: {e}")
        
        # 在线程池中执行
        executor.submit(run_batch_analysis)
        
        return jsonify({
            'success': True,
            'message': f'U.S. batch analysis started for {len(stock_codes)} symbols',
            'client_id': client_id,
            'market_distribution': market_distribution
        })
        
    except Exception as e:
        logger.error(f"启动全球批量流式分析失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """系统状态检查"""
    try:
        auth_enabled, auth_config = check_auth_config()
        supported_markets = analyzer.get_supported_markets() if analyzer else []
        
        return jsonify({
            'success': True,
            'status': 'ready',
            'message': 'U.S. stock analysis system is running',
            'analyzer_available': analyzer is not None,
            'auth_enabled': auth_enabled,
            'sse_support': True,
            'supported_markets': supported_markets,
            'version': 'Enhanced v3.1 (U.S.)',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze', methods=['POST'])
@require_auth
def analyze_stock():
    """单只股票分析 - 兼容接口（非流式，支持多市场）"""
    try:
        if not analyzer:
            return jsonify({
                'success': False,
                'error': 'Analyzer not initialised'
            }), 500
        
        data = request.json
        stock_code = data.get('stock_code', '').strip()
        enable_streaming = data.get('enable_streaming', False)
        
        if not stock_code:
            return jsonify({
                'success': False,
                'error': 'Stock symbol cannot be empty'
            }), 400
        
        # 验证股票代码格式
        is_valid, validation_message = analyzer.validate_stock_code(stock_code)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'股票代码格式错误: {validation_message}'
            }), 400
        
        # 检查是否有相同的分析正在进行
        with task_lock:
            if stock_code in analysis_tasks:
                return jsonify({
                    'success': False,
                    'error': f'股票 {stock_code} 正在分析中，请稍候'
                }), 429
            
            analysis_tasks[stock_code] = {
                'start_time': datetime.now(),
                'status': 'analyzing'
            }
        
        logger.info(f"开始全球分析股票: {stock_code}")
        
        try:
            # 执行分析
            report = analyzer.analyze_stock(stock_code, enable_streaming)
            
            # 清理数据中的NaN值
            cleaned_report = clean_data_for_json(report)
            
            logger.info(f"全球股票分析完成: {stock_code} ({report.get('market', 'Unknown').upper()})")
            
            return jsonify({
                'success': True,
                'data': cleaned_report,
                'message': f'股票 {stock_code} 全球分析完成'
            })
            
        finally:
            with task_lock:
                analysis_tasks.pop(stock_code, None)
        
    except Exception as e:
        with task_lock:
            analysis_tasks.pop(stock_code, None)
        
        logger.error(f"全球股票分析失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch_analyze', methods=['POST'])
@require_auth
def batch_analyze():
    """批量股票分析 - 兼容接口（非流式，支持多市场）"""
    try:
        if not analyzer:
            return jsonify({
                'success': False,
                'error': 'Analyzer not initialised'
            }), 500
        
        data = request.json
        stock_codes = data.get('stock_codes', [])
        
        if not stock_codes:
            return jsonify({
                'success': False,
                'error': 'Stock symbol list cannot be empty'
            }), 400
        
        if len(stock_codes) > 10:
            return jsonify({
                'success': False,
                'error': 'Batch analysis supports up to 10 symbols'
            }), 400
        
        # 验证所有股票代码
        invalid_codes = []
        market_distribution = {}
        
        for code in stock_codes:
            is_valid, validation_message = analyzer.validate_stock_code(code)
            if not is_valid:
                invalid_codes.append(f"{code}: {validation_message}")
            else:
                _, market = analyzer.normalize_stock_code(code)
                market_distribution[market] = market_distribution.get(market, 0) + 1
        
        if invalid_codes:
            return jsonify({
                'success': False,
                'error': f'以下股票代码格式错误: {"; ".join(invalid_codes)}'
            }), 400
        
        logger.info(f"开始全球批量分析 {len(stock_codes)} 只股票，市场分布: {market_distribution}")
        
        results = []
        failed_stocks = []
        
        # 使用线程池并发处理
        futures = {}
        for stock_code in stock_codes:
            future = executor.submit(analyzer.analyze_stock, stock_code, False)
            futures[future] = stock_code
        
        # 收集结果
        for future in futures:
            stock_code = futures[future]
            try:
                report = future.result(timeout=60)
                results.append(report)
                market = report.get('market', 'Unknown').upper()
                logger.info(f"✓ {stock_code} ({market}) 分析完成")
            except Exception as e:
                failed_stocks.append(stock_code)
                logger.error(f"❌ {stock_code} 分析失败: {e}")
        
        # 清理数据中的NaN值
        cleaned_results = clean_data_for_json(results)
        
        success_count = len(results)
        total_count = len(stock_codes)
        
        # 统计各市场成功数量
        success_by_market = {}
        for result in results:
            market = result.get('market', 'Unknown')
            success_by_market[market] = success_by_market.get(market, 0) + 1
        
        logger.info(f"全球批量分析完成，成功分析 {success_count}/{total_count} 只股票，市场分布: {success_by_market}")
        
        response_data = {
            'success': True,
            'data': cleaned_results,
            'message': f'全球批量分析完成，成功分析 {success_count}/{total_count} 只股票',
            'market_distribution': market_distribution,
            'success_by_market': success_by_market
        }
        
        if failed_stocks:
            response_data['failed_stocks'] = failed_stocks
            response_data['message'] += f'，失败股票: {", ".join(failed_stocks)}'
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"全球批量分析失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/task_status/<stock_code>', methods=['GET'])
@require_auth
def get_task_status(stock_code):
    """获取分析任务状态"""
    try:
        with task_lock:
            task_info = analysis_tasks.get(stock_code)
            
        if not task_info:
            return jsonify({
                'success': True,
                'status': 'not_found',
                'message': f'未找到股票 {stock_code} 的分析任务'
            })
        
        # 计算分析时长
        elapsed_time = (datetime.now() - task_info['start_time']).total_seconds()
        
        return jsonify({
            'success': True,
            'status': task_info['status'],
            'elapsed_time': elapsed_time,
            'client_id': task_info.get('client_id'),
            'market': task_info.get('market'),
            'message': f'股票 {stock_code} 正在分析中'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system_info', methods=['GET'])
def get_system_info():
    """获取系统信息"""
    try:
        with task_lock:
            active_tasks = len(analysis_tasks)
        
        with sse_lock:
            sse_clients_count = len(sse_manager.clients)
        
        # 检测配置的API
        configured_apis = []
        api_versions = {}
        
        if analyzer:
            for api_name, api_key in analyzer.api_keys.items():
                if api_name != 'notes' and api_key and api_key.strip():
                    configured_apis.append(api_name)
                    
                    # 检测API版本/状态
                    if api_name == 'openai':
                        try:
                            import openai
                            if hasattr(openai, 'OpenAI'):
                                api_versions[api_name] = "新版本"
                            else:
                                api_versions[api_name] = "旧版本"
                        except ImportError:
                            api_versions[api_name] = "未安装"
                    elif api_name == 'anthropic':
                        try:
                            import anthropic
                            api_versions[api_name] = "已安装"
                        except ImportError:
                            api_versions[api_name] = "未安装"
                    elif api_name == 'zhipu':
                        try:
                            import zhipuai
                            api_versions[api_name] = "已安装"
                        except ImportError:
                            api_versions[api_name] = "未安装"
        
        # 检测鉴权状态
        auth_enabled, auth_config = check_auth_config()
        
        # 获取支持的市场
        supported_markets = analyzer.get_supported_markets() if analyzer else []
        
        return jsonify({
            'success': True,
            'data': {
                'analyzer_available': analyzer is not None,
                'active_tasks': active_tasks,
                'max_workers': executor._max_workers,
                'sse_clients': sse_clients_count,
                'sse_support': True,
                'configured_apis': configured_apis,
                'api_versions': api_versions,
                'api_configured': len(configured_apis) > 0,
                'primary_api': analyzer.config.get('ai', {}).get('model_preference', 'openai') if analyzer else None,
                'auth_enabled': auth_enabled,
                'auth_configured': auth_config.get('password', '') != '',
                'supported_markets': supported_markets,
                'multi_market_support': True,
                'version': 'Enhanced v3.1-Multi-Market',
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validate_stock', methods=['POST'])
@require_auth  
def validate_stock():
    """验证股票代码接口"""
    try:
        if not analyzer:
            return jsonify({
                'success': False,
                'error': '分析器未初始化'
            }), 500
        
        data = request.json
        stock_code = data.get('stock_code', '').strip()
        
        if not stock_code:
            return jsonify({
                'success': False,
                'error': '股票代码不能为空'
            }), 400
        
        # 验证股票代码
        is_valid, message = analyzer.validate_stock_code(stock_code)
        normalized_code, market = analyzer.normalize_stock_code(stock_code)
        
        market_info = analyzer.market_config.get(market, {})
        
        return jsonify({
            'success': True,
            'data': {
                'is_valid': is_valid,
                'message': message,
                'original_code': stock_code,
                'normalized_code': normalized_code,
                'market': market,
                'market_info': {
                    'name': market.upper().replace('_', ''),
                    'currency': market_info.get('currency', 'Unknown'),
                    'enabled': market_info.get('enabled', True)
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

def main():
    """主函数"""
    print("🌍 启动全球股票分析系统（Multi-Market Edition）...")
    print("🌊 Server-Sent Events | 实时流式推送 | A股/港股/美股")
    print("=" * 70)
    
    # 检查依赖
    missing_deps = []
    
    try:
        import akshare
        print("   ✅ akshare: 已安装")
    except ImportError:
        missing_deps.append("akshare")
        print("   ❌ akshare: 未安装")
    
    try:
        import pandas
        print("   ✅ pandas: 已安装")
    except ImportError:
        missing_deps.append("pandas")
        print("   ❌ pandas: 未安装")
    
    try:
        import flask
        print("   ✅ flask: 已安装")
    except ImportError:
        missing_deps.append("flask")
        print("   ❌ flask: 未安装")
    
    try:
        import flask_cors
        print("   ✅ flask-cors: 已安装")
    except ImportError:
        missing_deps.append("flask-cors")
        print("   ❌ flask-cors: 未安装")
    
    # 检查AI依赖
    ai_deps = []
    try:
        import openai
        if hasattr(openai, 'OpenAI'):
            ai_deps.append("OpenAI (新版)")
        else:
            ai_deps.append("OpenAI (旧版)")
    except ImportError:
        pass
    
    try:
        import anthropic
        ai_deps.append("Claude")
    except ImportError:
        pass
    
    try:
        import zhipuai
        ai_deps.append("智谱AI")
    except ImportError:
        pass
    
    if ai_deps:
        print(f"   🤖 AI支持: {', '.join(ai_deps)}")
    else:
        print("   ⚠️  AI依赖: 未安装 (pip install openai anthropic zhipuai)")
    
    # 检查配置文件
    if os.path.exists('config.json'):
        print("   ✅ config.json: 已存在")
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_keys = config.get('api_keys', {})
                configured_apis = [name for name, key in api_keys.items() 
                                 if name != 'notes' and key and key.strip()]
                if configured_apis:
                    print(f"   🔑 已配置API: {', '.join(configured_apis)}")
                else:
                    print("   ⚠️  API密钥: 未配置")
                
                # 检查市场配置
                markets = config.get('markets', {})
                enabled_markets = [market for market, cfg in markets.items() if cfg.get('enabled', True)]
                if enabled_markets:
                    print(f"   🌍 启用市场: {', '.join([m.upper().replace('_', '') for m in enabled_markets])}")
                else:
                    print("   🌍 启用市场: 默认全部")
                
                # 检查Web鉴权配置
                web_auth = config.get('web_auth', {})
                if web_auth.get('enabled', False):
                    if web_auth.get('password'):
                        print(f"   🔐 Web鉴权: 已启用 (会话超时: {web_auth.get('session_timeout', 3600)}秒)")
                    else:
                        print("   ⚠️  Web鉴权: 已启用但未设置密码")
                else:
                    print("   🔓 Web鉴权: 未启用")
                    
        except Exception as e:
            print(f"   ❌ config.json: 格式错误 - {e}")
    else:
        print("   ⚠️  config.json: 不存在，将使用默认配置")
    
    if missing_deps:
        print(f"❌ 缺少必要依赖: {', '.join(missing_deps)}")
        print(f"请运行以下命令安装: pip install {' '.join(missing_deps)}")
        return
    
    print("=" * 70)
    
    # 初始化分析器
    if not init_analyzer():
        print("❌ 分析器初始化失败，程序退出")
        return
    
    print("✅ 全球股票分析系统初始化完成！")
    print("🌍 全球市场支持:")
    if analyzer:
        markets = analyzer.get_supported_markets()
        for market in markets:
            print(f"   - {market['name']} (货币: {market['currency']}, 时区: {market['timezone']})")
    
    print("🌊 SSE流式特性:")
    print("   - Server-Sent Events: 支持")
    print("   - 实时进度推送: 启用")
    print("   - 动态结果更新: 启用")
    print("   - 客户端连接管理: 自动化")
    print("   - 断线重连: 自动")
    print("   - 心跳检测: 启用")
    
    print("🔥 高并发特性:")
    print(f"   - 线程池: {executor._max_workers} 个工作线程")
    print("   - 异步分析: 支持")
    print("   - 任务队列: 支持")
    print("   - 重复请求防护: 启用")
    print("   - 批量并发优化: 启用")
    print("   - SSE连接池: 支持")
    print("   - 多市场并发: 支持")
    
    print("🔐 安全特性:")
    if analyzer:
        web_auth = analyzer.config.get('web_auth', {})
        if web_auth.get('enabled', False):
            if web_auth.get('password'):
                timeout_minutes = web_auth.get('session_timeout', 3600) // 60
                print(f"   - 密码鉴权: 已启用")
                print(f"   - 会话超时: {timeout_minutes} 分钟")
                print(f"   - 安全状态: 保护模式")
            else:
                print("   - 密码鉴权: 已启用但未设置密码")
                print("   - 安全状态: 配置不完整")
        else:
            print("   - 密码鉴权: 未启用")
            print("   - 安全状态: 开放模式")
    else:
        print("   - 鉴权配置: 无法检测")
    
    print("🤖 AI分析特性:")
    if analyzer:
        api_keys = analyzer.api_keys
        configured_apis = [name for name, key in api_keys.items() 
                          if name != 'notes' and key and key.strip()]
        if configured_apis:
            print(f"   - 已配置API: {', '.join(configured_apis)}")
            primary_api = analyzer.config.get('ai', {}).get('model_preference', 'openai')
            print(f"   - 主要API: {primary_api}")
            
            api_base = analyzer.config.get('ai', {}).get('api_base_urls', {}).get('openai')
            if api_base and api_base != 'https://api.openai.com/v1':
                print(f"   - 自定义API地址: {api_base}")
            
            model = analyzer.config.get('ai', {}).get('models', {}).get(primary_api, 'default')
            print(f"   - 使用模型: {model}")
            
            print("   - LLM深度分析: 完整支持")
            print("   - 流式AI推理: 支持")
            print("   - 多市场AI分析: 支持")
        else:
            print("   - API配置: 未配置")
            print("   - 分析模式: 高级规则分析")
    else:
        print("   - 分析器: 未初始化")
    
    print("   - 多模型支持: OpenAI/Claude/智谱AI")
    print("   - 智能切换: 启用")
    print("   - 版本兼容: 新旧版本自动适配")
    print("   - 规则分析备用: 启用")
    print("   - 跨市场分析: 支持")
    
    print("📋 分析配置:")
    if analyzer:
        params = analyzer.analysis_params
        weights = analyzer.analysis_weights
        print(f"   - 技术分析周期: {params.get('technical_period_days', 180)} 天")
        print(f"   - 财务指标数量: {params.get('financial_indicators_count', 25)} 项")
        print(f"   - 新闻分析数量: {params.get('max_news_count', 100)} 条")
        print(f"   - 分析权重: 技术{weights['technical']:.1f} | 基本面{weights['fundamental']:.1f} | 情绪{weights['sentiment']:.1f}")
        print(f"   - 多市场支持: {len(analyzer.get_supported_markets())} 个市场")
    else:
        print("   - 配置: 使用默认值")
    
    print("📋 性能优化:")
    print("   - 日志文件: 已禁用")
    print("   - JSON压缩: 启用")
    print("   - 缓存优化: 启用")
    print("   - NaN值清理: 启用")
    print("   - SSE消息队列: 启用")
    print("   - 多市场缓存: 分离")
    
    print("🌐 Web服务器启动中...")
    print("📱 请在浏览器中访问: http://localhost:5000")
    
    if analyzer and analyzer.config.get('web_auth', {}).get('enabled', False):
        print("🔐 首次访问需要密码验证")
    
    print("🔧 API接口文档:")
    print("   - GET  /api/status - 系统状态")
    print("   - GET  /api/sse?client_id=xxx - SSE流式接口")
    print("   - POST /api/analyze_stream - 单只股票流式分析")
    print("   - POST /api/batch_analyze_stream - 批量股票流式分析")
    print("   - POST /api/analyze - 单只股票分析 (兼容)")
    print("   - POST /api/batch_analyze - 批量股票分析 (兼容)")
    print("   - POST /api/validate_stock - 股票代码验证")
    print("   - GET  /api/task_status/<code> - 任务状态")
    print("   - GET  /api/system_info - 系统信息")
    print("   - GET  /login - 登录页面 (如启用鉴权)")
    print("   - GET  /logout - 退出登录")
    
    print("🌍 支持股票代码:")
    print("   - A股: 6位数字 (000001, 600036, 300019)")
    print("   - 港股: 5位数字 (00700, 00388, 01024)")
    print("   - 美股: 字母代码 (AAPL, TSLA, GOOGL)")
    
    print("🌊 SSE事件类型:")
    print("   - connected: 连接确认")
    print("   - log: 日志消息") 
    print("   - progress: 进度更新")
    print("   - scores_update: 评分更新")
    print("   - data_quality_update: 数据质量更新")
    print("   - partial_result: 部分结果")
    print("   - final_result: 最终结果")
    print("   - batch_result: 批量结果")
    print("   - analysis_complete: 分析完成")
    print("   - analysis_error: 分析错误")
    print("   - ai_stream: AI流式内容")
    print("   - heartbeat: 心跳")
    print("=" * 70)
    
    # 启动Flask服务器
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
            processes=1
        )
    except KeyboardInterrupt:
        print("\n👋 全球股票分析系统已关闭")
        executor.shutdown(wait=True)
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        executor.shutdown(wait=True)

if __name__ == '__main__':
    main()