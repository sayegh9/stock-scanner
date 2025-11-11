"""Enhanced web stock analyzer tuned for U.S. equities.

This module keeps the public interface that the Flask streaming server expects
while trimming the original multi-market implementation down to a concise,
English-only version. The class focuses on U.S. tickers, loads configuration
from ``config.json`` (falling back to sensible defaults), and exposes helper
methods for price, technical, fundamental, and sentiment analysis.

The analytical routines are intentionally lightweight so they function without
commercial data feeds. Whenever possible the analyzer fetches daily price
history via ``akshare``. If ``akshare`` is unavailable the methods raise a
clear error so operators can install the dependency or swap in an alternative
provider.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests import Response
from requests.exceptions import RequestException

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency
    yf = None

try:  # pragma: no cover - optional dependency
    import akshare as ak
except ImportError:  # pragma: no cover - optional dependency
    ak = None

try:  # pragma: no cover - optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class NewsProviderError(Exception):
    """Base error raised when a news provider fails."""


class RateLimitError(NewsProviderError):
    """Raised when a provider reports rate limiting."""

AK_PRICE_COLUMNS = {
    '\u65e5\u671f': 'date',
    '\u5f00\u76d8': 'open',
    '\u6536\u76d8': 'close',
    '\u6700\u9ad8': 'high',
    '\u6700\u4f4e': 'low',
    '\u6210\u4ea4\u91cf': 'volume',
}

AK_FUNDAMENTAL_KEYS = {
    '\u5e02\u76c8\u7387': 'pe_ratio',
    '\u6bcf\u80a1\u6536\u76ca': 'eps',
    '\u51c0\u8d44\u4ea7\u6536\u76ca\u7387': 'roe',
    '\u51c0\u5229\u6da6\u540c\u6bd4\u589e\u957f': 'net_profit_growth',
    '\u8425\u4e1a\u6536\u5165\u540c\u6bd4\u589e\u957f': 'revenue_growth',
}


@dataclass
class MarketInfo:
    """Description for a supported market.

    ``name`` and the other descriptive attributes default to empty strings so the
    analyzer can tolerate minimal configuration files that only toggle the
    ``enabled`` flag without providing human-readable metadata. This mirrors the
    behaviour of the legacy multi-market project where those fields were
    optional.
    """

    code: str
    name: str = ""
    currency: str = ""
    timezone: str = ""
    trading_hours: str = ""
    enabled: bool = True


class EnhancedWebStockAnalyzer:
    """U.S.-centric stock analyzer with a streaming-friendly API."""

    def __init__(self, config_file: str = "config.json") -> None:
        self.config_file = config_file
        self.config = self._load_config()

        self.analysis_weights = self.config.get("analysis_weights", {})
        self.analysis_params = self.config.get("analysis_params", {})
        allowed_markets = {"us_stock"}
        raw_markets = self.config.get("markets", {})
        if isinstance(raw_markets, dict):
            filtered_markets: Dict[str, Dict[str, str]] = {}
            for code, value in raw_markets.items():
                if code not in allowed_markets or not isinstance(value, dict):
                    continue
                sanitized = dict(value)
                sanitized.setdefault("name", "U.S. equities")
                sanitized.setdefault("currency", "USD")
                sanitized.setdefault("timezone", "America/New_York")
                sanitized.setdefault("trading_hours", "09:30-16:00")
                sanitized["enabled"] = bool(sanitized.get("enabled", True))
                filtered_markets[code] = sanitized
        else:
            filtered_markets = {}

        if "us_stock" not in filtered_markets:
            filtered_markets = self._default_config()["markets"]

        self.market_config = filtered_markets
        self.config["markets"] = filtered_markets
        self.streaming_config = self.config.get("streaming", {})
        self.cache_config = self.config.get("cache", {})
        self.api_keys = self.config.get("api_keys", {})

        self._price_cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self._fundamental_cache: Dict[str, Tuple[datetime, Dict[str, float]]] = {}
        self._news_cache: Dict[str, Tuple[datetime, Dict[str, List[dict]]]] = {}
        self._profile_cache: Dict[str, Dict[str, Any]] = {}
        self._sentiment_analyzer: Optional[SentimentIntensityAnalyzer] = None
        self._http_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0 Safari/537.36"
            )
        }

        self._log_config_summary()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> Dict:
        """Load configuration from disk or create defaults."""

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as fh:
                    config = json.load(fh)
                    logger.info("Loaded configuration from %s", self.config_file)
                    return config
            except json.JSONDecodeError:
                backup = f"{self.config_file}.invalid-{int(datetime.now().timestamp())}"
                os.rename(self.config_file, backup)
                logger.warning(
                    "Configuration file %s is invalid JSON. A backup was written to %s",
                    self.config_file,
                    backup,
                )
        else:
            logger.info("No configuration file found. Using default settings.")

        config = self._default_config()
        self._save_config(config)
        return config

    def _save_config(self, config: Dict) -> None:
        """Persist configuration to disk."""

        try:
            with open(self.config_file, "w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=4, ensure_ascii=False)
            logger.info("Saved configuration to %s", self.config_file)
        except OSError as exc:
            logger.error("Unable to write configuration: %s", exc)

    def _default_config(self) -> Dict:
        """Return a default configuration tuned for U.S. equities."""

        return {
            "api_keys": {
                "openai": "",
                "anthropic": "",
                "zhipu": "",
            },
            "ai": {
                "model_preference": "openai",
                "models": {
                    "openai": "gpt-4o-mini",
                    "anthropic": "claude-3-haiku-20240307",
                    "zhipu": "chatglm_turbo",
                },
                "max_tokens": 4000,
                "temperature": 0.7,
                "api_base_urls": {
                    "openai": "https://api.openai.com/v1",
                },
            },
            "analysis_weights": {
                "technical": 0.4,
                "fundamental": 0.4,
                "sentiment": 0.2,
            },
            "analysis_params": {
                "technical_period_days": 180,
                "max_news_count": 100,
                "financial_indicators_count": 25,
            },
            "cache": {
                "price_hours": 1,
                "fundamental_hours": 6,
                "news_hours": 2,
            },
            "streaming": {
                "enabled": True,
                "show_thinking": False,
                "delay": 0.05,
            },
            "markets": {
                "us_stock": {
                    "enabled": True,
                    "name": "U.S. equities",
                    "currency": "USD",
                    "timezone": "America/New_York",
                    "trading_hours": "09:30-16:00",
                },
            },
            "web_auth": {
                "enabled": False,
                "password": "",
                "session_timeout": 3600,
            },
        }

    def _log_config_summary(self) -> None:
        """Emit a short configuration summary to the console."""

        enabled_markets = [
            self._build_market_info(key, value)
            for key, value in self.market_config.items()
            if value.get("enabled", False)
        ]
        logger.info("Markets enabled: %s", ", ".join(m.name for m in enabled_markets) or "None")
        logger.info(
            "Streaming enabled: %s (delay %.2fs)",
            self.streaming_config.get("enabled", True),
            self.streaming_config.get("delay", 0.0),
        )

    def _build_market_info(self, code: str, value: Dict[str, str]) -> MarketInfo:
        """Safely instantiate :class:`MarketInfo` from configuration data."""

        market_fields = {field.name for field in fields(MarketInfo)}
        kwargs = {}

        for field_name in market_fields:
            if field_name == "code":
                kwargs[field_name] = code
            elif field_name in value:
                kwargs[field_name] = value[field_name]

        # Ensure we expose a readable market name even when the configuration
        # only toggles the ``enabled`` flag. Falling back to the code prevents
        # empty labels such as "Markets enabled: None" during startup logs.
        kwargs.setdefault("name", code)

        market = MarketInfo(**kwargs)

        # ``enabled`` was introduced after the initial release. If the dataclass
        # in a user's working copy predates that change, ensure the attribute is
        # still exposed so downstream code can rely on it.
        if "enabled" not in market_fields:
            setattr(market, "enabled", value.get("enabled", True))

        return market

    def get_ui_context(self) -> Dict[str, Any]:
        """Expose configuration highlights for the web dashboard."""

        streaming_enabled = bool(self.streaming_config.get("enabled", True))
        streaming_delay = float(self.streaming_config.get("delay", 0.0) or 0.0)
        show_thinking = bool(self.streaming_config.get("show_thinking", False))

        ai_config = self.config.get("ai", {})
        api_keys = {
            key: value
            for key, value in self.api_keys.items()
            if key not in {"notes"}
        }
        configured_keys = [key for key, value in api_keys.items() if value]
        model_preference = ai_config.get("model_preference", "openai")
        models = ai_config.get("models", {})
        preferred_model = models.get(model_preference, model_preference)

        enabled_markets: List[Dict[str, str]] = []
        for code, value in self.market_config.items():
            if not value.get("enabled", False):
                continue
            enabled_markets.append(
                {
                    "code": code,
                    "name": value.get("name") or code,
                    "currency": value.get("currency", ""),
                    "timezone": value.get("timezone", ""),
                    "trading_hours": value.get("trading_hours", ""),
                }
            )

        if enabled_markets:
            primary = enabled_markets[0]
            market_summary = " · ".join(
                filter(
                    None,
                    [
                        primary.get("name"),
                        primary.get("currency"),
                        primary.get("timezone"),
                    ],
                )
            )
        else:
            market_summary = "No markets enabled"

        weight_parts: List[str] = []
        for label, weight in self.analysis_weights.items():
            if isinstance(weight, (int, float)):
                weight_parts.append(f"{label.title()} {weight * 100:.0f}%")
        weight_summary = " · ".join(weight_parts)

        cache_parts: List[str] = []
        if isinstance(self.cache_config.get("price_hours"), (int, float)):
            cache_parts.append(f"Prices {self.cache_config['price_hours']}h")
        if isinstance(self.cache_config.get("fundamental_hours"), (int, float)):
            cache_parts.append(
                f"Fundamentals {self.cache_config['fundamental_hours']}h"
            )
        if isinstance(self.cache_config.get("news_hours"), (int, float)):
            cache_parts.append(f"News {self.cache_config['news_hours']}h")
        cache_summary = " · ".join(cache_parts)

        technical_days = self.analysis_params.get("technical_period_days")
        news_limit = self.analysis_params.get("max_news_count")

        return {
            "streaming": {
                "enabled": streaming_enabled,
                "delay": streaming_delay,
                "show_thinking": show_thinking,
            },
            "ai": {
                "preference": model_preference,
                "model": preferred_model,
                "has_keys": bool(configured_keys),
                "configured_keys": configured_keys,
            },
            "markets": enabled_markets,
            "market_summary": market_summary,
            "weights": weight_summary,
            "cache": cache_summary,
            "analysis": {
                "technical_days": technical_days,
                "news_limit": news_limit,
            },
        }

    def get_editable_config(self) -> Dict[str, Any]:
        """Return a sanitised snapshot of configurable settings for the UI."""

        ai_config = self.config.get("ai", {})
        models = ai_config.get("models", {})

        markets: Dict[str, Dict[str, Any]] = {}
        for code, value in self.market_config.items():
            if not isinstance(value, dict):
                continue
            markets[code] = {
                "enabled": bool(value.get("enabled", False)),
                "name": value.get("name", code),
                "currency": value.get("currency", ""),
                "timezone": value.get("timezone", ""),
                "trading_hours": value.get("trading_hours", ""),
            }

        return {
            "ai": {
                "model_preference": ai_config.get("model_preference", "openai"),
                "models": {
                    "openai": models.get("openai", ""),
                    "anthropic": models.get("anthropic", ""),
                    "zhipu": models.get("zhipu", ""),
                },
            },
            "analysis_weights": {
                key: float(value)
                for key, value in self.analysis_weights.items()
                if isinstance(value, (int, float))
            },
            "analysis_params": {
                "technical_period_days": self.analysis_params.get("technical_period_days"),
                "max_news_count": self.analysis_params.get("max_news_count"),
                "financial_indicators_count": self.analysis_params.get("financial_indicators_count"),
            },
            "markets": markets,
        }

    def update_runtime_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user-supplied updates into the configuration and persist them."""

        if not isinstance(updates, dict):
            raise ValueError("Configuration payload must be a JSON object")

        changed = False

        ai_updates = updates.get("ai")
        if isinstance(ai_updates, dict):
            ai_config = self.config.setdefault("ai", {})
            model_preference = ai_updates.get("model_preference")
            if isinstance(model_preference, str) and model_preference.strip():
                ai_config["model_preference"] = model_preference.strip()
                changed = True
            models_updates = ai_updates.get("models")
            if isinstance(models_updates, dict):
                models_config = ai_config.setdefault("models", {})
                for provider_key, model_value in models_updates.items():
                    if isinstance(model_value, str):
                        models_config[provider_key] = model_value.strip()
                        changed = True

        weight_updates = updates.get("analysis_weights")
        if isinstance(weight_updates, dict):
            combined = dict(self.analysis_weights)
            provided = False
            for key in ("technical", "fundamental", "sentiment"):
                if key not in weight_updates:
                    continue
                numeric = self._safe_numeric(weight_updates.get(key))
                if numeric is None:
                    continue
                if numeric > 1.0:
                    numeric = numeric / 100.0
                combined[key] = max(0.0, numeric)
                provided = True
            if provided:
                total = sum(combined.values())
                if total > 0:
                    normalised = {key: value / total for key, value in combined.items()}
                else:
                    normalised = combined
                self.config["analysis_weights"] = normalised
                changed = True

        params_updates = updates.get("analysis_params")
        if isinstance(params_updates, dict):
            params_config = self.config.setdefault("analysis_params", {})
            for key in ("technical_period_days", "max_news_count", "financial_indicators_count"):
                if key not in params_updates:
                    continue
                numeric = self._safe_numeric(params_updates.get(key))
                if numeric is None:
                    continue
                params_config[key] = int(max(1, round(numeric)))
                changed = True

        markets_updates = updates.get("markets")
        if isinstance(markets_updates, dict):
            markets_config = self.config.setdefault("markets", {})
            for code, payload in markets_updates.items():
                if code != "us_stock":
                    continue
                if not isinstance(payload, dict):
                    continue
                target = markets_config.setdefault(code, {"name": code, "enabled": True})
                if "enabled" in payload:
                    target["enabled"] = bool(payload.get("enabled"))
                for field in ("name", "currency", "timezone", "trading_hours"):
                    value = payload.get(field)
                    if isinstance(value, str):
                        target[field] = value
                changed = True

        if changed:
            self.analysis_weights = self.config.get("analysis_weights", self.analysis_weights)
            self.analysis_params = self.config.get("analysis_params", self.analysis_params)
            self.market_config = self.config.get("markets", self.market_config)
            self._save_config(self.config)

        return self.get_editable_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_sentiment_analyzer(self) -> Optional[SentimentIntensityAnalyzer]:
        """Return a shared VADER sentiment analyzer instance if available."""

        if self._sentiment_analyzer is None and SentimentIntensityAnalyzer is not None:
            try:  # pragma: no cover - depends on optional dependency
                self._sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as exc:
                logger.warning("Unable to initialise VADER sentiment analyzer: %s", exc)
                self._sentiment_analyzer = None
        return self._sentiment_analyzer

    @staticmethod
    def _estimate_keyword_sentiment(text: str) -> float:
        """Fallback keyword-based sentiment score in case VADER is unavailable."""

        lowered = text.lower()
        positive = sum(lowered.count(token) for token in ["beat", "surge", "record", "strong", "growth"])
        negative = sum(lowered.count(token) for token in ["miss", "slump", "weak", "lawsuit", "cut"])
        if positive == negative:
            return 0.0
        total = positive + negative
        score = (positive - negative) / max(total, 1)
        return float(max(-1.0, min(1.0, score)))

    @staticmethod
    def _safe_numeric(value: Any) -> Optional[float]:
        """Convert values to finite floats, returning ``None`` when invalid."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric

    def _http_get(self, url: str, *, params: Optional[Dict[str, Any]] = None, timeout: float = 6.0) -> Response:
        """Wrapper around ``requests.get`` with shared headers and short timeouts."""

        return requests.get(url, params=params, headers=self._http_headers, timeout=timeout)

    def _collect_yf_statements(self, ticker_instance: Any) -> Dict[str, Optional[pd.DataFrame]]:
        """Download yfinance statements with defensive fallbacks."""

        statements = {"income": None, "balance": None, "cashflow": None}

        def _maybe_call(attr_name: str) -> Optional[pd.DataFrame]:
            target = getattr(ticker_instance, attr_name, None)
            if callable(target):
                try:  # pragma: no cover - network dependent
                    return target()
                except Exception:
                    return None
            return target

        statements["income"] = _maybe_call("get_income_stmt") or _maybe_call("income_stmt")
        statements["balance"] = _maybe_call("get_balance_sheet") or _maybe_call("balance_sheet")
        statements["cashflow"] = _maybe_call("get_cashflow") or _maybe_call("cashflow")
        if isinstance(statements["income"], dict):
            statements["income"] = pd.DataFrame(statements["income"])
        if isinstance(statements["balance"], dict):
            statements["balance"] = pd.DataFrame(statements["balance"])
        if isinstance(statements["cashflow"], dict):
            statements["cashflow"] = pd.DataFrame(statements["cashflow"])
        return statements

    def _fetch_yahoo_quote_summary(self, stock_code: str) -> Tuple[Dict[str, float], List[str]]:
        """Lightweight fallback that scrapes Yahoo Finance quoteSummary endpoints."""

        params = {
            "modules": "financialData,defaultKeyStatistics,summaryDetail,price",
        }
        warnings: List[str] = []
        try:  # pragma: no cover - network dependent
            response = self._http_get(
                f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{stock_code}",
                params=params,
            )
        except RequestException as exc:
            warnings.append(f"Yahoo quoteSummary request failed: {exc}")
            return {}, warnings

        if response.status_code == 429:
            warnings.append("Yahoo quoteSummary rate limit exceeded")
            return {}, warnings

        try:
            response.raise_for_status()
        except RequestException as exc:
            warnings.append(f"Yahoo quoteSummary error: {exc}")
            return {}, warnings

        payload = response.json()
        result = payload.get("quoteSummary", {}).get("result")
        if not result:
            warnings.append("Yahoo quoteSummary returned no data")
            return {}, warnings

        node = result[0]
        fundamentals: Dict[str, float] = {}

        def _pull(section: str, key: str) -> Optional[float]:
            section_data = node.get(section) or {}
            value = section_data.get(key) if isinstance(section_data, dict) else {}
            if isinstance(value, dict) and "raw" in value:
                value = value["raw"]
            return self._safe_numeric(value)

        price_value = _pull("price", "regularMarketPrice") or _pull("financialData", "currentPrice")
        if price_value is not None:
            fundamentals["last_price"] = price_value
        eps = _pull("defaultKeyStatistics", "trailingEps")
        if price_value and eps and eps != 0:
            fundamentals["pe_ratio"] = price_value / eps
            fundamentals.setdefault("eps", eps)
        peg = _pull("defaultKeyStatistics", "pegRatio")
        if peg is not None:
            fundamentals["peg_ratio"] = peg
        roe = _pull("financialData", "returnOnEquity")
        if roe is not None:
            fundamentals["roe"] = roe * 100.0 if abs(roe) <= 10 else roe
        net_margin = _pull("financialData", "profitMargins")
        if net_margin is not None:
            fundamentals["net_margin"] = net_margin * 100.0 if abs(net_margin) <= 10 else net_margin
        operating_margin = _pull("financialData", "operatingMargins")
        if operating_margin is not None:
            fundamentals["operating_margin"] = (
                operating_margin * 100.0 if abs(operating_margin) <= 10 else operating_margin
            )
        revenue_growth = _pull("financialData", "revenueGrowth")
        if revenue_growth is not None:
            fundamentals["revenue_growth"] = revenue_growth * 100.0 if abs(revenue_growth) <= 10 else revenue_growth
        debt_to_equity = _pull("financialData", "debtToEquity")
        if debt_to_equity is not None:
            fundamentals["debt_to_equity"] = debt_to_equity
        current_ratio = _pull("financialData", "currentRatio")
        if current_ratio is not None:
            fundamentals["current_ratio"] = current_ratio

        return fundamentals, warnings

    def _fetch_yahoo_quote_snapshot(self, stock_code: str) -> Tuple[Dict[str, float], List[str]]:
        """Request Yahoo's quote endpoint for quick metrics as a final fallback."""

        warnings: List[str] = []
        params = {"symbols": stock_code}
        try:  # pragma: no cover - network dependent
            response = self._http_get(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                params=params,
            )
        except RequestException as exc:
            warnings.append(f"Yahoo quote request failed: {exc}")
            return {}, warnings

        if response.status_code == 429:
            warnings.append("Yahoo quote rate limit exceeded")
            return {}, warnings

        try:
            response.raise_for_status()
        except RequestException as exc:
            warnings.append(f"Yahoo quote error: {exc}")
            return {}, warnings

        payload = response.json()
        results = payload.get("quoteResponse", {}).get("result", [])
        if not results:
            warnings.append("Yahoo quote returned no data")
            return {}, warnings

        node = results[0]
        fundamentals: Dict[str, float] = {}

        def _pull(key: str) -> Optional[float]:
            return self._safe_numeric(node.get(key))

        price_value = _pull("regularMarketPrice") or _pull("postMarketPrice")
        if price_value is not None:
            fundamentals["last_price"] = price_value

        eps = _pull("epsTrailingTwelveMonths")
        if eps is not None:
            fundamentals.setdefault("eps", eps)

        pe = _pull("trailingPE")
        if pe is not None:
            fundamentals.setdefault("pe_ratio", pe)

        forward_pe = _pull("forwardPE")
        if forward_pe is not None:
            fundamentals.setdefault("forward_pe", forward_pe)

        roe = self._safe_numeric(node.get("returnOnEquity"))
        if roe is not None:
            fundamentals.setdefault("roe", roe * 100.0 if abs(roe) <= 10 else roe)

        revenue_growth = self._safe_numeric(node.get("revenueGrowth"))
        if revenue_growth is not None:
            fundamentals.setdefault(
                "revenue_growth",
                revenue_growth * 100.0 if abs(revenue_growth) <= 10 else revenue_growth,
            )

        profit_margin = self._safe_numeric(node.get("profitMargins"))
        if profit_margin is not None:
            fundamentals.setdefault(
                "net_margin",
                profit_margin * 100.0 if abs(profit_margin) <= 10 else profit_margin,
            )

        operating_margin = self._safe_numeric(node.get("operatingMargins"))
        if operating_margin is not None:
            fundamentals.setdefault(
                "operating_margin",
                operating_margin * 100.0 if abs(operating_margin) <= 10 else operating_margin,
            )

        debt_to_equity = self._safe_numeric(node.get("debtToEquity"))
        if debt_to_equity is not None:
            fundamentals.setdefault("debt_to_equity", debt_to_equity)

        current_ratio = self._safe_numeric(node.get("currentRatio"))
        if current_ratio is not None:
            fundamentals.setdefault("current_ratio", current_ratio)

        return fundamentals, warnings

    def _extract_statement_value(
        self, statement: Optional[pd.DataFrame], candidates: Iterable[str]
    ) -> Optional[float]:
        """Attempt to pull a numeric field from a financial statement."""

        if statement is None or not isinstance(statement, pd.DataFrame) or statement.empty:
            return None

        normalized_index = {str(index).strip().lower(): index for index in statement.index}
        for candidate in candidates:
            lookup = candidate.strip().lower()
            if lookup in normalized_index:
                series = statement.loc[normalized_index[lookup]]
                if isinstance(series, pd.Series) and not series.empty:
                    value = series.iloc[0]
                    numeric = self._safe_numeric(value)
                    if numeric is not None:
                        return numeric
        return None

    @staticmethod
    def _compute_growth(current: Optional[float], previous: Optional[float]) -> Optional[float]:
        """Return percentage growth given two comparable values."""

        if current is None or previous in (None, 0):
            return None
        return ((current - previous) / abs(previous)) * 100.0

    # ------------------------------------------------------------------
    # Market helpers
    # ------------------------------------------------------------------
    def detect_market(self, stock_code: str) -> str:
        """Return the market for a stock code. Defaults to ``us_stock``."""

        return "us_stock"

    def normalize_stock_code(self, stock_code: str) -> Tuple[str, str]:
        """Normalise a stock code and return ``(code, market)``."""

        cleaned = stock_code.strip().upper()
        if not cleaned:
            raise ValueError("Stock symbol cannot be empty")

        if not all(ch.isalnum() or ch in {".", "-"} for ch in cleaned):
            raise ValueError("Stock symbols may only contain letters, numbers, '.' or '-'")

        market = self.detect_market(cleaned)
        market_info = self.market_config.get(market, {})
        if not market_info.get("enabled", False):
            raise ValueError(f"Market {market} is disabled in the configuration")

        return cleaned, market

    def validate_stock_code(self, stock_code: str) -> Tuple[bool, str]:
        """Check whether a symbol looks like a U.S. ticker."""

        try:
            normalized, market = self.normalize_stock_code(stock_code)
        except ValueError as exc:
            return False, str(exc)

        if market != "us_stock":
            return False, "Only U.S. symbols are currently supported"

        if not (1 <= len(normalized) <= 7):
            return False, "U.S. tickers must contain between 1 and 7 characters"

        return True, "Valid U.S. ticker"

    def get_supported_markets(self) -> List[Dict[str, str]]:
        """Return metadata for all enabled markets."""

        markets = []
        for key, value in self.market_config.items():
            if value.get("enabled", False):
                markets.append({"code": key, **value})
        return markets

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def _fetch_company_profile(self, stock_code: str) -> Dict[str, Any]:
        """Retrieve and cache basic company metadata such as the display name."""

        cache_entry = self._profile_cache.get(stock_code)
        if cache_entry:
            return cache_entry

        profile: Dict[str, Any] = {"symbol": stock_code}

        if yf is not None:
            try:  # pragma: no cover - network dependent
                ticker = yf.Ticker(stock_code)
                info = {}
                try:
                    info = ticker.get_info() or {}
                except Exception as exc:
                    logger.debug("yfinance get_info unavailable for %s: %s", stock_code, exc)
                fast_info = getattr(ticker, "fast_info", None)
                fast_dict = {}
                if isinstance(fast_info, dict):
                    fast_dict = fast_info
                elif fast_info is not None:
                    fast_dict = getattr(fast_info, "__dict__", {})

                for source in (info, fast_dict):
                    name = source.get("shortName") or source.get("longName")
                    if name:
                        profile["name"] = name
                        break
                if "exchange" in info:
                    profile["exchange"] = info.get("exchange")
            except Exception as exc:
                logger.debug("Unable to fetch profile for %s via yfinance: %s", stock_code, exc)

        if ak is not None and "name" not in profile:
            try:  # pragma: no cover - network dependent
                listing = ak.stock_us_spot()
                if isinstance(listing, pd.DataFrame) and not listing.empty:
                    match = listing[listing["代码"].str.upper() == stock_code.upper()]
                    if not match.empty:
                        profile["name"] = match.iloc[0]["名称"]
            except Exception as exc:
                logger.debug("akshare profile lookup failed for %s: %s", stock_code, exc)

        self._profile_cache[stock_code] = profile
        return profile

    def get_stock_name(self, stock_code: str) -> str:
        """Return a human readable name for a ticker."""

        profile = self._fetch_company_profile(stock_code)
        return profile.get("name") or stock_code

    def get_stock_data(self, stock_code: str, days: Optional[int] = None) -> pd.DataFrame:
        """Fetch daily historical prices for ``stock_code``."""

        cache_key = stock_code
        cache_entry = self._price_cache.get(cache_key)
        expiry_hours = self.cache_config.get("price_hours", 1)
        if cache_entry and datetime.now() - cache_entry[0] < timedelta(hours=expiry_hours):
            return cache_entry[1].copy()

        period_days = days or self.analysis_params.get("technical_period_days", 180)
        end = datetime.utcnow()
        start = end - timedelta(days=period_days + 10)

        data: Optional[pd.DataFrame] = None
        source = ""
        rate_limited = False
        stooq_attempted = False

        if yf is not None:
            try:  # pragma: no cover - network dependent
                raw = yf.download(
                    tickers=stock_code,
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                )
                if not raw.empty:
                    raw = raw.rename(
                        columns={
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Adj Close": "adj_close",
                            "Volume": "volume",
                        }
                    )
                    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(None)
                    raw = raw.reset_index().rename(columns={"index": "date", "Date": "date"})
                    data = raw[["date", "open", "high", "low", "close", "volume"]].dropna()
                    source = "yfinance"
            except Exception as exc:  # pragma: no cover - network dependent
                message = str(exc)
                rate_limited = "rate" in message.lower() and "limit" in message.lower()
                if exc.__class__.__name__ == "YFRateLimitError":
                    rate_limited = True
                level = logger.warning
                if rate_limited:
                    level = logger.error
                level("yfinance price download failed for %s: %s", stock_code, exc)

        if (data is None or data.empty):
            stooq_attempted = True
            stooq_symbol = stock_code.lower()
            if "." not in stooq_symbol:
                stooq_symbol = f"{stooq_symbol}.us"
            stooq_url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
            try:  # pragma: no cover - network dependent
                raw = pd.read_csv(stooq_url)
                if not raw.empty and {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
                    raw = raw.rename(
                        columns={
                            "Date": "date",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )
                    raw["date"] = pd.to_datetime(raw["date"])
                    raw = raw.sort_values("date")
                    data = raw[["date", "open", "high", "low", "close", "volume"]].dropna()
                    source = "stooq"
            except Exception as exc:
                logger.warning("Failed to download price history for %s via stooq: %s", stock_code, exc)

        if (data is None or data.empty) and ak is not None:
            try:  # pragma: no cover - network dependent
                raw = ak.stock_us_hist(symbol=stock_code, period="daily", adjust="qfq")
                if raw is not None and not raw.empty:
                    raw = raw.rename(columns=AK_PRICE_COLUMNS)
                    raw["date"] = pd.to_datetime(raw["date"])
                    raw = raw.sort_values("date")
                    data = raw[["date", "open", "high", "low", "close", "volume"]].dropna()
                    source = "akshare"
            except Exception as exc:
                logger.error("Failed to pull price history for %s via akshare: %s", stock_code, exc)

        if data is None or data.empty:
            hint = []
            if rate_limited:
                hint.append("Yahoo Finance is rate limiting your IP. Wait a few minutes or reduce request frequency.")
            if ak is None:
                hint.append("Install akshare for an additional data source (pip install akshare).")
            if stooq_attempted and source != "stooq":
                hint.append("Stooq returned no data. Confirm the ticker trades in the U.S. market or try again later.")
            if not hint:
                hint.append("Check your internet connection or verify the ticker symbol is valid.")
            raise RuntimeError(
                "Unable to download price history. " + " ".join(hint)
            )

        data = data[data["date"] >= start]

        logger.info("Loaded %s price rows for %s from %s", len(data), stock_code, source or "unknown")

        self._price_cache[cache_key] = (datetime.now(), data.copy())
        return data

    def get_price_info(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Return summary pricing information used by the UI."""

        latest = price_data.iloc[-1]
        previous = price_data.iloc[-2] if len(price_data) > 1 else latest
        change = ((latest["close"] - previous["close"]) / previous["close"]) * 100 if previous["close"] else 0.0
        return {
            "current_price": float(latest["close"]),
            "price_change": float(change),
        }

    # ------------------------------------------------------------------
    # Technical analysis
    # ------------------------------------------------------------------
    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Compute basic technical indicators."""

        closing = price_data["close"].astype(float)
        volume = price_data["volume"].astype(float)

        ma20 = closing.rolling(window=20).mean().iloc[-1]
        ma50 = closing.rolling(window=50).mean().iloc[-1]
        ma200 = closing.rolling(window=200, min_periods=50).mean().iloc[-1]

        delta = closing.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(gain).rolling(window=14, min_periods=14).mean().iloc[-1]
        roll_down = pd.Series(loss).rolling(window=14, min_periods=14).mean().iloc[-1]
        rs = roll_up / roll_down if roll_down else np.inf
        rsi = 100 - (100 / (1 + rs)) if math.isfinite(rs) else 100.0

        avg_volume = volume.tail(20).mean()
        latest_volume = volume.iloc[-1]

        return {
            "ma20": float(ma20) if not math.isnan(ma20) else None,
            "ma50": float(ma50) if not math.isnan(ma50) else None,
            "ma200": float(ma200) if not math.isnan(ma200) else None,
            "rsi": float(rsi),
            "volume_trend": float(latest_volume / avg_volume) if avg_volume else 1.0,
        }

    def calculate_technical_score(self, indicators: Dict[str, Optional[float]]) -> float:
        """Combine technical indicators into a 0-100 score."""

        score = 50.0

        ma_values = [indicators.get("ma20"), indicators.get("ma50"), indicators.get("ma200")]
        if all(value for value in ma_values):
            if ma_values[0] >= ma_values[1] >= ma_values[2]:
                score += 20
            elif ma_values[0] < ma_values[1] < ma_values[2]:
                score -= 20

        rsi = indicators.get("rsi", 50)
        if 40 <= rsi <= 60:
            score += 5
        elif rsi > 70 or rsi < 30:
            score -= 10

        volume_trend = indicators.get("volume_trend", 1.0)
        if volume_trend > 1.2:
            score += 5
        elif volume_trend < 0.8:
            score -= 5

        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------
    def _fetch_yfinance_fundamentals(
        self, stock_code: str
    ) -> Tuple[Dict[str, float], str, List[str]]:
        """Return fundamental metrics sourced from yfinance where possible."""

        fundamentals: Dict[str, float] = {}
        provider_warnings: List[str] = []
        source_name = ""

        if yf is None:
            provider_warnings.append("yfinance package is not installed.")
            return fundamentals, source_name, provider_warnings

        try:  # pragma: no cover - network dependent
            ticker_instance = yf.Ticker(stock_code)
        except Exception as exc:
            provider_warnings.append(f"yfinance ticker initialisation failed: {exc}")
            logger.warning("Unable to initialise yfinance ticker for %s: %s", stock_code, exc)
            return fundamentals, source_name, provider_warnings

        info_sources: List[Tuple[Dict[str, Any], str]] = []
        statements: Dict[str, Optional[pd.DataFrame]] = {"income": None, "balance": None, "cashflow": None}
        earnings_dates: Optional[pd.DataFrame] = None

        summary_data, summary_warnings = self._fetch_yahoo_quote_summary(stock_code)
        if summary_data:
            fundamentals.update(summary_data)
            source_name = source_name or "yahoo-quote"
        provider_warnings.extend(summary_warnings)

        key_fields = {"pe_ratio", "eps", "roe", "revenue_growth", "net_margin", "operating_margin"}
        has_core_metrics = any(field in fundamentals for field in key_fields)

        if has_core_metrics:
            last_price = fundamentals.get("last_price")
        else:
            last_price = None

        if has_core_metrics and len(fundamentals) >= 5:
            return fundamentals, source_name or "yahoo-quote", provider_warnings

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                "info": pool.submit(lambda: ticker_instance.get_info() or {}),
                "fast": pool.submit(lambda: getattr(ticker_instance, "fast_info", {}) or {}),
                "statements": pool.submit(self._collect_yf_statements, ticker_instance),
                "earnings": pool.submit(
                    lambda: (
                        getattr(ticker_instance, "get_earnings_dates", lambda limit=8: None)(limit=8)
                        if hasattr(ticker_instance, "get_earnings_dates")
                        else None
                    )
                ),
            }

            for label, future in futures.items():
                try:  # pragma: no cover - network dependent
                    result = future.result(timeout=8)
                except Exception as exc:
                    provider_warnings.append(f"yfinance {label} retrieval failed: {exc}")
                    continue

                if label == "info" and isinstance(result, dict):
                    info_sources.append((result, "get_info"))
                elif label == "fast" and result:
                    if isinstance(result, dict):
                        info_sources.append((result, "fast_info"))
                    else:
                        info_sources.append((getattr(result, "__dict__", {}), "fast_info"))
                elif label == "statements" and isinstance(result, dict):
                    statements.update(result)
                elif label == "earnings" and isinstance(result, (pd.DataFrame, dict)):
                    earnings_dates = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

        mapping = {
            "trailingPE": "pe_ratio",
            "forwardPE": "forward_pe",
            "trailing_pe": "pe_ratio",
            "forward_pe": "forward_pe",
            "trailingEps": "eps",
            "forwardEps": "forward_eps",
            "returnOnEquity": "roe",
            "revenueGrowth": "revenue_growth",
            "earningsGrowth": "net_profit_growth",
            "earningsQuarterlyGrowth": "net_profit_growth_quarterly",
            "profitMargins": "net_margin",
            "grossMargins": "gross_margin",
            "operatingMargins": "operating_margin",
            "debtToEquity": "debt_to_equity",
            "currentRatio": "current_ratio",
        }

        percentage_keys = {
            "returnOnEquity",
            "revenueGrowth",
            "earningsGrowth",
            "earningsQuarterlyGrowth",
            "profitMargins",
            "grossMargins",
            "operatingMargins",
        }

        last_price = None
        for source_dict, source_label in info_sources:
            if not isinstance(source_dict, dict):
                continue
            if last_price is None:
                for price_key in ("lastPrice", "last_price", "regularMarketPrice", "regular_market_price"):
                    price_value = self._safe_numeric(source_dict.get(price_key))
                    if price_value is not None:
                        last_price = price_value
                        break
            for source_key, target_key in mapping.items():
                if target_key in fundamentals:
                    continue
                numeric_value = self._safe_numeric(source_dict.get(source_key))
                if numeric_value is None:
                    continue
                if source_key in percentage_keys and abs(numeric_value) <= 10:
                    numeric_value *= 100.0
                fundamentals[target_key] = numeric_value
                source_name = "yfinance"

        if last_price is None:
            price_candidates = [
                source_dict.get("regularMarketPrice")
                for source_dict, _ in info_sources
                if isinstance(source_dict, dict)
            ]
            for candidate in price_candidates:
                numeric = self._safe_numeric(candidate)
                if numeric is not None:
                    last_price = numeric
                    break

        income_stmt = statements.get("income")
        balance_sheet = statements.get("balance")
        cashflow_stmt = statements.get("cashflow")

        revenue_current = self._extract_statement_value(
            income_stmt, ["Total Revenue", "totalRevenue", "TotalRevenue"]
        )
        revenue_prev = None
        if isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
            normalized_index = {str(idx).strip().lower(): idx for idx in income_stmt.index}
            for candidate in ("total revenue", "totalRevenue"):
                lookup = candidate.lower()
                if lookup in normalized_index:
                    row = income_stmt.loc[normalized_index[lookup]]
                    if isinstance(row, pd.Series) and row.size > 1:
                        revenue_prev = self._safe_numeric(row.iloc[1])
                    break

        net_income_current = self._extract_statement_value(
            income_stmt, ["Net Income", "netIncome", "NetIncome"]
        )
        equity = self._extract_statement_value(
            balance_sheet,
            [
                "Total Stockholder Equity",
                "totalStockholderEquity",
                "Total Equity Gross Minority Interest",
            ],
        )
        total_assets = self._extract_statement_value(
            balance_sheet, ["Total Assets", "totalAssets"]
        )
        current_assets = self._extract_statement_value(
            balance_sheet,
            ["Total Current Assets", "totalCurrentAssets", "Current Assets"],
        )
        current_liabilities = self._extract_statement_value(
            balance_sheet,
            ["Total Current Liabilities", "totalCurrentLiabilities", "Current Liabilities"],
        )
        free_cash_flow = self._extract_statement_value(
            cashflow_stmt,
            ["Free Cash Flow", "freeCashFlow", "Free Cash Flow Net Income"],
        )

        if revenue_current is not None:
            fundamentals.setdefault("revenue", revenue_current)
        if net_income_current is not None:
            fundamentals.setdefault("net_income", net_income_current)
        if equity not in (None, 0) and net_income_current is not None:
            fundamentals.setdefault("roe", (net_income_current / equity) * 100.0)
        if total_assets not in (None, 0) and net_income_current is not None:
            fundamentals.setdefault("roa", (net_income_current / total_assets) * 100.0)
        if free_cash_flow is not None:
            fundamentals.setdefault("free_cash_flow", free_cash_flow)
        if revenue_current is not None and revenue_prev is not None:
            growth = self._compute_growth(revenue_current, revenue_prev)
            if growth is not None:
                fundamentals.setdefault("revenue_growth", growth)
        if current_assets is not None and current_liabilities not in (None, 0):
            fundamentals.setdefault("current_ratio", current_assets / current_liabilities)
        if equity not in (None, 0) and total_assets is not None:
            fundamentals.setdefault("debt_to_equity", (total_assets - equity) / equity)

        if earnings_dates is not None and not getattr(earnings_dates, "empty", False):
            try:
                if isinstance(earnings_dates, pd.DataFrame):
                    latest_eps = earnings_dates.get("epsActual")
                    if isinstance(latest_eps, pd.Series) and not latest_eps.empty:
                        eps_actual = self._safe_numeric(latest_eps.iloc[0])
                        if eps_actual is not None:
                            fundamentals.setdefault("eps", eps_actual)
            except Exception:
                provider_warnings.append("Unable to parse yfinance earnings history")

        if "pe_ratio" not in fundamentals and last_price and fundamentals.get("eps") not in (None, 0):
            fundamentals["pe_ratio"] = last_price / fundamentals["eps"]

        if not fundamentals:
            quote_summary, summary_warnings = self._fetch_yahoo_quote_summary(stock_code)
            fundamentals.update(quote_summary)
            provider_warnings.extend(summary_warnings)
            if fundamentals:
                source_name = "yahoo-quote"

        if not fundamentals:
            quote_snapshot, snapshot_warnings = self._fetch_yahoo_quote_snapshot(stock_code)
            fundamentals.update(quote_snapshot)
            provider_warnings.extend(snapshot_warnings)
            if quote_snapshot:
                source_name = "yahoo-quote"

        if fundamentals and not source_name:
            source_name = "yfinance"

        return fundamentals, source_name, provider_warnings

    def get_comprehensive_fundamental_data(self, stock_code: str) -> Dict:
        """Fetch or synthesise fundamental metrics."""

        cache_entry = self._fundamental_cache.get(stock_code)
        expiry_hours = self.cache_config.get("fundamental_hours", 6)
        if cache_entry and datetime.now() - cache_entry[0] < timedelta(hours=expiry_hours):
            return cache_entry[1]

        fundamentals: Dict[str, float] = {}
        source_name = ""
        provider_warnings: List[str] = []
        providers_attempted: List[str] = []

        if yf is not None:
            providers_attempted.append("yfinance")
            fetched, source_name, yf_warnings = self._fetch_yfinance_fundamentals(stock_code)
            fundamentals.update(fetched)
            provider_warnings.extend(yf_warnings)
            if source_name == "yahoo-quote" and "yahoo-quote" not in providers_attempted:
                providers_attempted.append("yahoo-quote")
        
        if not fundamentals and ak is not None and hasattr(ak, "stock_us_fundamental"):
            providers_attempted.append("akshare")
            try:  # pragma: no cover - network dependent
                info = ak.stock_us_fundamental(stock=stock_code)
                if info is not None and not info.empty:
                    info = info.set_index("item")
                    for source_key, target_key in AK_FUNDAMENTAL_KEYS.items():
                        if source_key in info.index:
                            try:
                                fundamentals[target_key] = float(info.loc[source_key, "value"])
                            except (TypeError, ValueError):
                                provider_warnings.append(
                                    f"Unable to parse {source_key} from akshare response."
                                )
                    if fundamentals and not source_name:
                        source_name = "akshare"
                else:
                    provider_warnings.append("akshare returned no rows for this ticker.")
            except Exception as exc:
                provider_warnings.append(f"akshare fundamental fetch failed: {exc}")
                logger.warning("Unable to fetch fundamentals for %s via akshare: %s", stock_code, exc)
        elif not fundamentals and ak is not None and not hasattr(ak, "stock_us_fundamental"):
            provider_warnings.append(
                "akshare installation does not provide stock_us_fundamental; skipping provider."
            )

        if fundamentals:
            logger.info(
                "Loaded %s fundamental metrics for %s using %s",
                len(fundamentals),
                stock_code,
                source_name or "unknown provider",
            )
        else:
            provider_warnings.append(
                "No fundamental metrics were retrieved; check data providers, ticker symbol, or rate limits."
            )
            logger.warning(
                "Fundamental fetch returned 0 metrics for %s (providers tried: %s)",
                stock_code,
                ", ".join(providers_attempted or ["none"]),
            )

        resolved_source = source_name or (
            "yahoo-quote"
            if "yahoo-quote" in providers_attempted
            else ("yfinance" if fundamentals else "unavailable")
        )

        data = {
            "financial_indicators": fundamentals,
            "metadata": {
                "source": resolved_source if fundamentals else "unavailable",
                "retrieved_at": datetime.now().isoformat(),
                "warnings": provider_warnings,
                "providers_attempted": providers_attempted,
            },
        }

        self._fundamental_cache[stock_code] = (datetime.now(), data)
        return data

    def calculate_fundamental_score(self, fundamental_data: Dict) -> float:
        """Convert fundamentals into a simple score."""

        indicators = fundamental_data.get("financial_indicators", {})
        if not indicators:
            return 50.0

        score = 50.0
        pe = indicators.get('pe_ratio')
        if pe and pe > 0:
            if pe < 20:
                score += 10
            elif pe > 40:
                score -= 10

        roe = indicators.get('roe')
        if roe:
            score += min(roe / 2, 15)

        growth = indicators.get('net_profit_growth')
        if growth:
            score += max(min(growth / 5, 10), -10)

        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------
    def get_comprehensive_news_data(self, stock_code: str, days: int = 30) -> Dict[str, List[dict]]:
        """Download and cache company news with Finnhub + NewsData fallbacks."""

        cache_entry = self._news_cache.get(stock_code)
        expiry_hours = self.cache_config.get("news_hours", 2)
        if cache_entry and datetime.now() - cache_entry[0] < timedelta(hours=expiry_hours):
            return cache_entry[1]

        lookback_days = max(1, int(days))
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        max_items = int(self.analysis_params.get("max_news_count", 100) or 100)

        aggregated = {
            "company_news": [],
            "announcements": [],
            "research_reports": [],
        }

        provider_payloads: Dict[str, Dict[str, List[dict]]] = {}
        provider_warnings: List[str] = []
        sources_used: List[str] = []
        providers_attempted: List[str] = []

        def _submit_news_job(pool, provider_name, func):
            providers_attempted.append(provider_name)
            return provider_name, pool.submit(func)

        futures: List[Tuple[str, Any]] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            if self.api_keys.get("finnhub"):
                futures.append(
                    _submit_news_job(
                        pool,
                        "Finnhub",
                        lambda: self._fetch_finnhub_company_news(
                            stock_code, start_date, end_date, max_items
                        ),
                    )
                )
            if self.api_keys.get("newsdata"):
                futures.append(
                    _submit_news_job(
                        pool,
                        "NewsData.io",
                        lambda: self._fetch_newsdata_company_news(
                            stock_code, start_date, end_date, max_items
                        ),
                    )
                )

            for provider_name, future in futures:
                try:  # pragma: no cover - network dependent
                    payload = future.result()
                    provider_payloads[provider_name] = payload
                except RateLimitError as exc:
                    provider_warnings.append(f"{provider_name} rate limit: {exc}")
                except NewsProviderError as exc:
                    provider_warnings.append(f"{provider_name} error: {exc}")
                except Exception as exc:  # pragma: no cover - defensive
                    provider_warnings.append(f"{provider_name} unexpected error: {exc}")

        primary_payload = provider_payloads.get("Finnhub")
        if primary_payload and sum(len(values) for values in primary_payload.values()):
            for key in aggregated:
                aggregated[key].extend(primary_payload.get(key, []))
            sources_used.append("Finnhub")
        elif "Finnhub" in providers_attempted and "Finnhub" not in provider_payloads:
            provider_warnings.append("Finnhub did not return any articles for this ticker.")

        secondary_payload = provider_payloads.get("NewsData.io")
        if secondary_payload:
            added = False
            for key in aggregated:
                before = len(aggregated[key])
                aggregated[key].extend(secondary_payload.get(key, []))
                if len(aggregated[key]) > before:
                    added = True
            if added:
                sources_used.append("NewsData.io")
        elif self.api_keys.get("newsdata") and "NewsData.io" not in providers_attempted:
            provider_warnings.append("NewsData.io API key configured but request was not attempted.")

        total_loaded = sum(len(items) for items in aggregated.values())
        if total_loaded < max_items // 2:
            yahoo_news = self._fetch_yfinance_news(stock_code, max_items)
            if yahoo_news:
                if "Yahoo Finance" not in providers_attempted:
                    providers_attempted.append("Yahoo Finance")
                for key in aggregated:
                    before_len = len(aggregated[key])
                    aggregated[key].extend(yahoo_news.get(key, []))
                    if len(aggregated[key]) > before_len and "Yahoo Finance" not in sources_used:
                        sources_used.append("Yahoo Finance")
            else:
                provider_warnings.append("Yahoo Finance news feed returned no articles for this ticker.")

        for bucket in list(aggregated.keys()):
            aggregated[bucket] = self._deduplicate_news_items(aggregated[bucket], max_items)

        metadata = {
            "sources": sources_used,
            "retrieved_at": datetime.utcnow().isoformat(),
            "warnings": provider_warnings,
            "attempted": providers_attempted,
        }

        if not any(len(items) for items in aggregated.values()):
            metadata.setdefault("warnings", []).append(
                "No news providers returned articles during this window."
            )

        payload = {**aggregated, "metadata": metadata}
        self._news_cache[stock_code] = (datetime.now(), payload)
        return payload

    def _deduplicate_news_items(self, items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Remove duplicate headlines while preserving recency order."""

        if not items:
            return []

        seen: set[str] = set()
        cleaned: List[Dict[str, Any]] = []

        def _normalise(value: Optional[str]) -> str:
            if not value:
                return ""
            return str(value).strip().lower()

        sorted_items = sorted(
            (item for item in items if isinstance(item, dict)),
            key=lambda item: item.get("published_at") or "",
            reverse=True,
        )

        for item in sorted_items:
            key = _normalise(item.get("url")) or _normalise(item.get("title"))
            if not key:
                key = f"item-{len(cleaned)}"
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(item)
            if limit and len(cleaned) >= limit:
                break

        return cleaned

    def _standardize_news_item(
        self,
        *,
        title: str,
        summary: str,
        source: str,
        url: str,
        published_at: Optional[datetime],
        tickers: Optional[List[str]] = None,
        provider: str = "",
    ) -> Dict[str, Any]:
        """Normalize disparate provider payloads into a common structure."""

        published_str = ""
        if published_at:
            if isinstance(published_at, datetime):
                published_str = published_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                published_str = str(published_at)

        return {
            "title": title or "",
            "summary": summary or "",
            "source": source or provider,
            "url": url or "",
            "published_at": published_str,
            "tickers": tickers or [],
            "provider": provider,
        }

    def _fetch_yfinance_news(self, stock_code: str, limit: int) -> Dict[str, List[dict]]:
        """Use yfinance's public news feed as a zero-config fallback."""

        if yf is None:
            return {}

        try:  # pragma: no cover - network dependent
            ticker = yf.Ticker(stock_code)
            news_items = getattr(ticker, "news", None) or []
        except Exception as exc:
            logger.debug("yfinance news fetch failed for %s: %s", stock_code, exc)
            return {}

        if not isinstance(news_items, list):
            return {}

        buckets = {
            "company_news": [],
            "announcements": [],
            "research_reports": [],
        }

        for raw in news_items:
            if not isinstance(raw, dict):
                continue
            title = raw.get("title") or ""
            summary = raw.get("summary") or raw.get("content") or ""
            url = raw.get("link") or raw.get("url") or ""
            provider = raw.get("publisher") or raw.get("source") or "Yahoo Finance"
            published = raw.get("providerPublishTime") or raw.get("pubDate")

            published_dt: Optional[datetime] = None
            try:
                timestamp = pd.to_datetime(published, utc=True, errors="coerce")
                if pd.notnull(timestamp):
                    published_dt = timestamp.to_pydatetime()
            except Exception:
                published_dt = None

            bucket = self._categorize_news_item(provider, raw.get("type"))
            if bucket not in buckets:
                bucket = "company_news"

            item = self._standardize_news_item(
                title=title,
                summary=summary,
                source=provider,
                url=url,
                published_at=published_dt,
                tickers=raw.get("relatedTickers") or [],
                provider="Yahoo Finance",
            )
            buckets[bucket].append(item)

        if limit:
            for key in list(buckets.keys()):
                buckets[key] = buckets[key][:limit]

        return buckets

    def _categorize_news_item(self, provider: str, category_hint: Optional[str]) -> str:
        """Map provider-specific category hints into UI buckets."""

        hint = (category_hint or "").lower()
        if not hint:
            return "company_news"

        if "press" in hint or "announcement" in hint or "earnings" in hint:
            return "announcements"
        if "research" in hint or "analysis" in hint:
            return "research_reports"

        if provider == "newsdata" and "business" not in hint:
            # NewsData categories are often lists joined by commas; treat non-business
            # labels as broader company news to avoid empty buckets.
            return "company_news"

        return "company_news"

    def _fetch_finnhub_company_news(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime,
        limit: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Download company news from Finnhub."""

        api_key = self.api_keys.get("finnhub")
        if not api_key:
            raise NewsProviderError("Finnhub API key is not configured")

        params = {
            "symbol": stock_code,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": api_key,
        }

        url = "https://finnhub.io/api/v1/company-news"
        response: Response = self._http_get(url, params=params, timeout=6)
        if response.status_code == 429:
            raise RateLimitError("Finnhub rate limit exceeded")
        if response.status_code >= 500:
            raise NewsProviderError(
                f"Finnhub server error ({response.status_code})"
            )
        try:
            response.raise_for_status()
        except RequestException as exc:
            raise NewsProviderError(f"Finnhub request failed: {exc}") from exc

        data = response.json()
        if not isinstance(data, list):
            raise NewsProviderError("Unexpected Finnhub response structure")

        buckets = {
            "company_news": [],
            "announcements": [],
            "research_reports": [],
        }

        for entry in data[: limit * 2]:  # Over-fetch to allow dedupe + categorisation
            if not isinstance(entry, dict):
                continue
            title = entry.get("headline")
            summary = entry.get("summary") or ""
            url_value = entry.get("url") or ""
            source = entry.get("source") or "Finnhub"
            timestamp = entry.get("datetime")
            published_at = None
            if isinstance(timestamp, (int, float)) and timestamp > 0:
                published_at = datetime.utcfromtimestamp(timestamp)
            category = entry.get("category") or ""

            bucket = self._categorize_news_item("finnhub", category)
            buckets[bucket].append(
                self._standardize_news_item(
                    title=title or url_value,
                    summary=summary,
                    source=source,
                    url=url_value,
                    published_at=published_at,
                    tickers=[stock_code],
                    provider="Finnhub",
                )
            )

        return buckets

    def _fetch_newsdata_company_news(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime,
        limit: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Download company news from NewsData.io."""

        api_key = self.api_keys.get("newsdata")
        if not api_key:
            raise NewsProviderError("NewsData.io API key is not configured")

        params = {
            "apikey": api_key,
            "q": stock_code,
            "language": "en",
            "category": "business",
            "from_date": start_date.strftime("%Y-%m-%d"),
            "to_date": end_date.strftime("%Y-%m-%d"),
            "page": 1,
            "pageSize": min(limit, 50),
        }

        url = "https://newsdata.io/api/1/news"
        response: Response = self._http_get(url, params=params, timeout=6)
        if response.status_code == 429:
            raise RateLimitError("NewsData.io rate limit exceeded")
        if response.status_code >= 500:
            raise NewsProviderError(
                f"NewsData.io server error ({response.status_code})"
            )
        try:
            response.raise_for_status()
        except RequestException as exc:
            raise NewsProviderError(f"NewsData.io request failed: {exc}") from exc

        payload = response.json()
        if payload.get("status") not in {"success", "ok"}:
            message = payload.get("message") or payload.get("results") or "Unknown error"
            raise NewsProviderError(f"NewsData.io returned an error: {message}")

        results = payload.get("results") or payload.get("data") or []
        if not isinstance(results, list):
            raise NewsProviderError("Unexpected NewsData.io response structure")

        buckets = {
            "company_news": [],
            "announcements": [],
            "research_reports": [],
        }

        for entry in results[: limit * 2]:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title")
            summary = entry.get("description") or entry.get("content") or ""
            url_value = entry.get("link") or entry.get("url") or ""
            source = entry.get("source_id") or entry.get("source") or "NewsData.io"
            published = entry.get("pubDate") or entry.get("publishedAt")
            category_hint = entry.get("category")
            if isinstance(category_hint, list):
                category_hint = ",".join(category_hint)

            buckets[self._categorize_news_item("newsdata", category_hint)].append(
                self._standardize_news_item(
                    title=title or url_value,
                    summary=summary,
                    source=source,
                    url=url_value,
                    published_at=published,
                    tickers=[stock_code],
                    provider="NewsData.io",
                )
            )

        return buckets

    def calculate_advanced_sentiment_analysis(self, news_data: Dict[str, Any]) -> Dict:
        """Generate sentiment statistics from news content."""

        buckets = {
            key: value for key, value in news_data.items() if isinstance(value, list)
        }
        metadata = news_data.get("metadata", {}) if isinstance(news_data, dict) else {}
        total_items = sum(len(items) for items in buckets.values())

        if not total_items:
            return {
                "overall_sentiment": 0.0,
                "confidence_score": 0.0,
                "sentiment_trend": "insufficient-data",
                "total_analyzed": 0,
                "sources": metadata.get("sources", []),
                "analyzer": "none",
            }

        analyzer = self._get_sentiment_analyzer()
        sentiment_scores: List[float] = []
        for items in buckets.values():
            for article in items:
                if not isinstance(article, dict):
                    continue
                title = (article.get("title") or "").strip()
                summary = (article.get("summary") or "").strip()
                content = f"{title}. {summary}".strip()
                if not content:
                    continue
                try:
                    if analyzer is not None:
                        score = analyzer.polarity_scores(content).get("compound", 0.0)
                    else:
                        score = self._estimate_keyword_sentiment(content)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Sentiment scoring failed for article: %s", exc)
                    continue
                sentiment_scores.append(float(max(-1.0, min(1.0, score))))

        if not sentiment_scores:
            return {
                "overall_sentiment": 0.0,
                "confidence_score": 0.0,
                "sentiment_trend": "insufficient-data",
                "total_analyzed": total_items,
                "sources": metadata.get("sources", []),
                "analyzer": "keyword" if analyzer is None else "vader",
            }

        overall_sentiment = float(np.mean(sentiment_scores))
        confidence = min(1.0, max(0.1, total_items / 20.0))
        if analyzer is None:
            confidence *= 0.6  # Heuristic: keyword fallback is less reliable

        if overall_sentiment >= 0.1:
            trend = "bullish"
        elif overall_sentiment <= -0.1:
            trend = "bearish"
        else:
            trend = "neutral"

        return {
            "overall_sentiment": overall_sentiment,
            "confidence_score": confidence,
            "sentiment_trend": trend,
            "total_analyzed": total_items,
            "sources": metadata.get("sources", []),
            "analyzer": "vader" if analyzer is not None else "keyword",
        }

    def calculate_sentiment_score(self, sentiment_analysis: Dict) -> Optional[float]:
        """Map sentiment statistics to a 0-100 score."""

        total_items = sentiment_analysis.get("total_analyzed", 0)
        if not total_items:
            return None

        base = 50.0 + sentiment_analysis.get("overall_sentiment", 0.0) * 50.0
        confidence = sentiment_analysis.get("confidence_score", 0.0)
        adjusted = base * (0.5 + confidence / 2)
        return max(0.0, min(100.0, adjusted))

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def calculate_comprehensive_score(self, scores: Dict[str, Optional[float]]) -> float:
        """Blend technical, fundamental, and sentiment scores."""

        weights = self.analysis_weights

        def _safe_value(value: Optional[float]) -> float:
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                return float(value)
            return 0.0

        return sum(
            _safe_value(scores.get(key)) * float(weights.get(key, 0.0))
            for key in ("technical", "fundamental", "sentiment")
        )

    def generate_recommendation(self, scores: Dict[str, Optional[float]], market: str) -> str:
        """Provide a qualitative recommendation."""

        composite_value = scores.get("comprehensive", 50.0)
        composite = (
            float(composite_value)
            if isinstance(composite_value, (int, float)) and not math.isnan(float(composite_value))
            else 50.0
        )
        if composite >= 80:
            return "Strong buy"
        if composite >= 65:
            return "Moderate buy"
        if composite >= 50:
            return "Hold"
        if composite >= 35:
            return "Reduce"
        return "Avoid"

    def _build_ai_analysis_prompt(self, analysis_context: Dict) -> str:
        """Compose an instruction prompt for the configured LLM."""

        stock_code = analysis_context.get("stock_code", "")
        stock_name = analysis_context.get("stock_name", stock_code or "the company")
        price_info = analysis_context.get("price_info", {})
        technicals = analysis_context.get("technical_analysis", {})
        fundamentals = analysis_context.get("fundamental_data", {}).get(
            "financial_indicators", {}
        )
        sentiment = analysis_context.get("sentiment_analysis", {})
        scores = analysis_context.get("scores", {})
        recommendation = analysis_context.get("recommendation", "Unknown")
        analysis_date = analysis_context.get("analysis_date")
        data_quality = analysis_context.get("data_quality", {})

        quality_messages = data_quality.get("messages") or []
        missing_sections: List[str] = []
        if not fundamentals:
            missing_sections.append("fundamental indicators")
        if sentiment.get("total_analyzed", 0) == 0:
            missing_sections.append("news & sentiment feed")

        def _format_dict(title: str, values: Dict) -> str:
            if not values:
                return f"{title}: No reliable data available."
            lines = [f"{title}:"]
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    numeric_value = float(value)
                    if math.isnan(numeric_value):
                        value_str = "N/A"
                    else:
                        value_str = f"{numeric_value:.4f}" if abs(numeric_value) < 1 else f"{numeric_value:.2f}"
                elif isinstance(value, (list, tuple, set)):
                    value_str = ", ".join(str(item) for item in value if item) or "N/A"
                elif value in (None, ""):
                    value_str = "N/A"
                else:
                    value_str = str(value)
                lines.append(f"- {key}: {value_str}")
            return "\n".join(lines)

        prompt_parts = [
            "You are a senior U.S. equity analyst tasked with producing an in-depth investment note.",
            "Summarise the opportunity in clear English for a professional audience.",
            "Blend quantitative metrics with qualitative insight and highlight catalysts, risks, and monitoring guidance.",
            "Use section headings (Overview, Financial Health, Technical View, Sentiment & News, Investment Outlook).",
            "Close with bullet-point action items for investors.",
            "Avoid fabricating data – if something is missing, call it out explicitly.",
            "Before writing, assess the data quality notes and explain any limitations to the reader.",
        ]

        if missing_sections:
            prompt_parts.append(
                "The following datasets are incomplete or missing: "
                + ", ".join(missing_sections)
                + ". Describe how this constrains the analysis."
            )

        prompt_parts.append(
            "If the 'financial_indicators' are empty, state clearly that financial health could not be assessed due to data availability and that the recommendation relies only on technical and sentiment signals."
        )

        prompt_parts.extend(
            [
                "",
                f"Ticker: {stock_name} ({stock_code})",
                f"Snapshot date: {analysis_date or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]
        )

        prompt_parts.extend(
            [
                _format_dict("Price snapshot", price_info),
                _format_dict("Scorecard (0-100 scale)", scores),
                _format_dict("Technical indicators", technicals),
                _format_dict("Fundamental indicators", fundamentals),
                _format_dict("Sentiment signals", sentiment),
                f"Model recommendation: {recommendation}",
            ]
        )

        if quality_messages:
            quality_lines = ["Data quality notes:"]
            for note in quality_messages:
                quality_lines.append(f"- {note}")
            prompt_parts.append("\n".join(quality_lines))

        return "\n".join(part for part in prompt_parts if part)

    def _rule_based_analysis(self, analysis_context: Dict) -> str:
        """Fallback narrative when LLM providers are unavailable."""

        stock_code = analysis_context.get("stock_code", "")
        stock_name = analysis_context.get("stock_name", stock_code or "the company")
        scores = analysis_context.get("scores", {})
        price_info = analysis_context.get("price_info", {})
        recommendation = analysis_context.get("recommendation", "Hold")

        def _score_text(key: str) -> str:
            value = scores.get(key)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                return f"{float(value):.1f}"
            return "N/A"

        lines = [
            f"Overview\n{stock_name} ({stock_code}) currently screens as a {recommendation.lower()} idea based on the blended scoring model.",
        ]

        if price_info:
            price_line = "Price data is unavailable."
            if price_info.get("current_price") is not None:
                price_line = (
                    f"The latest close is {price_info['current_price']:.2f}"
                )
                if price_info.get("price_change") is not None:
                    price_line += f", a {price_info['price_change']:.2f}% move over the selected window."
            lines.append(f"Market snapshot\n{price_line}")

        if scores:
            lines.append(
                "Score breakdown\n"
                f"Technical: {_score_text('technical')} · "
                f"Fundamental: {_score_text('fundamental')} · "
                f"Sentiment: {_score_text('sentiment')} · "
                f"Composite: {_score_text('comprehensive')}"
            )

        lines.append(
            "Outlook\n"
            "Monitor earnings revisions, price-volume trends, and material news flow to validate the signal."
        )

        return "\n\n".join(lines)

    def _call_ai_api(
        self,
        prompt: str,
        enable_streaming: bool,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Invoke the preferred AI provider with graceful fallbacks."""

        if not prompt.strip():
            return None

        preference = self.config.get("ai", {}).get("model_preference", "openai")
        providers = [preference]
        for candidate in ("openai", "anthropic", "zhipu"):
            if candidate not in providers:
                providers.append(candidate)

        for provider in providers:
            api_key = self.api_keys.get(provider)
            if not api_key:
                continue
            try:
                if provider == "openai":
                    return self._call_openai_api(prompt, enable_streaming, stream_callback)
                if provider == "anthropic":
                    return self._call_anthropic_api(prompt, enable_streaming, stream_callback)
                if provider == "zhipu":
                    return self._call_zhipu_api(prompt, enable_streaming, stream_callback)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("AI provider %s failed: %s", provider, exc)

        return None

    def _call_openai_api(
        self,
        prompt: str,
        enable_streaming: bool,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Call OpenAI's chat completion API with streaming support."""

        import importlib

        openai = importlib.import_module("openai")  # pragma: no cover - optional dependency

        api_key = self.api_keys.get("openai")
        if not api_key:
            return None

        config = self.config.get("ai", {})
        model = config.get("models", {}).get("openai", "gpt-4o-mini")
        max_tokens = config.get("max_tokens", 4000)
        temperature = config.get("temperature", 0.7)
        base_url = config.get("api_base_urls", {}).get("openai")

        messages = [
            {
                "role": "system",
                "content": "You are an experienced U.S. equity analyst who writes rigorous investment reports.",
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(openai, "OpenAI"):
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)

            def _normalise_content(value):
                if isinstance(value, list):
                    parts = []
                    for item in value:
                        text = getattr(item, "text", None)
                        if text is None and isinstance(item, dict):
                            text = item.get("text")
                        if text:
                            parts.append(text)
                    return "".join(parts)
                return value or ""

            if enable_streaming and stream_callback:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                chunks: List[str] = []
                for event in stream:
                    choices = getattr(event, "choices", None)
                    if not choices:
                        continue
                    for choice in choices:
                        delta_choice = getattr(choice, "delta", None)
                        content = getattr(delta_choice, "content", None) if delta_choice else None
                        normalised = _normalise_content(content)
                        if not normalised:
                            continue
                        stream_callback(normalised)
                        chunks.append(normalised)
                return "".join(chunks)

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = ""
            if response.choices:
                content = _normalise_content(response.choices[0].message.content)
            if enable_streaming and stream_callback and content:
                stream_callback(content)
            return content

        # Legacy openai library fallback
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url

        if enable_streaming and stream_callback:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            chunks = []
            for chunk in completion:
                if not chunk["choices"]:
                    continue
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    stream_callback(delta)
                    chunks.append(delta)
            return "".join(chunks)

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response["choices"][0]["message"].get("content", "")
        if enable_streaming and stream_callback and content:
            stream_callback(content)
        return content

    def _call_anthropic_api(
        self,
        prompt: str,
        enable_streaming: bool,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Call Anthropic's Claude models with optional streaming."""

        from anthropic import Anthropic  # pragma: no cover - optional dependency

        api_key = self.api_keys.get("anthropic")
        if not api_key:
            return None

        client = Anthropic(api_key=api_key)
        model = self.config.get("ai", {}).get("models", {}).get(
            "anthropic", "claude-3-haiku-20240307"
        )
        max_tokens = self.config.get("ai", {}).get("max_tokens", 4000)
        temperature = self.config.get("ai", {}).get("temperature", 0.7)

        if enable_streaming and stream_callback:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are an experienced U.S. equity analyst who writes rigorous investment reports.",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                chunks: List[str] = []
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta.get("text")
                        if delta:
                            stream_callback(delta)
                            chunks.append(delta)
                return "".join(chunks)

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are an experienced U.S. equity analyst who writes rigorous investment reports.",
            messages=[{"role": "user", "content": prompt}],
        )
        content = "".join(block.text for block in message.content if hasattr(block, "text"))
        if enable_streaming and stream_callback and content:
            stream_callback(content)
        return content

    def _call_zhipu_api(
        self,
        prompt: str,
        enable_streaming: bool,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Call ZhipuAI's chat completion endpoint."""

        from zhipuai import ZhipuAI  # pragma: no cover - optional dependency

        api_key = self.api_keys.get("zhipu")
        if not api_key:
            return None

        client = ZhipuAI(api_key=api_key)
        model = self.config.get("ai", {}).get("models", {}).get("zhipu", "chatglm_turbo")
        temperature = self.config.get("ai", {}).get("temperature", 0.7)

        if enable_streaming and stream_callback:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced U.S. equity analyst who writes rigorous investment reports.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=True,
            )
            chunks: List[str] = []
            for chunk in response:
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    stream_callback(delta)
                    chunks.append(delta)
            return "".join(chunks)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an experienced U.S. equity analyst who writes rigorous investment reports.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        choices = response.get("choices") or []
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", "")
        if enable_streaming and stream_callback and content:
            stream_callback(content)
        return content

    def generate_ai_analysis(
        self,
        analysis_context: Dict,
        enable_streaming: bool,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Produce an English narrative for the analysis."""

        if not any(self.api_keys.get(key) for key in ("openai", "anthropic", "zhipu")):
            logger.info("No AI API keys configured; using rule-based summary.")
            fallback = self._rule_based_analysis(analysis_context)
            if enable_streaming and stream_callback:
                stream_callback(fallback)
            return fallback

        prompt = self._build_ai_analysis_prompt(analysis_context)
        logger.info("Submitting analysis prompt to AI provider (%s)", self.config.get("ai", {}).get("model_preference", "openai"))
        response = self._call_ai_api(prompt, enable_streaming, stream_callback)
        if response:
            return response

        logger.warning("AI response unavailable. Falling back to rule-based analysis.")
        fallback = self._rule_based_analysis(analysis_context)
        if enable_streaming and stream_callback:
            stream_callback(fallback)
        return fallback

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def analyze_stock(
        self,
        stock_code: str,
        enable_streaming: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """Run a full analysis for ``stock_code``."""

        normalized_code, market = self.normalize_stock_code(stock_code)
        with ThreadPoolExecutor(max_workers=4) as pool:
            name_future = pool.submit(self.get_stock_name, normalized_code)
            price_future = pool.submit(self.get_stock_data, normalized_code)
            fundamental_future = pool.submit(
                self.get_comprehensive_fundamental_data, normalized_code
            )
            news_future = pool.submit(
                self.get_comprehensive_news_data, normalized_code
            )

            stock_name = name_future.result()
            price_data = price_future.result()
            fundamental_data = fundamental_future.result()
            news_data = news_future.result()

        price_info = self.get_price_info(price_data)

        technical_indicators = self.calculate_technical_indicators(price_data)
        technical_score = self.calculate_technical_score(technical_indicators)

        fundamental_score = self.calculate_fundamental_score(fundamental_data)

        sentiment_analysis = self.calculate_advanced_sentiment_analysis(news_data)
        news_metadata = news_data.get("metadata", {}) if isinstance(news_data, dict) else {}
        sentiment_score = self.calculate_sentiment_score(sentiment_analysis)

        scores = {
            "technical": technical_score,
            "fundamental": fundamental_score,
            "sentiment": sentiment_score,
        }
        scores["comprehensive"] = self.calculate_comprehensive_score(scores)

        recommendation = self.generate_recommendation(scores, market)

        fundamental_indicators = fundamental_data.get("financial_indicators", {})
        fundamental_count = len(fundamental_indicators)
        news_count = sentiment_analysis.get("total_analyzed", 0)
        news_sources = sentiment_analysis.get("sources") or news_metadata.get("sources") or []
        data_quality_messages: List[str] = []

        metadata_warnings = fundamental_data.get("metadata", {}).get("warnings") or []
        data_quality_messages.extend(metadata_warnings)

        raw_fundamental_source = (
            fundamental_data.get("metadata", {}).get("source") or ""
        ).strip()

        if fundamental_count == 0:
            data_quality_messages.append(
                "Fundamental data is unavailable; valuation metrics could not be assessed."
            )
        if fundamental_data.get("metadata", {}).get("source") in {"placeholder", "unavailable"}:
            data_quality_messages.append(
                "Fundamental source is unavailable – verify API credentials or provider availability."
            )
        if news_count == 0:
            data_quality_messages.append(
                "No recent news articles were retrieved. Sentiment score has been marked as N/A."
            )
            if not (self.api_keys.get("finnhub") or self.api_keys.get("newsdata")):
                data_quality_messages.append(
                    "News APIs are not configured. Add Finnhub or NewsData.io keys for richer sentiment coverage."
                )
        elif sentiment_analysis.get("analyzer") == "keyword":
            data_quality_messages.append(
                "Sentiment scoring fell back to keyword heuristics. Install vaderSentiment for higher fidelity results."
            )
        if price_info.get("current_price") is None:
            data_quality_messages.append(
                "Price snapshot is incomplete; confirm market data connectivity."
            )

        deduped_messages: List[str] = []
        for message in data_quality_messages:
            if message and message not in deduped_messages:
                deduped_messages.append(message)

        coverage_status = "complete"
        if fundamental_count == 0 and news_count == 0:
            coverage_status = "minimal"
        elif fundamental_count == 0 or news_count == 0:
            coverage_status = "partial"

        def _format_label(value: str) -> str:
            if not value or value in {"unavailable", "placeholder"}:
                return "Not available"
            return value.replace("_", " ").replace("-", " ").title()

        fundamental_status = "ok" if fundamental_count > 0 else "warn"
        fundamental_source_label = (
            _format_label(raw_fundamental_source)
            if fundamental_count > 0
            else "Not available"
        )

        sentiment_engine = (sentiment_analysis.get("analyzer") or "none").lower()
        if news_count == 0:
            sentiment_status = "warn"
            sentiment_label = "Not available"
        elif sentiment_engine == "vader":
            sentiment_status = "ok"
            sentiment_label = "VADER sentiment"
        elif sentiment_engine == "keyword":
            sentiment_status = "info"
            sentiment_label = "Keyword heuristics"
        else:
            sentiment_status = "info"
            sentiment_label = _format_label(sentiment_engine)

        data_quality = {
            "financial_indicators_count": fundamental_count,
            "total_news_count": news_count,
            "analysis_completeness": coverage_status,
            "market_coverage": "US",
            "fundamental_source": fundamental_source_label,
            "fundamental_source_key": raw_fundamental_source or ("none" if fundamental_count == 0 else "unknown"),
            "fundamental_status": fundamental_status,
            "fundamental_providers": fundamental_data.get("metadata", {}).get(
                "providers_attempted", []
            ),
            "sentiment_available": news_count > 0,
            "news_sources": news_sources,
            "sentiment_analyzer": sentiment_analysis.get("analyzer"),
            "sentiment_label": sentiment_label,
            "sentiment_status": sentiment_status,
            "messages": deduped_messages,
        }

        analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ai_analysis = self.generate_ai_analysis(
            {
                "stock_code": normalized_code,
                "stock_name": stock_name,
                "scores": scores,
                "price_info": price_info,
                "technical_analysis": technical_indicators,
                "fundamental_data": fundamental_data,
                "sentiment_analysis": sentiment_analysis,
                "recommendation": recommendation,
                "analysis_date": analysis_timestamp,
                "data_quality": data_quality,
            },
            enable_streaming,
            stream_callback,
        )

        report = {
            "stock_code": normalized_code,
            "original_code": stock_code,
            "stock_name": stock_name,
            "market": market,
            "market_info": self.market_config.get(market, {}),
            "analysis_date": analysis_timestamp,
            "price_info": price_info,
            "technical_analysis": technical_indicators,
            "fundamental_data": fundamental_data,
            "sentiment_analysis": sentiment_analysis,
            "scores": scores,
            "analysis_weights": self.analysis_weights,
            "recommendation": recommendation,
            "ai_analysis": ai_analysis,
            "data_quality": data_quality,
        }

        return report

    # ------------------------------------------------------------------
    # Conversational follow-up
    # ------------------------------------------------------------------
    def generate_chat_followup(
        self,
        report: Dict,
        conversation: List[Dict[str, str]],
        user_message: str,
        enable_streaming: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Answer follow-up questions about the latest report."""

        cleaned_history = [
            {"role": item.get("role", "user"), "content": item.get("content", "")}
            for item in conversation
            if item.get("content")
        ]
        if cleaned_history:
            cleaned_history = cleaned_history[-6:]

        if not any(self.api_keys.get(key) for key in ("openai", "anthropic", "zhipu")):
            logger.info("No AI API keys configured; using rule-based chat response.")
            response = self._rule_based_chat_response(report, user_message)
            if enable_streaming and stream_callback:
                stream_callback(response)
            return response

        prompt = self._build_chat_prompt(report, cleaned_history, user_message)
        reply = self._call_ai_api(prompt, enable_streaming, stream_callback)
        if reply:
            return reply

        logger.warning("Chat AI response unavailable. Falling back to rule-based reply.")
        response = self._rule_based_chat_response(report, user_message)
        if enable_streaming and stream_callback:
            stream_callback(response)
        return response

    def _rule_based_chat_response(self, report: Dict, question: str) -> str:
        """Fallback chat response using deterministic logic."""

        scores = report.get("scores", {}) if isinstance(report, dict) else {}
        rec = report.get("recommendation") if isinstance(report, dict) else None
        price_info = report.get("price_info", {}) if isinstance(report, dict) else {}
        data_quality = report.get("data_quality", {}) if isinstance(report, dict) else {}

        summary_parts = [
            "Automated response: live AI provider is unavailable.",
            f"Latest recommendation: {rec or 'N/A'}.",
            "Scores — "
            + ", ".join(
                f"{label.capitalize()}: {self._format_score_for_prompt(scores.get(label))}"
                for label in ("technical", "fundamental", "sentiment", "comprehensive")
                if label in scores
            ),
        ]

        if price_info:
            price_text = price_info.get("current_price")
            change = price_info.get("price_change")
            if price_text is not None:
                summary_parts.append(f"Last close: {price_text:.2f} {price_info.get('currency', 'USD')}")
            if change is not None:
                summary_parts.append(f"Daily change: {change:+.2f}%")

        messages = data_quality.get("messages") or []
        if messages:
            summary_parts.append("Data quality alerts: " + " | ".join(messages))

        summary_parts.append(
            "Question received: " + (question.strip() or "No question provided.")
        )

        return "\n".join(summary_parts)

    def _format_score_for_prompt(self, value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "N/A"
        return f"{value:.1f}"

    def _build_chat_prompt(
        self,
        report: Dict,
        conversation: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """Compose an instruction prompt for chat follow-ups."""

        stock_code = report.get("stock_code", "")
        stock_name = report.get("stock_name", "")
        recommendation = report.get("recommendation", "N/A")
        price_info = report.get("price_info", {})
        scores = report.get("scores", {})
        technical = report.get("technical_analysis", {})
        fundamentals = report.get("fundamental_data", {})
        sentiment = report.get("sentiment_analysis", {})
        data_quality = report.get("data_quality", {})

        lines = [
            "You are an expert U.S. equity analyst continuing a conversation about a stock report.",
            "Use only the provided report data when answering questions.",
            "Highlight any missing or low-quality data that affects the conclusion.",
            "If financial indicators are empty, clearly state that fundamentals were unavailable and rely on technical and sentiment inputs instead.",
        ]

        lines.append(
            "Report summary (structured JSON):\n" + json.dumps(
                {
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "recommendation": recommendation,
                    "price_info": price_info,
                    "scores": scores,
                    "technical_highlights": technical,
                    "fundamental_data": fundamentals,
                    "sentiment": sentiment,
                    "data_quality": data_quality,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        if conversation:
            lines.append("Conversation so far:")
            for turn in conversation:
                role = turn.get("role", "user").lower()
                content = turn.get("content", "")
                lines.append(f"{role}: {content}")

        lines.append(f"User follow-up question: {user_message}")
        lines.append(
            "Guidance: cite the relevant metrics from the report, address data-quality notes, and if something is missing, explicitly mention that limitation."
        )

        return "\n\n".join(lines)


def get_stock_analyzer() -> EnhancedWebStockAnalyzer:
    """Factory kept for legacy imports."""

    return EnhancedWebStockAnalyzer()


if __name__ == "__main__":  # pragma: no cover - manual testing helper
    analyzer = EnhancedWebStockAnalyzer()
    sample = analyzer.analyze_stock("AAPL")
    print(json.dumps(sample, indent=2, ensure_ascii=False))
