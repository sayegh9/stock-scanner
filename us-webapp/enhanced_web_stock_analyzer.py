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
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        self.market_config: Dict[str, Dict[str, str]] = self.config.get("markets", {})
        self.streaming_config = self.config.get("streaming", {})
        self.cache_config = self.config.get("cache", {})
        self.api_keys = self.config.get("api_keys", {})

        self._price_cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self._fundamental_cache: Dict[str, Tuple[datetime, Dict[str, float]]] = {}
        self._news_cache: Dict[str, Tuple[datetime, Dict[str, List[dict]]]] = {}

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
                "a_stock": {
                    "enabled": False,
                    "name": "A-share",
                    "currency": "CNY",
                    "timezone": "Asia/Shanghai",
                    "trading_hours": "09:30-15:00",
                },
                "hk_stock": {
                    "enabled": False,
                    "name": "Hong Kong",
                    "currency": "HKD",
                    "timezone": "Asia/Hong_Kong",
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
    def get_stock_name(self, stock_code: str) -> str:
        """Return a human readable name for a ticker."""

        return stock_code

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
    def get_comprehensive_fundamental_data(self, stock_code: str) -> Dict:
        """Fetch or synthesise fundamental metrics."""

        cache_entry = self._fundamental_cache.get(stock_code)
        expiry_hours = self.cache_config.get("fundamental_hours", 6)
        if cache_entry and datetime.now() - cache_entry[0] < timedelta(hours=expiry_hours):
            return cache_entry[1]

        fundamentals: Dict[str, float] = {}

        source_name = ""

        if yf is not None:
            try:  # pragma: no cover - network dependent
                ticker = yf.Ticker(stock_code)
                info = ticker.get_info()
                mapping = {
                    "trailingPE": "pe_ratio",
                    "forwardPE": "forward_pe",
                    "trailingEps": "eps",
                    "forwardEps": "forward_eps",
                    "returnOnEquity": "roe",
                    "revenueGrowth": "revenue_growth",
                    "earningsGrowth": "net_profit_growth",
                    "earningsQuarterlyGrowth": "net_profit_growth_quarterly",
                }
                for source_key, target_key in mapping.items():
                    value = info.get(source_key)
                    if value is None:
                        continue
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if source_key in {"returnOnEquity", "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth"}:
                        value *= 100.0
                    fundamentals[target_key] = value
                if fundamentals:
                    source_name = "yfinance"
            except Exception as exc:
                logger.warning("Unable to fetch yfinance fundamentals for %s: %s", stock_code, exc)

        if not fundamentals and ak is not None:
            try:  # pragma: no cover - network dependent
                info = ak.stock_us_fundamental(stock=stock_code)
                if not info.empty:
                    info = info.set_index("item")
                    for source_key, target_key in AK_FUNDAMENTAL_KEYS.items():
                        if source_key in info.index:
                            fundamentals[target_key] = float(info.loc[source_key, "value"])
                    if fundamentals and not source_name:
                        source_name = "akshare"
            except Exception as exc:
                logger.warning("Unable to fetch fundamentals for %s via akshare: %s", stock_code, exc)

        data = {
            "financial_indicators": fundamentals,
            "metadata": {
                "source": source_name or "placeholder",
                "retrieved_at": datetime.now().isoformat(),
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

        sources_used: List[str] = []
        fallback_required = False

        if self.api_keys.get("finnhub"):
            try:  # pragma: no cover - network dependent
                finnhub_payload = self._fetch_finnhub_company_news(
                    stock_code, start_date, end_date, max_items
                )
                for key, values in finnhub_payload.items():
                    aggregated[key].extend(values)
                if sum(len(values) for values in finnhub_payload.values()):
                    sources_used.append("Finnhub")
                else:
                    fallback_required = True
            except RateLimitError as exc:
                fallback_required = True
                logger.warning(
                    "Finnhub rate limit while fetching news for %s: %s", stock_code, exc
                )
            except NewsProviderError as exc:
                fallback_required = True
                logger.warning(
                    "Finnhub news download failed for %s: %s", stock_code, exc
                )
            except Exception as exc:  # pragma: no cover - defensive
                fallback_required = True
                logger.exception(
                    "Unexpected Finnhub error for %s: %s", stock_code, exc
                )
        else:
            fallback_required = True
            logger.debug("Finnhub API key missing; skipping Finnhub news fetch")

        should_try_newsdata = self.api_keys.get("newsdata") and (
            fallback_required or not sum(len(values) for values in aggregated.values())
        )

        if should_try_newsdata:
            try:  # pragma: no cover - network dependent
                newsdata_payload = self._fetch_newsdata_company_news(
                    stock_code, start_date, end_date, max_items
                )
                for key, values in newsdata_payload.items():
                    aggregated[key].extend(values)
                if sum(len(values) for values in newsdata_payload.values()):
                    sources_used.append("NewsData.io")
            except RateLimitError as exc:
                logger.warning(
                    "NewsData.io rate limit while fetching news for %s: %s", stock_code, exc
                )
            except NewsProviderError as exc:
                logger.warning(
                    "NewsData.io news download failed for %s: %s", stock_code, exc
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "Unexpected NewsData.io error for %s: %s", stock_code, exc
                )
        elif not self.api_keys.get("newsdata"):
            logger.debug("NewsData.io API key missing; skipping NewsData.io fallback")

        aggregated = {
            key: self._deduplicate_news_items(values, max_items)
            for key, values in aggregated.items()
        }

        total_items = sum(len(values) for values in aggregated.values())
        if total_items:
            logger.info(
                "Fetched %s news articles for %s via %s",
                total_items,
                stock_code,
                ", ".join(sources_used) if sources_used else "cache",
            )
        else:
            logger.info(
                "No recent news items found for %s. Providers attempted: %s",
                stock_code,
                ", ".join(sources_used) if sources_used else "none",
            )

        self._news_cache[stock_code] = (datetime.now(), aggregated)
        return aggregated

    def _deduplicate_news_items(
        self, items: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """Remove duplicate stories and enforce the configured limit."""

        if not items:
            return []

        seen: set[str] = set()
        cleaned: List[Dict[str, Any]] = []
        for item in sorted(
            items,
            key=lambda entry: entry.get("published_at") or "",
            reverse=True,
        ):
            unique_key = item.get("url") or (
                f"{item.get('title', '')}|{item.get('published_at', '')}"
            )
            if unique_key in seen:
                continue
            seen.add(unique_key)
            cleaned.append(item)
            if len(cleaned) >= limit:
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
        response: Response = requests.get(url, params=params, timeout=10)
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
        response: Response = requests.get(url, params=params, timeout=10)
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

    def calculate_advanced_sentiment_analysis(self, news_data: Dict[str, List[dict]]) -> Dict:
        """Generate sentiment statistics from news content."""

        total_items = sum(len(items) for items in news_data.values())
        sentiment_score = 0.0
        confidence = 0.0
        if total_items:
            sentiment_score = 0.0
            confidence = 0.5

        return {
            "overall_sentiment": sentiment_score,
            "confidence_score": confidence,
            "sentiment_trend": "neutral",
            "total_analyzed": total_items,
        }

    def calculate_sentiment_score(self, sentiment_analysis: Dict) -> float:
        """Map sentiment statistics to a 0-100 score."""

        base = 50.0 + sentiment_analysis.get("overall_sentiment", 0.0) * 50.0
        confidence = sentiment_analysis.get("confidence_score", 0.0)
        adjusted = base * (0.5 + confidence / 2)
        return max(0.0, min(100.0, adjusted))

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def calculate_comprehensive_score(self, scores: Dict[str, float]) -> float:
        """Blend technical, fundamental, and sentiment scores."""

        weights = self.analysis_weights
        return sum(
            scores.get(key, 0.0) * float(weights.get(key, 0.0))
            for key in ("technical", "fundamental", "sentiment")
        )

    def generate_recommendation(self, scores: Dict[str, float], market: str) -> str:
        """Provide a qualitative recommendation."""

        composite = scores.get("comprehensive", 50.0)
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

        def _format_dict(title: str, values: Dict) -> str:
            if not values:
                return f"{title}: No reliable data available."
            lines = [f"{title}:"]
            for key, value in values.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
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
            "",
            f"Ticker: {stock_name} ({stock_code})",
            f"Snapshot date: {analysis_date or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            _format_dict("Price snapshot", price_info),
            _format_dict("Scorecard (0-100 scale)", scores),
            _format_dict("Technical indicators", technicals),
            _format_dict("Fundamental indicators", fundamentals),
            _format_dict("Sentiment signals", sentiment),
            f"Model recommendation: {recommendation}",
        ]

        return "\n".join(part for part in prompt_parts if part)

    def _rule_based_analysis(self, analysis_context: Dict) -> str:
        """Fallback narrative when LLM providers are unavailable."""

        stock_code = analysis_context.get("stock_code", "")
        stock_name = analysis_context.get("stock_name", stock_code or "the company")
        scores = analysis_context.get("scores", {})
        price_info = analysis_context.get("price_info", {})
        recommendation = analysis_context.get("recommendation", "Hold")

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
                f"Technical: {scores.get('technical', 0):.1f} · "
                f"Fundamental: {scores.get('fundamental', 0):.1f} · "
                f"Sentiment: {scores.get('sentiment', 0):.1f} · "
                f"Composite: {scores.get('comprehensive', 0):.1f}"
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
        stock_name = self.get_stock_name(normalized_code)
        price_data = self.get_stock_data(normalized_code)
        price_info = self.get_price_info(price_data)

        technical_indicators = self.calculate_technical_indicators(price_data)
        technical_score = self.calculate_technical_score(technical_indicators)

        fundamental_data = self.get_comprehensive_fundamental_data(normalized_code)
        fundamental_score = self.calculate_fundamental_score(fundamental_data)

        news_data = self.get_comprehensive_news_data(normalized_code)
        sentiment_analysis = self.calculate_advanced_sentiment_analysis(news_data)
        sentiment_score = self.calculate_sentiment_score(sentiment_analysis)

        scores = {
            "technical": technical_score,
            "fundamental": fundamental_score,
            "sentiment": sentiment_score,
        }
        scores["comprehensive"] = self.calculate_comprehensive_score(scores)

        recommendation = self.generate_recommendation(scores, market)

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
            "data_quality": {
                "financial_indicators_count": len(fundamental_data.get("financial_indicators", {})),
                "total_news_count": sentiment_analysis.get("total_analyzed", 0),
                "analysis_completeness": "complete"
                if fundamental_data.get("financial_indicators")
                else "partial",
                "market_coverage": "US",
            },
        }

        return report


def get_stock_analyzer() -> EnhancedWebStockAnalyzer:
    """Factory kept for legacy imports."""

    return EnhancedWebStockAnalyzer()


if __name__ == "__main__":  # pragma: no cover - manual testing helper
    analyzer = EnhancedWebStockAnalyzer()
    sample = analyzer.analyze_stock("AAPL")
    print(json.dumps(sample, indent=2, ensure_ascii=False))
