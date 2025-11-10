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

import json
import logging
import math
import os
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import akshare as ak
except ImportError:  # pragma: no cover - optional dependency
    ak = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

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

        if ak is None:
            raise RuntimeError(
                "akshare is not installed. Install it or extend `get_stock_data` with a different provider."
            )

        period_days = days or self.analysis_params.get("technical_period_days", 180)
        start_date = (datetime.now() - timedelta(days=period_days + 10)).strftime("%Y-%m-%d")
        try:
            data = ak.stock_us_hist(symbol=stock_code, period="daily", adjust="qfq")
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Failed to pull price history for %s: %s", stock_code, exc)
            raise

        if data.empty:
            raise ValueError(f"No price data returned for {stock_code}")

        data = data.rename(columns=AK_PRICE_COLUMNS)
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date")
        if period_days:
            data = data[data["date"] >= datetime.now() - timedelta(days=period_days)]

        data = data[["date", "open", "high", "low", "close", "volume"]]
        data = data.dropna()

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

        if ak is not None:
            try:  # pragma: no cover - network dependent
                info = ak.stock_us_fundamental(stock=stock_code)
                if not info.empty:
                    info = info.set_index("item")
                    for source_key, target_key in AK_FUNDAMENTAL_KEYS.items():
                        if source_key in info.index:
                            fundamentals[target_key] = float(info.loc[source_key, "value"])
            except Exception as exc:
                logger.warning("Unable to fetch fundamentals for %s: %s", stock_code, exc)

        data = {
            "financial_indicators": fundamentals,
            "metadata": {
                "source": "akshare" if fundamentals else "placeholder",
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
        """Return cached news data. Placeholder implementation."""

        cache_entry = self._news_cache.get(stock_code)
        expiry_hours = self.cache_config.get("news_hours", 2)
        if cache_entry and datetime.now() - cache_entry[0] < timedelta(hours=expiry_hours):
            return cache_entry[1]

        news_data = {
            "company_news": [],
            "announcements": [],
            "research_reports": [],
        }

        self._news_cache[stock_code] = (datetime.now(), news_data)
        return news_data

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

    def generate_ai_analysis(
        self,
        analysis_context: Dict,
        enable_streaming: bool,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Produce an English narrative for the analysis."""

        stock_code = analysis_context.get("stock_code")
        stock_name = analysis_context.get("stock_name", stock_code)
        scores = analysis_context.get("scores", {})

        paragraphs = [
            f"Summary for {stock_name} ({stock_code})",
            f"Technical score: {scores.get('technical', 0):.1f} / 100",
            f"Fundamental score: {scores.get('fundamental', 0):.1f} / 100",
            f"Sentiment score: {scores.get('sentiment', 0):.1f} / 100",
            f"Composite score: {scores.get('comprehensive', 0):.1f} / 100",
        ]

        text = "\n\n".join(paragraphs)
        if enable_streaming and stream_callback:
            for chunk in paragraphs:
                stream_callback(chunk + "\n\n")
        return text

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

        ai_analysis = self.generate_ai_analysis(
            {
                "stock_code": normalized_code,
                "stock_name": stock_name,
                "scores": scores,
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
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
