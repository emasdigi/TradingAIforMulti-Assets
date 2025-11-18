"""Utilities for caching and retrieving recent news per trading asset.

This module provides helper functions to refresh a local cache of Google News
headlines for the configured trading symbols so they can be reused inside the
trading prompts without hitting the network every iteration. It relies on the
``search_google_news`` helper from ``bot.news_fetcher`` for the actual web
scraping.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


try:
    from config import config  # type: ignore
except ImportError:  # pragma: no cover - fallback when executed as script
    import sys
    from pathlib import Path

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.append(str(_PROJECT_ROOT))

    from config import config  # type: ignore

try:
    from .news_fetcher import search_google_news  # type: ignore
except ImportError:  # pragma: no cover - fallback for script execution
    from news_fetcher import search_google_news  # type: ignore


logger = logging.getLogger(__name__)

_NEWS_CACHE_FILE: Path = config.DATA_DIR / "news_cache.json"
_METADATA_KEY = "_metadata"
_METADATA_LAST_REFRESH = "last_refresh_at"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _default_date_range(hours: int = 24) -> tuple[str, str]:
    """Return (start_date, end_date) covering the trailing ``hours`` window."""
    end_dt = _now_utc()
    start_dt = end_dt - timedelta(hours=hours)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _ensure_cache_dir() -> None:
    _NEWS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_cache() -> Dict[str, Any]:
    if not _NEWS_CACHE_FILE.exists():
        return {}
    try:
        with open(_NEWS_CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
            logger.error("Unexpected data type in news cache file: %s", type(data))
            return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load news cache: %s", exc, exc_info=True)
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    _ensure_cache_dir()
    with open(_NEWS_CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)


def _build_default_queries() -> Dict[str, str]:
    queries: Dict[str, str] = {}
    if config.ASSET_MODE.lower() == "crypto":
        for symbol in config.SYMBOLS:
            coin = config.SYMBOL_TO_COIN.get(symbol, symbol)
            queries[coin] = f"{coin} cryptocurrency"
    elif config.ASSET_MODE.lower() == "idss":
        for symbol in config.SYMBOLS:
            coin = config.SYMBOL_TO_COIN.get(symbol, symbol)
            queries[coin] = f"{coin} Indonesia company news"
    return queries


def refresh_news_cache(
    *,
    max_results_per_asset: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    country: str = "US",
    language: str = "en-US",
    asset_queries: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fetch the latest headlines for each configured asset and persist them.

    Args:
        max_results_per_asset: Maximum number of headlines to cache per coin.
        start_date: Optional ``YYYY-MM-DD`` string. Defaults to 24h window.
        end_date: Optional ``YYYY-MM-DD`` string. Defaults to ``today``.
        country: Country code used to bias Google News results.
        language: Locale code for the query.
        asset_queries: Custom mapping of ``coin -> search query``. Falls back to
            ``coin + " cryptocurrency"`` for each configured symbol.
    Returns:
        The updated cache dictionary keyed by coin ticker.
    """

    if not start_date or not end_date:
        start_date, end_date = _default_date_range()

    queries = asset_queries or _build_default_queries()
    if config.ASSET_MODE.lower() == "idss":
        country = "ID"
        language = "id-ID"
    else:
        country = "US"
        language = "en-US"

    cache = _load_cache()
    updated_cache: Dict[str, List[Dict[str, Any]]] = {}
    metadata = cache.get(_METADATA_KEY)
    previous_refresh_at = (
        metadata.get(_METADATA_LAST_REFRESH)
        if isinstance(metadata, dict)
        else None
    )

    for coin, query in queries.items():
        try:
            articles = search_google_news(
                query=query,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results_per_asset,
                country=country,
                language=language,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error fetching news for %s: %s", coin, exc, exc_info=True)
            continue

        fetched_at = _now_utc().isoformat()
        normalized_articles: List[Dict[str, Any]] = []
        for article in articles:
            # Ensure the core fields exist before storing.
            title = article.get("title")
            link = article.get("link")
            if not title or not link:
                continue

            summary = article.get("summary") or article.get("snippet", "")
            sentiment = article.get("sentiment") or ""
            sentiment_confidence = article.get("sentiment_confidence")
            try:
                if sentiment_confidence is not None:
                    sentiment_confidence = float(sentiment_confidence)
            except (TypeError, ValueError):
                sentiment_confidence = None

            published_at = article.get("published_at") or article.get("date") or ""
            raw_date = article.get("raw_date", "")

            normalized_articles.append(
                {
                    "title": title,
                    "link": link,
                    "snippet": article.get("snippet", ""),
                    "date": published_at or raw_date,
                    "published_at": published_at,
                    "raw_date": raw_date,
                    "source": article.get("source", ""),
                    "summary": summary,
                    "sentiment": sentiment,
                    "sentiment_confidence": sentiment_confidence,
                    "fetched_at": fetched_at,
                }
            )

        if normalized_articles:
            cache[coin] = normalized_articles[:max_results_per_asset]
            updated_cache[coin] = cache[coin]

    refresh_timestamp = _now_utc().isoformat()
    if not isinstance(metadata, dict):
        metadata = {}
    metadata[_METADATA_LAST_REFRESH] = refresh_timestamp
    cache[_METADATA_KEY] = metadata

    if updated_cache or previous_refresh_at != refresh_timestamp:
        _save_cache(cache)

    return cache


def get_cached_news(
    coin: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Return cached news entries for a given coin (most recent first)."""
    cache = _load_cache()
    entries = cache.get(coin.upper(), cache.get(coin, []))
    if not isinstance(entries, list):
        return []
    return entries[:limit]


def get_cached_titles(coin: str, limit: int = 3) -> List[str]:
    """Convenience helper returning only the news headlines."""
    return [entry.get("title", "") for entry in get_cached_news(coin, limit=limit) if entry.get("title")]


def iter_cached_news(limit_per_asset: int = 3) -> Iterable[tuple[str, List[Dict[str, Any]]]]:
    """Yield ``(coin, news_list)`` pairs from the cache for all configured assets."""
    cache = _load_cache()
    for symbol in config.SYMBOLS:
        coin = config.SYMBOL_TO_COIN.get(symbol, symbol)
        entries = cache.get(coin.upper(), cache.get(coin, []))
        if isinstance(entries, list) and entries:
            yield coin, entries[:limit_per_asset]


def get_last_refresh_time() -> Optional[str]:
    """Return the last time the news cache was refreshed in ISO format."""
    cache = _load_cache()
    metadata = cache.get(_METADATA_KEY)
    if isinstance(metadata, dict):
        timestamp = metadata.get(_METADATA_LAST_REFRESH)
        if isinstance(timestamp, str) and timestamp.strip():
            return timestamp
    return None


def main() -> None:
    """Allow ``python -m bot.news_cache`` to refresh the cache quickly."""
    cache = refresh_news_cache()
    asset_count = sum(1 for key in cache.keys() if key != _METADATA_KEY)
    metadata = cache.get(_METADATA_KEY)
    last_refresh = (
        metadata.get(_METADATA_LAST_REFRESH)
        if isinstance(metadata, dict)
        else _now_utc().isoformat()
    )
    logger.info(
        "Updated news cache for %d assets at %s",
        asset_count,
        last_refresh,
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    logging.basicConfig(level=logging.INFO)
    main()
