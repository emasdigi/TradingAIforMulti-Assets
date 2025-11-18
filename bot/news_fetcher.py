"""Utility function for fetching recent Google News articles.

This module mirrors the scraping logic from `bot/news_mcp.py` but exposes a
single convenience function that can be imported and used without relying on
the MCP server infrastructure.
"""

from __future__ import annotations

import json
import logging
import random
import textwrap
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

try:  # Support both package and standalone execution
    from config import config  # type: ignore
except ImportError:  # pragma: no cover - fallback for direct invocation
    import config  # type: ignore

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/101.0.4951.54 Safari/537.36"
)

_SUMMARY_MODEL = "openai/gpt-4o-mini"
_MAX_CONTENT_CHARS = 6000
_SENTIMENT_LABELS = {"positive", "neutral", "negative"}

_RELATIVE_TIME_KEYWORDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 7 * 86400,
}
def _normalize_article_time(raw_value: str) -> Optional[str]:
    """Normalize Google News relative times to absolute UTC timestamps."""
    if not raw_value:
        return None

    raw = raw_value.strip()
    if not raw:
        return None

    # Direct ISO8601 or RFC3339 style timestamp
    date_formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%b %d, %Y",
        "%b %d %Y",
    ]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue

    lowered = raw.lower()

    if lowered in {"yesterday"}:
        dt = datetime.now(timezone.utc) - timedelta(days=1)
        return dt.isoformat()

    if "ago" in lowered:
        amount_str, unit = None, None
        words = lowered.replace(",", "").split()
        for idx, word in enumerate(words):
            if word.isdigit():
                amount_str = word
                if idx + 1 < len(words):
                    unit_candidate = words[idx + 1].rstrip("s")
                    if unit_candidate in _RELATIVE_TIME_KEYWORDS:
                        unit = unit_candidate
                        break
        if amount_str and unit:
            seconds = int(amount_str) * _RELATIVE_TIME_KEYWORDS[unit]
            dt = datetime.now(timezone.utc) - timedelta(seconds=seconds)
            return dt.isoformat()

    return None


def _make_request(url: str, headers: Dict[str, str]) -> requests.Response:
    """Perform an HTTP GET with a small randomized delay to reduce detection."""
    # Random delay before each request to avoid rate limiting.
    time.sleep(random.uniform(2, 6))
    return requests.get(url, headers=headers, timeout=10)


def _select_result_cards(soup: BeautifulSoup) -> List[BeautifulSoup]:
    """Return candidate Google News result containers."""
    candidate_selectors = [
        "div.SoaBEf",              # legacy desktop layout
        "div.Gx5Zad",              # common card wrapper
        "div.MjjYud",              # updated layout wrapper
        "article.Gx5Zad",          # article tags seen in some locales
        "div.NiLAwe",              # alternate news cards
    ]

    for selector in candidate_selectors:
        results = soup.select(selector)
        if results:
            return results

    fallback = soup.find_all("article")
    return fallback if fallback else []


def _extract_article_text(html: bytes) -> str:
    """Attempt to extract readable text from an article HTML payload."""
    soup = BeautifulSoup(html, "html.parser")

    # Prefer explicit <article> tags when available.
    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    lines = []
    for paragraph in paragraphs:
        text = paragraph.get_text(separator=" ", strip=True)
        if text and len(text.split()) >= 5:
            lines.append(text)

    article_text = " ".join(lines)
    if not article_text:
        # As a last resort, try meta description
        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description and meta_description.get("content"):
            article_text = meta_description["content"]

    return article_text[:_MAX_CONTENT_CHARS]


def _fetch_article_content(url: str, headers: Dict[str, str]) -> str:
    """Fetch the full article content for a given URL."""
    try:
        response = _make_request(url, headers)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch article content (%s): %s", url, exc)
        return ""

    try:
        return _extract_article_text(response.content)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse article content (%s): %s", url, exc)
        return ""


def _get_openrouter_client() -> Optional[OpenAI]:
    """Return an OpenRouter-enabled OpenAI client if credentials exist."""
    api_key = getattr(config, "OPENROUTER_API_KEY", "")
    base_url = getattr(config, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        logger.info("OpenRouter API key not configured; skipping summarization.")
        return None

    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as exc:  # pragma: no cover - instantiation failure
        logger.error("Failed to initialize OpenRouter client: %s", exc, exc_info=True)
        return None


def _summarize_and_score(
    client: OpenAI,
    *,
    query: str,
    article_text: str,
    source: str,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Summarize the article and provide a coarse sentiment label."""
    if not article_text:
        return None, None, None

    prompt = textwrap.dedent(
        f"""
        You are a financial news analyst. Summarize the following article in no more than 80 words
        with a focus on relevance to the query "{query}".

        Sentiment guidance (positive, neutral, or negative):
        - Evaluate strictly from the perspective of an investor in "{query}".
        - Treat developments that strengthen competitors, diminish pricing power, or otherwise
          increase risks for "{query}" as NEGATIVE.
        - Treat developments that enhance "{query}"'s outlook, profitability, or competitive
          advantage as POSITIVE.
        - If the impact is unclear or balanced, label it NEUTRAL.

        Provide a confidence score between 0 and 1 reflecting how clearly the article supports
        the sentiment label.

        Return ONLY valid JSON with these fields:
        {{
            "summary": "...",
            "sentiment": "positive|neutral|negative",
            "confidence": 0.0
        }}

        Article source: {source}
        Article content:
        {article_text}
        """
    ).strip()

    try:
        response = client.chat.completions.create(
            model=_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise financial news summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.error("Failed to summarize article via OpenRouter: %s", exc, exc_info=True)
        return None, None, None

    message = response.choices[0].message
    payload = getattr(message, "parsed", None)

    if payload is None:
        raw_content = message.content
        normalized: Optional[str] = None
        if raw_content:
            if isinstance(raw_content, list):
                normalized = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_content
                ).strip()
            else:
                normalized = str(raw_content).strip()

        if normalized:
            try:
                payload = json.loads(normalized)
            except json.JSONDecodeError:
                logger.warning(
                    "Summarization response not valid JSON; skipping. Raw content: %.120s",
                    normalized,
                )
                return None, None, None
        else:
            logger.warning("Summarization response missing payload and content; skipping.")
            return None, None, None

    if not isinstance(payload, dict):
        logger.warning("Summarization payload unexpected type %s; skipping.", type(payload))
        return None, None, None

    summary = str(payload.get("summary", "")).strip() or None
    sentiment = str(payload.get("sentiment", "")).lower()
    confidence_raw = payload.get("confidence")

    if sentiment not in _SENTIMENT_LABELS:
        sentiment = None

    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = None

    return summary, sentiment, confidence


def search_google_news(
    query: str,
    start_date: str,
    end_date: str,
    max_results: int = 3,
    *,
    country: str = "US",
    language: str = "en-US",
) -> List[Dict[str, Any]]:
    """Fetch recent Google News articles for a given query and date range.

    Args:
        query: Search query string (e.g., "Apple Stock").
        start_date: Inclusive start date in ``YYYY-MM-DD`` format.
        end_date: Inclusive end date in ``YYYY-MM-DD`` format.
        max_results: Maximum number of articles to return (defaults to 3).
        country: Two-letter country code used to bias results (defaults to ``US``).
        language: Locale code (e.g., ``en-US``) to request localized content.

    Returns:
        A list of dictionaries containing ``link``, ``title``, ``snippet``,
        ``date``, ``source``, ``content`` (full article text when available),
        ``summary`` (OpenRouter GPT-4o mini summary), ``sentiment`` and
        ``sentiment_confidence`` fields. The list length will not exceed
        ``max_results``.
    """

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    start_date_str = start_date_obj.strftime("%m/%d/%Y")
    end_date_str = end_date_obj.strftime("%m/%d/%Y")

    primary_lang = language.split("-")[0]
    accept_language = f"{language},{primary_lang};q=0.9"

    headers = {
        "User-Agent": _DEFAULT_USER_AGENT,
        "Accept-Language": accept_language,
    }

    news_results: List[Dict[str, Any]] = []
    page = 0
    max_pages = 0  # Only fetch the first page by default.

    llm_client = _get_openrouter_client()

    while page <= max_pages and len(news_results) < max_results:
        encoded_query = quote_plus(query)
        url = (
            "https://www.google.com/search?"
            f"q={encoded_query}"
            f"&tbs=cdr:1,cd_min:{start_date_str},cd_max:{end_date_str},lr:lang_1{primary_lang}"
            f"&tbm=nws&start={page * 10}"
            f"&hl={language}"
            f"&gl={country}"
            "&num=10"
        )

        try:
            response = _make_request(url, headers)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Request error while fetching Google News: %s", exc)
            break

        soup = BeautifulSoup(response.content, "html.parser")
        results_on_page = _select_result_cards(soup)

        if not results_on_page:
            break

        for element in results_on_page:
            if len(news_results) >= max_results:
                break

            try:
                link_tag = element.find("a", href=True)
                title_tag = (
                    element.select_one("div.MBeuO")
                    or element.select_one("h3")
                    or element.select_one("span")
                )
                snippet_tag = (
                    element.select_one(".GI74Re")
                    or element.select_one(".MUxGbd")
                    or element.find("p")
                )
                date_tag = element.select_one(".LfVVr") or element.find("time")
                source_tag = (
                    element.select_one(".NUnG9d span")
                    or element.select_one(".vr1PYe")
                    or element.select_one(".SVJrMe")
                )

                link = link_tag["href"] if link_tag else None
                title = title_tag.get_text(strip=True) if title_tag else None
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                raw_date = ""
                if date_tag and hasattr(date_tag, "get_text"):
                    raw_date = date_tag.get_text(strip=True)
                if not raw_date and date_tag and hasattr(date_tag, "get"):
                    raw_date = date_tag.get("datetime", "")
                published_at = _normalize_article_time(raw_date)
                source = source_tag.get_text(strip=True) if source_tag else ""

                if not link or not title:
                    raise ValueError("Missing link or title")
            except (AttributeError, TypeError, KeyError, ValueError) as exc:
                logger.warning("Error parsing a Google News result: %s", exc)
                continue

            link = link.replace("/url?esrc=s&q=&rct=j&sa=U&url=", "")
            content = _fetch_article_content(link, headers)
            summary = None
            sentiment = None
            confidence = None

            if llm_client:
                summary_input = ""
                if isinstance(content, str) and content.strip():
                    summary_input = f"{title}\n\n{content}" if title else content
                elif title:
                    summary_input = title

                if summary_input:
                    summary, sentiment, confidence = _summarize_and_score(
                        llm_client,
                        query=query,
                        article_text=summary_input,
                        source=source,
                    )

            news_results.append(
                {
                    "link": link,
                    "title": title,
                    "snippet": snippet,
                    "date": published_at or raw_date,
                    "published_at": published_at,
                    "raw_date": raw_date,
                    "source": source,
                    "content": content,
                    "summary": summary,
                    "sentiment": sentiment,
                    "sentiment_confidence": confidence,
                }
            )

        # The default behaviour mirrors the original implementation: do not
        # paginate beyond the first page. If this ever changes, adjust
        # ``max_pages`` accordingly.
        break

    return news_results[:max_results]


__all__ = ["search_google_news"]

if __name__ == "__main__":  # pragma: no cover - convenience manual run
    # results = search_google_news(
    #     query="Bitcoin",
    #     start_date="2025-11-09",
    #     end_date="2025-11-10",
    #     max_results=3,
    # )

    results = search_google_news(
        query="TLKM",
        start_date="2025-11-10",
        end_date="2025-11-11",
        max_results=3,
        country="ID",
        language="id-ID",
    )

    print(results)