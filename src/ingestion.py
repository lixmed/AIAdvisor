from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.database import get_vectorstore


logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_cache() -> dict[str, Any]:
    settings = get_settings()
    if not settings.cache_file.exists():
        return {}
    try:
        return json.loads(settings.cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Cache file is corrupted. Starting with an empty cache.")
        return {}


def _save_cache(payload: dict[str, Any]) -> None:
    settings = get_settings()
    settings.cache_file.parent.mkdir(parents=True, exist_ok=True)
    settings.cache_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _upsert_cache_section(section: str, value: Any) -> None:
    cache = _load_cache()
    cache[section] = {
        "saved_at": _utc_now().isoformat(),
        "value": value,
    }
    _save_cache(cache)


def _get_cached_section(section: str) -> Any | None:
    cache = _load_cache()
    return cache.get(section, {}).get("value")


def _http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            )
        }
    )
    return session


def _latest_close(ticker: str) -> float | None:
    try:
        history = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if history.empty:
            return None
        close_series = history["Close"].dropna()
        if close_series.empty:
            return None
        return float(close_series.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to fetch ticker %s: %s", ticker, exc)
        return None


def _normalize_numeric_string(value: str) -> float:
    cleaned = (
        value.replace(",", "")
        .replace("،", "")
        .replace("جنيه", "")
        .replace("EGP", "")
        .strip()
    )
    return float(cleaned)


def _extract_price(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return _normalize_numeric_string(match.group(1))
            except ValueError:
                continue
    return None


def fetch_fx_rates() -> dict[str, dict[str, Any]]:
    snapshot_date = _utc_now().date().isoformat()
    direct_pairs = {
        "USD": ["USDEGP=X", "EGP=X"],
        "EUR": ["EUREGP=X"],
        "GBP": ["GBPEGP=X"],
        "SAR": ["SAREGP=X"],
        "AED": ["AEDEGP=X"],
    }
    cross_pairs = {
        "EUR": "EURUSD=X",
        "GBP": "GBPUSD=X",
        "SAR": "SARUSD=X",
        "AED": "AEDUSD=X",
    }

    usd_egp = None
    usd_ticker = None
    for ticker in direct_pairs["USD"]:
        usd_egp = _latest_close(ticker)
        if usd_egp:
            usd_ticker = ticker
            break

    if not usd_egp:
        raise RuntimeError("Unable to fetch USD/EGP rate from Yahoo Finance.")

    rates: dict[str, dict[str, Any]] = {
        "USD": {
            "currency": "USD",
            "egp_per_unit": round(usd_egp, 4),
            "source_ticker": usd_ticker,
            "retrieved_at": _utc_now().isoformat(),
            "date": snapshot_date,
            "calculation_method": "direct",
        }
    }

    for currency in ("EUR", "GBP", "SAR", "AED"):
        direct_value = None
        direct_ticker = None
        for ticker in direct_pairs[currency]:
            direct_value = _latest_close(ticker)
            if direct_value:
                direct_ticker = ticker
                break

        if direct_value:
            rates[currency] = {
                "currency": currency,
                "egp_per_unit": round(direct_value, 4),
                "source_ticker": direct_ticker,
                "retrieved_at": _utc_now().isoformat(),
                "date": snapshot_date,
                "calculation_method": "direct",
            }
            continue

        cross_value = _latest_close(cross_pairs[currency])
        if not cross_value:
            raise RuntimeError(f"Unable to compute {currency}/EGP from Yahoo Finance.")

        rates[currency] = {
            "currency": currency,
            "egp_per_unit": round(cross_value * usd_egp, 4),
            "source_ticker": cross_pairs[currency],
            "retrieved_at": _utc_now().isoformat(),
            "date": snapshot_date,
            "calculation_method": "cross_via_usd",
        }

    _upsert_cache_section("fx_rates", rates)
    return rates


def fetch_global_gold_price() -> dict[str, Any]:
    settings = get_settings()
    price = _latest_close(settings.yahoo_gold_ticker)
    if not price:
        raise RuntimeError("Unable to fetch global gold futures price.")

    payload = {
        "ticker": settings.yahoo_gold_ticker,
        "gold_usd_per_ounce": round(price, 2),
        "retrieved_at": _utc_now().isoformat(),
    }
    _upsert_cache_section("global_gold_price", payload)
    return payload


def fetch_egyptian_gold_prices(fx_rates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    settings = get_settings()
    session = _http_session()
    timestamp = _utc_now().isoformat()

    try:
        response = session.get(settings.egypt_gold_price_url, timeout=settings.request_timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(soup.stripped_strings)

        prices = {
            "24k": _extract_price(
                text,
                [
                    r"عيار\s*24[^0-9]{0,25}([\d,]+(?:\.\d+)?)",
                    r"24k[^0-9]{0,25}([\d,]+(?:\.\d+)?)",
                ],
            ),
            "21k": _extract_price(
                text,
                [
                    r"عيار\s*21[^0-9]{0,25}([\d,]+(?:\.\d+)?)",
                    r"21k[^0-9]{0,25}([\d,]+(?:\.\d+)?)",
                ],
            ),
            "18k": _extract_price(
                text,
                [
                    r"عيار\s*18[^0-9]{0,25}([\d,]+(?:\.\d+)?)",
                    r"18k[^0-9]{0,25}([\d,]+(?:\.\d+)?)",
                ],
            ),
            "gold_pound": _extract_price(
                text,
                [
                    r"الجنيه\s*الذهب[^0-9]{0,25}([\d,]+(?:\.\d+)?)",
                    r"gold\s*pound[^0-9]{0,25}([\d,]+(?:\.\d+)?)",
                ],
            ),
        }

        if not prices["21k"] or not prices["24k"]:
            raise RuntimeError("Gold scraping did not return the required karat prices.")

        if not prices["18k"]:
            prices["18k"] = round(prices["24k"] * 18 / 24, 2)
        if not prices["gold_pound"]:
            prices["gold_pound"] = round(prices["21k"] * 8, 2)

        payload = {
            **prices,
            "currency": "EGP",
            "source": "local_scrape",
            "source_url": settings.egypt_gold_price_url,
            "retrieved_at": timestamp,
            "date": _utc_now().date().isoformat(),
            "confidence": "high",
        }
        _upsert_cache_section("egyptian_gold_prices", payload)
        return payload

    except Exception as exc:  # noqa: BLE001
        logger.warning("Local Egyptian gold scraping failed: %s", exc)

    cached = _get_cached_section("egyptian_gold_prices")
    if cached:
        cached["source"] = "cached_last_known_price"
        cached["fallback_reason"] = "local_scrape_failed"
        cached["retrieved_at"] = timestamp
        cached["confidence"] = "medium"
        return cached

    global_gold = fetch_global_gold_price()
    usd_egp = fx_rates["USD"]["egp_per_unit"]
    pure_24k_per_gram = (global_gold["gold_usd_per_ounce"] * usd_egp) / 31.1034768
    localized_24k = round(pure_24k_per_gram * settings.local_gold_premium_factor, 2)

    derived = {
        "24k": localized_24k,
        "21k": round(localized_24k * 21 / 24, 2),
        "18k": round(localized_24k * 18 / 24, 2),
        "gold_pound": round(localized_24k * 21 / 24 * 8, 2),
        "currency": "EGP",
        "source": "derived_from_global_gold_and_usd_egp",
        "source_url": settings.egypt_gold_price_url,
        "retrieved_at": timestamp,
        "date": _utc_now().date().isoformat(),
        "confidence": "estimated",
        "fallback_reason": "local_scrape_failed_no_cache",
    }
    return derived


def fetch_top_financial_news(limit: int = 8) -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.news_api_key:
        logger.warning("NEWS_API_KEY is not configured. News ingestion will be skipped.")
        return []

    params = {
        "q": settings.global_news_query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": limit,
        "apiKey": settings.news_api_key,
    }

    try:
        response = _http_session().get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        normalized = [
            {
                "title": article.get("title", "").strip(),
                "description": (article.get("description") or "").strip(),
                "source": (article.get("source") or {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", _utc_now().isoformat()),
            }
            for article in articles
            if article.get("title")
        ]
        _upsert_cache_section("financial_news", normalized)
        return normalized
    except Exception as exc:  # noqa: BLE001
        logger.warning("News fetch failed: %s", exc)
        return _get_cached_section("financial_news") or []


def _build_fx_documents(fx_rates: dict[str, dict[str, Any]]) -> list[Document]:
    summary_lines = [
        "ملخص أحدث أسعار صرف الجنيه المصري مقابل العملات الرئيسية:",
    ]
    documents: list[Document] = []

    arabic_names = {
        "USD": "الدولار الأمريكي",
        "EUR": "اليورو",
        "GBP": "الجنيه الإسترليني",
        "SAR": "الريال السعودي",
        "AED": "الدرهم الإماراتي",
    }

    for code, payload in fx_rates.items():
        summary_lines.append(f"- {arabic_names[code]}: {payload['egp_per_unit']:.4f} جنيه لكل 1 {code}")
        documents.append(
            Document(
                page_content=(
                    f"سعر صرف {arabic_names[code]} مقابل الجنيه المصري يبلغ "
                    f"{payload['egp_per_unit']:.4f} جنيه لكل وحدة واحدة من {code}. "
                    f"تاريخ القراءة: {payload['date']}. "
                    f"طريقة الحساب: {payload['calculation_method']}. "
                    f"رمز المصدر: {payload['source_ticker']}."
                ),
                metadata={
                    "asset": code,
                    "asset_class": "fx",
                    "country": "Egypt",
                    "date": payload["date"],
                    "source": "yfinance",
                    "ticker": payload["source_ticker"],
                },
            )
        )

    documents.append(
        Document(
            page_content="\n".join(summary_lines),
            metadata={
                "asset": "FX_BASKET",
                "asset_class": "fx",
                "country": "Egypt",
                "date": next(iter(fx_rates.values()))["date"],
                "source": "yfinance",
            },
        )
    )
    return documents


def _build_gold_documents(gold_prices: dict[str, Any], fx_rates: dict[str, dict[str, Any]]) -> list[Document]:
    global_gold = _get_cached_section("global_gold_price") or {}
    gold_source = gold_prices["source"]
    date = gold_prices["date"]
    usd_egp = fx_rates["USD"]["egp_per_unit"]

    content = (
        "أحدث أسعار الذهب في مصر بالجنيه المصري للجرام الواحد أو للوحدة القياسية:\n"
        f"- عيار 24: {gold_prices['24k']:.2f} جنيه\n"
        f"- عيار 21: {gold_prices['21k']:.2f} جنيه\n"
        f"- عيار 18: {gold_prices['18k']:.2f} جنيه\n"
        f"- الجنيه الذهب: {gold_prices['gold_pound']:.2f} جنيه\n"
        f"- سعر الدولار المرجعي مقابل الجنيه: {usd_egp:.4f}\n"
        f"- مصدر البيانات: {gold_source}\n"
        f"- مستوى الثقة: {gold_prices.get('confidence', 'unknown')}\n"
        f"- تاريخ القراءة: {date}\n"
    )
    if global_gold:
        content += f"- سعر الذهب العالمي المرجعي: {global_gold['gold_usd_per_ounce']:.2f} دولار للأوقية\n"

    return [
        Document(
            page_content=content,
            metadata={
                "asset": "GOLD_EGYPT",
                "asset_class": "gold",
                "country": "Egypt",
                "date": date,
                "source": gold_source,
            },
        )
    ]


def _build_news_documents(news_items: list[dict[str, Any]]) -> list[Document]:
    documents: list[Document] = []
    for item in news_items:
        headline = item["title"]
        lower_headline = headline.lower()
        topic = "macro"
        if "gold" in lower_headline:
            topic = "gold"
        elif "forex" in lower_headline or "currency" in lower_headline or "dollar" in lower_headline:
            topic = "forex"

        documents.append(
            Document(
                page_content=(
                    "خبر اقتصادي عالمي قد يؤثر في الأسواق الناشئة والذهب والعملات.\n"
                    f"العنوان: {item['title']}\n"
                    f"الملخص: {item['description'] or 'لا يوجد وصف مختصر متاح.'}\n"
                    f"المصدر: {item['source']}\n"
                    f"تاريخ النشر: {item['published_at']}\n"
                    f"الرابط: {item['url']}"
                ),
                metadata={
                    "asset": topic.upper(),
                    "asset_class": "news",
                    "country": "Global",
                    "date": item["published_at"][:10],
                    "source": item["source"],
                    "topic": topic,
                    "url": item["url"],
                },
            )
        )
    return documents


def build_market_documents() -> list[Document]:
    fx_rates = fetch_fx_rates()
    gold_prices = fetch_egyptian_gold_prices(fx_rates)
    news_items = fetch_top_financial_news()

    documents: list[Document] = []
    documents.extend(_build_fx_documents(fx_rates))
    documents.extend(_build_gold_documents(gold_prices, fx_rates))
    documents.extend(_build_news_documents(news_items))
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "، ", " "],
    )
    return splitter.split_documents(documents)


def _document_id(doc: Document, index: int) -> str:
    metadata_string = json.dumps(doc.metadata, ensure_ascii=False, sort_keys=True)
    fingerprint = hashlib.sha1(
        f"{doc.page_content}|{metadata_string}|{index}".encode("utf-8")
    ).hexdigest()
    return fingerprint


def ingest_market_data() -> dict[str, Any]:
    documents = build_market_documents()
    if not documents:
        raise RuntimeError("No documents were produced by the ingestion pipeline.")

    chunked_documents = chunk_documents(documents)
    ids = [_document_id(doc, index) for index, doc in enumerate(chunked_documents)]
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunked_documents, ids=ids)

    result = {
        "base_documents": len(documents),
        "chunked_documents": len(chunked_documents),
        "ingested_at": _utc_now().isoformat(),
    }
    logger.info("Ingestion complete: %s", result)
    return result

