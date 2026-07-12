from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value or ""


def _get_int(name: str, default: int) -> int:
    return int(_get_env(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(_get_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    groq_api_key: str
    news_api_key: str
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_admin_db: str
    embedding_model: str
    groq_model: str
    llm_temperature: float
    pgvector_collection: str
    chat_history_table: str
    retriever_k: int
    retriever_fetch_k: int
    chunk_size: int
    chunk_overlap: int
    request_timeout: int
    egypt_gold_price_url: str
    yahoo_gold_ticker: str
    global_news_query: str
    local_gold_premium_factor: float
    cache_file: Path

    @property
    def sqlalchemy_connection_uri(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def psycopg_connection_uri(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def admin_connection_uri(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_admin_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cache_dir = BASE_DIR / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        openai_api_key=_get_env("OPENAI_API_KEY", required=False),
        groq_api_key=_get_env("GROQ_API_KEY", required=False),
        news_api_key=_get_env("NEWS_API_KEY", required=False),
        postgres_host=_get_env("POSTGRES_HOST", "localhost"),
        postgres_port=_get_int("POSTGRES_PORT", 5432),
        postgres_db=_get_env("POSTGRES_DB", "financial_ai"),
        postgres_user=_get_env("POSTGRES_USER", "postgres"),
        postgres_password=_get_env("POSTGRES_PASSWORD", "postgres"),
        postgres_admin_db=_get_env("POSTGRES_ADMIN_DB", "postgres"),
        embedding_model=_get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        groq_model=_get_env("GROQ_MODEL", "llama-3.3-70b-versatile"),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.1),
        pgvector_collection=_get_env("PGVECTOR_COLLECTION", "financial_market_knowledge"),
        chat_history_table=_get_env("CHAT_HISTORY_TABLE", "financial_chat_history"),
        retriever_k=_get_int("RETRIEVER_K", 6),
        retriever_fetch_k=_get_int("RETRIEVER_FETCH_K", 18),
        chunk_size=_get_int("CHUNK_SIZE", 900),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
        request_timeout=_get_int("REQUEST_TIMEOUT", 20),
        egypt_gold_price_url=_get_env("EGYPT_GOLD_PRICE_URL", "https://www.isagha.com/prices"),
        yahoo_gold_ticker=_get_env("YAHOO_GOLD_TICKER", "GC=F"),
        global_news_query=_get_env(
            "GLOBAL_NEWS_QUERY",
            '(gold OR forex OR inflation OR "emerging markets" OR Egypt)',
        ),
        local_gold_premium_factor=_get_float("LOCAL_GOLD_PREMIUM_FACTOR", 1.04),
        cache_file=cache_dir / "latest_market_snapshot.json",
    )
