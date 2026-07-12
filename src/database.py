from __future__ import annotations

from functools import lru_cache

import psycopg
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_postgres import PGVector, PostgresChatMessageHistory

from src.config import get_settings


_CHAT_HISTORY_CACHE: dict[str, PostgresChatMessageHistory] = {}


def get_sync_connection() -> psycopg.Connection:
    settings = get_settings()
    connection = psycopg.connect(settings.psycopg_connection_uri, autocommit=True)
    return connection


def ensure_chat_history_schema() -> None:
    settings = get_settings()
    with get_sync_connection() as connection:
        PostgresChatMessageHistory.create_tables(connection, settings.chat_history_table)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:  
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model, 
        model_kwargs={"device": "cpu"}, 
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> PGVector:
    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.pgvector_collection,
        connection=settings.sqlalchemy_connection_uri,
        use_jsonb=True,
    )


def get_session_history(session_id: str) -> PostgresChatMessageHistory:
    if session_id not in _CHAT_HISTORY_CACHE:
        settings = get_settings()
        connection = get_sync_connection()
        _CHAT_HISTORY_CACHE[session_id] = PostgresChatMessageHistory(
            settings.chat_history_table,
            session_id,
            sync_connection=connection,
        )
    return _CHAT_HISTORY_CACHE[session_id]


def initialize_database() -> None:
    """Ensure the chat history table exists before the app starts."""
    ensure_chat_history_schema()