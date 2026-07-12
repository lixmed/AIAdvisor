from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from psycopg import sql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_postgres import PostgresChatMessageHistory  # noqa: E402

from src.config import get_settings  # noqa: E402


SQL_COMMANDS = r"""
-- Run against an admin database such as postgres
CREATE DATABASE financial_ai;

-- Then connect to the target database
\c financial_ai;

CREATE EXTENSION IF NOT EXISTS vector;
"""


def database_exists(connection: psycopg.Connection, database_name: str) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
        return cursor.fetchone() is not None


def create_database_if_missing(connection: psycopg.Connection, database_name: str) -> None:
    if database_exists(connection, database_name):
        print(f"قاعدة البيانات `{database_name}` موجودة بالفعل.")
        return

    with connection.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    print(f"تم إنشاء قاعدة البيانات `{database_name}`.")


def enable_vector_extension(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("تم تفعيل امتداد `vector` بنجاح.")


def create_chat_history_table(connection: psycopg.Connection, table_name: str) -> None:
    PostgresChatMessageHistory.create_tables(connection, table_name)
    print(f"تم التأكد من وجود جدول الذاكرة `{table_name}`.")


def main() -> None:
    settings = get_settings()

    with psycopg.connect(settings.admin_connection_uri, autocommit=True) as admin_connection:
        create_database_if_missing(admin_connection, settings.postgres_db)

    with psycopg.connect(settings.psycopg_connection_uri, autocommit=True) as app_connection:
        enable_vector_extension(app_connection)
        create_chat_history_table(app_connection, settings.chat_history_table)

    print("\nأوامر SQL المرجعية:")
    print(SQL_COMMANDS.replace("financial_ai", settings.postgres_db))


if __name__ == "__main__":
    main()
