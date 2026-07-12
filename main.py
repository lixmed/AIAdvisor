from __future__ import annotations

import argparse
import logging
import sys
import uuid

from src.database import initialize_database
from src.ingestion import ingest_market_data
from src.rag_chain import answer_question


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("financial_ai_agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Financial AI Chatbot for Egyptian and global markets.",
    )
    parser.add_argument(
        "--session-id",
        default=str(uuid.uuid4()),
        help="Persistent chat session identifier.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip refreshing market data before starting the chat loop.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Refresh market data and exit without opening the chat loop.",
    )
    return parser.parse_args()


def refresh_knowledge_base() -> None:
    result = ingest_market_data()
    print(
        "تم تحديث قاعدة المعرفة بنجاح. "
        f"عدد المستندات الأساسية: {result['base_documents']} | "
        f"عدد المقاطع المخزنة: {result['chunked_documents']}"
    )


def chat_loop(session_id: str) -> None:
    print("\nمرحباً بك في المساعد المالي الذكي.")
    print(f"معرّف الجلسة الحالي: {session_id}")
    print("اكتب سؤالك المالي، أو اكتب /ingest لتحديث البيانات، أو exit للخروج.\n")

    while True:
        try:
            question = input("أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nتم إنهاء الجلسة.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "خروج"}:
            print("تم إنهاء الجلسة.")
            break
        if question == "/ingest":
            try:
                refresh_knowledge_base()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ingestion failed")
                print(f"تعذر تحديث البيانات حالياً: {exc}")
            continue

        try:
            response = answer_question(question=question, session_id=session_id)
            print(f"\nالمستشار المالي: {response}\n")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chat invocation failed")
            print(f"\nحدث خطأ أثناء معالجة السؤال: {exc}\n")


def main() -> int:
    args = parse_args()

    try:
        initialize_database()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Database initialization failed")
        print(f"تعذر تهيئة قاعدة البيانات: {exc}")
        return 1

    if not args.skip_ingestion:
        try:
            refresh_knowledge_base()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Initial ingestion failed")
            print(
                "فشل تحديث البيانات عند التشغيل. "
                "يمكنك المتابعة بالمخزون الموجود أو تشغيل /ingest لاحقاً.\n"
                f"سبب الخطأ: {exc}"
            )

    if args.ingest_only:
        return 0

    chat_loop(session_id=args.session_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

