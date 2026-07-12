from __future__ import annotations

import json
from operator import itemgetter
from typing import List

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import get_settings
from src.database import get_session_history, get_vectorstore


class TickerExtraction(BaseModel):
    tickers: List[str] = Field(description="List of stock ticker symbols extracted from the question")


def fetch_predictions_for_question(question: str, llm: ChatGroq) -> str:
    """Extracts tickers from the query and fetches predictions from the local API."""
    try:
        extractor = llm.with_structured_output(TickerExtraction)
        result = extractor.invoke(question)
        if not result.tickers:
            return ""
        
        response = requests.post(
            "http://localhost:8000/predict", 
            json={"tickers": result.tickers},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return json.dumps(data.get("predictions", []), ensure_ascii=False, indent=2)
        else:
            return ""
    except Exception as e:
        return ""


def _get_live_market_data() -> str:
    """Read the latest market snapshot from cache to guarantee the LLM has exact numbers."""
    from src.config import get_settings
    import json
    settings = get_settings()
    if not settings.cache_file.exists():
        return "الأسعار المباشرة غير متوفرة حالياً."
    
    try:
        data = json.loads(settings.cache_file.read_text(encoding="utf-8"))
        fx = data.get("fx_rates", {}).get("value", {})
        gold = data.get("egyptian_gold_prices", {}).get("value", {})
        
        usd_price = fx.get("USD", {}).get("egp_per_unit", "غير متوفر")
        eur_price = fx.get("EUR", {}).get("egp_per_unit", "غير متوفر")
        gold_24k = gold.get("24k", "غير متوفر")
        gold_21k = gold.get("21k", "غير متوفر")
        gold_pound = gold.get("gold_pound", "غير متوفر")
        
        return (f"أسعار السوق الحالية والمؤكدة:\n"
                f"- الدولار الأمريكي: {usd_price} جنيه\n"
                f"- اليورو: {eur_price} جنيه\n"
                f"- جرام الذهب عيار 24: {gold_24k} جنيه\n"
                f"- جرام الذهب عيار 21: {gold_21k} جنيه\n"
                f"- الجنيه الذهب: {gold_pound} جنيه\n")
    except Exception:
        return "خطأ في قراءة الأسعار المباشرة."


def _format_documents(documents, question: str = "") -> str:
    live_prices = _get_live_market_data()
    
    formatted = [f"--- الأسعار المباشرة (استخدم هذه الأرقام حصراً إذا سُئلت عن السعر) ---\n{live_prices}\n--- أخبار وسياق إضافي ---"]
    for index, doc in enumerate(documents[:5], start=1):
        metadata = doc.metadata or {}
        formatted.append(f"[{index}] {metadata.get('asset', 'خبر')}: {doc.page_content}")
    
    return "\n\n".join(formatted)

def _build_llm() -> ChatGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is required to run the chat model.")
    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )


def build_rag_chain() -> RunnableWithMessageHistory:
    settings = get_settings()
    llm = _build_llm()
    retriever = get_vectorstore().as_retriever(
        search_type="mmr",  
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "filter": {
                "$or": [
                    {"asset": {"$in": ["USD", "EUR", "GBP", "GOLD_EGYPT", "FX_BASKET"]}},
                    {"asset_class": {"$in": ["fx", "gold"]}},
                    {"topic": {"$in": ["forex", "gold", "macro"]}}
                ]
            }  
        }
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """أجب مباشرة على سؤال المستخدم كمستشار مالي محترف. إياك أن تبدأ إجابتك بـ "أنا خبير مالي" أو "بناء على خبرتي" أو تشرح دورك. ادخل في صلب الموضوع فوراً.

        قواعد صارمة جداً (يجب الالتزام بها حرفياً):
        1. تحدث بثقة كإنسان طبيعي. إياك أن تذكر مصطلحات مثل (API، RAG، خوارزميات، قاعدة بيانات، النظام، التحليل الكمي، أو بيانات مرفقة).
        2. استخدم أسعار الذهب والدولار الموجودة في قسم (الأسعار المباشرة) حصراً. لا تقل أبداً "لا تتوفر لدي معلومات" لأن الأرقام موجودة أمامك بالفعل.
        3. إذا تم تزويدك بتوقعات لأسهم (مثل احتمالية الشراء ومستوى القناعة)، قم بصياغتها كأنها نابعة من تحليلك الشخصي للسهم. 
        4. ادمج التوقعات مع الوضع الاقتصادي (Guardrail). إذا كانت الأسواق العالمية تشهد تقلبات سلبية بناءً على الأخبار المرفقة، حذر المستخدم حتى لو كانت مؤشرات السهم إيجابية.
        5. تجاهل تماماً أي أخبار أو بيانات في السياق لا تتعلق بسؤال المستخدم مباشرة (تجنب الحشو).
        6. يجب أن تكون الإجابة باللغة العربية الفصحى السليمة 100%. إياك أن تستخدم كلمات إنجليزية وسط الكلام العربي (مثل diversify وغيرها).
        7. اختم دائماً بنصيحة استثمارية أو مالية قصيرة ومفيدة.
        """
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """سؤال المستخدم: {question}

    أخبار وأسعار السوق المتوفرة حالياً:
    {context}

    التحليل الكمي لسلوك الأسهم المطلوبة (فارغ إذا لم يطلب المستخدم أسهماً):
    {predictions}

    أعطني تحليلك الآن بناءً على المعطيات السابقة دون ذكر مصادرها:"""
        ),
    ])


    base_chain = (
        {
            "context": lambda x: _format_documents(
                retriever.invoke(x["question"]), 
                question=x["question"]  # ← نمرر السؤال عشان الـ fallback يشتغل
            ),
            "predictions": lambda x: fetch_predictions_for_question(x["question"], llm),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


def answer_question(question: str, session_id: str) -> str:
    chain = build_rag_chain()
    return chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )