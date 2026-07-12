import streamlit as st
from src.rag_chain import build_rag_chain
from src.ingestion import ingest_market_data
from src.database import initialize_database
import uuid 
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# إعدادات الصفحة
st.set_page_config(
    page_title="المستشار المالي الذكي",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# العنوان والوصف
st.title("Financial AI Agent")
st.markdown("""
مرحباً بك في مساعدك المالي الشخصي للأسواق المصرية والعالمية.
اسأل عن **الذهب، الدولار، اليورو، أو أي أخبار اقتصادية** وسأقوم بتحليلها لك بناءً على أحدث البيانات.
""")

# تهيئة الداتابيز مرة واحدة عند بدء التشغيل
@st.cache_resource
def init_db():
    try:
        initialize_database()
        return True
    except Exception as e:
        st.error(f"⚠️ تحذير: لم يتم الاتصال بقاعدة البيانات: {e}")
        return False

# تشغيل التهيئة
db_ready = init_db()

# الشريط الجانبي للتحكم
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # زر تحديث البيانات
    if st.button("🔄 تحديث بيانات السوق الآن", use_container_width=True):
        with st.spinner("جاري جلب أسعار الذهب والعملات والأخبار..."):
            try:
                result = ingest_market_data()
                st.success(f"✅ تم التحديث: {result['chunked_documents']} معلومة جديدة")
            except Exception as e:
                st.error(f"❌ خطأ في التحديث: {e}")
    
    st.divider()
    
    # معلومات الجلسة
    st.info("💡 البوت يتذكر سياق محادثتك. ابدأ سؤالك بـ 'ما رأيك في...' للحصول على تحليل أعمق.")
    


# إدارة حالة الشات (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    # بناء الـ Chain مرة واحدة وتخزينها في الذاكرة
    st.session_state.chain = build_rag_chain()

# عرض الرسائل السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# مدخلات المستخدم
if prompt := st.chat_input("اسأل عن سعر الذهب، الدولار، أو أي خبر اقتصادي..."):
    # 1. عرض رسالة المستخدم
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. توليد الرد
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳")
        
        try:
           
            response = st.session_state.chain.invoke(
                {"question": prompt},
                config={"configurable": {"session_id": str(uuid.uuid4())}}
            )
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"⚠️ حدث خطأ أثناء المعالجة: {e}")
            message_placeholder.markdown("عذراً، واجهت مشكلة تقنية. يرجى المحاولة لاحقاً.")