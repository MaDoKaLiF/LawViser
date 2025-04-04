import streamlit as st
from embedding import EmbeddingHandler
from model import ModelHandler
from generation import AnswerGenerator

st.set_page_config(page_title="이용약관 → 관련 법률 매칭기", page_icon="📘")

st.title("📘 이용약관 관련 법률 검색기")

# 사용자 입력
clause = st.text_area("📄 이용약관 조항 입력", height=200)

db_option = st.selectbox("💽 사용할 법률 DB", ["subscribe", "housing"])

if st.button("🔍 분석 시작") and clause.strip():
    # 경로 설정
    db_folder = f"./{db_option}_db"
    faiss_path = f"{db_folder}/faiss_index.bin"
    ref_path = f"{db_folder}/document_references.pkl"

    # 모델 및 핸들러
    embedder = EmbeddingHandler()
    model = ModelHandler()
    agent = AnswerGenerator(
        model_handler=model,
        embedding_handler=embedder,
        faiss_index_path=faiss_path,
        document_reference_path=ref_path
    )

    # 분석 실행
    result = agent.find_related_laws(clause)

    st.markdown("### 📚 관련된 법률 조항")
    for i, law in enumerate(result["related_laws"], 1):
        st.markdown(f"**{i}.** {law}")
