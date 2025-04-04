import streamlit as st
import os
import tempfile
import json

from tools.classifier import classifier
from RAG.rag import run_rag
from lm.final_answer import final_answer
from tools.parse_answer import parse_answer
from tools.email_module import send_email_naver

st.set_page_config(page_title="AI 이용약관 분석기", layout="wide")
st.title("📜 LawVisor: AI 이용약관 분석기")

st.markdown("""
<style>
.highlight-box {
    background-color: #f0f8ff;
    padding: 1.2em;
    border-left: 6px solid #4CAF50;
    border-radius: 8px;
    font-size: 16px;
    line-height: 1.6;
}
</style>

<div class="highlight-box">
📄 <strong>이용약관 자동 분석기</strong>는 다음의 기능을 제공합니다:

1️⃣ <strong>도메인 자동 분류</strong>  
→ 업로드한 약관이 <span style='color:green'><b>임대차</b></span> / <span style='color:blue'><b>구독 서비스</b></span> / <span style='color:gray'><b>기타</b></span> 중 어디에 속하는지 자동 판단합니다.

2️⃣ <strong>관련 법률 검색 (RAG)</strong>  
→ 약관의 각 조항에 대해 관련된 <b>법률 조항</b>을 검색하여 제공합니다.

3️⃣ <strong>유리/불리 판단</strong>  
→ 1번과 2번을 기반으로, 사용자의 입력 약관이 <b>소비자에게 유리한지 / 불리한지</b>를 판단합니다.

4️⃣ <strong>문의 메일 작성</strong>  
→ 3번의 결과를 바탕으로, 사용자에게 불리한 조항에 대해 <b>문의 메일</b>을 자동으로 작성할 수 있습니다.

</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("🔽 분석할 이용약관 파일을 업로드하세요 (PDF, TXT, PNG, JPG, JPEG)", type=["pdf", "txt", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # save input file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
    st.write("🚀 분석을 시작하려면 아래 버튼을 눌러주세요.")

    if st.button("🔍 분석 시작"):
        with st.spinner("Classifier 실행 중..."):
            classified_output_path = classifier(tmp_file_path)

        with st.spinner("RAG 실행 중..."):
            rag_output_path = run_rag(classified_output_path)

        with st.spinner("최종 답변 생성 중..."):
            final_response_file = final_answer(rag_output_path)
            parsed_response_file = parse_answer(final_response_file)

        # save state of results
        if os.path.exists(rag_output_path):
            with open(rag_output_path, "r", encoding="utf-8") as f:
                st.session_state['rag_results'] = json.load(f)
        else:
            st.error("❌ RAG 결과 파일을 찾을 수 없습니다.")

        if os.path.exists(parsed_response_file):
            with open(parsed_response_file, "r", encoding="utf-8") as f:
                st.session_state['parsed_results'] = json.load(f)
        else:
            st.error("❌ 최종 결과 파일을 찾을 수 없습니다.")

        st.session_state['rag_output_path'] = rag_output_path
        st.session_state['final_response_file'] = final_response_file
        st.session_state['parsed_response_file'] = parsed_response_file

# print results
if 'rag_results' in st.session_state and 'parsed_results' in st.session_state:
    st.success("✅ 분석 완료! 관련 법률 결과는 아래와 같습니다.")
    rag_items = st.session_state['rag_results'].get("results", [])
    parsed_results = st.session_state['parsed_results']

    for idx, (item, parsed_result) in enumerate(zip(rag_items, parsed_results)):
        parsed_item = parsed_result["responses"]
        st.markdown(f"### 📄 조항 {item['index'] + 1}")
        st.code(item['clause'], language="text")
        st.markdown(f"**📌 분석기의 판단:  {parsed_item[0]['label']}**")
        st.markdown(f"{parsed_item[0]['reason']}.")
        st.markdown("**📚 관련된 법률 조항:**")
        for i, law in enumerate(item.get("related_laws", []), 1):
            st.markdown(f"{i}. {law}")
        st.markdown("---")

    # sending mail UI
    st.write("🚀 관련 문의 메일을 보내려면 아래에 이메일을 입력하고 버튼을 눌러주세요.")
    recipient_email = st.text_input("📫 수신자 이메일을 입력하세요:")

    if st.button("📨 문의 시작"):
        with st.spinner("💌 메일 전송을 시도합니다..."):
            if recipient_email:
                final_path = st.session_state.get("final_response_file")
                parsed_path = st.session_state.get("parsed_response_file")
                rag_path = st.session_state.get("rag_output_path")

                if final_path and parsed_path:
                    send_email_naver(
                        recipient=recipient_email,
                        path1=parsed_path,
                        path2=rag_path
                    )
                    st.success(f"메일이 {recipient_email} 로 전송되었습니다!")
                else:
                    st.error("메일 전송에 필요한 파일이 없습니다. 먼저 분석을 완료해주세요.")
            else:
                st.warning("이메일 주소를 입력해주세요!")
        
