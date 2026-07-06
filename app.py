# app.py
import streamlit as st
import os
from groq import Groq
from rag_core import load_model, load_rag_data, rag_answer

st.set_page_config(page_title="ADsP RAG 챗봇", page_icon="📚")

@st.cache_resource
def init():
    model = load_model()
    chunks, embeddings = load_rag_data("adsp_rag")
    chunk_texts = [c["text"] for c in chunks]
    return model, chunks, embeddings, chunk_texts

model, chunks, chunk_embeddings, chunk_texts = init()

# [수정] 스트림릿 클라우드 보안 환경에 맞게 Secret 적용
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.title("📚 ADsP 기출문제 RAG 챗봇")
st.caption(f"로딩된 기출문제: {len(chunks)}개")
st.divider()

with st.sidebar:
    st.title("💡 질문 예시")
    examples = [
        "과적합이란 무엇인가요?",
        "의사결정나무의 장단점은?",
        "k-means 클러스터링이란?",
        "회귀분석과 분류의 차이는?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["preset"] = ex
            st.rerun()
    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력 및 토글 유지
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "chunks" in msg:
            with st.expander("📄 참고한 기출문제"):
                for c in msg["chunks"]:
                    st.markdown(f"**[{c['rank']}위]** 유사도: `{c['similarity']:.4f}`")
                    st.text(c["text"][:300] + "...")
                    st.divider()

# [수정] 입력 로직의 무한 루프 방지 처리
preset = st.session_state.pop("preset", None)
chat_input = st.chat_input("ADsP 관련 질문을 입력하세요...")
user_input = chat_input if chat_input else preset

if user_input:
    # 1. 유저 메시지 출력 및 저장
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. 어시스턴트 답변 생성 및 출력
    with st.chat_message("assistant"):
        with st.spinner("기출문제 검색 중..."):
            result = rag_answer(user_input, chunk_texts, chunk_embeddings, model, client)
        st.write(result["answer"])
        with st.expander("📄 참고한 기출문제"):
            for c in result["chunks"]:
                st.markdown(f"**[{c['rank']}위]** 유사도: `{c['similarity']:.4f}`")
                st.text(c["text"][:300] + "...")
                st.divider()

    # 3. 어시스턴트 메시지 저장 후 리런하여 상태 고정
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "chunks": result["chunks"],
    })
    st.rerun()
