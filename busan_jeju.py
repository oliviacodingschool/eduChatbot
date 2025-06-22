import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os
import json

# 페이지 설정
st.set_page_config(page_title="제주AI챗봇")
st.title("🌱 4학년1반 제주 AI챗봇!")
st.markdown("<h3 style='color:#28a745;'>제주도의 지리정보를 알려드려요!</h3>", unsafe_allow_html=True)

# 디바이스 설정 (GPU 우선)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = SentenceTransformer('jhgan/ko-sbert-sts')
    model.to(torch.device(DEVICE))
    return model

model = load_model()

# 지식 데이터 로딩
with open("jeju_busan_json.txt", "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)

if "history" not in st.session_state:
    st.session_state["history"] = []

# 버튼 나란히
col1, col2 = st.columns([1, 1])
with col1:
    질문하기 = st.button("질문하기")
with col2:
    if st.button("초기화"):
        st.session_state["history"] = []
        st.success("기록이 초기화되었습니다!")

# 질문 입력
user_input = st.text_input("무엇이 궁금한가요?")

# FAISS 인덱스 구축
def build_faiss_index(data):
    search_sentences = [d["search"] for d in data]
    embeddings = model.encode(search_sentences, convert_to_numpy=True, device=DEVICE)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, data

# 질문 처리
if 질문하기 and user_input:
    if "1인당" in user_input and "온실가스" in user_input:
        matched_answer = knowledge_data[0]["full"]
    else:
        index, data_list = build_faiss_index(knowledge_data)
        query_vec = model.encode([user_input], convert_to_numpy=True, device=DEVICE)
        D, I = index.search(np.array(query_vec), k=1)

        distance = D[0][0]
        if distance > 500.0:
            matched_answer = "잘 이해되지 않아요. 다시 질문해 주세요!"
        else:
            matched_answer = data_list[I[0][0]]["full"]

    # 챗봇 답변 스타일링 출력
    answer_html = f"""
    <div style="
        border: 1.5px solid #87ceeb;
        border-radius: 10px;
        padding: 15px;
        max-width: 700px;
        font-size: 16px;
        line-height: 1.4;
        white-space: pre-wrap;
    ">
        <span>💡 <strong>챗봇:</strong></span>
        <div style="margin-left: 3.5em;">
            {matched_answer}
        </div>
    </div>
    """
    st.markdown(answer_html, unsafe_allow_html=True)

    st.session_state["history"].insert(0, (user_input, matched_answer))

# 이전 질문 기록
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("📜 이전 질문 기록")
    for idx, (prev_q, prev_a) in enumerate(st.session_state["history"], 1):
        with st.expander(f"Q{idx}: {prev_q}", expanded=False):
            st.markdown(prev_a)
