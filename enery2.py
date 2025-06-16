import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os

st.set_page_config(page_title="재생에너지 AI 챗봇")

# CUDA 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로딩
@st.cache_resource
def load_model():
    model = SentenceTransformer('kykim/bert-kor-base')
    model.to(torch.device(DEVICE))
    return model

model = load_model()

def load_knowledge(file_path="energy2.txt"):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

sentences = load_knowledge()

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

# UI
st.title("🌱 재생에너지2 AI 챗봇")
st.markdown("<h3 style='color:#28a745;'>1인당 물 사용량, 온실가스, 탄소 배출량 등에 대해 알려드려요</h3>", unsafe_allow_html=True)

# 초기화 버튼
if st.button("초기화"):
    st.session_state["history"] = []
    st.success("기록이 초기화되었습니다!")

user_input = st.text_input("무엇이 궁금한가요?")

# FAISS 인덱스 구축
def build_faiss_index(sentences):
    embeddings = model.encode(sentences, convert_to_numpy=True, device=DEVICE)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences

# 질문 처리
if st.button("질문하기") and user_input:
    index, searchable_sentences = build_faiss_index(KNOWLEDGE)

    query_vec = model.encode([user_input], convert_to_numpy=True, device=DEVICE)
    D, I = index.search(np.array(query_vec), k=1)

    if D[0][0] > 500.0:
        matched_answer = "잘 이해되지 않아요. 다시 질문해 주세요!"
    else:
        matched_answer = searchable_sentences[I[0][0]]

    st.markdown(f"**챗봇:** {matched_answer}")
    st.session_state["history"].insert(0, (user_input, matched_answer))

# 이전 질문 기록
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("📜 이전 질문 기록")
    for idx, (prev_q, prev_a) in enumerate(st.session_state["history"], 1):
        with st.expander(f"Q{idx}: {prev_q}", expanded=False):
            st.markdown(prev_a)
