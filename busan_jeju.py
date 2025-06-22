import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os

# 페이지 설정
st.set_page_config(page_title="제주AI챗봇")
st.title("🌱 4학년1반 제주 AI챗봇!")
st.markdown("<h3 style='color:#28a745;'>제주도의 지리정보를 알려드려요!</h3>", unsafe_allow_html=True)

# 디바이스 설정 (GPU 우선)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 캐싱 로딩
@st.cache_resource
def load_model():
    model = SentenceTransformer('jhgan/ko-sbert-sts')
    model.to(torch.device(DEVICE))
    return model

model = load_model()

# 지식 불러오기 함수
def load_knowledge(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# 지식 파일 경로
file_path = "energy3.txt"
sentences = load_knowledge(file_path)

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

# 초기화 버튼
if st.button("초기화"):
    st.session_state["history"] = []
    st.success("기록이 초기화되었습니다!")

# 사용자 질문 입력
user_input = st.text_input("무엇이 궁금한가요?")

# FAISS 인덱스 구축 함수
def build_faiss_index(sentences):
    embeddings = model.encode(sentences, convert_to_numpy=True, device=DEVICE)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences

# 질문 처리 로직
if st.button("질문하기") and user_input:

    # 1. 키워드 조건 우선 답변
    if "1인당" in user_input and "온실가스" in user_input:
        matched_answer = sentences[1] if len(sentences) > 1 else "데이터가 부족해요."
    elif "세계" in user_input and "온실가스" in user_input:
        matched_answer = sentences[3] if len(sentences) > 3 else "데이터가 부족해요."
    else:
        # 2. FAISS를 통한 문장 유사도 검색
        index, searchable_sentences = build_faiss_index(sentences)
        query_vec = model.encode([user_input], convert_to_numpy=True, device=DEVICE)
        D, I = index.search(np.array(query_vec), k=1)

        # 3. 유사도 거리 기준 설정 (낮을수록 유사)
        distance = D[0][0]
        if distance > 500.0:
            matched_answer = "잘 이해되지 않아요. 다시 질문해 주세요!"
        else:
            matched_answer = searchable_sentences[I[0][0]]

    # 답변 출력 및 기록 저장
    st.markdown(f"**챗봇:** {matched_answer}")
    st.session_state["history"].insert(0, (user_input, matched_answer))

# 질문 히스토리 출력
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("📜 이전 질문 기록")
    for idx, (prev_q, prev_a) in enumerate(st.session_state["history"], 1):
        with st.expander(f"Q{idx}: {prev_q}", expanded=False):
            st.markdown(prev_a)
