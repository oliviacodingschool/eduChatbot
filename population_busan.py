import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import torch
import random

st.set_page_config(page_title="구포초등학교 AI 챗봇")

# CUDA 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로딩
@st.cache_resource
def load_model():
    model = SentenceTransformer('kykim/bert-kor-base')
    model.to(torch.device(DEVICE))
    return model

model = load_model()

# 지식 파일 로드 (4문장)
def load_knowledge(file_path="population_busan.txt"):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

FULL_KNOWLEDGE = load_knowledge()

# 시도 관련 태그
TAG_SENTENCES = [
    "시도", "전국", "대한민국", "우리나라",
    "경기도", "서울", "경상남도", "경상북도", "대구",
    "충청남도", "인천", "전라남도", "전북특별자치도", "충청북도",
    "강원도", "대전", "광주", "울산", "제주도", "세종"
]

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

# UI
st.title("🕊️ 구포4-1반 초등 AI 챗봇")
st.markdown("<h3 style='color:#0078D7;'>인구, 면적 데이터를 담고 있는 챗봇이에요</h3>", unsafe_allow_html=True)

# 초기화 버튼
if st.button("초기화"):
    st.session_state["history"] = []
    st.success("기록이 초기화되었습니다!")

user_input = st.text_input("무엇이 궁금한가요?")

# FAISS 인덱스 구축 함수
def build_faiss_index(sentences):
    embeddings = model.encode(sentences, convert_to_numpy=True, device=DEVICE)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences

# 질문 처리
if st.button("질문하기") and user_input:
    if not FULL_KNOWLEDGE or len(FULL_KNOWLEDGE) < 4:
        st.error("지식 파일에 필요한 문장이 부족합니다.")
    else:
        # 면적 질문일 경우 4번째 문장 출력
        if "면적" in user_input:
            matched_answer = random.choice([FULL_KNOWLEDGE[3]])

        # 시도 관련 질문일 경우 3번째 문장 출력
        elif any(tag in user_input for tag in TAG_SENTENCES):
            matched_answer = FULL_KNOWLEDGE[2]

        # 일반 질문: 1~2번째 문장 대상으로 검색
        else:
            knowledge_subset = FULL_KNOWLEDGE[:3]  # 0, 1, 2번째 문장만
            index, searchable_sentences = build_faiss_index(knowledge_subset)

            query_vec = model.encode([user_input], convert_to_numpy=True, device=DEVICE)
            D, I = index.search(np.array(query_vec), k=1)

            if D[0][0] > 500.0:
                matched_answer = "질문이 잘 이해되지 않습니다. 다른 방식으로 질문해주세요. 6Quiz를 활용해봐요!"
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
