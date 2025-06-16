import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

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

# 지식 문장 직접 삽입
KNOWLEDGE = [
    "전 세계 국가별 이산화탄소 배출량은 중국 144억톤, 미국 64억톤, 인도 35억톤, 유럽 34억톤입니다.",
    "전 세계 국가별 재생에너지 사용량 비중은 중국은 32%, 미국은 21%입니다.",
    "대한민국 항목별 재생에너지 사용량 순위는 태양광이 3323만, 바이오 1192만, 수력 372만 MWh(메가와트)입니다.",
    "대한민국 사람 한 명이 1년에 쓰는 에너지는 1990년에는 2000톤, 2000년에는 4000톤, 2010년에는 5000톤, 2020년에는 5500톤입니다."
]

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

# UI
st.title("🌱 재생에너지 AI 챗봇")
st.markdown("<h3 style='color:#28a745;'>탄소 배출량과 재생에너지 데이터를 알려드려요</h3>", unsafe_allow_html=True)

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
