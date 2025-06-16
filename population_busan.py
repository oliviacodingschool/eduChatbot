import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import torch

st.set_page_config(page_title="초등학생 AI 챗봇")

# CUDA 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = SentenceTransformer('kykim/bert-kor-base')
    model.to(torch.device(DEVICE))  # 모델을 CUDA로 이동
    return model

model = load_model()

def load_knowledge(file_path="population_busan.txt"):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

sentences = load_knowledge()

@st.cache_resource
def build_faiss_index(sentences):
    embeddings = model.encode(sentences, convert_to_numpy=True, device=DEVICE)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = build_faiss_index(sentences)

st.title("📘 구포4-1반 초등 AI 챗봇")
st.markdown("인구, 면적 데이터를 담고 있는 챗봇이에요")

user_input = st.text_input("무엇이 궁금한가요?")
if st.button("질문하기") and user_input:
    query_vec = model.encode([user_input], convert_to_numpy=True, device=DEVICE)
    D, I = index.search(np.array(query_vec), k=2)
    best_score = D[0][0]
    matched_answer = sentences[I[0][0]]

    tag_sentences = [
        "시도", "전국", "대한민국", "우리나라",
        "경기도", "서울", "경상남도", "경상북도", "대구",
        "충청남도", "인천", "전라남도", "전북특별자치도", "충청북도",
        "강원도", "대전", "광주", "울산", "제주도", "세종"
    ]

    if best_score > 500.0:
        st.markdown("**챗봇:** 질문이 잘 이해되지 않습니다. 다른 방식으로 질문해주세요. 6Quiz를 활용해봐요!")
    else:
        if any(tag in user_input for tag in tag_sentences):
            matched_answer = ("경기도 1,369.9만 명, 서울 933.6만 명, 부산 325.9만 명, "
                              "경상남도 321.9만 명, 경상북도 252.3만 명, 대구 236.0만 명, "
                              "충청남도 213.6만 명, 인천 303.1만 명, 전라남도 178.5만 명, "
                              "전북특별자치도 173.4만 명, 충청북도 159.1만 명, 강원도 151.3만 명, "
                              "대전 143.9만 명, 광주 140.2만 명, 울산 109.5만 명, 제주도 66.8만 명, "
                              "세종 39.2만 명 순서입니다.")
        st.markdown(f"**챗봇:** {matched_answer}")
