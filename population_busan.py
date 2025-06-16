import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import torch

st.set_page_config(page_title="구포초등학교 AI 챗봇")

# CUDA 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로딩
@st.cache_resource
def load_model():
    model = SentenceTransformer('kykim/bert-kor-base')
    model.to(torch.device(DEVICE))
    return model

model = load_model()

# 지식 파일 로드
def load_knowledge(file_path="population_busan.txt"):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

sentences = load_knowledge()

# FAISS 인덱스 구축
@st.cache_resource
def build_faiss_index(sentences):
    embeddings = model.encode(sentences, convert_to_numpy=True, device=DEVICE)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = build_faiss_index(sentences)

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
if st.button("질문하기") and user_input:
    # 면적 질문 고정 응답
    if "면적" in user_input:
        area_text = (
            "부산 구군별 면적 순위를 비교하겠습니다. 단위는 제곱미터입니다.\n\n"
            "기장군 21.8만, 강서구 18.0만, 금정구 6.5만, 해운대구 5.1만,\n"
            "사하구 4.1만, 북구 3.9만, 사상구 3.6만, 부산진구 3.0만, 남구 2.7만,\n"
            "동래구 1.7만, 영도구 1.4만, 서구 1.4만, 연제구 1.2만, 수영구 1.0만,\n"
            "동구 1.0만, 중구 0.3만입니다.\n\n총 면적은 77001만 제곱미터입니다."
        )
        st.markdown(f"**챗봇:** {area_text}")
        st.session_state["history"].insert(0, (user_input, area_text))

    else:
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
            fallback_msg = "질문이 잘 이해되지 않습니다. 다른 방식으로 질문해주세요. 6Quiz를 활용해봐요!"
            st.markdown(f"**챗봇:** {fallback_msg}")
            st.session_state["history"].insert(0, (user_input, fallback_msg))
        else:
            if any(tag in user_input for tag in tag_sentences):
                matched_answer = (
                    "경기도 1,369.9만 명, 서울 933.6만 명, 부산 325.9만 명, "
                    "경상남도 321.9만 명, 경상북도 252.3만 명, 대구 236.0만 명, "
                    "충청남도 213.6만 명, 인천 303.1만 명, 전라남도 178.5만 명, "
                    "전북특별자치도 173.4만 명, 충청북도 159.1만 명, 강원도 151.3만 명, "
                    "대전 143.9만 명, 광주 140.2만 명, 울산 109.5만 명, 제주도 66.8만 명, "
                    "세종 39.2만 명 순서입니다."
                )
            st.markdown(f"**챗봇:** {matched_answer}")
            st.session_state["history"].insert(0, (user_input, matched_answer))

# 이전 질문 기록 표시
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("📜 이전 질문 기록")
    for idx, (prev_q, prev_a) in enumerate(st.session_state["history"], 1):
        with st.expander(f"Q{idx}: {prev_q}", expanded=False):
            st.markdown(prev_a)
