import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 페이지 설정
st.set_page_config(page_title="학생용 AI 챗봇", layout="centered")
st.title("📘 학생용 AI 챗봇")
st.markdown("궁금한 걸 입력해 보세요!")

# 모델 로딩 (캐시로 메모리 효율화)
@st.cache_resource
def load_model():
    return SentenceTransformer('jhgan/ko-sbert-sts')

model = load_model()

# 질문-답변 데이터 (카테고리별 분류 포함)
FAQ_DATA = {
    "면적": {
        "부산 면적은 얼마인가요?": "부산의 면적은 약 770km²(제곱킬로미터)입니다.",
        "제주도 면적은 얼마인가요?": "제주도 면적은 1846km²(제곱키로미터)입니다."
    },
    "인구": {
        "부산 인구가 얼마인가요?": "부산 인구를 알려드립니다. 2025년 기준 부산의 인구수는 남자 158만 5597명 여자 167만 3622명이고, 남녀를 합친 총 인구수는 약 325만 9219명입니다.",
        "제주도 인구가 얼마인가요?": "제주도 인구를 알려드립니다. 2025년 기준 남자 33만 3318명 여자 33만 3925명이고, 남녀를 합친 총 인구수는 약 66만 7739명입니다."
    },
    "기온": {
        "2024년 부산 평균 기온은?": "2024년 부산 평균 기온에 대해 알려드릴게요. 봄 14.7도, 여름 25.9도, 가을 20도, 겨울 6.1도 입니다.",
        "2024년 제주도 평균 기온은?": "2024년 제주도 평균 기온에 대해 알려드릴게요. 봄 15.5도, 여름 27.3도, 가을 21.2도, 겨울 8.7도 입니다."
    },
    "강수량": {
        "부산 최근 강수량은?": "2022년부터 2024년까지 3년간 계절별 부산 강수량 평균은 봄 382mm, 여름 750mm, 가을 438mm, 겨울 149mm 입니다.",
        "제주 최근 강수량은?": "2022년부터 2024년까지 3년간 계절별 제주도 강수량 평균은 봄 415mm, 여름 692mm, 가을 395mm, 겨울 204mm입니다."
    }
}

# 카테고리별 FAISS 인덱스와 답변 저장
@st.cache_resource
def build_indexes():
    category_indexes = {}
    category_answers = {}

    for category, qa in FAQ_DATA.items():
        questions = list(qa.keys())
        answers = list(qa.values())
        embeddings = model.encode(questions)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

        category_indexes[category] = index
        category_answers[category] = answers

    return category_indexes, category_answers

category_indexes, category_answers = build_indexes()

# 답변 검색 함수
def find_best_answer(user_input, threshold=0.45):
    user_vec = model.encode([user_input])
    best_score = 100
    best_answer = None

    for category, index in category_indexes.items():
        D, I = index.search(np.array(user_vec), k=1)
        score = D[0][0]
        if score < best_score:
            best_score = score
            best_answer = category_answers[category][I[0][0]]

    if best_score > threshold:
        return "잘 모르겠어요. 다시 질문해 주세요.", best_score
    return best_answer, best_score

# 사용자 입력
user_question = st.text_input("질문을 입력해 보세요.", placeholder="예: 부산의 인구는 얼마인가요?")

if user_question:
    with st.spinner("답변을 찾는 중..."):
        answer, score = find_best_answer(user_question)
    st.markdown(f"**📌 답변:** {answer}")
    st.caption(f"(유사도 거리: {score:.4f})")  # 거리 작을수록 유사함
    st.caption(f"(유사도 거리: {:.4f})")  # 거리 작을수록 유사함
