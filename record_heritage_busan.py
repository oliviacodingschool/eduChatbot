import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch
import random

# 모델 로드
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# JSON 데이터 로드
@st.cache_data
def load_data():
    with open("busan_heritage.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

heritage_data = load_data()

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# 타이틀
st.title("🏛️ 부산 문화유산 챗봇")

question = st.text_input("궁금한 걸 물어보세요. 예: '조선시대 유형문화유산 알려줘'")
search = st.button("질문하기")

if search:
    if not question:
        st.warning("질문을 입력해주세요.")
    else:
        # 필터링: 종류 조건
        if "유형문화유산" in question:
            filtered_data = [item for item in heritage_data if "유형문화유산" in item.get("종류", "")]
        elif "무형유산" in question:
            filtered_data = [item for item in heritage_data if "무형유산" in item.get("종류", "")]
        else:
            filtered_data = heritage_data

        # 주소 조건이 있다면 필터링
        areas = ['동래구', '사하구', '금정구', '서구', '북구', '수영구', '부산진구', '강서구',
                 '남구', '영도구', '기장군', '사상구', '해운대구', '동구']
        area_matches = [area for area in areas if area in question]
        if area_matches:
            filtered_data = [item for item in filtered_data if any(area in item.get("주소", "") for area in area_matches)]

        if not filtered_data:
            st.error("해당 조건에 맞는 문화유산을 찾을 수 없습니다.")
        else:
            # 문장 생성
            sentences = [
                f"{item.get('이름', '이름 없음')}는 {item.get('시대', '시대 정보 없음')} 시대의 {item.get('종류', '종류 정보 없음')}이며, {item.get('주소', '주소 정보 없음')}에 있다."
                for item in filtered_data
            ]

            # 임베딩 계산
            heritage_embeddings = model.encode(sentences, convert_to_tensor=True)
            question_embedding = model.encode(question, convert_to_tensor=True)
            scores = util.cos_sim(question_embedding, heritage_embeddings)[0]

            # 상위 20개 추출
            top_n = 20
            top_n_indices = torch.topk(scores, top_n).indices.tolist()
            top_n_filtered_data = [filtered_data[i] for i in top_n_indices]
            top_n_scores = [scores[i].item() for i in top_n_indices]

            # 랜덤 선택
            combined = list(zip(top_n_filtered_data, top_n_scores))
            random.shuffle(combined)
            selected, best_score = combined[0]

            # 기록 저장
            st.session_state.history.append({
                "질문": question,
                "답변": selected,
                "유사도": best_score
            })

# 기록 보여주기 (최신 것이 위에 오도록)
for record in reversed(st.session_state.history):
    st.markdown(f"""
### 🧾 질문: {record['질문']}
#### 🏷️ {record['답변']['이름']}
- 📍 주소: {record['답변']['주소']}
- 📜 시대: {record['답변'].get('시대', '정보 없음')}
- 🏛️ 종류: {record['답변'].get('종류', '정보 없음')}
- 📅 지정 날짜: {record['답변'].get('지정날짜', '정보 없음')}
- 📐 수량/면적: {record['답변'].get('수량/면적', '정보 없음')}
- 👤 소유자: {record['답변'].get('소유자', '정보 없음')}
- 🛠️ 관리자: {record['답변'].get('관리자', '정보 없음')}
- 🔍 유사도 점수: `{record['유사도']:.2f}`
---
""")
