import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch
import random

# 모델 로드 (캐싱)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# JSON 데이터 로드 (캐싱)
@st.cache_data
def load_data():
    with open("busan_heritage.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

heritage_data = load_data()

# Streamlit 인터페이스
st.title("🏛️ 부산 문화유산 챗봇")
question = st.text_input("궁금한 걸 물어보세요. 종류, 시대, 주소, 지정날짜를 알 수 있어요. 예: '조선시대 유형문화유산 알려줘'")
search = st.button("질문하기")

if search:
    if not question:
        st.warning("질문을 입력해주세요.")
    else:
        # 질문 내 의도 파악
        if "유형문화유산" in question:
            filtered_data = [item for item in heritage_data if "유형문화유산" in item.get("종류", "")]
        elif "무형유산" in question:
            filtered_data = [item for item in heritage_data if "무형유산" in item.get("종류", "")]
        else:
            filtered_data = heritage_data  # 전체 검색

        if not filtered_data:
            st.error("해당 조건에 맞는 문화유산을 찾을 수 없습니다.")
        else:
            # 문장 만들기
            sentences = [
                f"{item.get('이름', '이름 없음')}는 {item.get('시대', '시대 정보 없음')} 시대의 {item.get('종류', '종류 정보 없음')}이며, {item.get('주소', '주소 정보 없음')}에 있다."
                for item in filtered_data
            ]

            # 문장 임베딩
            heritage_embeddings = model.encode(sentences, convert_to_tensor=True)

            # 질문 임베딩 및 유사도 계산
            question_embedding = model.encode(question, convert_to_tensor=True)
            scores = util.cos_sim(question_embedding, heritage_embeddings)[0]

            # 상위 N개의 유사도 높은 항목을 선택 (예: 5개)
            top_n = 20
            top_n_indices = torch.topk(scores, top_n).indices.tolist()
            top_n_scores = scores[top_n_indices].tolist()
            top_n_filtered_data = [filtered_data[i] for i in top_n_indices]
            
            # 랜덤으로 하나를 선택
            selected = random.choice(top_n_filtered_data)
            best_score = max(top_n_scores)  # 가장 높은 유사도 점수
            
            best_index = torch.argmax(scores).item()
            best_score = scores[best_index].item()
            selected = filtered_data[best_index]

            # 결과 출력
            st.markdown(f"""
### 🏷️ {selected['이름']}
- 📍 주소: {selected['주소']}
- 📜 시대: {selected['시대'] or '정보 없음'}
- 🏛️ 종류: {selected['종류']}
- 📅 지정 날짜: {selected.get('지정날짜', '정보 없음')}
- 📐 수량/면적: {selected.get('수량/면적', '정보 없음')}
- 👤 소유자: {selected.get('소유자', '정보 없음')}
- 🛠️ 관리자: {selected.get('관리자', '정보 없음')}
- 🔍 유사도 점수: `{best_score:.2f}`
""")
