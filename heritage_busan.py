import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch

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

# 세션 상태 초기화
if "shown_names" not in st.session_state:
    st.session_state["shown_names"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

# 구 리스트
districts = ['동래구', '사하구', '금정구', '서구', '북구', '수영구', '부산진구', '강서구',
             '남구', '영도구', '기장군', '사상구', '해운대구', '동구']

# Streamlit UI
st.title("🏛️ 부산 문화유산 챗봇")
question = st.text_input("궁금한 걸 물어보세요. 예: '조선시대 해운대구 유형문화유산 알려줘'")
search = st.button("질문하기")

if st.button("초기화"):
    st.session_state["shown_names"] = []
    st.session_state["history"] = []
    st.success("초기화되었습니다!")

if search:
    if not question:
        st.warning("질문을 입력해주세요.")
    else:
        # 1. 이미 보여준 이름 제외
        filtered_data = [
            item for item in heritage_data 
            if item.get("이름") not in st.session_state["shown_names"]
        ]

        # 2. 유산 종류 필터
        if "유형문화유산" in question:
            filtered_data = [item for item in filtered_data if "유형문화유산" in item.get("종류", "")]
        elif "무형유산" in question or "무형문화유산" in question:
            filtered_data = [item for item in filtered_data if "무형유산" in item.get("종류", "") or "무형문화유산" in item.get("종류", "")]

        # 3. 지역 필터
        selected_districts = [d for d in districts if d in question]
        if selected_districts:
            filtered_data = [item for item in filtered_data if any(d in item.get("주소", "") for d in selected_districts)]

        # 필터 후 확인
        if not filtered_data:
            st.error("더 이상 조건에 맞는 새로운 문화유산을 찾을 수 없습니다.")
        else:
            # 문장 생성
            sentences = [
                f"{item.get('이름', '이름 없음')}는 {item.get('시대', '시대 정보 없음')} 시대의 {item.get('종류', '종류 정보 없음')}이며, {item.get('주소', '주소 정보 없음')}에 있다."
                for item in filtered_data
            ]

            # 임베딩 및 유사도 계산
            heritage_embeddings = model.encode(sentences, convert_to_tensor=True)
            question_embedding = model.encode(question, convert_to_tensor=True)
            scores = util.cos_sim(question_embedding, heritage_embeddings)[0]

            # 유사도 높은 순 정렬
            scored_items = sorted(
                zip(filtered_data, scores.tolist()), 
                key=lambda x: (-x[1], x[0]["이름"])  # 유사도 높은 순, 이름 기준 tie-break
            )

            # 이미 보여준 것 다시 제외 (이중검사)
            scored_items = [pair for pair in scored_items if pair[0]["이름"] not in st.session_state["shown_names"]]

            if not scored_items:
                st.error("질문 조건에 맞는 새로운 항목이 없습니다.")
            else:
                selected, best_score = scored_items[0]
                st.session_state["shown_names"].append(selected["이름"])

                answer = f"""
### 🏷️ {selected['이름']}
- 📍 주소: {selected['주소']}
- 📜 시대: {selected['시대'] or '정보 없음'}
- 🏛️ 종류: {selected['종류']}
- 📅 지정 날짜: {selected.get('지정날짜', '정보 없음')}
- 📐 수량/면적: {selected.get('수량/면적', '정보 없음')}
- 👤 소유자: {selected.get('소유자', '정보 없음')}
- 🛠️ 관리자: {selected.get('관리자', '정보 없음')}
- 🔍 유사도 점수: `{best_score:.2f}`
"""
                st.markdown(answer)
                st.session_state["history"].insert(0, (question, answer))

# 이전 질문 기록
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("📜 이전 질문 기록")
    for idx, (prev_q, prev_a) in enumerate(st.session_state["history"], 1):
        with st.expander(f"Q{idx}: {prev_q}", expanded=False):
            st.markdown(prev_a)
