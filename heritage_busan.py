import streamlit as st
import json
import random
from sentence_transformers import SentenceTransformer, util
import torch
import re

# 모델 로드
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# 데이터 로드
@st.cache_data
def load_data():
    with open("busan_heritage.json", "r", encoding="utf-8") as f:
        return json.load(f)

heritage_data = load_data()

# 세션 상태 초기화
if "shown_names" not in st.session_state:
    st.session_state["shown_names"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

# 지역 목록
districts = ['동래구', '사하구', '금정구', '서구', '북구', '수영구', '부산진구', '강서구',
             '남구', '영도구', '기장군', '사상구', '해운대구', '동구']

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
        # --- 1차 필터: 이미 보여준 항목 제거
        filtered = [
            item for item in heritage_data 
            if item.get("이름") not in st.session_state["shown_names"]
        ]

        # --- 2차 필터: 유형/무형 키워드 확인
        if "유형" in question:
            filtered = [item for item in filtered if "유형" in item.get("종류", "")]
        elif "무형" in question:
            filtered = [item for item in filtered if "무형" in item.get("종류", "")]
             
        # 3. 지역 필터 
        selected_districts = [d for d in districts if d in question]
        
        def exact_district_match(district, address):
            pattern = r'\b' + re.escape(district) + r'\b'
            return re.search(pattern, address) is not None
        
        if selected_districts:
            filtered_data = [
                item for item in filtered_data
                if any(exact_district_match(d, item.get("주소", "")) for d in selected_districts)
            ]

        # --- 필터 후 결과 없음
        if not filtered:
            st.error("더 이상 조건에 맞는 문화유산을 찾을 수 없습니다.")
        else:
            # 문장화
            sentences = [
                f"{item.get('이름', '이름 없음')}는 {item.get('시대', '시대 정보 없음')} 시대의 {item.get('종류', '종류 정보 없음')}이며, {item.get('주소', '주소 정보 없음')}에 있다."
                for item in filtered
            ]

            # 임베딩 & 유사도
            heritage_embeddings = model.encode(sentences, convert_to_tensor=True)
            question_embedding = model.encode(question, convert_to_tensor=True)
            similarities = util.cos_sim(question_embedding, heritage_embeddings)[0]

            # 결과와 점수 zip
            scored = list(zip(filtered, similarities.tolist()))

            # 유사도 기준 정렬 → 동일 점수는 무작위 셔플
            scored.sort(key=lambda x: x[1], reverse=True)

            final_sorted = []
            i = 0
            while i < len(scored):
                same_score_group = [scored[i]]
                j = i + 1
                while j < len(scored) and abs(scored[j][1] - scored[i][1]) < 1e-6:
                    same_score_group.append(scored[j])
                    j += 1
                random.shuffle(same_score_group)
                final_sorted.extend(same_score_group)
                i = j

            # 최종 선택
            selected, score = final_sorted[0]
            st.session_state["shown_names"].append(selected["이름"])

            answer = f"""
### 🏷️ {selected['이름']}
- 📍 주소: {selected['주소']}
- 📜 시대: {selected.get('시대', '정보 없음')}
- 🏛️ 종류: {selected.get('종류', '정보 없음')}
- 📅 지정 날짜: {selected.get('지정날짜', '정보 없음')}
- 📐 수량/면적: {selected.get('수량/면적', '정보 없음')}
- 👤 소유자: {selected.get('소유자', '정보 없음')}
- 🛠️ 관리자: {selected.get('관리자', '정보 없음')}
- 🔍 유사도 점수: `{score:.2f}`
"""
            st.markdown(answer)
            st.session_state["history"].insert(0, (question, answer))

# --- 이전 질문 기록
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("📜 이전 질문 기록")
    for idx, (prev_q, prev_a) in enumerate(st.session_state["history"], 1):
        with st.expander(f"Q{idx}: {prev_q}", expanded=False):
            st.markdown(prev_a)
