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
st.markdown("""
<style>
    table.custom-table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
        font-family: Arial, sans-serif;
    }
    table.custom-table th, table.custom-table td {
        border: 1px solid #ddd;
        padding: 8px 12px;
        text-align: center;
    }
    table.custom-table th {
        background-color: #0078d7;
        color: white;
        font-weight: bold;
    }
    table.custom-table tr:nth-child(even){background-color: #f2f2f2;}
</style>

<table class="custom-table">
    <thead>
        <tr>
            <th>구분</th>
            <th>내용</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>유산 종류</td><td>유형문화유산, 무형유산</td></tr>
        <tr><td>시대</td><td>조선시대, 삼한시대, 일제강점기, 기타, 백제, 고려시대, 삼국시대, 대한제국시대, 가야</td></tr>
        <tr><td>지역</td><td>동래구, 사하구, 금정구, 서구, 북구, 수영구, 부산진구, 강서구, 남구, 영도구, 기장군, 사상구, 해운대구, 동구</td></tr>
    </tbody>
</table>
""", unsafe_allow_html=True)

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
        # 3차 필터링: 주소 기반
        if "강서구" in question:
            selected_districts = ["강서구"]
        elif "서구" in question:
            # '강서구'가 없고 '서구'가 있을 때만 서구로 필터링
            selected_districts = ["서구"]
        else:
            # 그 외 지역은 기존 방식 유지
            selected_districts = [d for d in districts if d in question]
        
        if selected_districts:
            filtered = [
                item for item in filtered
                if any(d in item.get("주소", "") for d in selected_districts)
            ]

        # 시대 리스트
        eras = ['조선시대', '삼한시대', '일제강점기', '기타', '백제', '고려시대', '삼국시대', '대한제국시대', '가야']
        
        # 시대 필터
        matched_era = None
        for era in eras:
            if era in question:
                matched_era = era
                break
        
        if matched_era:
            filtered = [item for item in filtered if item.get("시대") == matched_era]



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
