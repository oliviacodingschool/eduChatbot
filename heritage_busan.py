import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch

# 모델 로드 (캐싱하여 속도 개선)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# JSON 데이터 로딩
@st.cache_data
def load_data():
    with open("busan_heritage.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

heritage_data = load_data()

# 설명 없이 문장 만들기
sentences = []
for item in heritage_data:
    name = item.get('이름', '이름 없음')
    era = item.get('시대', '시대 정보 없음')
    kind = item.get('종류', '종류 정보 없음')
    addr = item.get('주소', '주소 정보 없음')
    sent = f"{name}는 {era} 시대의 {kind}이며, {addr}에 있다."
    sentences.append(sent)

# 문장 임베딩
heritage_embeddings = model.encode(sentences, convert_to_tensor=True)

# Streamlit 인터페이스
st.title("🏛️ 부산 문화유산 챗봇")
question = st.text_input("궁금한 걸 물어보세요. 예: '조선시대 문화유산 알려줘'")

if question:
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, heritage_embeddings)[0]
    best_index = torch.argmax(scores).item()
    best_score = scores[best_index].item()
    selected = heritage_data[best_index]

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
