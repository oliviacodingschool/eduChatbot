import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
st.set_page_config(page_title="초등학생 AI 챗봇")

@st.cache_resource
def load_model():
    return SentenceTransformer("jhgan/ko-sbert-sts")

model = load_model()

def load_knowledge(file_path="knowledge.txt"):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

sentences = load_knowledge()

@st.cache_resource
def build_faiss_index(sentences):
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = build_faiss_index(sentences)

st.title("📘 초등학생 AI 챗봇")
st.markdown("내가 배운 지식으로만 대답해요!")

user_input = st.text_input("무엇이 궁금한가요?")
if st.button("질문하기") and user_input:
    query_vec = model.encode([user_input])
    D, I = index.search(np.array(query_vec), k=1)
    best_score = D[0][0]
    matched_answer = sentences[I[0][0]]

    if best_score < 0.1:
        st.markdown(f"**챗봇:** {matched_answer}")
    else:
        st.markdown("**챗봇:** 다른 방식으로 질문해줄래요?")
