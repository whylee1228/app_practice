# rag_core.py
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_rag_data(filepath_prefix="adsp_rag"):
    with open(f"{filepath_prefix}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(f"{filepath_prefix}_embeddings.npy")
    return chunks, embeddings

def search_similar_chunks(query, chunk_texts, chunk_embeddings, model, top_k=3):
    query_embedding = model.encode(query)
    query_norm  = query_embedding / np.linalg.norm(query_embedding)
    chunks_norm = chunk_embeddings / np.linalg.norm(
        chunk_embeddings, axis=1, keepdims=True)
    similarities = np.dot(chunks_norm, query_norm)
    top_indices  = np.argsort(similarities)[::-1][:top_k]
    return [
        {"rank": r+1, "chunk_id": int(i),
         "similarity": float(similarities[i]), "text": chunk_texts[i]}
        for r, i in enumerate(top_indices)
    ]

def rag_answer(query, chunk_texts, chunk_embeddings, model, client, top_k=3):
    results = search_similar_chunks(
        query, chunk_texts, chunk_embeddings, model, top_k)

    context = ""
    for r in results:
        context += f"\n--- 관련 문제 {r['rank']} (유사도: {r['similarity']:.4f}) ---\n"
        context += r["text"] + "\n"

    system_prompt = f"""당신은 ADsP 시험 전문 튜터입니다.
아래 기출문제를 참고해서 답하세요.

[참고 기출문제]
{context}

[규칙]
- 반드시 위 기출문제를 근거로 답하세요
- 없으면 "기출문제에서 찾을 수 없습니다"라고 하세요
- 한국어로 답하세요"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
        max_tokens=1000,
    )
    return {
        "answer": response.choices[0].message.content,
        "chunks": results,
    }