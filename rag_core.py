import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def load_model():
    # [팁] 만약 배포 후 서버가 터진다면, 메모리를 덜 먹는 한국어 특화 모델로 교체하는 것을 추천합니다.
    # 예: return SentenceTransformer('jhgan/ko-sroberta-multitask')
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_rag_data(filepath_prefix="adsp_rag"):
    # 파일이 없는 경우를 대비한 안전장치 추가
    try:
        with open(f"{filepath_prefix}_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        embeddings = np.load(f"{filepath_prefix}_embeddings.npy")
        return chunks, embeddings
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"❌ 기출문제 데이터 파일을 찾을 수 없습니다! "
            f"깃허브에 {filepath_prefix}_chunks.pkl와 {filepath_prefix}_embeddings.npy를 올리셨는지 확인해주세요."
        ) from e

def search_similar_chunks(query, chunk_texts, chunk_embeddings, model, top_k=3):
    query_embedding = model.encode(query)
    
    # 분모가 0이 되어 나눗셈 에러가 나는 것을 방지하기 위한 아주 작은 값(1e-9) 추가 (안전장치)
    query_norm  = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    chunks_norm = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-9)
    
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
아래 기출문제를 참고해서 사용자의 질문에 답변하세요.

[참고 기출문제]
{context}

[규칙]
- 반드시 제공된 기출문제를 근거로만 답변하세요.
- 만약 기출문제 내용만으로 답을 알 수 없다면, 억지로 지어내지 말고 반드시 "제공된 기출문제에서 관련 내용을 찾을 수 없습니다"라고만 답변하세요.
- 친절한 한국어로 답하세요."""

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
