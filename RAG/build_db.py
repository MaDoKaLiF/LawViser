"""
build_db.py
-------------------------------
[RAG 시스템 내 역할]
▶️ 사전 구축 단계에서 사용되는 문서 벡터 DB 생성
▶️ 각 법 조항 전체를 임베딩하여 FAISS 인덱스를 구축
▶️ 결과물:
    - faiss_index.bin: FAISS 벡터 인덱스
    - document_references.pkl: 각 벡터에 대응되는 조항 전체 텍스트
-------------------------------
"""

import os
import json
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from embedding import EmbeddingHandler

# ✅ 폴더 경로 설정
data_dir = "./legal_data/legal_data_subscribe"

# ✅ 폴더 내 모든 .json 파일 탐색
json_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(".json")]

# ✅ 데이터 로딩 및 조항 단위 결합
all_documents = []
for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            if all(key in item for key in ["law_name", "legal_provisions", "text"]) and item["text"].strip():
                combined = f"[{item['law_name']}] {item['legal_provisions']} - {item['text']}"
                all_documents.append(combined)

# ✅ 임베딩 준비
embedding_handler = EmbeddingHandler()
index = faiss.IndexFlatIP(768)  # cosine similarity
document_references = []
all_vectors = []

# ✅ 조항 단위 임베딩
for doc in tqdm(all_documents, desc="Embedding legal provisions"):
    embedding = embedding_handler.get_embedding(doc)
    all_vectors.append(embedding)
    document_references.append(doc)

# ✅ 벡터 정규화 및 FAISS 인덱스 구축
vectors = np.array(all_vectors).astype("float32")
faiss.normalize_L2(vectors)
index.add(vectors)

# ✅ 저장
faiss.write_index(index, "faiss_index.bin")
with open("document_references.pkl", "wb") as f:
    pickle.dump(document_references, f)

print("✅ 조항 단위 FAISS 인덱스 구축 완료!")
