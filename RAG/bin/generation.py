"""
generation.py
-------------------------------
[RAG 시스템 내 역할]
▶️ 사용자의 질문(query)을 받아 관련 문서를 MongoDB 또는 FAISS에서 검색하고,
    이를 기반으로 LLM에게 적절한 답변 생성을 요청하는 핵심 모듈
▶️ 문서 검색에는 두 가지 방식 사용:
    - 키워드 기반 검색 (KeyBERT + MongoDB)
    - 벡터 유사도 기반 검색 (MongoDB $vectorSearch 또는 FAISS)
▶️ 검색된 문서를 문맥으로 프롬프트에 포함시켜 답변 생성
-------------------------------
"""

import os
import re
import pickle
import torch
import faiss
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
from embedding import EmbeddingHandler
from keybert import KeyBERT

class AnswerGenerator:
    def __init__(self, model_handler, embedding_handler: EmbeddingHandler):
        self.model_handler = model_handler
        self.embedding_handler = embedding_handler
        self.device = model_handler.device
        self.kw_model = KeyBERT()

        # MongoDB 연결 시도
        try:
            self.collection = self._connect_to_mongo()
        except Exception as e:
            print(f"⚠️ MongoDB 연결 실패: {e}")
            self.collection = None

        # FAISS 인덱스와 참조 문서 로드
        self.index = faiss.read_index("/root/NEW/med_faiss_index.bin")
        with open("/root/NEW/document_references.pkl", "rb") as f:
            self.document_references = pickle.load(f)

    def _connect_to_mongo(self):
        uri = os.getenv(
            "MONGO_URI",
            "mongodb+srv://cofla2020:dnfl2014!@cluster0.e9ecc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        )
        client = MongoClient(uri)
        db = client["hack"]
        return db["embedding"]

    def _extract_keywords(self, text, top_n=3):
        keywords = self.kw_model.extract_keywords(
            text, top_n=top_n, keyphrase_ngram_range=(1, 2), stop_words='english'
        )
        return [kw[0] for kw in keywords]

    def _find_documents_by_keywords(self, keywords, limit=2):
        if not self.collection:
            return []
        documents = []
        for keyword in keywords:
            query = {"text": {"$regex": keyword, "$options": "i"}}
            for doc in self.collection.find(query).limit(limit):
                documents.append(doc["text"])
        return documents

    def _vector_search_documents_mongo(self, query_vector, limit=4):
        if not self.collection:
            return []
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "exact": False,
                    "limit": limit,
                    "numCandidates": 100,
                    "path": "embedding",
                    "queryVector": query_vector
                }
            }
        ]
        return [doc["text"] for doc in self.collection.aggregate(pipeline)]

    def _vector_search_documents_faiss(self, query_vector, limit=5):
        query_vector = torch.tensor(query_vector).reshape(1, -1).numpy().astype('float32')
        faiss.normalize_L2(query_vector)
        _, indices = self.index.search(query_vector, limit)
        return [self.document_references[i] for i in indices[0]]

    def _prepare_prompt(self, question, contexts):
        context_text = " ".join(contexts)
        return f"Q: {question}\nContext: {context_text}\nA:"

    def generate_answer_and_collect_results(self, question, data, top_k=5, idx=0):
        # 1. 질문 임베딩
        query_vector = self.embedding_handler.get_embedding(question)

        # 2. 키워드 기반 문서 검색 (MongoDB)
        keyword_docs = self._extract_keywords(question)
        keyword_results = self._find_documents_by_keywords(keyword_docs)

        # 3. 벡터 유사도 기반 문서 검색 (MongoDB 또는 FAISS)
        vector_results = self._vector_search_documents_mongo(query_vector)
        if not vector_results:
            vector_results = self._vector_search_documents_faiss(query_vector)

        # 4. 문맥 병합 및 중복 제거
        combined_contexts = list(dict.fromkeys(keyword_results + vector_results))[:6]

        # 5. 프롬프트 생성
        prompt = self._prepare_prompt(question, combined_contexts)
        inputs = self.model_handler.tokenizer(
            prompt, truncation=True, max_length=1024, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 6. LLM으로 답변 생성
        generated = self.model_handler.model.generate(**inputs, max_new_tokens=100)
        decoded = self.model_handler.tokenizer.decode(generated[0], skip_special_tokens=True)

        # 7. 후처리
        answer_parts = re.split(r'\s*A:\s*', decoded)
        answer = answer_parts[-1].strip() if len(answer_parts) > 1 else decoded.strip()
        ground_truth = data[idx]['response']

        print(f'answer: {answer}')

        return {
            "question": question,
            "answer": answer,
            "contexts": combined_contexts,
            "ground_truth": ground_truth
        }
