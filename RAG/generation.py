"""
generation.py
-------------------------------
[RAG 시스템 내 역할]
▶️ 사용자의 질문(query)을 받아 관련 문서를 FAISS에서 검색하고,
    이를 기반으로 LLM에게 적절한 답변 생성을 요청하는 핵심 모듈
▶️ 문서 검색 방식:
    - 벡터 유사도 기반 검색 (FAISS)
▶️ 검색된 문서를 문맥으로 프롬프트에 포함시켜 답변 생성
-------------------------------
"""

import os
import re
import pickle
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from RAG.embedding import EmbeddingHandler
from keybert import KeyBERT


class AnswerGenerator:
    def __init__(
        self,
        model_handler,
        embedding_handler: EmbeddingHandler,
        faiss_index_path: str = "med_faiss_index.bin",
        document_reference_path: str = "document_references.pkl"
    ):
        self.model_handler = model_handler
        self.embedding_handler = embedding_handler
        self.device = model_handler.device

        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"❌ FAISS 인덱스 파일이 존재하지 않습니다: {faiss_index_path}")
        if not os.path.exists(document_reference_path):
            raise FileNotFoundError(f"❌ 문서 참조 파일이 존재하지 않습니다: {document_reference_path}")

        self.index = faiss.read_index(faiss_index_path)
        with open(document_reference_path, "rb") as f:
            self.document_references = pickle.load(f)

    def _vector_search_documents_faiss(self, query_vector, limit=5):
        query_vector = torch.tensor(query_vector).reshape(1, -1).numpy().astype('float32')
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, limit)
        return [(self.document_references[i], scores[0][j]) for j, i in enumerate(indices[0])]

    
    def _keyword_search_documents(self, query, top_n=3, limit=5):
        """
        KeyBERT로 query에서 키워드 추출 → 매핑 사전을 거쳐 확장 → 키워드 포함 문서 필터링
        """
        kw_model = KeyBERT()
        raw_keywords = kw_model.extract_keywords(query, top_n=top_n, stop_words=None)
        base_keywords = [kw[0] for kw in raw_keywords]

        # ✅ 동의어 사전 정의
        keyword_map = {
            "환불": ["청약철회", "청약 철회", "계약 철회", "환불"],
            "결제 취소": ["청약철회", "결제 취소"],
            "계약 해지": ["계약의 해제", "계약 해제", "청약철회", "계약 해지"],
            "자동 결제": ["정기결제", "자동갱신", "자동 결제"],
            "해지 불가": ["청약철회 제한", "청약철회 금지", "해지 불가"]
        }

        # ✅ 확장된 키워드 집합 생성
        expanded_keywords = set()
        for kw in base_keywords:
            expanded_keywords.add(kw)
            if kw in keyword_map:
                expanded_keywords.update(keyword_map[kw])

        # ✅ 키워드 포함 문서 필터링
        matched_chunks = []
        for chunk in self.document_references:
            if any(ek in chunk for ek in expanded_keywords):
                matched_chunks.append(chunk)
            if len(matched_chunks) >= limit:
                break
        return matched_chunks



    def find_related_laws(self, clause_text, top_k=5):
        query_vector = self.embedding_handler.get_embedding(clause_text)

        # 벡터 기반 결과: [(문장, 점수)]
        vector_scored = self._vector_search_documents_faiss(query_vector, limit=top_k * 2)

        # 키워드 기반 결과: [문장들]
        keyword_results = self._keyword_search_documents(clause_text, top_n=3, limit=top_k)

        # keyword match된 것들은 점수 1.0으로 가정
        keyword_scored = [(chunk, 1.0) for chunk in keyword_results]

        # 벡터 + 키워드 통합 후 중복 제거 (문장 기준)
        all_results_dict = {}
        for chunk, score in keyword_scored + vector_scored:
            if chunk not in all_results_dict:
                all_results_dict[chunk] = score
            else:
                # 이미 있으면 더 높은 점수로 (보수적 통합)
                all_results_dict[chunk] = max(all_results_dict[chunk], score)

        # 정렬
        sorted_results = sorted(all_results_dict.items(), key=lambda x: -x[1])
        top_chunks = [chunk for chunk, _ in sorted_results[:top_k]]

        return {
            "input_clause": clause_text,
            "related_laws": top_chunks
        }


