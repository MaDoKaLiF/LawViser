"""
embedding.py
-------------------------------
[RAG 시스템 내 역할]
▶️ 입력된 텍스트(query 또는 문서)를 임베딩 벡터로 변환
▶️ 주로 generation.py에서 사용자 질문(query)을 벡터화할 때 사용됨
▶️ 법률 포함 다양한 한국어 텍스트로 학습된 KR-BERT 기반 모델(snunlp/KR-Medium)을 사용
-------------------------------
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class EmbeddingHandler:
    def __init__(self, model_name="snunlp/KR-Medium"):
        # GPU 사용 가능 여부에 따라 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 한국어 BERT 기반 사전학습 모델과 토크나이저 불러오기
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def get_embedding(self, text):
        # 입력 텍스트를 토크나이즈하고 모델 인퍼런스를 통해 임베딩 생성
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 모든 토큰의 평균을 임베딩으로 사용 (mean pooling)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        return embedding.astype(np.float32)
