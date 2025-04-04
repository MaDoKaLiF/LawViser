"""
model.py
-------------------------------
[RAG 시스템 내 역할]
▶️ LLM(예: kosaul_v0.2)을 로드하는 모듈
▶️ generation.py에서 답변 생성을 위해 사용됨
▶️ 필요 시 4-bit 양자화를 적용해 메모리 효율적인 실행 가능
▶️ tokenizer와 model을 함께 관리하며, 필요한 설정(pad_token 등)도 포함
-------------------------------
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ModelHandler:
    def __init__(self, model_name="ingeol/kosaul_v0.2", use_4bit=False):
        # GPU가 가능하면 CUDA 사용, 아니면 CPU 사용
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 사전학습된 토크나이저 불러오기
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # eos_token이 설정되어 있지 않다면 추가
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

        # pad_token은 eos_token과 동일하게 설정
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 모델 로드 (양자화 옵션 선택)
        if use_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quant_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto"
            )

        # 토크나이저와 모델의 vocab size가 다를 경우 모델 임베딩 크기 조정
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
