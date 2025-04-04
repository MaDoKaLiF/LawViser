"""
main.py
-------------------------------
[RAG 시스템 내 역할]
▶️ RAG 파이프라인의 전체 실행을 담당하는 메인 스크립트
▶️ 질문 데이터를 불러오고, LLM과 임베딩 모델을 초기화한 뒤
    각 질문에 대해 적절한 문서를 검색하고 답변을 생성
▶️ 결과를 CSV로 저장하며, 필요 시 RAGAS를 통한 성능 평가도 수행 가능
-------------------------------
"""

from datasets import load_from_disk
from embedding import EmbeddingHandler
from model import ModelHandler
from generation import AnswerGenerator
from evaluation import RAGASEvaluator
import logging
import pandas as pd
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# ✅ 데이터 로드
logging.info("데이터셋을 불러오는 중입니다...")
dataset = load_from_disk("/root/YBIGTA_X_KUBIG_Hackathon/combined_dataset")['train']

# ✅ 모델 및 핸들러 초기화
model_handler = ModelHandler()
embedding_handler = EmbeddingHandler()
answer_generator = AnswerGenerator(model_handler, embedding_handler)

results = []

# ✅ 질문마다 답변 생성
for i in tqdm(range(len(dataset))):
    question = dataset[i]["question"]
    logging.info(f"[{i+1}] 질문 처리 중: {question}")
    
    result = answer_generator.generate_answer_and_collect_results(
        question, dataset, idx=i
    )
    
    logging.info(f"생성된 답변: {result}")
    results.append(result)

# ✅ 결과를 CSV로 저장
result_df = pd.DataFrame(results)
result_df.to_csv("submission.csv", index=False)
logging.info("답변 결과가 submission.csv에 저장되었습니다.")

# ✅ (선택) RAGAS 평가 수행
# logging.info("RAGAS 평가 수행 중...")
# evaluator = RAGASEvaluator()
# evaluator.evaluate_with_ragas(results)
# logging.info("평가 완료.")
