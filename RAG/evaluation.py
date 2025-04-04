"""
evaluation.py
-------------------------------
[RAG 시스템 내 역할]
▶️ 생성된 답변의 품질을 평가하는 모듈
▶️ RAGAS(Retrieval-Augmented Generation Assessment Scores)를 이용하여
    다음 네 가지 지표 기반으로 자동 평가 수행:
    - answer_relevancy: 답변이 질문과 얼마나 관련 있는지
    - context_precision: 문맥의 정확도
    - context_recall: 문맥이 충분히 포함되었는지
    - faithfulness: 문맥에 기반한 사실성
▶️ OpenAI API를 통해 평가 수행 (gpt-3.5 등 활용됨)
-------------------------------
"""

import openai
import logging
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness, context_recall
import os
from datasets import Dataset

class RAGASEvaluator:
    def __init__(self):
        # 환경 변수에서 OpenAI API 키를 불러옵니다
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다.\n"
                "터미널에서 다음 명령어를 실행해 주세요:\n"
                "export OPENAI_API_KEY='your-key-here'"
            )

    def evaluate_with_ragas(self, results):
        # 평가용 데이터셋 구성 (question, answer, contexts, ground_truth)
        data_samples = {
            'question': [res["question"] for res in results],
            'answer': [res["answer"] for res in results],
            'contexts': [res["contexts"] for res in results],
            'ground_truth': [res["ground_truth"] for res in results]
        }
        dataset = Dataset.from_dict(data_samples)

        try:
            # RAGAS 평가 수행 (총 4개 지표)
            score = evaluate(
                dataset,
                metrics=[answer_relevancy, context_precision, faithfulness, context_recall]
            )
            logging.info("평가 점수:\n%s", score)
            return score
        except Exception as e:
            logging.error(f"RAGAS 평가 도중 오류 발생: {e}")
            return None
