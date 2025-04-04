import os
import argparse
import json
from RAG.embedding import EmbeddingHandler
from RAG.model import ModelHandler
from RAG.generation import AnswerGenerator

def get_db_folder(label_text=None):
    if label_text == 0:
        return "./RAG/housing_db"
    elif label_text == 1:
        return "./RAG/subscribe_db"
    elif label_text == 2:
        return "./RAG/other_db"
    else:
        print("💽 사용할 DB를 선택하세요:")
        print("0: housing (주택임대차, 상가건물임대차)")
        print("1: subscribe (구독 서비스 관련)")
        db_choice = input("입력 (0 또는 1): ").strip()
        if db_choice == "0":
            return "./RAG/housing_db"
        elif db_choice == "1":
            return "./RAG/subscribe_db"
        elif db_choice == "2":
            return "./RAG/other_db"
        else:
            print("❌ 잘못된 입력입니다. 0 또는 1만 입력 가능합니다.")
            return None

def init_answer_generator(db_folder):
    faiss_index_path = os.path.join(db_folder, "faiss_index.bin")
    document_reference_path = os.path.join(db_folder, "document_references.pkl")

    embedding_handler = EmbeddingHandler()
    model_handler = ModelHandler()
    return AnswerGenerator(
        model_handler=model_handler,
        embedding_handler=embedding_handler,
        faiss_index_path=faiss_index_path,
        document_reference_path=document_reference_path
    )

def run_rag(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    db_folder = get_db_folder(data.get("final_classification"))
    if not db_folder:
        return

    print(f"📁 선택한 DB 폴더: {db_folder}")
    answer_generator = init_answer_generator(db_folder)

    # 도메인 출력
    domain_map = {0: "🏠 [도메인: 주택임대차]", 1: "📰 [도메인: 구독 서비스]", 2: "📜 [도메인: 기타]"}
    print(domain_map.get(data.get("final_classification"), "❌ 알 수 없는 도메인입니다."))

    section_inputs = data.get("section_inputs", [])
    all_results = []

    for idx, clause in enumerate(section_inputs):
        print(f"\n📄 조항 {idx}:")
        print(clause.strip())

        try:
            rag_result = answer_generator.find_related_laws(clause.strip(), top_k=2)
            related_laws = rag_result["related_laws"]
            print("\n📚 관련된 법률 조항:")
            for i, law in enumerate(related_laws, 1):
                print(f"{i}. {law}")
            print("\n" + "-"*60 + "\n")

            all_results.append({
                "index": idx,
                "clause": clause.strip(),
                "related_laws": related_laws
            })
        except Exception as e:
            print(f"⚠️ 오류 발생 (조항 {idx}): {e}")
            all_results.append({
                "index": idx,
                "clause": clause.strip(),
                "related_laws": [],
                "error": str(e)
            })

    # 결과 저장
    output_path = os.path.join(os.path.dirname(json_path), "rag_results.json")
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump({"results": all_results}, out_file, ensure_ascii=False, indent=2)

    print(f"\n✅ RAG 결과가 저장되었습니다: {output_path}")
    
    return output_path


def interactive_mode():
    db_folder = get_db_folder()
    if not db_folder:
        return

    answer_generator = init_answer_generator(db_folder)
    print("📘 [이용약관 → 관련 법률 조항 검색기 시작]")
    print("아래에 이용약관의 조항을 입력하세요. 종료하려면 'exit' 입력\n")

    while True:
        clause = input("📄 이용약관 조항: ").strip()
        if clause.lower() in {"exit", "quit"}:
            print("🛑 시스템을 종료합니다.")
            break
        try:
            result = answer_generator.find_related_laws(clause, top_k=5)
            print("\n📚 관련된 법률 조항:")
            for i, law in enumerate(result["related_laws"], 1):
                print(f"{i}. {law}")
            print("\n" + "-"*60 + "\n")
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="문제 조항이 저장된 JSON 파일 경로")
    args = parser.parse_args()

    if args.json:
        if not os.path.exists(args.json):
            print("❌ 입력 JSON 파일이 존재하지 않습니다.")
            return
        run_rag(args.json)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
