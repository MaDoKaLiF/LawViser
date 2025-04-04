from tools.classifier import classifier
from RAG.rag import run_rag
from lm.final_answer import final_answer
from tools.parse_answer import parse_answer
from tools.email_module import send_email_naver

if __name__ == "__main__":
    # 1. Prompt the user to enter the file path for classification
    input_path = input("📂 Enter the file path to classify: ").strip()

    # 2. Run the classifier
    classified_output = classifier(input_path)

    # 3. Run the RAG process
    rag_output = run_rag(classified_output)

    # 4. Generate the final answer
    final_response_file = final_answer(rag_output)
    
    # 5. parsing
    parsed_response_file = parse_answer(final_response_file)

    # 6. Send an email with the final output JSON file
    send_email_naver(path1=parsed_response_file, path2=rag_output)

