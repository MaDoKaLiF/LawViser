import json
import os
import re
from collections import Counter

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util


api_key = "up_KE4F4IAvUTKqQ6feOuA09jfyyLohD"
output_file = "./results/result.json"

# 1. OCR and text extraction (e.g., for PDFs)
def document_parse(filename):
    ext = os.path.splitext(filename)[1]  # e.g., ".pdf"
    if ext in ['.pdf', '.png', '.PNG', '.jpg', '.jpeg']:
        url = "https://api.upstage.ai/v1/document-digitization"
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {"document": open(filename, "rb")}
        data = {"ocr": "force", "base64_encoding": "['table']", "model": "document-parse"}
        response = requests.post(url, headers=headers, files=files, data=data)
        
        response_json = response.json()
        soup = BeautifulSoup(response_json['content']['html'], "html.parser")
        plain_text = soup.get_text(separator="\n")  # Extract text using newline as delimiter
        print("plain_text:", plain_text)
        
        with open("./results/response.json", "w", encoding="utf-8") as f:
            json.dump(plain_text, f, ensure_ascii=False, indent=4)
    else:
        with open(filename, "r", encoding="utf-8") as f:
            plain_text = f.read()

    return plain_text


def classifier(filename):
    os.makedirs("./results", exist_ok=True)
    # filename = input("Enter the filename: ")
    plain_text = document_parse(filename)
    
    # 2. Split the document into sections using regular expressions
    #    - The pattern must include the format "제 [number] 조 ([content in parentheses])"
    pattern = r"(제\s*\d+\s*조\s*\([^)]*\).*?)(?=^제\s*\d+\s*조|\Z)"
    sections = re.findall(pattern, plain_text, flags=re.MULTILINE | re.DOTALL)

    for idx, sec in enumerate(sections, start=1):
        print(f"Section {idx}:\n{sec.strip()}\n{'='*40}\n")

    # 3. Define classification labels and compute their embeddings
    #    - Classes: 0: 임대차, 1: 전자상거래, 2: 기타
    class_labels = ["임대차", "전자상거래", "기타"]

    # Load the SentenceTransformer model
    embed_model = SentenceTransformer("nlpai-lab/KURE-v1", trust_remote_code=True)

    # Compute label embeddings (as tensors)
    label_embeddings = embed_model.encode(class_labels, convert_to_tensor=True)

    # 4. Compute embeddings for each section and classify based on cosine similarity
    threshold = 0.35  # If similarity is below this value, classify as "기타" (Others)

    section_results = []
    for idx, sec in enumerate(sections, start=1):
        sec_embedding = embed_model.encode(sec, convert_to_tensor=True)
        cosine_scores = util.cos_sim(sec_embedding, label_embeddings)
        max_score = cosine_scores.max().item()
        best_label_idx = cosine_scores.argmax().item()
        # If the similarity score is below the threshold, classify as "기타"
        if max_score < threshold:
            classification = "기타"
        else:
            classification = class_labels[best_label_idx]
        section_results.append(classification)
        print(f"Section {idx} classification result: {classification} (similarity: {max_score:.3f})")

    # 5. Determine the final classification by voting (majority vote)
    vote_counts = Counter(section_results)
    final_classification = vote_counts.most_common(1)[0][0]
    print("Final classification result:", final_classification)

    # Dictionary to map label strings to integers
    label_to_int = {
        "임대차": 0,
        "전자상거래": 1,
        "기타": 2
    }

    # Convert the final classification string to an integer
    final_classification_int = label_to_int.get(final_classification, -1)
    print("Final classification integer result:", final_classification_int)

    # 6. Save the results
    output_data = {
        "section_inputs": sections,
        "section_results": section_results,
        "final_classification": final_classification_int
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("Results have been saved to result.json.")

    return output_file
