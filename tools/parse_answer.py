import json
import re

output_file = "./results/parsed_data.json"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def parse_responses(data):
    results = []
    for item in data:
        responses = item.get("responses", "")
        match = re.findall(r"A:\s*(유리|불리)\s*-\s*(.*?)(?=\.|\n|$)", responses)

        parsed_item = {"idx": item.get("idx", -1), "responses": []}
        if match:
            label, reason = match[-1]  # 마지막 매치만 사용
            if not reason.strip():
                reason = "근거 없음"
            parsed_item["responses"].append({"label": label, "reason": reason})
        else:
            parsed_item["responses"].append({"label": "없음", "reason": "없음"})
        
        results.append(parsed_item)
    
    return results

def parse_answer(final_response_file):
    file_path = final_response_file
    data = load_json(file_path)
    parsed_results = parse_responses(data)
    save_json(output_file, parsed_results)
    print(f"Parsed results saved to {output_file}")

    return output_file

