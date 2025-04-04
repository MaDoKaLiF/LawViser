import json
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from bs4 import BeautifulSoup
import os
import re
from collections import Counter

class RestrictTokensLogitsProcessor:
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)
    
    def __call__(self, input_ids, scores):
        # scores: (batch_size, vocab_size)
        mask = torch.full_like(scores, float("-inf"))
        for idx in self.allowed_token_ids:
            mask[:, idx] = scores[:, idx]
        return mask

api_key = "up_KE4F4IAvUTKqQ6feOuA09jfyyLohD"
filename = "주택임대차 표준계약서_임대차.pdf"  # File name (e.g., "Rental Agreement Standard Contract.pdf")
ext = os.path.splitext(filename)[1]  # ".pdf" or ".txt"
if ext == '.pdf':
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": open(filename, "rb")}
    data = {"ocr": "force", "base64_encoding": "['table']", "model": "document-parse"}
    response = requests.post(url, headers=headers, files=files, data=data)
    
    response_json = response.json()
    soup = BeautifulSoup(response_json['content']['html'], "html.parser")
    plain_text = soup.get_text(separator="\n")  # Use newline as the delimiter for text extraction
    print("plain_text", plain_text)

    with open("response.json", "w", encoding="utf-8") as f:
        json.dump(plain_text, f, ensure_ascii=False, indent=4)
    
elif ext == '.txt':   
    with open(filename, "r", encoding="utf-8") as f:
        plain_text = f.read()
ocr_input = plain_text 

# -------------------------
# Clause Splitting (must include the format "제 [number] 조 ([content in parentheses])")
# -------------------------
pattern = r"(제\s*\d+\s*조\s*\([^)]*\).*?)(?=^제\s*\d+\s*조|\Z)"
sections = re.findall(pattern, plain_text, flags=re.MULTILINE | re.DOTALL)
for idx, sec in enumerate(sections, start=1):
    print(f"Section {idx}:\n{sec.strip()}\n{'='*40}\n")

with open("n_shot_prompts_class.json", "r", encoding="utf-8") as f:
    n_shot_data = json.load(f)
n_shot_text = "\n\n".join([entry["prompt"] for entry in n_shot_data["n_shot_prompts"]])

model_name = "ingeol/kosaul_v0.2" 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate a prompt for each section
prompts = []
for sec in sections:
    prompt = f"""{n_shot_text}

USER: The above is an exemplary answer. Classify the subject of the following contract by assigning the corresponding number:
0: Rental Agreement
1: E-commerce
2: Others

{sec.strip()}
ASSISTANT:"""
    prompts.append(prompt)

# Calculate allowed tokens using the tokenizer (tokens corresponding to 0, 1, 2)
allowed_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(3)]
logits_processor = LogitsProcessorList([RestrictTokensLogitsProcessor(allowed_tokens)])

# -------------------------
# Settings for Batch Processing
# -------------------------
batch_size = 8  # Adjust batch size according to available GPU memory
section_results = []

# Process prompts in batches
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        logits_processor=logits_processor,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode the result for each prompt in the batch
    for j in range(outputs.shape[0]):
        classification = tokenizer.decode(outputs[j][input_ids.shape[1]:]).strip()
        section_results.append(classification)
        print(f"Section {i + j + 1} classification result: {classification}")

# Determine the final classification by majority voting
vote_counts = Counter(section_results)
final_classification = vote_counts.most_common(1)[0][0]
print("Final classification result:", final_classification)

# Save the results to a JSON file
output_data = {
    "section_inputs": sections,
    "section_results": section_results,
    "final_classification": final_classification
}
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("Results have been saved to result.json.")
