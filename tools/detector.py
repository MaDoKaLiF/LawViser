import json
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import re

output_file = "./results/detection_results.json"
model_name = "ingeol/kosaul_v0.2"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
# def generate_response(model_name, inputs, section_inputs_list):
#     responses = {}
#     for idx, (input, section_input) in enumerate(zip(inputs, section_inputs_list)):
#         tokenized = tokenizer(input, return_tensors="pt").to("cuda")
#         with torch.no_grad():
#             output = model.generate(**tokenized, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=1.0)
#         response = tokenizer.decode(output[0], skip_special_tokens=True)
#         match = re.search(r"ASSISTANT:\s*(.*)", response, re.DOTALL)
#         assistant_response = match.group(1) if match else ""
#         responses[f"{idx}"] = {
#             "section": assistant_response,
#             "output": response,
#         }

#     return responses


def generate_response(model_name, inputs, section_inputs_list):
    gpu_count = torch.cuda.device_count()
    llm = LLM(
            model=model_name,
            tensor_parallel_size=gpu_count,
            max_num_seqs=4,
            max_num_batched_tokens=512,
            gpu_memory_utilization=0.9,
            dtype="bfloat16"
        )
    
    sampling_params = SamplingParams(max_tokens=128)
    with torch.no_grad():
        all_outputs = llm.generate(inputs, sampling_params)

    responses = {}
    for idx, (outputs, section_input) in enumerate(zip(all_outputs, section_inputs_list)):
        responses[f"{idx}"] = {
            "section": section_input,
            "output": outputs.outputs[0].text,
        }
    
    return responses


def detector(classified_output):
    # load classified data
    with open(classified_output, "r", encoding="utf-8") as f:
        data = json.load(f)

    section_inputs_list = data.get("section_inputs", [])
    classified_label = str(data.get("final_classification"))
    class_labels = {'0': "임대차", '1': "전자상거래", '2': "기타"}

    classified_label_text = class_labels.get(classified_label)

    # load n-shot prompts
    prompt_filename = f"./tool1/n_shot_prompts/n_shot_prompts_{classified_label}.json"
    with open(prompt_filename, "r", encoding="utf-8") as f:
        n_shot_data = json.load(f)        
    n_shot_text = "\n\n".join([entry["prompt"] for entry in n_shot_data["n_shot_prompts"]])    

    
    # formatting input prompts
    inputs = []
    for section_input in section_inputs_list:
        prompt = (
            n_shot_text + "\n\n" +
            section_input + "\n" +
            f"USER: 위 내용은 {classified_label_text} 약관 조항 중 하나이다. 이 조항이 계약자에게 문제가 될 가능성이 있는지 판단하시오.\n" +
            "(0) 예 (1) 아니오\n" +
            "ASSISTANT: "
        )
        inputs.append(prompt)
    
    # results
    response_data = generate_response(model_name, inputs, section_inputs_list)

    print(response_data)

    output_data = {
        "classified_label": classified_label,
        "classified_label_text": classified_label_text,
        "responses": response_data,
    }

    # save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"결과가 {output_file} 파일에 저장되었습니다.")

    return output_file

