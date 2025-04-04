import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from tqdm import tqdm

def log_args(file_path, **kwargs):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    data.append(kwargs)
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def prompt_processing(rag_output):
    # load JSON file
    with open(rag_output, "r", encoding="utf-8") as file:
        data = json.load(file)

    # load n-shot prompts
    prompt_filename = f"./lm/n_shot_prompts/n_shot_prompts_0.json"
    with open(prompt_filename, "r", encoding="utf-8") as f:
        n_shot_data = json.load(f)        
    n_shot_text = "\n\n".join([entry["prompt"] for entry in n_shot_data["n_shot_prompts"]])
    
    # formatting prompts
    prompts = []
    for result in data["results"]:
        q = result["clause"]
        t1 = result["related_laws"][0]
        text = f"{n_shot_text}\n\nQ: {q}\nContext: {t1}\nContext을 참고하여 Q의 유불리를 판단하고, 다음 형식을 무조건 지켜서 답하시오. [유리/불리] - [근거]\nA:"
        prompts.append(text)

    return prompts

def load_lora_state(model, lora_path):
    # Load LoRA state dictionary
    lora_state_dict = torch.load(lora_path, map_location="cpu")
    model_state_dict = model.state_dict()
    updated_lora_state_dict = {}

    for key in lora_state_dict.keys():
        new_key = "module." + key if "module." + key in model_state_dict else key
        updated_lora_state_dict[new_key] = lora_state_dict[key]

    # Load LoRA parameters into the model
    model.load_state_dict(updated_lora_state_dict, strict=False)  # Allow missing keys
    
    return model

def final_answer(rag_output):
    # Model and LoRA paths
    base_model_name = "ingeol/kosaul_v0.2"
    lora_model_path = "./lora_weights"  # Path to the LoRA weight folder

    # # Load the base model
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load LoRA weights
    chkpoints = []
    for root, dirs, files in os.walk(lora_model_path):
        for file in files:
            chkpoints.append(os.path.join(root, file))

    for chkpoint in chkpoints:
        if chkpoint.endswith(".pth"):
            model = load_lora_state(model, chkpoint)
        else:
            raise ValueError(f"Unknown checkpoint format: {chkpoint}")

    # Inference function
    def generate_response(prompts):
        progress_bar = tqdm(
            enumerate(prompts),
            total=len(prompts),
            desc=f"Final processing...",
            leave=False,
        )
        responses = []
        for idx, prompt in progress_bar:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=256)
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(response)

        return responses

    # def generate_response_batch(prompts, batch_size=2):
    #     responses = []
        
    #     for i in tqdm(range(0, len(prompts), batch_size), desc="Final processing..."):
    #         batch = prompts[i : i + batch_size]
    #         inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            
    #         with torch.no_grad():
    #             outputs = model.generate(**inputs, max_new_tokens=256)
            
    #         responses.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
    #     return responses

    # Run test
    prompts = prompt_processing(rag_output)
    responses = generate_response(prompts)
    output_file = "./results/final_output.json"
    for idx, response in enumerate(responses):
        log_args(output_file, idx=idx, responses=response)
    print(response)

    return output_file
