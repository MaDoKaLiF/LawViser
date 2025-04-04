import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType, FullStateDictConfig

from transformers import AutoModelForCausalLM
from tqdm import tqdm
import gc
import re
from collections import Counter
import time

from utils.utils import log_args


def create_labels(input_ids, rank, tokenizer):
    a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]  # Encode "A"
    colon_token_id = tokenizer.encode(":", add_special_tokens=False)[0]  # Encode ":"

    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone().to(torch.int64)

    for i in range(batch_size):
        last_a_colon_position = -1
        for j in range(seq_len - 1):
            if input_ids[i, j] == a_token_id and input_ids[i, j + 1] == colon_token_id:
                last_a_colon_position = j

        if last_a_colon_position != -1:
            labels[i, :last_a_colon_position + 2] = -100  # Mask up to and including "A:"
        else:
            labels[i, :] = -100  # Mask entire sequence if "A:" is not found
    return labels


def test_metric(args, predictions, answers):
    correct, total = 0, 0

    for idx, (pred, ref) in enumerate(zip(predictions, answers), 1):
        with open(f"{args.exp_name}/{args.log_save}/debug_predictions.log", "a") as log_file:  # for debug
            log_file.write(f"Prediction: {pred}\nReference: {ref}\n\n")

        q_start_idx = pred.find("Q: ")
        if q_start_idx != -1:
            pred = pred[:q_start_idx]  # Keep only the text before the first "Q: "

        if args.task == "cladder":  # First yes / no as a answer
            matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
            pred_answer = matches[-1].lower() if matches else None
            ref_answer = ref
            if pred_answer == ref.lower():
                correct += 1

        elif args.task == "logiqa":
            matches = re.findall(r"\s*(\d+)", pred)
            pred_answer = int(matches[-1]) if matches else None
            ref_answer = ref
            if pred_answer == ref:
                correct += 1

        elif args.task == "arc_challenge":
            matches = list(re.finditer(r"\b(A|B|C|D)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None
            ref_answer = ref
            if pred_answer == ref:
                correct += 1
        elif args.task == "anli_r1":
            matches = list(re.finditer(r"\b(0|1|2)\b", pred))
            pred_answer = matches[0].group(1) if matches else None
            ref_answer = str(ref)
            if pred_answer == ref_answer:
                correct += 1

        elif args.task == "aqua_rat":
            matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None
            ref_answer = ref
            if pred_answer == ref:
                correct += 1

        elif args.task == "cqa":
            matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None
            ref_answer = ref
            if pred_answer == ref:
                correct += 1

        elif args.task in ["asdiv", "svamp", "numglue"]:
            matches = re.findall(r"-?\d+\.?\d*", pred)
            pred_answer = matches[-1] if matches else None
            ref_match = re.search(r"-?\d+\.?\d*", ref)
            ref_answer = ref_match.group(0) if ref_match else None
            if pred_answer == ref_answer:
                correct += 1

        elif args.task == "gsm8k":
            # 1. Check if prediction has #### format
            hash_match = re.search(r"####\s*([-+]?\d*\.?\d+)", pred)

            if hash_match:
                # If #### format is found, extract number after first ####
                pred_answer = hash_match.group(1).strip()
            else:
                # If no #### format, check if prediction is just a number
                pred = pred.strip()
                if re.match(r"^[-+]?\d*\.?\d+$", pred):
                    # If prediction is just a number, use it directly
                    pred_answer = pred
                else:
                    # Otherwise, extract the last number from the text
                    matches = re.findall(r"[-+]?\d*\.?\d+", pred)
                    pred_answer = matches[-1] if matches else None

            # Reference answer pattern remains the same
            ref_match = re.search(r"####\s*([-+]?\d*\.?\d+)", ref)
            ref_answer = ref_match.group(1).strip() if ref_match else None

            if pred_answer == ref_answer:
                correct += 1

        else:
            raise NotImplementedError

        # For debug output
        target_save = f"{args.exp_name}/{args.log_save}/pred.txt"
        with open(target_save, 'a+') as new_train_f:
            print(f"{pred} \n pred: {pred_answer} gt: {ref_answer}", file=new_train_f, end="\n\n")

        total += 1

    return correct, total


def get_DDP_copy_model(args, model, rank):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    # Get FULL state dict
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        full_state_dict = model.state_dict()
    dist.barrier()

    if args.model_name == "Qwen/Qwen2.5-3B":
        test_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True,
                                                          attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.bfloat16)
    else:
        test_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    test_model.load_state_dict(full_state_dict)  # Load FULL state dict
    test_model = test_model.to(rank)
    test_model = DDP(test_model, device_ids=[rank], output_device=rank)  # Wrap with DDP

    # NOTE I tried FSDP but FSDP had difficulty emptying model and cache from VRAM. Using DDP is OK in inference, if model can fit in single GPU. Modify test_batch_size if OOM
    return test_model


def test(args, model, rank, world_size, train_loader, test_loader, tokenizer, gen_length, target_save, cur_epoch,
         log_epoch):
    test_start_event = torch.cuda.Event(enable_timing=True)
    test_end_event = torch.cuda.Event(enable_timing=True)
    model_start_event = torch.cuda.Event(enable_timing=True)
    model_end_event = torch.cuda.Event(enable_timing=True)
    model_start_time = time.time()
    if args.evaluate:  # When called from inference.py
        test_model = model
    else:
        test_model = get_DDP_copy_model(args, model, rank)
    test_model.eval()
    model_elapsed_time = time.time() - model_start_time
    if rank == 0:
        time_log_file = f"{args.exp_name}/{args.log_save}/elapsed_time_log.json"
        log_args(time_log_file, epoch=log_epoch, model_load_time=model_elapsed_time)

    test_start_time = time.time()
    # Access the underlying model if wrapped with FSDP
    generate_fn = test_model.module.generate if hasattr(test_model, 'module') else test_model.generate

    eval_progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=f"Eval [Rank {rank}]",
        position=rank + 1,
        leave=False,
        disable=(rank != 0),  # Only show progress bar on rank 0
    )

    # test loss
    loss_sum = 0.0
    tokens_sum = 0
    # accuracy
    correctsum, totalsum = 0, 0
    n_shot_correctsum, n_shot_totalsum = 0, 0
    sc_correctsum, sc_totalsum = 0, 0

    # Create log files for different prediction types
    if rank == 0:
        # Create log files for different prediction types
        base_pred_file = f"{args.exp_name}/{args.log_save}/base_predictions.txt"
        n_shot_pred_file = f"{args.exp_name}/{args.log_save}/n_shot_predictions.txt"
        sc_pred_file = f"{args.exp_name}/{args.log_save}/sc_predictions.txt"

        # Clear existing files if they exist
        for file_path in [base_pred_file, n_shot_pred_file, sc_pred_file]:
            with open(file_path, 'w') as f:
                f.write(f"Epoch: {log_epoch}\n\n")

    with torch.no_grad():
        for batch_i, data in eval_progress_bar:
            input_ids = data["input_ids"].to(rank)
            attention_mask = data["attention_mask"].to(rank)
            answers = data["answers"]  # List of correct answers

            ## Test loss
            labels = create_labels(input_ids, rank, tokenizer)
            labels.to(rank)

            # forward pass (cross-entropy)
            loss_outputs = test_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = loss_outputs.loss

            batch_token_count = (labels != -100).sum().item()

            loss_sum += loss.item() * batch_token_count
            tokens_sum += batch_token_count

            ## Accuracy
            # pred generation
            outputs = generate_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + gen_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # greedy decoding
                temperature=None,
                top_k=None,
                top_p=None
            )

            generated_tokens = outputs[:, input_ids.shape[-1]:]
            predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Decode the original input for logging
            if rank == 0:
                original_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            # Compute correct and total using test_metric
            correct, total = test_metric(args, predictions, answers)
            correctsum += correct
            totalsum += total

            # Log base predictions
            if rank == 0:
                with open(base_pred_file, 'a') as f:
                    for idx, (inp, pred, ans) in enumerate(zip(original_inputs, predictions, answers)):
                        f.write(f"Example {batch_i * args.test_batch_size + idx + 1}\n")
                        f.write(f"Input: {inp}\n")
                        f.write(f"Prediction: {pred}\n")
                        f.write(f"Answer: {ans}\n")
                        # Extract the model's final answer for clearer comparison
                        if args.task == "cladder":
                            matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
                            pred_answer = matches[-1].lower() if matches else "None"
                        elif args.task == "anli_r1":
                            matches = list(re.finditer(r"\b(0|1|2)\b", pred))
                            pred_answer = matches[0].group(1) if matches else "None"

                        elif args.task == "aqua_rat":
                            matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                            pred_answer = matches[-1].group(1) if matches else "None"

                        elif args.task == "logiqa":
                            matches = re.findall(r"\s*(\d+)", pred)
                            pred_answer = matches[-1] if matches else "None"
                        elif args.task in ["arc_challenge", "cqa"]:
                            matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                            pred_answer = matches[-1].group(1) if matches else "None"
                        else:  # GSM8K, ASDIV, SVAMP
                            matches = re.findall(r"-?\d+\.?\d*", pred)
                            pred_answer = matches[-1] if matches else "None"
                        f.write(f"Extracted Answer: {pred_answer}\n")
                        f.write(
                            f"Correct: {pred_answer == ans if args.task != 'gsm8k' else pred_answer == ans.split('####')[1].strip()}\n")
                        f.write("\n" + "-" * 50 + "\n\n")

            # log flops
            eos_token_id = tokenizer.eos_token_id
            actual_lengths = 0
            actual_c = 0
            for seq in generated_tokens:
                eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    actual_length = eos_positions[0].item() + 1
                else:
                    actual_length = seq.size(0)
                actual_lengths += actual_length
                actual_c += 1

            actual_gen_len = actual_lengths
            flops_log_file = f"{args.exp_name}/{args.log_save}/flops_test_log_{rank}.json"
            log_args(flops_log_file, epoch=cur_epoch, idx=batch_i, split="test", batch=actual_c,
                     input=actual_c * input_ids.size(1), output=actual_gen_len)

            # n-shot
            if args.n_shot > 0:
                decoded_input_ids = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                combined_prompts = [f"{args.n_shot_input}\n{decoded_input}" for decoded_input in decoded_input_ids]
                tokenized = tokenizer(combined_prompts, return_tensors="pt", padding="max_length", truncation=True,
                                      max_length=args.max_length + args.prompt_tokenized_len)

                # log_truncation_warnings(args, combined_prompts, tokenizer, is_n_shot=True)

                n_shot_input_ids = tokenized.input_ids.to(rank)
                n_shot_attention_mask = tokenized.attention_mask.to(rank)

                n_shot_outputs = generate_fn(
                    input_ids=n_shot_input_ids,
                    attention_mask=n_shot_attention_mask,
                    max_length=n_shot_input_ids.size(1) + gen_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,  # greedy decoding
                    temperature=None,
                    top_k=None,
                    top_p=None
                )

                n_shot_generated_tokens = n_shot_outputs[:, n_shot_input_ids.shape[-1]:]
                n_shot_predictions = tokenizer.batch_decode(n_shot_generated_tokens, skip_special_tokens=True)

                # Log n-shot predictions
                if rank == 0:
                    with open(n_shot_pred_file, 'a') as f:
                        for idx, (inp, pred, ans) in enumerate(zip(original_inputs, n_shot_predictions, answers)):
                            f.write(f"Example {batch_i * args.test_batch_size + idx + 1}\n")
                            f.write(f"Input: {inp}\n")
                            f.write(f"N-shot Prediction: {pred}\n")
                            f.write(f"Answer: {ans}\n")
                            # Extract the model's final answer for clearer comparison
                            if args.task == "cladder":
                                matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
                                pred_answer = matches[-1].lower() if matches else "None"
                            elif args.task == "logiqa":
                                matches = re.findall(r"\s*(\d+)", pred)
                                pred_answer = matches[-1] if matches else "None"
                            elif args.task in ["arc_challenge", "cqa", "aqua_rat"]:
                                matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                                pred_answer = matches[-1].group(1) if matches else "None"
                            elif args.task == "anli_r1":
                                matches = list(re.finditer(r"\b(0|1|2)\b", pred))
                                pred_answer = matches[0].group(1) if matches else "None"

                            else:  # GSM8K, ASDIV, SVAMP
                                matches = re.findall(r"-?\d+\.?\d*", pred)
                                pred_answer = matches[-1] if matches else "None"
                            f.write(f"Extracted Answer: {pred_answer}\n")
                            f.write(
                                f"Correct: {pred_answer == ans if args.task != 'gsm8k' else pred_answer == ans.split('####')[1].strip()}\n")
                            f.write("\n" + "-" * 50 + "\n\n")

                n_shot_correct, n_shot_total = test_metric(args, n_shot_predictions, answers)
                n_shot_correctsum += n_shot_correct
                n_shot_totalsum += n_shot_total

            # self-consistency
            if args.self_consistency > 0:
                sc_predictions_list = []
                for sc_idx in range(args.self_consistency):
                    outputs = generate_fn(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + gen_length,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.7,
                        top_k=40,
                        # top_p=0.9
                    )
                    generated_tokens = outputs[:, input_ids.shape[-1]:]
                    predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    sc_predictions_list.append(predictions)

                    # Log self-consistency predictions
                    if rank == 0:
                        with open(sc_pred_file, 'a') as f:
                            for idx, (inp, pred, ans) in enumerate(zip(original_inputs, predictions, answers)):
                                f.write(f"Example {batch_i * args.test_batch_size + idx + 1}, SC Sample {sc_idx + 1}\n")
                                f.write(f"Input: {inp}\n")
                                f.write(f"SC Prediction: {pred}\n")
                                f.write(f"Answer: {ans}\n")
                                # Extract the model's final answer for clearer comparison
                                if args.task == "cladder":
                                    matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
                                    pred_answer = matches[-1].lower() if matches else "None"
                                elif args.task == "logiqa":
                                    matches = re.findall(r"\s*(\d+)", pred)
                                    pred_answer = matches[-1] if matches else "None"
                                elif args.task in ["arc_challenge", "cqa", "aqua_rat"]:
                                    matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                                    pred_answer = matches[-1].group(1) if matches else "None"
                                elif args.task == "anli_r1":
                                    matches = list(re.finditer(r"\b(0|1|2)\b", pred))
                                    pred_answer = matches[0].group(1) if matches else "None"

                                else:  # GSM8K, ASDIV, SVAMP
                                    matches = re.findall(r"-?\d+\.?\d*", pred)
                                    pred_answer = matches[-1] if matches else "None"
                                f.write(f"Extracted Answer: {pred_answer}\n")
                                f.write(
                                    f"Correct: {pred_answer == ans if args.task != 'gsm8k' else pred_answer == ans.split('####')[1].strip()}\n")
                                f.write("\n" + "-" * 50 + "\n\n")

                # Log majority vote results
                if rank == 0:
                    majority_votes = calculate_majority_votes(sc_predictions_list, args)
                    with open(sc_pred_file, 'a') as f:
                        for idx, (vote, ans) in enumerate(zip(majority_votes, answers)):
                            f.write(f"Example {batch_i * args.test_batch_size + idx + 1}, MAJORITY VOTE\n")
                            f.write(f"Final SC Vote: {vote}\n")
                            f.write(f"Answer: {ans}\n")
                            is_correct = vote == ans
                            if args.task == "gsm8k":
                                ref_match = re.search(r"####\s*([-+]?\d*\.?\d+)", ans)
                                ref_answer = ref_match.group(1).strip() if ref_match else None
                                is_correct = vote == ref_answer
                            f.write(f"Correct: {is_correct}\n")
                            f.write("\n" + "=" * 50 + "\n\n")

                sc_correct, sc_total = get_majority_vote(sc_predictions_list, answers, args)
                sc_correctsum += sc_correct
                sc_totalsum += sc_total

    # Convert to tensors with integer type for counts
    loss_tensor = torch.tensor(loss_sum, dtype=torch.float32, device=rank)
    tokens_tensor = torch.tensor(tokens_sum, dtype=torch.float32, device=rank)
    correct_tensor = torch.tensor(correctsum, dtype=torch.int64, device=rank)
    total_tensor = torch.tensor(totalsum, dtype=torch.int64, device=rank)

    # Gather correct and total counts from all processes
    gathered_loss = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    gathered_tokens = [torch.zeros_like(tokens_tensor) for _ in range(world_size)]
    gathered_correct = [torch.zeros_like(correct_tensor) for _ in range(world_size)]
    gathered_total = [torch.zeros_like(total_tensor) for _ in range(world_size)]

    dist.all_gather(gathered_loss, loss_tensor)
    dist.all_gather(gathered_tokens, tokens_tensor)
    dist.all_gather(gathered_correct, correct_tensor)
    dist.all_gather(gathered_total, total_tensor)

    if args.n_shot > 0:
        n_shot_correct_tensor = torch.tensor(n_shot_correctsum, dtype=torch.int64, device=rank)
        n_shot_total_tensor = torch.tensor(n_shot_totalsum, dtype=torch.int64, device=rank)

        n_shot_gathered_correct = [torch.zeros_like(n_shot_correct_tensor) for _ in range(world_size)]
        n_shot_gathered_total = [torch.zeros_like(n_shot_total_tensor) for _ in range(world_size)]
        dist.all_gather(n_shot_gathered_correct, n_shot_correct_tensor)
        dist.all_gather(n_shot_gathered_total, n_shot_total_tensor)

    if args.self_consistency > 0:
        sc_correct_tensor = torch.tensor(sc_correctsum, dtype=torch.int64, device=rank)
        sc_total_tensor = torch.tensor(sc_totalsum, dtype=torch.int64, device=rank)

        sc_gathered_correct = [torch.zeros_like(sc_correct_tensor) for _ in range(world_size)]
        sc_gathered_total = [torch.zeros_like(sc_total_tensor) for _ in range(world_size)]
        dist.all_gather(sc_gathered_correct, sc_correct_tensor)
        dist.all_gather(sc_gathered_total, sc_total_tensor)

    if rank == 0:
        total_loss = sum(t.item() for t in gathered_loss)
        total_tokens = sum(t.item() for t in gathered_tokens)
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        # Sum up all correct and total counts
        total_correct = sum(t.item() for t in gathered_correct)
        total_total = sum(t.item() for t in gathered_total)
        average_accuracy = total_correct / total_total if total_total > 0 else 0.0

        log_data = {"epoch": log_epoch, "loss": avg_loss, "accuracy": average_accuracy}

        if args.n_shot > 0:
            n_shot_total_correct = sum(t.item() for t in n_shot_gathered_correct)
            n_shot_total_total = sum(t.item() for t in n_shot_gathered_total)
            n_shot_average_accuracy = n_shot_total_correct / n_shot_total_total if n_shot_total_total > 0 else 0.0
            log_data["n_shot_accuracy"] = n_shot_average_accuracy

        if args.self_consistency > 0:
            sc_total_correct = sum(t.item() for t in sc_gathered_correct)
            sc_total_total = sum(t.item() for t in sc_gathered_total)
            sc_average_accuracy = sc_total_correct / sc_total_total if sc_total_total > 0 else 0.0
            log_data["sc_accuracy"] = sc_average_accuracy
            log_data["self_consistency_samples"] = args.self_consistency

        test_file = f"{args.exp_name}/{args.log_save}/eval_log.json"
        log_args(test_file, **log_data)
        print(f"[Evaluation] loss: {avg_loss:.4f}  Accuracy: {average_accuracy * 100:.2f}%")
        if args.n_shot > 0:
            print(f"{args.n_shot}-shot Evaluation Accuracy: {n_shot_average_accuracy * 100:.2f}%")
        if args.self_consistency > 0:
            print(
                f"Self-consistency ({args.self_consistency} samples) Evaluation Accuracy: {sc_average_accuracy * 100:.2f}%")

    # delete test_model and cache
    del test_model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    test_elapsed_time = time.time() - test_start_time
    if rank == 0:
        time_log_file = f"{args.exp_name}/{args.log_save}/elapsed_time_log.json"
        log_args(time_log_file, epoch=log_epoch, test_time=test_elapsed_time)

    return average_accuracy if rank == 0 else None


def calculate_majority_votes(sc_predictions_list, args):
    """Helper function to calculate the majority vote for each example in the batch"""
    # Transpose the list of lists to get all predictions for each example
    example_predictions = list(zip(*sc_predictions_list))
    majority_votes = []

    for predictions in example_predictions:
        vote_dict = {}
        for pred in predictions:
            q_start_idx = pred.find("Q: ")
            if q_start_idx != -1:
                pred = pred[:q_start_idx]  # Keep only the text before the first "Q: "

            # Extract answers differently based on the task
            if args.task == "cladder":  # last yes / no as a answer
                matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
                pred_answer = matches[-1].lower() if matches else None
            elif args.task == "logiqa":
                matches = re.findall(r"\s*(\d+)", pred)
                pred_answer = int(matches[-1]) if matches else None
            elif args.task == "arc_challenge":
                matches = list(re.finditer(r"\b(A|B|C|D)\b", pred))
                pred_answer = matches[-1].group(1) if matches else None
            elif args.task == "cqa":
                matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                pred_answer = matches[-1].group(1) if matches else None
            elif args.task == "anli_r1":
                matches = list(re.finditer(r"\b(0|1|2)\b", pred))
                pred_answer = matches[0].group(1) if matches else None

            elif args.task == "aqua_rat":
                matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                pred_answer = matches[-1].group(1) if matches else None

            else:  # GSM8K, ASDIV, SVAMP
                matches = re.findall(r"-?\d+\.?\d*", pred)
                pred_answer = matches[-1] if matches else None

            if pred_answer is not None:
                vote_dict[str(pred_answer)] = vote_dict.get(str(pred_answer), 0) + 1

        # Select the answer with the highest votes as the final answer
        if vote_dict:
            final_answer = max(vote_dict.items(), key=lambda x: x[1])[0]
            if args.task == "logiqa":
                final_answer = int(final_answer)
            majority_votes.append(final_answer)
        else:
            majority_votes.append("No majority found")

    return majority_votes


def get_majority_vote(predictions_list, answers, args):
    correct, total = 0, 0
    for sample_predictions, ref in zip(zip(*predictions_list), answers):
        # Aggregate voting results for each sample
        vote_dict = {}
        for pred in sample_predictions:
            q_start_idx = pred.find("Q: ")
            if q_start_idx != -1:
                pred = pred[:q_start_idx]  # Keep only the text before the first "Q: "

            # Extract answers differently based on the task
            if args.task == "cladder":  # last yes / no as a answer
                matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
                pred_answer = matches[-1].lower() if matches else None
            elif args.task == "logiqa":
                matches = re.findall(r"\s*(\d+)", pred)
                pred_answer = int(matches[-1]) if matches else None
            elif args.task == "arc_challenge":
                matches = list(re.finditer(r"\b(A|B|C|D)\b", pred))
                pred_answer = matches[-1].group(1) if matches else None
            elif args.task == "cqa":
                matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                pred_answer = matches[-1].group(1) if matches else None
            elif args.task == "anli_r1":
                matches = list(re.finditer(r"\b(0|1|2)\b", pred))
                pred_answer = matches[0].group(1) if matches else None
            elif args.task == "aqua_rat":
                matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                pred_answer = matches[-1].group(1) if matches else None


            else:  ## GSM8K, ASDIV, SVAMP
                matches = re.findall(r"-?\d+\.?\d*", pred)
                pred_answer = matches[-1] if matches else None

            if pred_answer is not None:
                vote_dict[str(pred_answer)] = vote_dict.get(str(pred_answer), 0) + 1

        # Select the answer with the highest votes as the final answer
        if vote_dict:
            final_answer = max(vote_dict.items(), key=lambda x: x[1])[0]
            if args.task == "logiqa":
                final_answer = int(final_answer)
                if final_answer == ref:
                    correct += 1
            elif args.task == "cladder":
                if final_answer.lower() == ref.lower():
                    correct += 1
            elif args.task in ["arc_challenge", "cqa"]:
                if final_answer == ref:
                    correct += 1
            elif args.task == "aqua_rat":
                if final_answer and final_answer == ref:
                    correct += 1
            elif args.task == "anli_r1":
                if final_answer and str(final_answer) == str(ref):
                    correct += 1
            elif args.task in ["asdiv", "svamp", "numglue"]:
                ref_match = re.search(r"-?\d+\.?\d*", ref)
                ref_answer = ref_match.group(0) if ref_match else None
                if final_answer.strip() == str(ref_answer).strip():
                    correct += 1
            else:
                ref_match = re.search(r"####\s*([-+]?\d*\.?\d+)", ref)
                ref_answer = ref_match.group(1).strip() if ref_match else None
                if final_answer.strip() == str(ref_answer).strip():
                    correct += 1

        total += 1
    return correct, total
