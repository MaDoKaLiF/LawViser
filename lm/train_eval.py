import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
from tqdm import tqdm
import gc
from types import SimpleNamespace
import time

from utils.utils import get_model_tokenizer, get_dataloader, get_optimizer_scheduler, setup, cleanup, save_model, save_lora_model
from utils.utils_test import create_labels, test, log_args

def train(args, model, rank, world_size, train_loader, test_loader, tokenizer, optimizer, scheduler, epoch, sampler=None):
    model.train()
    model_forward = model.custom_forward if hasattr(model, 'custom_forward') else model.forward # custom forward if exists

    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    # Wrap the dataloader in tqdm for progress tracking
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Pretrain | Epoch {epoch} [Rank {rank}]" if args.pretrain else f"Epoch {epoch} [Rank {rank}]",
        position=rank,
        leave=False,
        disable=(rank != 0),  # Only show progress bar on rank 0
    )

    log_interval = max(1, round(len(train_loader) / args.log_divisor))
    if args.save_divisor != 0:
        save_interval = max(1, round(len(train_loader) / args.save_divisor))

    for batch_idx, data in progress_bar:
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = create_labels(input_ids, rank, tokenizer)

        input_ids, attention_mask, labels = input_ids.to(rank), attention_mask.to(rank), labels.to(rank)   

        output = model_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss / args.grad_accum  # Scale loss by accumulation steps
        loss.backward()
          
        if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients
            
        scheduler.step()  # Update learning rate
        # Track loss and sample count
        ddp_loss[0] += loss.item() * args.grad_accum  # Unscaled loss
        ddp_loss[1] += len(input_ids)  # Track total samples

        # saving model
        if args.save_divisor != 0:
            if (batch_idx+1) % save_interval == 0 or (batch_idx + 1) == len(train_loader):
                if args.lora:
                    save_lora_model(args, epoch, batch_idx, model, rank)
                else:
                    save_model(args, model, rank)
        elif args.save_divisor == 0:
            if epoch == args.epochs and (batch_idx + 1) == len(train_loader):
                if args.lora:
                    save_lora_model(args, epoch, batch_idx, model, rank)
                else:
                    save_model(args, model, rank)

        # logging loss
        if (batch_idx+1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            if rank == 0:
                avg_loss = (ddp_loss[0] / ddp_loss[1]).item()
                progress_bar.set_postfix({"Loss": avg_loss})
                trainfile = f"{args.exp_name}/{args.log_save}/train_log.json"
                log_args(trainfile, epoch=epoch, step=batch_idx, loss=avg_loss)

            dist.barrier()
            ddp_loss = torch.zeros(2).to(rank)  # Reset loss tracking


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # Set the device for this rank

    args.evaluate = False
    
    if args.method == "vanilla":
        model, tokenizer = get_model_tokenizer(args, args.model_name, rank)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    train_loader, sampler_train, test_loader, sampler_test = get_dataloader(args, model, tokenizer, rank, world_size)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    optimizer, scheduler = get_optimizer_scheduler(args, model, train_loader)
    
    init_start_event.record()
    for epoch in range(1, args.max_eval_epoch + 1):
        train(args, model, rank, world_size, train_loader, test_loader, tokenizer, optimizer, scheduler, epoch, sampler=sampler_train)
    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='FSDP Finetune')
    parser.add_argument("--config", type=str, required=True, help="Config file location")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, required=True,help='random seed')
    parser.add_argument('--method', type=str, default="vanilla", help='train method default: vanilla')
    parser.add_argument('--call_lora', type=bool, default=False, help='if lora, just use the config since initialized is done.')
    parser.add_argument('--use_auxiliary', type=bool, default=False, help='if True, pretrain with auxiliary dataset')
    parser.add_argument('--pretrain', type=bool, default=False, help='signal for pretrain')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    params = json.load(open(args.config))

    if args.call_lora: # if called from train_eval_lora, initialized is done just use the loaded config
        args = SimpleNamespace(**params)
        
    else:
        args.batch_size = params["batch_size"]
        args.test_batch_size = params["test_batch_size"]
        args.grad_accum = params["grad_accumulation"]
        args.gen_length = params["gen_length"]
        args.eval_save = params["eval_save"]
        args.log_save = params["log_save"]
        args.ckpt_save = params["ckpt_save"]
        args.log_divisor = params["log_divisor"]
        args.save_divisor = params["save_divisor"]
        args.eval_divisor = params["eval_divisor"]
        args.toy_eval_divisor = params.get("toy_eval_divisor", 5)
        args.task = params["task"]
        args.epochs = params["epochs"]
        args.min_eval_epoch = params["min_eval_epoch"]
        args.max_eval_epoch = params["max_eval_epoch"]
        args.lr = params["lr"]
        args.weight_decay = params["weight_decay"]
        args.warm_up_ratio = params["warm_up_ratio"]
        args.warm_up_steps = params["warm_up_steps"]
        args.scheduler = params["scheduler"] # cosine or linear
        args.optimizer = params["optimizer"] # AdamW or Adam
        args.precision = params["precision"] # bf16 or fp16
        args.model_name = params["model_name"]
        args.n_shot = params["n_shot"]
        args.self_consistency = params["self_consistency"]
        args.max_length = params["max_length"]
        args.lora = params.get("lora", False)
        if args.use_auxiliary:
            args.epochs_pretrain = params["epochs_pretrain"]
        if args.method != "vanilla": # Load method specific args
            with open(f'configs_method/{args.method}.json', 'r') as method_json_file:
                method_json = json.load(method_json_file)
            for key, value in method_json.items():
                setattr(args, key, value)

        exp_name_base = params["exp_name"]
        args.exp_name = f"{exp_name_base}_{args.method}_{args.seed}"

        if os.path.exists(args.exp_name):
            while True:
                args.exp_name = args.exp_name + "_"
                if os.path.exists(args.exp_name) == False: break

        os.makedirs(args.exp_name, exist_ok=True)
        os.makedirs(f"{args.exp_name}/{args.log_save}", exist_ok=True)
        os.makedirs(f"{args.exp_name}/{args.ckpt_save}", exist_ok=True)

        configfile = f"{args.exp_name}/{args.log_save}/config.json"

        # Dump the dictionary to a JSON file
        args_dict = vars(args)
        with open(configfile, "w") as json_file:
            json.dump(args_dict, json_file, indent=4)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
