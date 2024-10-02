import argparse
import math
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers.trainer import get_scheduler
from datasets import Dataset
from openrlhf.datasets import RewardDataset
from openrlhf.models import Actor
from openrlhf.trainer import DPOTrainer
from openrlhf.utils import prepare_dataset, get_strategy, get_tokenizer
from openrlhf.datasets.utils import read_all_shard_and_evaluate
import re

def remove(str, map, direction):
    if direction == 0:
        if str.startswith(map):
            str = str[len(map):]
    elif direction == 1:
        if str.endswith(map):
            str = str[:-len(map)]
    return str

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # prepare for data and dataset
    train_data, eval_data, test_data = prepare_dataset(args.dataset, strategy, ratio=args.ratio)
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_eval_samples, len(eval_data))))

    if args.label != 'None':
        test_id2acc = read_all_shard_and_evaluate(args.label)
        dataset_list = [train_data, eval_data, test_data]
        for idx, data in enumerate(dataset_list):
            new_data = {'tag':[],'test_id':[],'chosen':[],'rejected':[],'conversations':[],'count':[],'type':[]}
            for i in range(len(data)):
                if data[i]['test_id'] not in test_id2acc.keys():
                    continue
                for k in ['tag', 'test_id',]:
                    new_data[k].append(data[i][k])
                new_data['count'].append(1)
                new_data['type'].append('None')
                # chosen = 'r1' if test_id2acc[data[i]['test_id']] == 1 else 'r2'
                # rejected = 'r2' if test_id2acc[data[i]['test_id']] == 1 else 'r1'
                chosen = 'r2' if test_id2acc[data[i]['test_id']] == 1 else 'r1'
                rejected = 'r1' if test_id2acc[data[i]['test_id']] == 1 else 'r2'
                new_data['chosen'].append(data[i][chosen])
                new_data['rejected'].append(data[i][rejected])
                new_data['conversations'].append([{'from':'human', 'value':data[i]['prompt']}])
            dataset_list[idx] = Dataset.from_dict(new_data)
        train_data, eval_data, test_data = dataset_list
    
    if len(args.filter) > 1:
        if 'length' in args.filter:
            new_data = {'tag':[],'test_id':[],'chosen':[],'rejected':[],'conversations':[],'count':[],'type':[]}
            ratio = 1 if ',' not in args.filter else float(args.filter.split(',')[-1]) 
            for i in range(len(train_data)):
                answers = [re.findall(r'回答1：(.*?)回答2', train_data[i]['conversations'][0]['value'],  re.DOTALL)[0].strip(), remove(re.findall(r'回答2：(.*)', train_data[i]['conversations'][0]['value'],  re.DOTALL)[0].strip(), '<\|im_end\|>', direction=1)]
                chosen = train_data[i]['chosen_id'] - 1
                if ratio == -1: # ratio == -1 时直接取相反的样本
                    if len(answers[chosen]) * 2 < len(answers[1-chosen]):
                        for k in ['tag', 'test_id','count','type','chosen','rejected','conversations']:
                            new_data[k].append(train_data[i][k])
                else:
                    if len(answers[chosen]) > len(answers[1-chosen]) * 2 or i >= len(train_data) * ratio:
                        for k in ['tag', 'test_id','count','type','chosen','rejected','conversations']:
                            new_data[k].append(train_data[i][k])
            train_data = Dataset.from_dict(new_data)
        if 'no_critic' in args.filter:
            new_data = {'tag':[], 'test_id':[],'chosen':[],'rejected':[],'conversations':[],'count':[],'type':[]}
            for i in range(len(train_data)):
                for k in ['tag', 'test_id', 'count', 'type', 'conversations']:
                    new_data[k].append(train_data[i][k])
                new_data['chosen'].append(f'{{更好的回答: {train_data[i]["chosen_id"]}}}')
                new_data['rejected'].append(f'{{更好的回答: {3-train_data[i]["chosen_id"]}}}')
            train_data = Dataset.from_dict(new_data)
        if 'best_of_n' in args.filter:
            new_data = {'tag':[], 'test_id':[],'chosen':[],'rejected':[],'conversations':[],'count':[],'type':[], 'pair_type':[]}
            for i in range(len(train_data)):
                if train_data[i]['pair_type'] == 'best_of_n':
                    for k in ['tag', 'test_id', 'count', 'type', 'conversations','pair_type','chosen','rejected']:
                        new_data[k].append(train_data[i][k])
            train_data = Dataset.from_dict(new_data)

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    # import pdb
    # pdb.set_trace()

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # load weights for ref model
    ref_model = Actor(
        args.pretrain if args.ref_pretrain == '' else args.ref_pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
    )
    if args.ref_offload:
        ref_model._offload = True

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    train_dataset = RewardDataset(
        train_data, tokenizer, args.max_len, strategy, is_dpo=True, raw = 'p2' in args.dataset
    )
    eval_dataset = RewardDataset(
        eval_data, tokenizer, args.max_len, strategy, is_dpo=True, raw = 'p2' in args.dataset
    )

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # strategy prepare
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
        # strategy.load_checkpoint(args.save_path + '/rm_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        beta=args.beta,
        max_epochs=args.max_epochs,
    )

    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=100000000000)
    parser.add_argument("--max_eval_samples", type=int, default=100000000000)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

    # custom dataset key name
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default=None)
    parser.add_argument("--rejected_key", type=str, default=None)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument("--label", type=str, default="None")
    parser.add_argument("--filter", type=str, default="None")
    parser.add_argument("--sft_loss", type=float, default=0.0)
    parser.add_argument("--sft_loss_float", type=float, default=1.0)
    parser.add_argument("--dpo_loss", type=float, default=1.0)
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--ref_pretrain", type=str, default='')

    args = parser.parse_args()
    train(args)
