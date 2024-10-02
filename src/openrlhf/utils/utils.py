import os
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

from openrlhf.utils import DeepspeedStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        strategy.print(f"dataset: {dataset}")
        # local dir with python script or common local file
        if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset
                data_type = os.path.splitext(files)[1][1:]
            else:
                path = Path(dataset)
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                strategy.print(f"script: {script}")
                strategy.print(f"files: {files}")
                # For dir, follow python script or first file type
                data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
            # reformat data type
            if data_type in ["json", "jsonl"]:
                data_type = "json"
            elif data_type == "txt":
                data_type = "text"
            elif data_type.endswith(".py"):
                # load local dir with python script
                files = None
            if data_type.endswith(".py"):
                strategy.print(f"load {dataset} with script {data_type}")
            else:
                strategy.print(f"load {files} from {dataset}")
            data = load_dataset(data_type, data_files=files)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip())
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]
            data = load_dataset(dataset)
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data:
            train_data_list.append(data["train"].select(range(min(max_count, len(data["train"])))))
        else:
            train_data_list.append(data.select(range(min(max_count, len(data)))))  # train will contains eval? TODO

        if return_eval:
            if "test" in data:
                eval_data = data["test"].select(range(min(int(max_count * 0.1), len(data["test"]))))
            elif "validation" in data:
                eval_data = data["validation"].select(range(min(int(max_count * 0.1), len(data["validation"]))))
            elif "train" in data:
                eval_data = data["train"].select(range(min(int(max_count * 0.1), int(len(data["train"]) * 0.01))))
            else:
                eval_data = data.select(range(min(int(max_count * 0.1), int(len(data) * 0.001))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset

def prepare_single_dataset(data_addr, strategy, max_count=5000000):
    data = load_from_disk(data_addr)
    test_data = None
    train_dataset = data["train"].select(range(min(max_count, len(data["train"]))))
    strategy.print(f"Train data: {len(train_dataset)}")
    if "test" in data:
        test_data = data["test"].select(range(min(int(max_count * 0.1), len(data["test"]))))
        strategy.print(f"Test data: {len(test_data)}")
    if "validation" in data:
        eval_data = data["validation"].select(range(min(int(max_count * 0.1), len(data["validation"]))))
        strategy.print(f"Validation data: {len(eval_data)}")
    else:
        eval_data = data["train"].select(range(min(int(max_count * 0.1), int(len(data["train"]) * 0.01))))
        strategy.print(f"Using some of train data as validation: {len(eval_data)}")

    return train_dataset, eval_data, test_data

def ratio_datasets(dataset, dataset2, ratio):
    if int(ratio*len(dataset)) < len(dataset2):
        return concatenate_datasets([dataset, dataset2.select(range(int(ratio*len(dataset))))])
    else:
        return concatenate_datasets([dataset.select(range(int(len(dataset2)/ratio))), dataset2])

def prepare_dataset(data_addr, strategy, max_count=5000000, ratio=1):
    if ',' in data_addr:
        data_addrs = data_addr.split(',')
        train_dataset, eval_data, test_data = None, None, None
        for data_addr in data_addrs:
            if train_dataset is None:
                train_dataset, eval_data, test_data = prepare_single_dataset(data_addr, strategy, max_count=max_count)
            else:
                train_dataset2, eval_data2, test_data2 = prepare_single_dataset(data_addr, strategy, max_count=max_count)
                train_dataset, eval_data, test_data = ratio_datasets(train_dataset, train_dataset2, ratio), ratio_datasets(eval_data, eval_data2, ratio), test_data
                # train_dataset, eval_data, test_data = ratio_datasets(train_dataset, train_dataset2, ratio), ratio_datasets(eval_data, eval_data2, ratio), ratio_datasets(test_data, test_data2, ratio)
        return train_dataset, eval_data, test_data
    else:
        return prepare_single_dataset(data_addr, strategy, max_count=5000000)

    
