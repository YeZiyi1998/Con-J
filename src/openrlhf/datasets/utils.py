import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_from_disk
import json
import random

def load_data(path, is_test=False, is_arrow=False, max_samples=-1):
    if is_arrow:
        data = load_from_disk(path)
        if is_test == False and max_samples != -1:
            data = data.shuffle(2021)
            data = data.select(range(0, min(max_samples, len(data))))
    else:
        data = []
        with open(path, "r") as f:
            for line in f:
                line_json = json.loads(line)
                if line_json['acc'] == 1 or is_test:
                    data.append(line_json)
        if is_test == False and max_samples != -1:
            random.shuffle(data)
            data = data[:max_samples]
    return data

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

def exist_and_not_none(d, key):
    return key in d and d[key] is not None

def read_all_shard_and_evaluate(folder):
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import os,json
    pd.options.display.precision = 4

    acc_dict, acc_dict1, acc_dict2 = defaultdict(list), defaultdict(list), defaultdict(list)
    table, overall_acc = [], []
    overall_acc1, overall_acc2 = [], []
    test_id2acc = {}
    
    for fname in os.listdir(folder):
        if fname.endswith("json") or fname.endswith("jsonl"):
            with open(folder + "/" + fname, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    tag_k = 'tags' if 'tags' in data.keys() and len(data['tags']) > 0 else 'tag'
                    for i in range(len(data[tag_k])):
                        tag = data[tag_k][i]
                        if 'acc' in data.keys():
                            acc = data["acc"][i]
                            acc_dict[tag].append(acc)
                            overall_acc.append(acc)
                            if 'test_id' in data.keys():
                                test_id2acc[data['test_id'][i]] = acc
                        else:
                            if len(data["acc1"]) > 0:
                                acc_dict1[tag].append(data["acc1"][i])
                                overall_acc1.append(data["acc1"][i])
                            if len(data["acc2"]) > 0:
                                acc_dict2[tag].append(data["acc2"][i])
                                overall_acc2.append(data["acc2"][i])
  
    if len(overall_acc) > 0:
        sort_keys = sorted(list(acc_dict.keys()))
        for tag in sort_keys:
            accs = acc_dict[tag]
            table.append([tag, np.mean(accs), np.std(accs), len(accs)])
            # self.strategy.print(f"{tag:<20s}{np.mean(accs):.4f}")
        table.append(["Overall", np.mean(overall_acc), np.std(overall_acc), len(overall_acc)])
        # self.strategy.print(f"{str(Overall):<20s}{np.mean(overall_acc):.4f}")
        df = pd.DataFrame(table, columns=['Tag', 'mean', 'std', 'size'])
    else:
        sort_keys = sorted(list(acc_dict1.keys()))
        for tag in sort_keys:
            accs = acc_dict1[tag]
            accs2 = acc_dict2[tag]
            table.append([tag, np.mean(accs), np.std(accs), len(accs), np.mean(accs2), np.std(accs2)])
            # self.strategy.print(f"{tag:<20s}{np.mean(accs):.4f}")
        table.append(["Overall", np.mean(overall_acc1), np.std(overall_acc1), len(overall_acc1), np.mean(overall_acc2), np.std(overall_acc2)])
        # self.strategy.print(f"{str(Overall):<20s}{np.mean(overall_acc):.4f}")
        df = pd.DataFrame(table, columns=['Tag', 'mean', 'std', 'size', 'mean2', 'std2'])
    print(df)

    return test_id2acc
